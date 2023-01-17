import os

import cv2

import json

import random

import numpy as np

import mxnet as mx

import pandas as pd

import gluoncv as gcv

from multiprocessing import cpu_count

from multiprocessing.dummy import Pool





def load_dataset(root):

    csv = pd.read_csv(os.path.join(root, "train.csv"))

    data = {}

    for i in csv.index:

        key = csv["image_id"][i]

        bbox = json.loads(csv["bbox"][i])

        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], 0.0]

        if key in data:

            data[key].append(bbox)

        else:

            data[key] = [bbox]

    return sorted(

        [(k, os.path.join(root, "train", k + ".jpg"), v) for k, v in data.items()],

        key=lambda x: x[0]

    )



def load_image(path):

    with open(path, "rb") as f:

        buf = f.read()

    return mx.image.imdecode(buf)



def get_batches(dataset, batch_size, width=512, height=512, net=None, ctx=mx.cpu()):

    batches = len(dataset) // batch_size

    if batches * batch_size < len(dataset):

        batches += 1

    sampler = Sampler(width, height, net)

    with Pool(cpu_count() * 2) as p:

        for i in range(batches):

            start = i * batch_size

            samples = p.map(sampler, dataset[start:start+batch_size])

            stack_fn = [gcv.data.batchify.Stack()]

            pad_fn = [gcv.data.batchify.Pad(pad_val=-1)]

            if net is None:

                batch = gcv.data.batchify.Tuple(*(stack_fn + pad_fn))(samples)

            else:

                batch = gcv.data.batchify.Tuple(*(stack_fn * 6 + pad_fn))(samples)

            yield [x.as_in_context(ctx) for x in batch]



def gauss_blur(image, level):

    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))



def gauss_noise(image):

    for i in range(image.shape[2]):

        c = image[:, :, i]

        diff = 255 - c.max();

        noise = np.random.normal(0, random.randint(1, 6), c.shape)

        noise = (noise - noise.min()) / (noise.max() - noise.min())

        noise = diff * noise

        image[:, :, i] = c + noise.astype(np.uint8)

    return image





class Sampler:

    def __init__(self, width, height, net=None, **kwargs):

        self._net = net

        if net is None:

            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultValTransform(width, height, **kwargs)

        else:

            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultTrainTransform(width, height, net=net, **kwargs)



    def __call__(self, data):

        raw = load_image(data[1])

        bboxes = np.array(data[2])

        if not self._net is None:

            raw = raw.asnumpy()

            blur = random.randint(0, 3)

            if blur > 0:

                raw = gauss_blur(raw, blur)

            raw = gauss_noise(raw)

            raw = mx.nd.array(raw)

            h, w, _ = raw.shape

            raw, flips = gcv.data.transforms.image.random_flip(raw, py=0.5)

            bboxes = gcv.data.transforms.bbox.flip(bboxes, (w, h), flip_y=flips[1])

        res = self._transform(raw, bboxes)

        return [mx.nd.array(x) for x in res]

import mxnet as mx

import gluoncv as gcv





def load_model(path, ctx=mx.cpu()):

    net = gcv.model_zoo.yolo3_darknet53_custom(["wheat"], pretrained_base=False)

    net.set_nms(post_nms=150)

    net.load_parameters(path, ctx=ctx)

    return net

import os

import time

import random

import mxnet as mx

import pandas as pd

import gluoncv as gcv



max_epochs = 8

learning_rate = 0.0001

batch_size = 16

img_s = 512

threshold = 0.1

context = mx.gpu()



print("Loading model...")

model = load_model("/kaggle/input/global-wheat-detection-models/global-wheat-yolo3-darknet53.params", ctx=context)



print("Loading test images...")

test_images = [

    (os.path.join(dirname, filename), os.path.splitext(filename)[0])

        for dirname, _, filenames in os.walk('/kaggle/input/global-wheat-detection/test') for filename in filenames

]



print("Pseudo labaling...")

pseudo_set = []

for path, image_id in test_images:

    print(path)

    raw = load_image(path)

    x, _ = gcv.data.transforms.presets.yolo.transform_test(raw, short=img_s)

    classes, scores, bboxes = model(x.as_in_context(context))

    bboxes[0, :, 0::2] = (bboxes[0, :, 0::2] / x.shape[3]).clip(0.0, 1.0) * raw.shape[1]

    bboxes[0, :, 1::2] = (bboxes[0, :, 1::2] / x.shape[2]).clip(0.0, 1.0) * raw.shape[0]

    label = [

        [round(x) for x in bboxes[0, i].asnumpy().tolist()] + [0.0] for i in range(classes.shape[1])

            if model.classes[int(classes[0, i].asscalar())] == "wheat" and scores[0, i].asscalar() > threshold

    ]

    if len(label) > 0:

        pseudo_set.append((image_id, path, label))

    

print("Loading training set...")

training_set = load_dataset("/kaggle/input/global-wheat-detection") + pseudo_set



print("Re-training...")

trainer = mx.gluon.Trainer(model.collect_params(), "Nadam", {

    "learning_rate": learning_rate

})

for epoch in range(max_epochs):

    ts = time.time()

    random.shuffle(training_set)

    training_total_L = 0.0

    training_batches = 0

    for x, objectness, center_targets, scale_targets, weights, class_targets, gt_bboxes in get_batches(training_set, batch_size, width=img_s, height=img_s, net=model, ctx=context):

        training_batches += 1

        with mx.autograd.record():

            obj_loss, center_loss, scale_loss, cls_loss = model(x, gt_bboxes, objectness, center_targets, scale_targets, weights, class_targets)

            L = obj_loss + center_loss + scale_loss + cls_loss

            L.backward()

        trainer.step(x.shape[0])

        training_batch_L = mx.nd.mean(L).asscalar()

        if training_batch_L != training_batch_L:

            raise ValueError()

        training_total_L += training_batch_L

        print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" % (

            epoch, training_batches, training_batch_L, training_total_L / training_batches, time.time() - ts

        ))

    training_avg_L = training_total_L / training_batches

    print("[Epoch %d]  training_loss %.10f  duration %.2fs" % (epoch + 1, training_avg_L, time.time() - ts))



print("Inference...")

results = []

for path, image_id in test_images:

    print(path)

    raw = load_image(path)

    x, _ = gcv.data.transforms.presets.yolo.transform_test(raw, short=img_s)

    classes, scores, bboxes = model(x.as_in_context(context))

    bboxes[0, :, 0::2] = (bboxes[0, :, 0::2] / x.shape[3]).clip(0.0, 1.0) * raw.shape[1]

    bboxes[0, :, 1::2] = (bboxes[0, :, 1::2] / x.shape[2]).clip(0.0, 1.0) * raw.shape[0]

    bboxes[0, :, 2:4] -= bboxes[0, :, 0:2]

    results.append({

        "image_id": image_id,

        "PredictionString": " ".join([

            " ".join([str(x) for x in [scores[0, i].asscalar()] + [round(x) for x in bboxes[0, i].asnumpy().tolist()]])

                for i in range(classes.shape[1])

                    if model.classes[int(classes[0, i].asscalar())] == "wheat" and scores[0, i].asscalar() > threshold

        ])

    })

pd.DataFrame(results, columns=['image_id', 'PredictionString']).to_csv('submission.csv', index=False)
