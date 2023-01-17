# Load packages

import h5py

import os



import matplotlib.pyplot as plt

import numpy as np

import xml.etree.ElementTree as etree



from skimage.io import imread

from skimage.transform import resize



from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input

from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input

from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error

from tensorflow.keras.models import Model
# Parse the xml annotation file and retrieve the path to each image, its size and annotations

def extract_xml_annotation(filename):

    """Parse the xml file

    :param filename: str

    """

    z = etree.parse(filename)

    objects = z.findall('./object')

    size = (int(float(z.find('.//width').text)), int(float(z.find('.//height').text)))

    fname = z.find('./filename').text

    dicts = [{obj.find('name').text: [int(float(obj.find('bndbox/xmin').text)),

                                      int(float(obj.find('bndbox/ymin').text)),

                                      int(float(obj.find('bndbox/xmax').text)),

                                      int(float(obj.find('bndbox/ymax').text))]}

             for obj in objects]

    return {'size': size, 'filename': fname, 'objects': dicts}
%%time

# Filters annotations keeping only those we are interested in.

# We only keep images in which there is a single item

annotations = []



filters = ['dog', 'cat', 'bird', 'cow', 'horse']

idx2labels = {k: v for k, v in enumerate(filters)}

labels2idx = {v: k for k, v in idx2labels.items()}



ANNOTATION_DIR = '../input/pascal-voc-2012/VOC2012/Annotations/'

for filename in sorted(os.listdir(ANNOTATION_DIR)):

    annotation = extract_xml_annotation(os.path.join(ANNOTATION_DIR, filename))

    

    new_objects = []

    for obj in annotation['objects']:

        if list(obj.keys())[0] in filters:

            new_objects.append(obj)

    

    if len(new_objects) == 1:

        annotation['class'] = list(new_objects[0].keys())[0]

        annotation['bbox'] = list(new_objects[0].values())[0]

        annotation.pop('objects')

        annotations.append(annotation)
print(f'Number of images with annotations: {len(annotations)}.')
print(f'Example contents of one annotation: {annotations[0]}')
print(f'Correspondence between indices and labels: {idx2labels}')
model = ResNet50(include_top=False, weights="imagenet")

input_tensor = model.layers[0].input

output_tensor = model.layers[-2].output

model_conv = Model(input_tensor, output_tensor)
def predict_batch(model, img_batch_path, img_size=None):

    img_list = []

    

    for im_path in img_batch_path:

        img = imread(im_path)

        if img_size:

            img = resize(img, img_size,

                         mode='reflect',

                         preserve_range=True)

            

        img = img.astype('float32')

        img_list.append(img)

    

    try:

        img_batch = np.stack(img_list, axis=0)

    except:

        raise ValueError('''When both img_size and crop_size are None, all images

            in image_paths must have the same shapes.''')

    

    return model(preprocess_input(img_batch)).numpy()
# Test the model

IMG_DIR = '../input/pascal-voc-2012/VOC2012/JPEGImages/'

IMG_PATH = os.path.join(IMG_DIR, annotations[0]['filename'])

output = predict_batch(model_conv, [IMG_PATH], (1000, 224))

print(f'Shape of the output: {output.shape}')
def compute_representations(annotations):

    batch_size = 32

    batches = []

    

    n_batches = len(annotations) // 32 + 1

    for batch_idx in range(n_batches):

        batch_bgn = batch_idx * 32

        batch_end = min(len(annotations), (batch_idx + 1) * 32)

        img_names = []

        for annotation in annotations[batch_bgn:batch_end]:

            img_path = os.path.join(IMG_DIR, annotation['filename'])

            img_names.append(img_path)

        batch = predict_batch(model_conv, img_names, img_size=(224, 224))

        batches.append(batch)

        print(f'Batch {batch_idx + 1}/{n_batches} prepared')

    return np.vstack(batches)
%%time

reprs = compute_representations(annotations)
# Serialize representations

#h5f = h5py.File('voc_representations.h5', 'w')

#h5f.create_dataset('reprs', data=reprs)

#h5f.close()
img_resize = 224

num_classes = len(labels2idx.keys())



def tensorize_ground_truth(annotations):

    all_boxes = []

    all_cls = []

    for idx, annotation in enumerate(annotations):

        # Build a one-hot encoding of the class

        cls = np.zeros((num_classes))

        cls_idx = labels2idx[annotation['class']]

        cls[cls_idx] = 1.0

        

        coords = annotation['bbox']

        size = annotation['size']

        

        # Resize the image

        x1, y1, x2, y2 = (coords[0] * img_resize / size[0],

                          coords[1] * img_resize / size[1],

                          coords[2] * img_resize / size[0],

                          coords[3] * img_resize / size[1])

        

        # Compute center of the box and its height and width

        cx, cy = ((x2 + x1) / 2, (y2 + y1) / 2)

        w = x2 - x1

        h = y2 - y1

        boxes = np.array([cx, cy, w, h])

        all_boxes.append(boxes)

        all_cls.append(cls)

    

    # Stack everything into two big np tensors

    return np.vstack(all_cls), np.vstack(all_boxes)
classes, boxes = tensorize_ground_truth(annotations)
print(f'Classes shape: {classes.shape}, Boxes shape: {boxes.shape}')
def interpret_output(cls, boxes, img_size=(500, 333)):

    cls_idx = np.argmax(cls)

    confidence = cls[cls_idx]

    classname = idx2labels[cls_idx]

    cx, cy = boxes[0], boxes[1]

    w, h = boxes[2], boxes[3]

    

    small_box = [max(0, cx - w / 2), max(0, cy - h / 2),

                 min(img_resize, cx + w / 2), min(img_resize, cy + h / 2)]

    fullsize_box = [int(small_box[0] * img_size[0] / img_resize),

                    int(small_box[1] * img_size[1] / img_resize),

                    int(small_box[2] * img_size[0] / img_resize),

                    int(small_box[3] * img_size[1] / img_resize)]

    output = {'class': classname, 'confidence': confidence, 'bbox': fullsize_box}

    return output
print(f'Original annotation:\n {annotations[0]}')

print(f'Interpreted output:\n {interpret_output(classes[0], boxes[0], img_size=annotations[0]["size"])}')
def IoU(boxA, boxB):

    # Find the intersecting box coordinates

    x0 = max(boxA[0], boxB[0])

    y0 = max(boxA[1], boxB[1])

    x1 = min(boxA[2], boxB[2])

    y1 = min(boxA[3], boxB[3])

    

    # Compute the area of intersection rectangle

    inter_area = max(x1 - x0, 0) * max(y1 - y0, 0)

    

    # Compute the area of each box

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    

    # Compute the intersection over union by taking the intersection

    # area and dividing it by the sum of areas minus the intersection areas

    return inter_area / float(boxA_area + boxB_area - inter_area)
original = annotations[0]

interpreted = interpret_output(classes[0], boxes[0], img_size=annotations[0]["size"])

print(f'IoU of the original versus the interpreted bounding box: {IoU(original["bbox"], interpreted["bbox"])}.')
def plot_IoU(boxA, boxB, img_size=(10, 10)):

    """Plot the IoU measure

    """

    iou = IoU(boxA, boxB)

    

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.set_ylim(0, img_size[1])

    ax.set_xlim(0, img_size[0])

    ax.add_patch(plt.Rectangle((boxA[0], boxA[1]),

                               boxA[2] - boxA[0],

                               boxA[3] - boxA[1],

                               color='blue', alpha=0.5))

    ax.add_patch(plt.Rectangle((boxB[0], boxB[1]),

                               boxB[2] - boxB[0],

                               boxB[3] - boxB[1],

                               color='red', alpha=0.5))

    ax.set_title(f'IoU: {iou}')

    return fig, ax
fix, ax = plot_IoU([2, 2, 8, 8], [3, 3, 7, 9])

plt.show()
def classif_local_stupid_model(num_classes):

    """Stupid model that averages all the spatial information

    

    The goal of this model is to show that it's a very bad idea to destroy

    the spatial information with GlobalAveragePooling2D layer if our

    goal is to do object localization.

    """

    model_input = Input(shape=(7, 7, 2048))

    x = GlobalAveragePooling2D()(model_input)

    x = Dropout(0.2)(x)

    x = Dense(1000)(x)

    head_classes = Dense(num_classes, activation='softmax', name='head_classes')(x)

    head_boxes = Dense(4, name='head_boxes')(x)

    

    model = Model(inputs=model_input, outputs=[head_classes, head_boxes], name='resnet_loc')

    model.compile(optimizer='adam', loss=[categorical_crossentropy, mean_squared_error], 

                  loss_weights=[1., 0.01])

    return model
model = classif_local_stupid_model(num_classes)
num = 64

inputs = reprs[0:num]

out_cls, out_boxes = classes[0:num], boxes[0:num]



print(f'Input batch shape: {inputs.shape}')

print(f'Ground truth batch shapes: {out_cls.shape} and {out_boxes.shape}')
out = model.predict(inputs)

print(f'Output model batch shapes: {out[0].shape} and {out[1].shape}')
history = model.fit(inputs, [out_cls, out_boxes], batch_size=10, epochs=10)
plt.plot(np.log(history.history["head_boxes_loss"]), label="Bounding boxes log-loss")

plt.plot(np.log(history.history["head_classes_loss"]), label="Classes log-loss")

plt.plot(np.log(history.history["loss"]), label="Log-loss")

plt.legend(loc="best")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.show()
def patch(axis, bbox, display_txt, color):

    coords = (bbox[0], bbox[1]), bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1

    axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))

    axis.text(bbox[0], bbox[1], display_txt, color='white', bbox={'facecolor':color, 'alpha':0.5})

    

def plot_annotations(img_path, annotation=None, ground_truth=None):

    img = imread(img_path)

    plt.imshow(img)

    current_axis = plt.gca()

    if ground_truth:

        text = "GT " + ground_truth["class"]

        patch(current_axis, ground_truth["bbox"], text, "red")

    if annotation:

        conf = f'{annotation["confidence"]:0.2f} '

        text = conf + annotation["class"]

        patch(current_axis, annotation["bbox"], text, "blue")

    plt.axis('off')



def display(model, index, ground_truth=True):

    res = model.predict(reprs[index][np.newaxis])

    output = interpret_output(res[0][0], res[1][0], img_size=annotations[index]["size"])

    plot_annotations(IMG_DIR + annotations[index]["filename"], 

                     output, annotations[index] if ground_truth else None)
plt.figure(figsize=(12, 10))

for i in range(15):

    plt.subplot(3, 5, i + 1)

    display(model, np.random.randint(reprs.shape[0]))

plt.show()
def accuracy_and_iou(preds, trues, threshold=0.5):

    sum_valid, sum_accurate, sum_iou = 0, 0, 0

    num = len(preds)

    for pred, true in zip(preds, trues):

        iou_value = IoU(pred["bbox"], true["bbox"])

        if pred["class"] == true["class"] and iou_value > threshold:

            sum_valid = sum_valid + 1

        sum_iou = sum_iou + iou_value

        if pred["class"] == true["class"]:

            sum_accurate = sum_accurate + 1

    return sum_accurate / num, sum_iou / num, sum_valid / num
def compute_acc(model, train=True):

    if train:

        beg, end = 0, (9 * len(annotations) // 10)

        split_name = "Train"

    else:

        beg, end = (9 * len(annotations)) // 10, len(annotations) 

        split_name = "Test"

    res = model.predict(reprs[beg:end])

    outputs = []

    for index, (classes, boxes) in enumerate(zip(res[0], res[1])):

        output = interpret_output(classes, boxes,

                                  img_size=annotations[index]["size"])

        outputs.append(output)

    

    acc, iou, valid = accuracy_and_iou(outputs, annotations[beg:end],

                                       threshold=0.5)

    

    print(f'{split_name} acc: {acc:0.3f}, mean IoU: {iou:0.3f}, acc_valid: {valid:0.3f}')
compute_acc(model, train=True)

compute_acc(model, train=False)
# Keep last examples for test

test_num = reprs.shape[0] // 10

train_num = reprs.shape[0] - test_num

test_inputs = reprs[train_num:]

test_cls, test_boxes = classes[train_num:], boxes[train_num:]

print(f'Length of the train set: {train_num}, length of the test set: {test_num}')
model = classif_local_stupid_model(num_classes)
batch_size = 32

inputs = reprs[0:train_num]

out_cls, out_boxes = classes[0:train_num], boxes[0:train_num]



history = model.fit(inputs, y=[out_cls, out_boxes],

                    validation_data=(test_inputs, [test_cls, test_boxes]),

                    batch_size=batch_size, epochs=20, verbose=2)
compute_acc(model, train=True)

compute_acc(model, train=False)
plt.figure(figsize=(12, 10))

for i in range(15):

    plt.subplot(3, 5, i + 1)

    display(model, np.random.randint(reprs.shape[0]))

plt.show()
def classif_local_model(num_classes):

    model_input = Input(shape=(7, 7, 2048))

    x = GlobalAveragePooling2D()(model_input)

    x = Dropout(0.2)(x)

    x = Dense(1000)(x)

    head_classes = Dense(num_classes, activation='softmax', name='head_classes')(x)

    

    y = Convolution2D(4, (1, 1), activation='relu', name='hidden_conv')(model_input)

    y = Flatten()(y)

    y = Dropout(0.2)(y)

    head_boxes = Dense(4, name='head_boxes')(y)

    

    model = Model(inputs=model_input, outputs=[head_classes, head_boxes], name='resnet_loc')

    model.compile(optimizer='adam', loss=[categorical_crossentropy, 'mse'], loss_weights=[1., 1 / (224 * 224)])

    return model
model = classif_local_model(num_classes)
history = model.fit(inputs, y=[out_cls, out_boxes],

                    validation_data=(test_inputs, [test_cls, test_boxes]),

                    batch_size=batch_size, epochs=20, verbose=2)
compute_acc(model, train=True)

compute_acc(model, train=False)
plt.figure(figsize=(12, 10))

for i in range(15):

    plt.subplot(3, 5, i + 1)

    display(model, np.random.randint(reprs.shape[0]))

plt.show()