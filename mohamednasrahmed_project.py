# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import scipy.misc
import pydicom 
import glob
import sys
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import pylab
#!git clone https://github.com/i-pan/rsna18-retinanet-starter.git
!git clone https://github.com/fizyr/keras-retinanet
os.chdir("keras-retinanet") 
!python setup.py build_ext --inplace    
!pip install . --user
DATA_DIR = "/kaggle/input/"
ROOT_DIR = "/kaggle/working/"
# converted training set DICOMs to PNGs, it should be part of the data environment
train_pngs_dir = os.path.join(DATA_DIR, "rsna-pneu-train-png/orig/")
test_dicoms_dir  = os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_2_test_images/") 
s = glob.glob('/kaggle/input/rsna-pneu-train-png/orig/*.png')
print((s[0])[39:-4])   
data = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"))
#data = data.drop_duplicates()
#data
ss = []
for i in s:
    ss.append(i[39:-4])

l_rm = []
for i in range(len(data['patientId'])):
    if data['patientId'][i] not in ss:
        l_rm.append(i)
        
data = data.drop(l_rm, axis = 0) 
data2 = data.drop_duplicates()
#data2
data1 =  pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")) 
# dropping passed values 
data1 = data1.drop(l_rm, axis = 0) 
#data1
#bbox_info = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"))
#detailed_class_info = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"))
#detailed_class_info = detailed_class_info.drop_duplicates()
# To get started, we'll train on positives only
positives = data2[data2["class"] == "Lung Opacity"]

# Annotations file should have no header and columns in the following order:
# filename, x1, y1, x2, y2, class 
positives = positives.merge(data1, on="patientId")
positives = positives[["patientId", "x", "y", "width", "height", "Target"]]
positives["patientId"] = [os.path.join(train_pngs_dir, "{}.png".format(_)) for _ in positives.patientId]
positives["x1"] = positives["x"] 
positives["y1"] = positives["y"] 
positives["x2"] = positives["x"] + positives["width"]
positives["y2"] = positives["y"] + positives["height"]
positives["Target"] = "opacity"
del positives["x"], positives["y"], positives["width"], positives["height"]

negatives = data2[data2["class"] == "Normal"]

negatives = negatives.merge(data1, on="patientId")
negatives = negatives[["patientId", "x", "y", "width", "height", "Target"]]
negatives["patientId"] = [os.path.join(train_pngs_dir, "{}.png".format(_)) for _ in negatives.patientId]
negatives["x1"] = negatives["x"] 
negatives["y1"] = negatives["x"]  
negatives["x2"] = negatives["x"]  
negatives["y2"] = negatives["x"]  
negatives["Target"] = "normal"
del negatives["x"], negatives["y"], negatives["width"], negatives["height"]
annotations = positives
#annotations = annotations.append(negatives)

# Before we save to CSV, we have to do some manipulating to make sure
# bounding box coordinates are saved as integers and not floats 
# Note: This is only necessary if you include negatives in your annotations
annotations = annotations.fillna(88888)
annotations["x1"] = annotations.x1.astype("int32").astype("str") 
annotations["y1"] = annotations.y1.astype("int32").astype("str") 
annotations["x2"] = annotations.x2.astype("int32").astype("str") 
annotations["y2"] = annotations.y2.astype("int32").astype("str")
annotations = annotations.replace({"88888": None}) 

annotations = annotations[["patientId", "x1", "y1", "x2", "y2", "Target"]]
annotations.to_csv(os.path.join(ROOT_DIR, "annotations.csv"), index=False, header=False)
annotations
classes_file = pd.DataFrame({"class": ["opacity"], "label": [0]}) 
classes_file.to_csv(os.path.join(ROOT_DIR, "classes.csv"), index=False, header=False)
#classes_file = pd.DataFrame({"class": ["normal","opacity"], "label": [0,1]}) 
#classes_file.to_csv(os.path.join(ROOT_DIR, "classes.csv"), index=False, header=False)
classes_file
!pip install progressbar2
%%writefile /kaggle/working/keras-retinanet/keras_retinanet/utils/eval.py
"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os
import time

import cv2
#pip install progressbar2
import progressbar
#assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_inferences = [None for i in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        inference_time = time.time() - start

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        all_inferences[i] = inference_time

    return all_detections, all_inferences


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_inferences = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    # inference time
    inference_time = np.sum(all_inferences) / generator.size()

    return average_precisions, inference_time
!python /kaggle/working/keras-retinanet/keras_retinanet/bin/train.py --backbone "resnet50" --image-min-side 256 --image-max-side 256 --batch-size 32 --random-transform --epochs 50 --steps 100 csv /kaggle/working/annotations.csv /kaggle/working/classes.csv
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_01.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_02.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_03.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_04.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_05.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_06.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_07.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_08.h5
!rm /kaggle/working/keras-retinanet/snapshots/resnet50_csv_09.h5
!python /kaggle/working/keras-retinanet/keras_retinanet/bin/convert_model.py /kaggle/working/keras-retinanet/snapshots/resnet50_csv_11.h5 /kaggle/working/keras-retinanet/converted_model.h5 
from keras_retinanet.models import load_model 

retinanet = load_model(os.path.join(ROOT_DIR, "keras-retinanet/converted_model.h5"), 
                       backbone_name="resnet50")
retinanet.summary()
def preprocess_input(x):
    x = x.astype("float32")
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.680
    return x
test_dicoms = glob.glob(os.path.join(test_dicoms_dir, "*.dcm"))
test_patient_ids = [_.split("/")[-1].split(".")[0] for _ in test_dicoms]
test_predictions = [] 
for i, dcm_file in enumerate(test_dicoms): 
    sys.stdout.write("Predicting images: {}/{} ...\r".format(i+1, len(test_dicoms)))
    sys.stdout.flush() 
    # Load DICOM and extract pixel array 
    dcm = pydicom.read_file(dcm_file)
    arr = dcm.pixel_array
    # Make 3-channel image
    img = np.zeros((arr.shape[0], arr.shape[1], 3))
    for channel in range(img.shape[-1]):
        img[..., channel] = arr 
    # Resize 
    # Change image size if necessary!
    scale_factor = 256. / img.shape[0]
    img = zoom(img, [scale_factor, scale_factor, 1], order=1, prefilter=False)
    # Preprocess with ImageNet mean subtraction
    img = preprocess_input(img) 
    prediction = retinanet.predict_on_batch(np.expand_dims(img, axis=0))
    test_predictions.append(prediction)
test_pred_df = pd.DataFrame() 
for i, pred in enumerate(test_predictions):
    # Take top 5 
    # Should already be sorted in descending order by score
    bboxes = pred[0][0][:5]
    scores = pred[1][0][:5]
    # -1 will be predicted if nothing is detected
    detected = scores > -1 
    if np.sum(detected) == 0: 
        continue
    else:
        bboxes = bboxes[detected]
        bboxes = [box / scale_factor for box in bboxes]
        scores = scores[detected]
    individual_pred_df = pd.DataFrame() 
    for j, each_box in enumerate(bboxes): 
        # RetinaNet output is [x1, y1, x2, y2] 
        tmp_df = pd.DataFrame({"patientId": [test_patient_ids[i]], 
                               "x": [each_box[0]],  
                               "y": [each_box[1]], 
                               "w": [each_box[2]-each_box[0]],
                               "h": [each_box[3]-each_box[1]],
                               "score": [scores[j]]})
        individual_pred_df = individual_pred_df.append(tmp_df) 
    test_pred_df = test_pred_df.append(individual_pred_df) 

test_pred_df.head()
test_pred_df.head(20)
threshold = 0.35

list_of_pids = [] 
list_of_preds = [] 
for pid in np.unique(test_pred_df.patientId): 
    tmp_df = test_pred_df[test_pred_df.patientId == pid]
    tmp_df = tmp_df[tmp_df.score >= threshold]
    # Skip if empty
    if len(tmp_df) == 0:
        continue
    predictionString = " ".join(["{} {} {} {} {}".format(row.score, row.x, row.y, row.w, row.h) for rownum, row in tmp_df.iterrows()])
    list_of_preds.append(predictionString)
    list_of_pids.append(pid) 

positives = pd.DataFrame({"patientId": list_of_pids, 
                          "PredictionString": list_of_preds}) 

negatives = pd.DataFrame({"patientId": list(set(test_patient_ids) - set(list_of_pids)), 
                          "PredictionString": [""] * (len(test_patient_ids)-len(list_of_pids))})

submission = positives.append(negatives)
#positives['PredictionString'][2]
positives
def pd_2dict(pd):
    a = {'patientId':pd['patientId'][1], 'PredictionString':pd['PredictionString'][1]}
    return a
def post_process(df):
    lst = []
    box_lst = df["PredictionString"].split()
    for i in range(0,len(box_lst),5):
        temp_lst = box_lst[i+1:i+5]
        lst.append(temp_lst)
    output = {'patientId': df["patientId"], 'boxes': lst}    
    return output
def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    img_loc = os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_2_test_images/%s.dcm"%(data['patientId']))
    d = pydicom.read_file(img_loc)
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        box = [float(b) for b in box]
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')
    
def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
b = pd_2dict(positives)
b = post_process(b)
b
draw(b)
retinanet.save('retinanet_1.h5')
weights = retinanet.save_weights(os.path.join(ROOT_DIR, "weights_retinanet"))
#change dcm to png

import cv2
import os
import pydicom

inputdir = os.path.join("/kaggle/input/", 'rsna-pneumonia-detection-challenge/stage_2_train_images/')
outdir = '../outputs/rsna-pneumonia-detection-challenge/stage_2_train_png_images/'
#os.mkdir(outdir)

train_list = [ f for f in  os.listdir(inputdir)]

for f in train_list:   # remove "[:10]" to convert all images 
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image
import cv2 
  
# Save image in set directory 
# Read RGB image 
img = cv2.imread('/kaggle/input/rsna-pneu-train-png/orig/01027bc3-dc40-4165-a6c3-d6be2cb7ca34.png')
img2 = cv2.imread('/kaggle/input/rsna-pneu-train-png/orig/000db696-cf54-4385-b10b-6b16fbb3f985.png')
print(img2)