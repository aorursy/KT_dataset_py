import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob
from sklearn.model_selection import KFold
data_file_path = '/kaggle/input/rsna-pneumonia-detection-challenge'
# to save the model to the current directory
file_path = '/kaggle/working'
# Temporaary files stores just for this session
temp_file_path = '/kaggle/temp'
!git clone https://www.github.com/matterport/Mask_RCNN.git
!pip install tensorflow-gpu==1.13.1
!pip install tensorflow==1.13.1
!pip install keras==2.0.8
os.chdir('Mask_RCNN')
!python3 setup.py install
import tensorflow as tf
print(tf.__version__)
ls
sys.path.append(os.path.join(file_path, 'Mask_RCNN'))
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
import mrcnn
train_dicom_dir = os.path.join(data_file_path, 'stage_2_train_images')
test_dicom_dir = os.path.join(data_file_path, 'stage_2_test_images')
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
!ls -lh mask_rcnn_coco.h5
Weights_file_path = "mask_rcnn_coco.h5"
def get_dicom_fps(dicom_dir):
    # To get a list of dicom images
    dicom_img = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_img))

def parse_dataset(dicom_dir, annotations): 
    # returns a list of all images
    image_fps = get_dicom_fps(dicom_dir)
    # annotates the list of images obtained
    image_annotations = {fp: [] for fp in image_fps}
    
    for index, row in annotations.iterrows(): 
        
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
        
    return image_fps, image_annotations 
class PneumoniaConfig(Config):
    """Configuration for training Lung Opacity location detection on the RSNA pneumonia dataset. Created a
    sub-class that inherits from config class and override properties that need to be changed.
    """
    # Name the configurations. 
    NAME = 'Lung_Opacity'  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    # Train on 2 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    IMAGES_PER_GPU = 8
    
    

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 32
    VALIDATION_STEPS = 8

    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    # Here, we leave it as resnet 101
    BACKBONE = 'resnet50'
    
    # Length of square anchor side in pixels
#     RPN_ANCHOR_SCALES = (32, 64, 128, 256)
#     BACKBONE_STRIDES = [4, 8, 16, 32]
    

    # Number of classification classes (including background)
    NUM_CLASSES = 2  # background and 1 pneumonia class
    
    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 32

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 3
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 3
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.8

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.1
        
config = PneumoniaConfig()
config.display()
class PneumoniaDataset(utils.Dataset):
    """
    Dataset class for training automated loation detection for lung opacities on the RSNA pneumonia dataset. For overriding the Base dataset class in Matterport's Mask R-CNN.
    
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        # For parent class
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # add images 
        for i, fp in enumerate(image_fps):
            
            # Adding annotations
            annotations = image_annotations[fp]
            
            # Add image function (self, source, image_id, path, anntations)
            self.add_image('pneumonia', 
                           image_id=i, 
                           path=fp, 
                           annotations=annotations, 
                           orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for pneumonia dataset, and pass to this function
        if you encounter images not in your dataset.
        """
        info = self.image_info[image_id]
        return info['path']

    
    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,3] Numpy array.
        """
        info = self.image_info[image_id]
        
        fp = info['path']
        
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
 

        return image

    
    def load_mask(self, image_id):
        """
        Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """


        info = self.image_info[image_id]
        
        annotations = info['annotations']
        
        count = len(annotations)
        
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
            
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            
            for i, a in enumerate(annotations):
                
                if a['Target'] == 1:
                    
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
                    
        return mask.astype(np.bool), class_ids.astype(np.int32)
annotations = pd.read_csv(os.path.join(data_file_path, 'stage_2_train_labels.csv'))
annotations
annotations.head(20)
image_fps, image_annotations = parse_dataset(train_dicom_dir, annotations=annotations)
ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
image = ds.pixel_array # get image array
ds
# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024
image_fps_list = list(image_fps)

# Seeding to get same results in case of shuffling the data
random.seed(42)

random.shuffle(image_fps_list)

# Validation size
val_size = 8100
image_fps_validate1 = image_fps_list[:val_size]
image_fps_training = image_fps_list[val_size:]

random.shuffle(image_fps_list)

# Validation size
test_size = 4050
image_fps_validate = image_fps_validate1[:test_size]
image_fps_test = image_fps_validate1[test_size:]

print(len(image_fps_training), len(image_fps_validate))
print(image_fps_validate[:5])
print(image_fps_test[:5])
# prepare the training dataset
dataset_train = PneumoniaDataset(image_fps_training, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()
# Show annotation(s) for a DICOM image 
test_fp = random.choice(image_fps_training)
image_annotations[test_fp]
# prepare the validation dataset
dataset_val = PneumoniaDataset(image_fps_validate, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()
# Load and display random sample and their bounding boxes

class_ids = [0]
while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)

plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])

for i in range(mask.shape[2]):
    masked += image[:,:,0] * mask[:,:,i]
    
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(image_fp)
print(class_ids)
# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# test on the same image as above
imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
plt.figure(figsize=(30, 12))
_ = plt.imshow(imggrid[:, :, 0], cmap='gray')
model = modellib.MaskRCNN(mode='training', config=config, model_dir=file_path)

# Exclude the last layers because they require a matching
# number of classes
model.load_weights(Weights_file_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])
%%time
LEARNING_RATE = 0.005

model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=7,
            layers='all',
            augmentation=augmentation)
history = model.keras_model.history.history.copy()
print(history)
%%time
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=15,
            layers='all',
            augmentation=augmentation)
s = model.keras_model.history.history
for k in s: history[k] = history[k] + s[k]
%%time
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=20,
            layers='all',
            augmentation=augmentation)
s = model.keras_model.history.history
for k in s: history[k] = history[k] + s[k]
epochs = range(1,len(next(iter(history.values())))+1)
pd.DataFrame(history, index=epochs)
plt.figure(figsize=(10,5))
plt.subplot(111)
plt.plot(epochs, history["loss"], label="Training loss")
plt.plot(epochs, history["val_loss"], label="Validation loss")
plt.legend()
plt.show()
# select trained model 
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))
    
fps = []
# Pick last directory
for d in dir_names:
    dir_name = os.path.join(model.model_dir, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))
class InferenceConfig(PneumoniaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=file_path)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors
dataset = dataset_val
fig = plt.figure(figsize=(10, 30))

for i in range(6):

    image_id = random.choice(dataset.image_ids)
    
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    
    print(original_image.shape)
    plt.subplot(6, 2, 2*i + 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
    
    plt.subplot(6, 2, 2*i + 2)
    results = model.detect([original_image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])

# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)
# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.95):
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
        file.write("patientId,PredictionString\n")

        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)
                        out_str += bboxes_str

            file.write(out_str+"\n")
test_results = os.path.join(file_path, 'test_results.csv')
predict(image_fps_test, filepath=test_results)
print(test_results)
# Against Submission
submission_fp = os.path.join(file_path, 'submission.csv')
predict(test_image_fps, filepath=submission_fp)
print(submission_fp)
test = pd.read_csv(test_results, names=['patientId', 'PredictionString'])
test.head(60)
output = pd.read_csv(submission_fp, names=['patientId', 'PredictionString'])
output.head(60)
def visualize(): 
    image_id = random.choice(test_image_fps)
    ds = pydicom.read_file(image_id)
    
    # original image 
    image = ds.pixel_array
    
    # assume square image 
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 
    resized_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    patient_id = os.path.splitext(os.path.basename(image_id))[0]
    print(patient_id)

    results = model.detect([resized_image])
    r = results[0]
    for bbox in r['rois']: 
        print(bbox)
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2]  * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        width = x2 - x1 
        height = y2 - y1 
        print("x {} y {} h {} w {}".format(x1, y1, width, height))
    plt.figure() 
    plt.imshow(image, cmap=plt.cm.gist_gray)

visualize()
visualize()
visualize()
visualize()
!rm -rf /kaggle/working/Mask_RCNN