import numpy as np

import pandas as pd

import skimage.io as sio

import matplotlib.pyplot as plt

import cv2

import os



from mrcnn.config import Config

from mrcnn import utils

from mrcnn import model as modellib

from mrcnn import visualize



from imgaug import augmenters as iaa



class Nucleus(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):        

        self.add_class("nucleus", 1, "nucleus")

        assert subset in ["train_images", "test_images", "val"]

        

        image_ids = next(os.walk(dataset_dir))[1]



        if subset == "val":

            image_ids = VAL_IMAGE_IDS

        else:

            # Get image ids from directory names

            image_ids = next(os.walk(dataset_dir+"/"+subset))[1]

            if subset == "train":

                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))



        for image_id in image_ids:

            if (subset == 'val'):

                subset = "train_images" 

            self.add_image(

                "nucleus",

                image_id=image_id,

                path=os.path.join(dataset_dir, subset +"/"+ str(image_id), "images/{}.png".format(image_id)))



    def load_mask(self, image_id):

        info = self.image_info[image_id]

        

        df = pd.read_csv('input/train_masks.csv')

        

        img = sio.imread('input/train_images/' + str(info["id"]) +

                            '/images/' + str(info["id"]) + ".png")



        my_array = np.zeros((img.shape[0] * img.shape[1]))



        test_rows = df.loc[df['ImageId'] == image_id]

        length = len(test_rows.index)



        for i in range(0, length):

            encoded = test_rows.iloc[i]['EncodedPixels']

            self.draw_single_instance(my_array, encoded)



        tim = np.reshape(my_array, (img.shape[0], img.shape[1]))



        final = np.transpose(tim)

        mask = np.stack(final, axis=-1)



        return mask, np.ones([mask.shape], dtype=np.int32)





    def image_reference(self, image_id):

        """lol wut"""



    def draw_single_instance(self, npArray, data_string):

        data = data_string.split(' ')



        for start, run in zip(data[0::2], data[1::2]):

            start = int(start) - 1  # some people start at 1 (handle one indexing)

            run = int(run)

            # pretty much this just means: let that segment starting at start with length run be 1 (marking it as part of the instance)

            npArray[start: start + run] = 1



def train(model, dataset_dir, subset):

    """Train the model."""

    # Training dataset.

    dataset_train = Nucleus()

    dataset_train.load_nucleus(dataset_dir, subset)

    dataset_train.prepare()



    # *** This training schedule is an example. Update to your needs ***



    # If starting from imagenet, train heads only for a bit

    # since they have random weights

    print("Train network heads")

    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=20,

                augmentation=augmentation,

                layers='heads')



    print("Train all layers")

    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=40,

                augmentation=augmentation,

                layers='all')



def train(model, dataset_dir, subset):

    """Train the model."""

    # Training dataset.

    dataset_train = Nucleus()

    dataset_train.load_nucleus(dataset_dir, subset)

    dataset_train.prepare()



    # Validation dataset

    dataset_val = Nucleus()

    dataset_val.load_nucleus(dataset_dir, "val")

    dataset_val.prepare()



    # Image augmentation

    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html

    augmentation = iaa.SomeOf((0, 2), [

        iaa.Fliplr(0.5),

        iaa.Flipud(0.5),

        iaa.OneOf([iaa.Affine(rotate=90),

                   iaa.Affine(rotate=180),

                   iaa.Affine(rotate=270)]),

        iaa.Multiply((0.8, 1.5)),

        iaa.GaussianBlur(sigma=(0.0, 5.0))

    ])



    # *** This training schedule is an example. Update to your needs ***



    # If starting from imagenet, train heads only for a bit

    # since they have random weights

    print("Train network heads")

    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=20,

                augmentation=augmentation,

                layers='heads')



    print("Train all layers")

    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=40,

                augmentation=augmentation,

                layers='all')



def rle_decode(rle, shape):

    """Decodes an RLE encoded list of space separated

    numbers and returns a binary mask."""

    rle = list(map(int, rle.split()))

    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])

    rle[:, 1] += rle[:, 0]

    rle -= 1

    mask = np.zeros([shape[0] * shape[1]], np.bool)

    for s, e in rle:

        assert 0 <= s < mask.shape[0]

        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)

        mask[s:e] = 1

    # Reshape and transpose

    mask = mask.reshape([shape[1], shape[0]]).T

    return mask





def mask_to_rle(image_id, mask, scores):

    "Encodes instance masks to submission format."

    assert mask.ndim == 3, "Mask must be [H, W, count]"

    # If mask is empty, return line with image ID only

    if mask.shape[-1] == 0:

        return "{},".format(image_id)

    # Remove mask overlaps

    # Multiply each instance mask by its score order

    # then take the maximum across the last dimension

    order = np.argsort(scores)[::-1] + 1  # 1-based descending

    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)

    # Loop over instance masks

    lines = []

    for o in order:

        m = np.where(mask == o, 1, 0)

        # Skip if empty

        if m.sum() == 0.0:

            continue

        rle = rle_encode(m)

        lines.append("{}, {}".format(image_id, rle))

    return "\n".join(lines)





############################################################

#  Detection

############################################################



def detect(model, dataset_dir, subset):

    """Run detection on images in the given directory."""

    print("Running on {}".format(dataset_dir))



    # Create directory

    if not os.path.exists(RESULTS_DIR):

        os.makedirs(RESULTS_DIR)

    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())

    submit_dir = os.path.join(RESULTS_DIR, submit_dir)

    os.makedirs(submit_dir)



    # Read dataset

    dataset = Nucleus()

    dataset.load_nucleus(dataset_dir, subset)

    dataset.prepare()

    # Load over images

    submission = []

    for image_id in dataset.image_ids:

        # Load image and run detection

        image = dataset.load_image(image_id)

        # Detect objects

        r = model.detect([image], verbose=0)[0]

        # Encode image to RLE. Returns a string of multiple lines

        source_id = dataset.image_info[image_id]["id"]

        rle = mask_to_rle(source_id, r["masks"], r["scores"])

        submission.append(rle)

        # Save image with masks

        visualize.display_instances(

            image, r['rois'], r['masks'], r['class_ids'],

            dataset.class_names, r['scores'],

            show_bbox=False, show_mask=False,

            title="Predictions")

        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))



    # Save to csv file

    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)

    file_path = os.path.join(submit_dir, "submit.csv")

    with open(file_path, "w") as f:

        f.write(submission)

    print("Saved to ", submit_dir)



VAL_IMAGE_IDS = [

    "0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2",

    "92f31f591929a30e4309ab75185c96ff4314ce0a7ead2ed2c2171897ad1da0c7",

    "1e488c42eb1a54a3e8412b1f12cde530f950f238d71078f2ede6a85a02168e1f",

    "c901794d1a421d52e5734500c0a2a8ca84651fb93b19cec2f411855e70cae339",

    "8e507d58f4c27cd2a82bee79fe27b069befd62a46fdaed20970a95a2ba819c7b",

    "60cb718759bff13f81c4055a7679e81326f78b6a193a2d856546097c949b20ff",

    "da5f98f2b8a64eee735a398de48ed42cd31bf17a6063db46a9e0783ac13cd844",

    "9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32",

    "1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df",

    "97126a9791f0c1176e4563ad679a301dac27c59011f579e808bbd6e9f4cd1034",

    "e81c758e1ca177b0942ecad62cf8d321ffc315376135bcbed3df932a6e5b40c0",

    "f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81",

    "0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1",

    "3ab9cab6212fabd723a2c5a1949c2ded19980398b56e6080978e796f45cbbc90",

    "ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716",

    "bb61fc17daf8bdd4e16fdcf50137a8d7762bec486ede9249d92e511fcb693676",

    "e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b",

    "947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050",

    "cbca32daaae36a872a11da4eaff65d1068ff3f154eedc9d3fc0c214a4e5d32bd",

    "f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3",

    "4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06",

    "f73e37957c74f554be132986f38b6f1d75339f636dfe2b681a0cf3f88d2733af",

    "a4c44fc5f5bf213e2be6091ccaed49d8bf039d78f6fbd9c4d7b7428cfcb2eda4",

    "cab4875269f44a701c5e58190a1d2f6fcb577ea79d842522dcab20ccb39b7ad2",

    "8ecdb93582b2d5270457b36651b62776256ade3aaa2d7432ae65c14f07432d49",

]



ROOT_DIR = os.path.abspath("")



COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")



RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")



DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class NucleusConfig(Config):

    """Configuration for training on the nucleus segmentation dataset."""

    # Give the configuration a recognizable name

    NAME = "nucleus"



    # Adjust depending on your GPU memory

    IMAGES_PER_GPU = 6



    # Number of classes (including background)

    NUM_CLASSES = 1 + 1  # Background + nucleus



    # Number of training and validation steps per epoch

    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU

    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)



    # Don't exclude based on confidence. Since we have two classes

    # then 0.5 is the minimum anyway as it picks between nucleus and BG

    DETECTION_MIN_CONFIDENCE = 0



    # Backbone network architecture

    # Supported values are: resnet50, resnet101

    BACKBONE = "resnet50"



    # Input image resizing

    # Random crops of size 512x512

    IMAGE_RESIZE_MODE = "crop"

    IMAGE_MIN_DIM = 512

    IMAGE_MAX_DIM = 512

    IMAGE_MIN_SCALE = 2.0



    # Length of square anchor side in pixels

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)



    # ROIs kept after non-maximum supression (training and inference)

    POST_NMS_ROIS_TRAINING = 1000

    POST_NMS_ROIS_INFERENCE = 2000



    # Non-max suppression threshold to filter RPN proposals.

    # You can increase this during training to generate more propsals.

    RPN_NMS_THRESHOLD = 0.9



    # How many anchors per image to use for RPN training

    RPN_TRAIN_ANCHORS_PER_IMAGE = 64



    # Image mean (RGB)

    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])



    # If enabled, resizes instance masks to a smaller size to reduce

    # memory load. Recommended when using high-resolution images.

    USE_MINI_MASK = True

    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask



    # Number of ROIs per image to feed to classifier/mask heads

    # The Mask RCNN paper uses 512 but often the RPN doesn't generate

    # enough positive proposals to fill this and keep a positive:negative

    # ratio of 1:3. You can increase the number of proposals by adjusting

    # the RPN NMS threshold.

    TRAIN_ROIS_PER_IMAGE = 128



    # Maximum number of ground truth instances to use in one image

    MAX_GT_INSTANCES = 200



    # Max number of final detections per image

    DETECTION_MAX_INSTANCES = 400





class NucleusInferenceConfig(NucleusConfig):

    # Set batch size to 1 to run one image at a time

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    # Don't resize imager for inferencing

    IMAGE_RESIZE_MODE = "pad64"

    # Non-max suppression threshold to filter RPN proposals.

    # You can increase this during training to generate more propsals.

    RPN_NMS_THRESHOLD = 0.7

config = NucleusConfig()

model = modellib.MaskRCNN(mode="training", config=config, model_dir="")

weights_path = COCO_WEIGHTS_PATH

# Download weights file

if not os.path.exists(weights_path):

    utils.download_trained_weights(weights_path)



model.load_weights(weights_path, by_name=True, exclude=[

    "mrcnn_class_logits", "mrcnn_bbox_fc",

    "mrcnn_bbox", "mrcnn_mask"])



train(model, "input/", "train_images")
