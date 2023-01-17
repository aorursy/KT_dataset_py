# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Set matplotlib backend

# This has to be done before other importa that might

# set it, but only if we're running in script mode

# rather than being imported.

if __name__ == '__main__':

    import matplotlib

    # Agg backend runs without a display

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt



import os

import sys

import json

import datetime

import numpy as np

import skimage.io

from imgaug import augmenters as iaa

# Results directory

# Save submission files here

RESULTS_DIR = os.path.join("/kaggle/working/")



# The dataset doesn't have a standard train/val split, so I picked

# a variety of images to surve as a validation set.

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
# change directory to the dataset where our

# custom scripts are found

os.chdir("/kaggle/input/mrcnntools/")



from config import Config

import utils

import model as modellib

import visualize



os.chdir("/kaggle/working/")
# Path to trained weights file

COCO_WEIGHTS_PATH = os.path.join("<need to add to dataset>/mask_rcnn_coco.h5")
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

class NucleusDataset(utils.Dataset):



    def load_nucleus(self, dataset_dir, subset):

        """Load a subset of the nuclei dataset.



        dataset_dir: Root directory of the dataset

        subset: Subset to load. Either the name of the sub-directory,

                such as stage1_train, stage1_test, ...etc. or, one of:

                * train: stage1_train excluding validation images

                * val: validation images from VAL_IMAGE_IDS

        """

        # Add classes. We have one class.

        # Naming the dataset nucleus, and the class nucleus

        self.add_class("nucleus", 1, "nucleus")



        # Which subset?

        # "val": use hard-coded list above

        # "train": use data from stage1_train minus the hard-coded list above

        # else: use the data from the specified sub-directory

        assert subset in ["train", "val", "stage1_train", "test_images", "stage2_test"]

        subset_dir = "stage1_train" if subset in ["train", "val"] else subset

        dataset_dir = os.path.join(dataset_dir, subset_dir)

        if subset == "val":

            image_ids = VAL_IMAGE_IDS

        elif subset=='test_images':

            image_ids=['f01a9742c43a69f087700a43893f713878e537bae8e44f76b957f09519601ad6',

                        'f0a75e0322f11cead4219aa530673fe5eef67580fb6fccc254963c9fc6b58aa1',

                        'f0c9f135c62572f3669a75b2c735e4477dc77fac85e653426ee2b3bcfbed7aaf',

                        'f0d0ab13ff53adc3c4d57e95a5f83d80b06f2cbc0bf002b52cf7b496612e0ce4',

                        'f113626a04125d97b27f21b45a0ce9a686d73dee7b5dbc0725d49194ba0203bd',

                        'f20eb4592e7d3cf58d421a9c34832d33adcdcbd0e17b7bf009a013847608da27',

                        'f26f4c2c70c38fe12e00d5a814d5116691f2ca548908126923fd76ddd665ed24',

                        'f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81',

                        'f35ab34528e3e2d2589d24cbffc0e10024dfc474a68585d0b5feb7b05aa0067f',

                        'f43169e3d8b4f71e687945b9e72cbfdfe2e40e68842568e6a30c60d64c1378b6',

                        'f487cc82271cf84b4414552aa8b0a9d82d902451ebe8e8bc639d4121c1672ff7',

                        'f4b7c24baf69b8752c49d0eb5db4b7b5e1524945d48e54925bff401d5658045d',

                        'f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3',

                        'f4faa3a409014db1865074c5f66a0255f71ae3faba03265da0b3b91f68e8a8f0',

                        'f534b43bf37ff946a310a0f08315d76c3fb3394681cf523acef7c0682240072a',

                        'f5effed21f671bbf4551ecebb7fe95f3be1cf09c16a60afe64d2f0b95be9d1eb',

                        'f67e72b7fe0b1e3648ea745ffd395c80705c89b0c0c48227991fe6f5815b2a18',

                        'f6863b83d75e5927b30e2e326405b588293283c25aaef2251b30c343296b9cb1',

                        'f6b16c885c0b2bc0d0eb2bb2eeb0a2753ebafb5a7a91da10e89b0b0478984637',

                        'f6cb37ebf29c225284c8415962f7287abe7007fae8fe3d8a3899b608b832d7d5',

                        'f728de04267283f0b4daab9a840e7433b2c6034baf195fd526850439c9297687',

                        'f73e37957c74f554be132986f38b6f1d75339f636dfe2b681a0cf3f88d2733af',

                        'f7e5dcfc9c93183c668c5a4ab028d5faad54fb54298711f2caae0508aa978300',

                        'f7eaaf420b5204c4a42577428b7cd897a53ef07b759ccbba3ed30a3548ca5605',

                        'f81ca7ee25e733ff37240c34c8e3044d9937bb0166e315952ebde3f237ecb86f',

                        'f8e74d4006dd68c1dbe68df7be905835e00d8ba4916f3b18884509a15fdc0b55',

                        'f93ec5e683d81005ffc2a84a1c0299b2406ad14b764b824e013f7ca3a13833b5',

                        'f952cc65376009cfad8249e53b9b2c0daaa3553e897096337d143c625c2df886',

                        'f9ac03b0344ce8c48bc058448541f9211a1e5f4c94fdaf633dd534328d8610ab',

                        'f9ea1a1159c33f39bbe5f18bb278d961188b40508277eab7c0b4b91219b37b5d',

                        'fa73f24532b3667718ede7ac5c2e24ad7d3cae17b0a42ed17bbb81b15c28f4ae',

                        'fa751ff3a6332c95cb5cb1d28563553914295e9e7d35c4b6bd267241e8a0787c',

                        'fac507fa4d1649e8b24c195d990f1fc3ca3633d917839e1751a9d412a14ab5e3',

                        'fadeb0ab092833f27daaeb3e24223eb090f9536b83f68cde8f49df7c544f711b',

                        'fc22db33a2495f58f118bc182c0087e140df14ccb8dad51373e1a54381f683de',

                        'fc345dac2205deb169bd70197f07f053bada80b61ffa69fdfb490758323ead69',

                        'fc5452f612a0f972fe55cc677055ede662af6723b5c1615ad539b8a4bd279bdb',

                        'fc9269fb2e651cd4a32b65ae164f79b0a2ea823e0a83508c85d7985a6bed43cf',

                        'fd8065bcb1afdbed19e028465d5d00cd2ecadc4558de05c6fa28bea3c817aa22',

                        'fdda64c47361b0d1a146e5b7b48dc6b7de615ea80b31f01227a3b16469589528',

                        'fe80a2cf3c93dafad8c364fdd1646b0ba4db056cdb7bdb81474f957064812bba',

                        'fe9adb627a6f45747c5a8223b671774791ededf9364f6544be487c540107fa4f',

                        'fec226e45f49ab81ab71e0eaa1248ba09b56a328338dce93a43f4044eababed5',

                        'feffce59a1a3eb0a6a05992bb7423c39c7d52865846da36d89e2a72c379e5398',

                        'ff3407842ada5bc18be79ae453e5bdaa1b68afc842fc22fa618ac6e6599d0bb3',

                        'ff3e512b5fb860e5855d0c05b6cf5a6bcc7792e4be1f0bdab5a00af0e18435c0']

        else:

            # Get image ids from directory names

            print(dataset_dir)

            try:

                image_ids = next(os.walk(dataset_dir))[1]

            except:

                pass

            if subset == "train":

                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))



        # Add images

        for image_id in image_ids:

            self.add_image(

                "nucleus",

                image_id=image_id,

                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))



    def load_mask(self, image_id):

        """Generate instance masks for an image.

       Returns:

        masks: A bool array of shape [height, width, instance count] with

            one mask per instance.

        class_ids: a 1D array of class IDs of the instance masks.

        """

        info = self.image_info[image_id]

        # Get mask directory from image path

        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")



        # Read mask files from .png image

        mask = []

        for f in next(os.walk(mask_dir))[2]:

            if f.endswith(".png"):

                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)

                mask.append(m)

        mask = np.stack(mask, axis=-1)

        # Return mask, and array of class IDs of each instance. Since we have

        # one class ID, we return an array of ones

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)



    def image_reference(self, image_id):

        """Return the path of the image."""

        info = self.image_info[image_id]

        if info["source"] == "nucleus":

            return info["id"]

        else:

            super(self.__class__, self).image_reference(image_id)

############################################################

#  Training

############################################################



def train(model, dataset_dir, subset):

    """Train the model."""

    # Training dataset.

    dataset_train = NucleusDataset()

    dataset_train.load_nucleus(dataset_dir, subset)

    dataset_train.prepare()



    # Validation dataset

    dataset_val = NucleusDataset()

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



############################################################

#  RLE Encoding

############################################################



def rle_encode(mask):

    """Encodes a mask in Run Length Encoding (RLE).

    Returns a string of space-separated values.

    """

    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"

    # Flatten it column wise

    m = mask.T.flatten()

    # Compute gradient. Equals 1 or -1 at transition points

    g = np.diff(np.concatenate([[0], m, [0]]), n=1)

    # 1-based indicies of transition points (where gradient != 0)

    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1

    # Convert second index in each pair to lenth

    rle[:, 1] = rle[:, 1] - rle[:, 0]

    return " ".join(map(str, rle.flatten()))





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

    dataset = NucleusDataset()

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



############################################################

#  Command Line

############################################################



if __name__ == '__main__':

    import argparse



    # Parse command line arguments

    parser = argparse.ArgumentParser(

        description='Mask R-CNN for nuclei counting and segmentation')

    parser.add_argument("command",

                        metavar="<command>",

                        help="'train' or 'detect'")

    parser.add_argument('--dataset', required=False,

                        metavar="/path/to/dataset/",

                        help='Root directory of the dataset')

    parser.add_argument('--weights', required=True,

                        metavar="/path/to/weights.h5",

                        help="Path to weights .h5 file or 'coco'")

    parser.add_argument('--logs', required=False,

                        default=DEFAULT_LOGS_DIR,

                        metavar="/path/to/logs/",

                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--subset', required=False,

                        metavar="Dataset sub-directory",

                        help="Subset of dataset to run prediction on")

    args = parser.parse_args()



    # Validate arguments

    if args.command == "train":

        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "detect":

        assert args.subset, "Provide --subset to run prediction on"



    print("Weights: ", args.weights)

    print("Dataset: ", args.dataset)

    if args.subset:

        print("Subset: ", args.subset)

    print("Logs: ", args.logs)



    # Configurations

    if args.command == "train":

        config = NucleusConfig()

    else:

        config = NucleusInferenceConfig()

    config.display()



    # Create model

    if args.command == "train":

        model = modellib.MaskRCNN(mode="training", config=config,

                                  model_dir=args.logs)

    else:

        model = modellib.MaskRCNN(mode="inference", config=config,

                                  model_dir=args.logs)



    # Select weights file to load

    if args.weights.lower() == "coco":

        weights_path = COCO_WEIGHTS_PATH

        # Download weights file

        if not os.path.exists(weights_path):

            utils.download_trained_weights(weights_path)

    elif args.weights.lower() == "last":

        # Find last trained weights

        weights_path = model.find_last()

    elif args.weights.lower() == "imagenet":

        # Start from ImageNet trained weights

        weights_path = model.get_imagenet_weights()

    else:

        weights_path = args.weights



    # Load weights

    print("Loading weights ", weights_path)

    if args.weights.lower() == "coco":

        # Exclude the last layers because they require a matching

        # number of classes

        model.load_weights(weights_path, by_name=True, exclude=[

            "mrcnn_class_logits", "mrcnn_bbox_fc",

            "mrcnn_bbox", "mrcnn_mask"])

    else:

        model.load_weights(weights_path, by_name=True)



    # Train or evaluate

    if args.command == "train":

        train(model, args.dataset, args.subset)

    elif args.command == "detect":

        detect(model, args.dataset, args.subset)

    else:

        print("'{}' is not recognized. "

              "Use 'train' or 'detect'".format(args.command))
# load weights

weights_path = "../input/mrcnntools/mask_rcnn_coco.h5"

model = modellib.MaskRCNN(mode="inference", config=NucleusInferenceConfig(), model_dir='/kaggle/working/logs/')

model.load_weights(weights_path, by_name=True, exclude=[

    "mrcnn_class_logits", "mrcnn_bbox_fc",

    "mrcnn_bbox", "mrcnn_mask"])



# detect nuclei

detect(model, '/kaggle/input/2019-fire171-asn11-image-segmentation-challenge-v2/test_images/', 'test_images')