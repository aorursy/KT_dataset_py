!pip install -q -U torch torchvision -f https://download.pytorch.org/whl/torch_stable.html 

!pip install -q -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

!pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog

from detectron2 import model_zoo



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



%matplotlib inline
# Loading the default config

cfg = get_cfg()





# Merging config from a YAML file

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))





# Downloading and loading pretrained weights

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")





# Changing some other configs

cfg.MODEL.DEVICE = 'cpu' # setting device to CPU as no training is required as per now

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # setting threshold for this model





# Defining the Predictor

predictor = DefaultPredictor(cfg)
def show_image(im, height=16, width=10):

    """

    Function to display an image

    

    Args:

        im ([numpy.ndarray])

        height ([int] or None)

        width ([int] or None)

    """

    plt.figure(figsize=(16,10))

    plt.imshow(im)

    plt.axis("off")

    plt.show()
def get_predicted_labels(classes, scores, class_names):

    """

    Function to return the name of predicted classes along with accuracy scores

    

    Args:

        classes (list[int] or None)

        scores (list[float] or None)

        class_names (list[str] or None)

    Returns:

        list[str] or None

    """

    labels = None

    if classes is not None and class_names is not None and len(class_names) > 1:

        labels = [class_names[i] for i in classes]

        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        return labels

    else:

        return "No object identified"
# Download image as input_image.jpg

# !wget https://images.unsplash.com/photo-1585574123552-aac232a58514 -O input_image.jpg



!wget https://cdn-images-1.medium.com/max/872/1*EYFejGUjvjPcc4PZTwoufw.jpeg -O input_image.jpg



# Read image

im = mpimg.imread("input_image.jpg")



# Show image

show_image(im)
# Predicting image

outputs = predictor(im)





# Extracting other data from the predicted image

scores = outputs["instances"].scores

classes = outputs["instances"].pred_classes

class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes





# Obtaining a list of predicted class labels using the utility function created earlier

predicted_labels = get_predicted_labels(classes, scores, class_names)





# Creating the Visualizer for visualizing the bounding boxes

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

output_im = v.get_image()[:, :, ::-1] # image with bounding box and lables defined





# Displaying the output

print(f"Predicted Objects: {predicted_labels}")

show_image(output_im, outputs)