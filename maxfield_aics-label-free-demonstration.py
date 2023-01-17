# Setup environment
!pip install quilt --user
!pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip install git+https://github.com/JacksonMaxfield/pytorch_fnet.git
    
print("Completed environment setup.")
# Install the data and models
import quilt
quilt.install("aics/cell_line_samples")
quilt.install("aics/label_free")
# Install the data and models
# The prior cell will fail this will succeed
import quilt
quilt.install("aics/cell_line_samples")
quilt.install("aics/label_free")
# Import required packages
import json
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import fnet
import fnet.transforms

plt.set_cmap("bone")
# Import the data and models
from quilt.data.aics import cell_line_samples
from quilt.data.aics import label_free

# Predict function, random image and model choice by default
def predict(image_name=None, model_name=None):
    # chose image
    if image_name is None:
        imgs = [i for i in cell_line_samples._data_keys() if i != "README"]
        image_name = np.random.choice(imgs, 1)[0]

    chosen_img = cell_line_samples[image_name]
    
    # chose model
    if model_name is None:
        models = label_free._group_keys()
        model_name = np.random.choice(models, 1)[0]
    
    chosen_model = label_free[model_name]
        
    # load image
    source_image = tifffile.imread(chosen_img())
    print("Loaded image...")
    
    # load model
    model = fnet.fnet_model.Model()
    model.load_state(chosen_model.model(), gpu_ids=0)
    print("Loaded model...")
    
    # load model opts
    with open(chosen_model.train_options(), "r") as read_in:
        model_opts = json.load(read_in)
    
    # prep image
    brightfield = source_image[:, 3, :, :]
    minified = fnet.transforms.prep_ndarray(brightfield, model_opts["transform_signal"])[0]
    minified = minified[:32, :160, :160]
    tensor_in = fnet.transforms.ndarray_to_tensor(minified)
    
    print("Prepped image..")
    
    # predict
    with torch.no_grad():
        tensor_out = model.predict(tensor_in)
    predicted_image = fnet.transforms.tensor_to_ndarray(tensor_out)
    
    # results
    results = {"Source Brightfield: {}".format(image_name): minified,
               "Predicted Structure: {}".format(model_name): predicted_image}
    
    # plot brightfield and predicted
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    for label, ax in zip(results.keys(), axes.flat):
        ax.imshow(results[label].max(0))
        ax.set_title(label)
        ax.axis("off")
print("Model choices:")
label_free
print("Image choices:")
cell_line_samples
# If none provided the function chooses a random image and model
predict(image_name=None, model_name=None)
