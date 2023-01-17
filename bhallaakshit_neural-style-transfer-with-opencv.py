# Import Libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load base image
img = cv2.imread("../input/neural-style-transfer/images/images/Yuki.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()
# Total of 9 style images
style_images = os.listdir("../input/neural-style-transfer/art/art")
len(style_images)
# Load style images
style_images = np.array(style_images)
style_images = style_images.reshape(3, 3)

style_dic = {}
fig, axs = plt.subplots(3, 3, figsize = (15, 15))
for i in range(3):
    for j in range(3):
        path = os.path.join("../input/neural-style-transfer/art/art", style_images[i, j])
        img_st = cv2.imread(path)
        img_st = cv2.cvtColor(img_st, cv2.COLOR_BGR2RGB)
        
        axs[i, j].imshow(img_st)
        axs[i, j].set(xticks = [], yticks = [])
        
        text = style_images[i, j][:-4]
        axs[i, j].set_title(text)
        
        style_dic[text] = img_st
# List of models for each image
models = [
    "candy.t7",
    "composition_vii.t7",
    "feathers.t7",
    "la_muse.t7",
    "mosaic.t7",
    "starry_night.t7",
    "the_scream.t7",
    "the_wave.t7",
    "udnie.t7"
]
# Apply Neural Style Transfer
output_dic = {}
for i, mod in enumerate(models):
    text = str(mod)[:-3]
    style = cv2.imread(os.path.join("../input/neural-style-transfer/art/art", text + ".jpg"))

    # Load Neural Transfer Style Model
    model_neural = cv2.dnn.readNetFromTorch(os.path.join("../input/neural-style-transfer", mod))

    # Resize and make smaller (takes forever to perform on large images)
    height, width = img.shape[0], img.shape[1]
    new_height = 640
    new_width = int((new_height/height) * width)
    new_img = cv2.resize(
        img, 
        (new_width, new_height),
        interpolation = cv2.INTER_AREA
    )

    # Create a blob from the image and perform a forward pass considering the mean (R, G, B) values from ImageNet 
    R, G, B = 103.93, 116.77, 123.68
    input_blob = cv2.dnn.blobFromImage(
        new_img,
        1,
        (new_width, new_height),
        (R, G, B),
        swapRB = False,
        crop = False
    )
    model_neural.setInput(input_blob)
    output = model_neural.forward()
    
    # Reshape the output Tensor
    output = output.reshape(output.shape[1], output.shape[2], output.shape[3])
    # Adding the mean (R, G, B) values from ImageNet
    output[0] += R
    output[1] += G
    output[2] += B
   
    # Normalize output
    output /= 255
    # Ensure range of values in between (0, 1)
    output = output.clip(0, 1)
    output = output.transpose(1, 2, 0)
    
    output_dic[text] = output
fig, axs = plt.subplots(9, 3, figsize = (20, 20))
for i, text in enumerate(output_dic):
    axs[i, 0].imshow(img)
    axs[i, 0].set(xticks = [], yticks = [])
    axs[i, 0].set_title("Base Image")
    
    axs[i, 1].imshow(style_dic[text])
    axs[i, 1].set(xticks = [], yticks = [])
    axs[i, 1].set_title(text + " (Style)")    
        
    axs[i, 2].imshow(output_dic[text])
    axs[i, 2].set(xticks = [], yticks = [])
    axs[i, 2].set_title("Output Image")
    
fig.tight_layout()
