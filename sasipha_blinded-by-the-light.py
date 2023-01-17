import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

original_image = "/kaggle/input/aeroscape/aeroscapes/JPEGImages/000001_001.jpg"
label_image_semantic = "/kaggle/input/aeroscape/aeroscapes/SegmentationClass/000001_001.png"

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow( Image.open(original_image))
axs[0].grid(False)

label_image_semantic = Image.open(label_image_semantic)
label_image_semantic = np.asarray(label_image_semantic)
axs[1].imshow(label_image_semantic)
axs[1].grid(False)

kaggle_commit = True

epochs = 20
if kaggle_commit:
    epochs = 5
    
from keras_segmentation.models.unet import vgg_unet

n_classes = 12 # Aerial Semantic Segmentation Drone Dataset tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle
model = vgg_unet(n_classes=n_classes ,  input_height=416, input_width=608  )

model.train( 
    train_images =  "/kaggle/input/aeroscape/aeroscapes/JPEGImages",
    train_annotations = "/kaggle/input/aeroscape/aeroscapes/SegmentationClass",
    checkpoints_path = "vgg_unet" , epochs=epochs
)
import time
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

start = time.time()

input_image = "/kaggle/input/aeroscape/aeroscapes/JPEGImages/000001_001.jpg"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="out.png"
)

fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)

img_orig = Image.open(input_image)
axs[0].imshow(img_orig)
axs[0].set_title('original image-001.jpg')
axs[0].grid(False)

axs[1].imshow(out)
axs[1].set_title('prediction image-out.png')
axs[1].grid(False)

validation_image = "/kaggle/input/aeroscape/aeroscapes/Visualizations/000001_001.png"
axs[2].imshow( Image.open(validation_image))
axs[2].set_title('true label image-001.png')
axs[2].grid(False)

done = time.time()
elapsed = done - start