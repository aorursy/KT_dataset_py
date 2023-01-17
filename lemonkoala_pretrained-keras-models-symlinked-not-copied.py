import matplotlib.pyplot as plt
import numpy             as np
import os

from keras.preprocessing               import image
from keras.applications.resnet50       import ResNet50, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
inputs_dir = "/kaggle/input"
models_dir = os.path.expanduser(os.path.join("~", ".keras", "models"))
os.makedirs(models_dir)
for file in os.listdir(inputs_dir):
    if file.endswith(".json") or file.endswith(".h5"):
        os.symlink(
            os.path.join(inputs_dir, file),
            os.path.join(models_dir, file)
        )
!ls  ~/.keras/models
fig, ax = plt.subplots(1, figsize=(12, 10))
img = image.load_img('../input/Kuszma.JPG')
img = image.img_to_array(img)
ax.imshow(img / 255.) 
ax.axis('off')
plt.show()
resnet = ResNet50()
img = image.load_img('../input/Kuszma.JPG', target_size=(224, 224))
img = image.img_to_array(img)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
preds = resnet.predict(x)
decode_predictions(preds, top=5)