# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image
Image(filename='/kaggle/input/cat-photo/Turkish_Van_Cat.jpg') 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
img_path_cat = '/kaggle/input/cat-photo/Turkish_Van_Cat.jpg'
img = image.load_img(img_path_cat, target_size=(224, 224))
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
X = preprocess_input(X)
#Check the shape of X
X.shape

model = ResNet50(weights='imagenet')
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
def classify(img_path):
    display(Image(filename=img_path))
    
    img = image.load_img(img_path, target_size=(224, 224))

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)

    preds = model.predict(X)
    print('Predicted:', decode_predictions(preds, top=3)[0])
classify(img_path_cat)

classify("/kaggle/input/bbqjpg/BBQ.jpg")
classify('/kaggle/input/osakacastlejpg/osaka.jpg')
classify('/kaggle/input/otanijpg/otani.jpg')
