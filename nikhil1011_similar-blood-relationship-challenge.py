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
!pip install git+https://github.com/rcmalli/keras-vggface.git
import gc

import cv2

import glob

from PIL import Image

from keras_vggface.utils import preprocess_input



def read_img(path):

    img = cv2.imread(path)

    #print(img)

    img = np.array(img).astype(np.float)

    return preprocess_input(img,version=2)
from collections import defaultdict

#keeps all photos path in a dictionary

allPhotos = defaultdict(list)

for family in glob.glob("../input/recognizing-faces-in-the-wild/train/*"):

    for mem in glob.glob(family+'/*'):

        for photo in glob.glob(mem+'/*'):

            allPhotos[mem].append(photo)



#list of all members with valid photo

ppl = list(allPhotos.keys())

len(ppl)
data = pd.read_csv('../input/recognizing-faces-in-the-wild/train_relationships.csv')

data.p1 = data.p1.apply( lambda x: '../input/recognizing-faces-in-the-wild/train/'+x )

data.p2 = data.p2.apply( lambda x: '../input/recognizing-faces-in-the-wild/train/'+x )

print(data.shape)

data.head()
data['result']=0

for ind in data.index: 

    if data['p1'][ind] in ppl and data['p2'][ind] in ppl:

        

        data['result'][ind]=1

    else:

        data['result'][ind]=0

     
data.head()
data.result.value_counts()
from sklearn.model_selection import train_test_split

y=data['result']

X=data.drop(['result'],axis=1)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
from numpy import load

from keras.models import load_model

model = load_model("../input/facenet/keras-facenet-20190708t084110z-001/keras-facenet/model/facenet_keras.h5")
submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')

submission['p1'] = submission.img_pair.apply( lambda x: '../input/recognizing-faces-in-the-wild/test/'+x.split('-')[0] )

submission['p2'] = submission.img_pair.apply( lambda x: '../input/recognizing-faces-in-the-wild/test/'+x.split('-')[1] )

print(submission.shape)

submission.head()
from PIL import Image

from tqdm import tqdm

from numpy import dot

from numpy.linalg import norm

from numpy import expand_dims

from numpy import asarray

def extract_face(model, photo , required_size=(160, 160)):

    image = Image.open(photo)

    image = image.resize(required_size)

    face_array = asarray(image)

    face_pixels = face_array.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean) / std

    samples = expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)

    return yhat[0]

    



probs = []

img1=[]

img2=[]

for i,j in tqdm([ (0,500),(500,1000),(1000,1500),(1500,2000),(2000,2500),

                 (2500,3000),(3000,3500),(3500,4000),(4000,4500),(4500,5000),(5000,5310) ]):

    for photo in submission.p1.values[i:j]:

            imgs1=extract_face(model,photo)

            img1.append(imgs1)

    for photo in submission.p2.values[i:j]:

            imgs2=extract_face(model,photo)

            img2.append(imgs2)       

len(img1)
len(img2)
probs = []

for i in range(len(img1)):

    a=img1[i]

    b=img2[i]

    cos_sim = dot(a, b)/(norm(a)*norm(b))

    probs.append(np.squeeze(cos_sim))

    
submission['is_related']=probs
submission.drop( ['p1','p2'],axis=1,inplace=True )

submission.head()
submission.to_csv('submission.csv',index=False)