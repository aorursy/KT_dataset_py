# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow==1.14
# install keras facenet vgg 

!pip install git+https://github.com/rcmalli/keras-vggface.git
from keras.models import Model

from keras.optimizers import Adam, Adagrad, RMSprop

from keras import regularizers



import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()



print(tf.__version__)
!pip install mtcnn
# example of face detection with mtcnn

from matplotlib import pyplot as plt

from PIL import Image

from numpy import asarray

from mtcnn.mtcnn import MTCNN

from numpy import expand_dims



# extract a single face from a given photograph

def extract_face(filename, required_size=(224, 224)):

    # load image from file

    pixels = plt.imread(filename)

    # create the detector, using default weights

    detector = MTCNN()

    # detect faces in the image

    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face

    x1, y1, width, height = results[0]['box']

    x2, y2 = x1 + width, y1 + height

    # extract the face

    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size

    image = Image.fromarray(face)

    image = image.resize(required_size)

    face_array = asarray(image)

    return face_array



bashar2 = "../input/sample-images/bashar.jpg"

bashar1 = "../input/sample-images/8655252.jpg"

# chanImage = "../input/sample-images/channing_tatum-779x1024.jpg"

pixelsArr = plt.imread(bashar1)

pixelsArr2 = plt.imread(bashar2)

image = Image.fromarray(pixelsArr)

image2 = Image.fromarray(pixelsArr2)

plt.imshow(image)

plt.show()

plt.imshow(image2)

# show the plot

plt.show()



# load the photo and extract the face

pixels = extract_face(bashar1)

# plot the extracted face

plt.imshow(pixels)

# show the plot

plt.show()



# load the photo and extract the face

pixels = extract_face(bashar2)

# plot the extracted face

plt.imshow(pixels)

# show the plot

plt.show()
from keras_vggface.vggface import VGGFace

from keras_vggface.utils import preprocess_input

from keras_vggface.utils import decode_predictions

model = VGGFace(model='resnet50')



# summarize input and output shape

print('Inputs: %s' % model.inputs)

print('Outputs: %s' % model.outputs)
def predict(image):

    # load the photo and extract the face

    pixels = extract_face(image)

    # convert one face into samples

    pixels = pixels.astype('float32')

    sample = expand_dims(pixels, axis=0)

    # prepare the face for the model, e.g. center pixels

    sample = preprocess_input(sample, version=2)

    # perform prediction

    yhat = model.predict(sample)

    print(yhat.shape)

    # convert prediction into names

    results = decode_predictions(yhat)

    # display most likely results

    for result in results[0]:

        print('%s: %.3f%%' % (result[0], result[1]*100))
# predict(chanImage)

predict(bashar1) # classification (matches nearest existing embedding in the data set)
embeddingModel = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# extract faces and calculate face embeddings for a list of photo files

def get_embeddings(filenames):

	# extract faces

	faces = [extract_face(f) for f in filenames]

	# convert into an array of samples

	samples = asarray(faces, 'float32')

	# prepare the face for the model, e.g. center pixels

	samples = preprocess_input(samples, version=2)

	# perform prediction

	yhat = embeddingModel.predict(samples)

	return yhat
from scipy.spatial.distance import cosine

# determine if a candidate face is a match for a known face

def is_match(known_embedding, candidate_embedding, thresh=0.5):

    # calculate distance between embeddings

    score = cosine(known_embedding, candidate_embedding)

    if score <= thresh:

        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))

        return True

    else:

        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

        return False

import urllib.request

print('Beginning file download with urllib2...')

url = 'https://scontent.fybz2-2.fna.fbcdn.net/v/t1.0-9/28167929_10156263330182425_507706998966515633_n.jpg?_nc_cat=104&_nc_oc=AQmf2vMyahqW2OSAjy4qxZe7qEChxWF6waTI4PAp_zg5lxedsFYWc8qzkQM0_FbCwzM&_nc_ht=scontent.fybz2-2.fna&oh=6091e1c655995d5a186790cfad6ebe29&oe=5E5F3808'

testImage = '/tmp/xx1.jpg'

# celeb that my image was classifed as above 

#url2 = 'https://scontent.fybz2-2.fna.fbcdn.net/v/t1.0-9/16649540_768402856641047_233173073172558227_n.jpg?_nc_cat=107&_nc_eui2=AeEY6DYphag_bpbaB7sujXJRF8DW8mDAQjb3gL-IAVsFn8IwDQ7Lx1pk_zcIQ0BAWpjlpZfeEgV5-fGhtdNaKeXzEtE84x7lMaaoQWOlOpJLrg&_nc_oc=AQnr9XEH8uOeur5ab_IyXcFkH2Oam6LA8-dfMxmnGR3Jc3D_10d75KkUBw86bHB6tjo&_nc_ht=scontent.fybz2-2.fna&oh=edfcdcb91144440a9713e56f3e1ea437&oe=5E5B55F9'

url2 = 'http://www.gstatic.com/tv/thumb/persons/259192/259192_v9_ba.jpg'

negTestImage = '/tmp/neg1.jpg'



urllib.request.urlretrieve(url, testImage)

urllib.request.urlretrieve(url2, negTestImage)



pixelsArr3 = plt.imread(testImage)

image3 = Image.fromarray(pixelsArr3)

plt.imshow(image3)

plt.show()



pixelsArr3 = plt.imread(negTestImage)

image3 = Image.fromarray(pixelsArr3)

plt.imshow(image3)

plt.show()



# load the photo and extract the face

pixels = extract_face(testImage)

# plot the extracted face

plt.imshow(pixels)

# show the plot

plt.show()



# load the photo and extract the face

pixels = extract_face(negTestImage)

# plot the extracted face

plt.imshow(pixels)

# show the plot

plt.show()

embeddings = get_embeddings([bashar1])



db = {}

db['bashar'] = { 'imagePath': bashar1, 'embedding': embeddings[0] }





def findMatch(searchEmbedding):

    for k in db:

        if is_match(db[k]['embedding'], searchEmbedding):

            print('match found in db.., name: ' + k)

            return k


searchEmb1 = get_embeddings([bashar2])

findMatch(searchEmb1)
searchEmb2 = get_embeddings([testImage])

findMatch(searchEmb2)
se3 = get_embeddings([negTestImage])

findMatch(se3)