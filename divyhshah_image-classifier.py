!pip install gTTS
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

from keras.applications.vgg16 import VGG16

model = VGG16()

from gtts import gTTS
print(model.summary())
path='../input/sample-images/3722572342_6904d11d52.jpg'

image = load_img(path,target_size=(224,224))

photo=image
image = img_to_array(image)
image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
image = preprocess_input(image)
y_hat = model.predict(image)
label = decode_predictions(y_hat)
label = label[0][0]
text=label[1]

text_list=text.split('_')

mytext="".join(text_list)
import IPython

from IPython.display import Image, display

from PIL import Image

import os

language='en-in'



obj = gTTS(text=mytext,lang=language,slow=False)

obj.save("output.mp3")
IPython.display.display(IPython.display.Image(path))

print(mytext)

IPython.display.display(IPython.display.Audio('output.mp3'))