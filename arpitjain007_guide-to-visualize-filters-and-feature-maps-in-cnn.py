 # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import Model

from matplotlib import pyplot

from numpy import expand_dims

from matplotlib import pyplot



import warnings

warnings.filterwarnings('ignore')
#Load the model

model = VGG16()



# Summary of the model

model.summary()
for layer in model.layers:

    

    if 'conv' not in layer.name:

        continue    

    filters , bias = layer.get_weights()

    print(layer.name , filters.shape)
# retrieve weights from the second hidden layer

filters , bias = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them

f_min, f_max = filters.min(), filters.max()

filters = (filters - f_min) / (f_max - f_min)
n_filters =6

ix=1

fig = pyplot.figure(figsize=(20,15))

for i in range(n_filters):

    # get the filters

    f = filters[:,:,:,i]

    for j in range(3):

        # subplot for 6 filters and 3 channels

        pyplot.subplot(n_filters,3,ix)

        pyplot.imshow(f[:,:,j] ,cmap='gray')

        ix+=1

#plot the filters 

pyplot.show()
for i in range(len(model.layers)):

    layer = model.layers[i]

    if 'conv' not in layer.name:

        continue    

    print(i , layer.name , layer.output.shape)
model = Model(inputs=model.inputs , outputs=model.layers[1].output)
image = load_img("../input/emma_watson.jpg" , target_size=(224,224))



# convert the image to an array

image = img_to_array(image)

# expand dimensions so that it represents a single 'sample'

image = expand_dims(image, axis=0)
image = preprocess_input(image)
#calculating features_map

features = model.predict(image)



fig = pyplot.figure(figsize=(20,15))

for i in range(1,features.shape[3]+1):



    pyplot.subplot(8,8,i)

    pyplot.imshow(features[0,:,:,i-1] , cmap='gray')

    

pyplot.show()
model2 = VGG16()
blocks = [ 2, 5 , 9 , 13 , 17]

outputs = [model2.layers[i].output for i in blocks]



model2 = Model( inputs= model2.inputs, outputs = outputs)
feature_map = model2.predict(image)



for i,fmap in zip(blocks,feature_map):

    fig = pyplot.figure(figsize=(20,15))

    #https://stackoverflow.com/a/12444777

    fig.suptitle("BLOCK_{}".format(i) , fontsize=20)

    for i in range(1,features.shape[3]+1):



        pyplot.subplot(8,8,i)

        pyplot.imshow(fmap[0,:,:,i-1] , cmap='gray')

    

pyplot.show()