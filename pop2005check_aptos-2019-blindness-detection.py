import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#from keras.preprocessing import image

import cv2

from tqdm import tqdm 

from PIL import Image, ImageEnhance
N=3662 # number of images 



df_train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
df_train.head()
(counts,bin_edges,_)=plt.hist(df_train['diagnosis'],align='left',bins=5,edgecolor='purple',linewidth=1.5,color='lavender')

plt.title('Histogram of diabetic retinopathy diagnosis', fontsize=14,weight='bold')

plt.xticks(bin_edges[:-1], np.arange(0,5,1))



ax=plt.gca()

ax.set_xlabel('Diabetic Retinopathy Diagnosis',fontsize=12);

ax.set_ylabel('Frequency',fontsize=12);



df_train['diagnosis'].value_counts()
paths='../input/aptos2019-blindness-detection/train_images/'+df_train['id_code'][:5]+'.png'

plt.figure(figsize=(20,4))

for index, (path,label) in enumerate(zip(paths,df_train['diagnosis'][:5])):

    plt.subplot(1,5,index+1)

    plt.imshow(Image.open(path))

    plt.title('Diagnosis: %i\n' % label, fontsize = 20)
def preprocess_image(path, desired_size=224):

    '''

    resize image to desired size x desired size

    and also increase contrast by 1.5x

    '''

    im = Image.open(path)

    im = im.resize((desired_size, )*2,resample=Image.LANCZOS)

    

    # increase contrast of the images

    enhancer = ImageEnhance.Brightness(im)

    factor = 1.5 #factor > 1 increases the contrast

    im_output = enhancer.enhance(factor)

    return im_output
%%time 



# reading in images into the array pics_data

# https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter



# create an empty 4d array to store the images

pics_data = np.empty((N, 224, 224,3), dtype=np.uint8)



for i, image_name in enumerate(tqdm(df_train['id_code'][:N])):

    pics_data[i, :, :,:] = preprocess_image(f'../input/aptos2019-blindness-detection/train_images/{image_name}.png')

plt.figure(figsize=(20,4))

for index, (path,label) in enumerate(zip(paths,df_train['diagnosis'][:5])):

    plt.subplot(1,5,index+1)

    plt.imshow(preprocess_image(path))

    plt.title('Diagnosis: %i\n' % label, fontsize = 20)
# store corresponding diagnoses in an array named y_data

y_data=np.array(df_train['diagnosis'][:N])



# print the shape of y_data array

y_data.shape
# print the shape of pics_data

pics_data.shape
# reshape y_data from 1D --> 2D 

y_data2D=y_data.reshape(N,1)



# reshape pics_data from 1D --> 2D 

pics_data2D=pics_data.reshape(N,150528)
# split dataset into training and test sets in a way that is blind to the programmer

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(pics_data2D, y_data2D, test_size=0.25, random_state=0)
%%time



from sklearn.linear_model import LogisticRegression

# create logistic regressor

logisticRegr = LogisticRegression(random_state=0)

# train logistic regressor using training sets

logisticRegr.fit(x_train, y_train.ravel()) 
predictions = logisticRegr.predict(x_test) # predict entire test set

y_test_reshaped=y_test.reshape(916,)

df = pd.DataFrame({'Actual':y_test_reshaped,'Predicted':predictions})

df.tail()
# evaluating the algorithm



from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
%%time



from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults

logisticRegr = LogisticRegression(random_state=0,max_iter=1000)

logisticRegr.fit(x_train, y_train.ravel()) # train regressor 

# evaluating the algorithm

predictions = logisticRegr.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
%%time



from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults

logisticRegr = LogisticRegression(random_state=0,solver='saga')

logisticRegr.fit(x_train, y_train.ravel()) # train regressor 

# evaluating the algorithm

predictions = logisticRegr.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
%%time

from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=20,random_state=0) # creates neural network

# hidden_layer_sizes creates 3 layers of 10 nodes each; just try different combos and see what is best

#max_iter = number of iterations of epochs (cycles of feed-forward and back propagation)

mlp.fit(x_train, y_train.ravel())

# make predictions to our test data

predictions=mlp.predict(x_test)

# evaluating the neural net algorithm



from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
%%time

from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=0) # creates neural network

# hidden_layer_sizes creates 3 layers of 10 nodes each; just try different combos and see what is best

#max_iter = number of iterations of epochs (cycles of feed-forward and back propagation)

mlp.fit(x_train, y_train.ravel())

# make predictions to our test data

predictions=mlp.predict(x_test)

# evaluating the neural net algorithm



from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))