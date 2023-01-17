import pandas as pd

import numpy as np

import cv2

import os

import matplotlib.pyplot as plt

from sklearn.utils import shuffle



os.listdir('../input/fruits-360_dataset/fruits-360/Training')[0:10]
path = '../input/fruits-360_dataset/fruits-360/Training/Clementine/206_100.jpg'



im = cv2.imread(path)

plt.imshow(im)
b,g,r = cv2.split(im)



im2 = cv2.merge([r,g,b])



plt.imshow(im2)
grayim = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)



plt.imshow(grayim,cmap = 'gray')
grayim
grayim.shape
grayim = grayim.astype('float')/255

grayim
grayim = grayim.flatten()

grayim = pd.Series(grayim)

grayim
path = '../input/fruits-360_dataset/fruits-360/Training/'



cols = np.arange(grayim.shape[0])

df = pd.DataFrame(columns = cols)

labelcol = []



fruitlist = os.listdir(path)

x = 0



for f in fruitlist[0:9] : 

    fruitpath = '%s%s' % (path,f)

    

    imagelist = os.listdir(fruitpath)

    

    for i in imagelist:

        imagepath = '%s/%s' % (fruitpath,i)

    

        image = cv2.imread(imagepath)

    

        b,g,r = cv2.split(image)

        image = cv2.merge([r,g,b])

    

        imagegray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    

        imagegray = imagegray.astype('float')/255

    

        imagegray = imagegray.flatten()

    

        df.loc[x] = imagegray

    

        x = df.shape[0] + 1

        labelcol.append(f)

    
df['label'] = labelcol

df
df['label'].value_counts(normalize = True)
df = shuffle(df).reset_index(drop = True)

df
# transpose data set and shuffle

df_t = shuffle(df.transpose())

# transpose back to normal

df = df_t.transpose()
from sklearn.model_selection import train_test_split



# Create X and Y variables

X = df.drop('label',axis = 1)

y = df['label']



# create our test and training set

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,stratify = y)

from sklearn.svm import SVC



svm_model = SVC().fit(X_train,y_train)



trainscore = svm_model.score(X_train,y_train)

testscore = svm_model.score(X_test,y_test)

print('Training score: {:.3f}\nTest score: {:.3f}'.format(trainscore,testscore))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_depth = 5).fit(X_train,y_train)



train = rf.score(X_train,y_train)

test = rf.score(X_test,y_test)



print('Training score: {:.3f}\nTest Score: {:.3f}'.format(train,test))