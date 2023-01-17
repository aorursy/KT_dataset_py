# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import json

from matplotlib import pyplot as plt

from skimage import color

from skimage.feature import hog

from sklearn import svm

from sklearn.metrics import classification_report,accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
f = open(r'../input/shipsnet.json')

dataset = json.load(f)

f.close()



dataset.keys()
data = np.array(dataset['data']).astype('uint8')

img_length = 80

data = data.reshape(-1,3,img_length,img_length).transpose([0,2,3,1])



plt.imshow(data[51])
data_gray = [ color.rgb2gray(i) for i in data]

plt.imshow(data_gray[51])
ppc = 16

hog_images = []

hog_features = []

for image in data_gray:

    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)

    hog_images.append(hog_image)

    hog_features.append(fd)
plt.imshow(hog_images[51])
labels =  np.array(dataset['labels']).reshape(len(dataset['labels']),1)
clf = svm.SVC()

hog_features = np.array(hog_features)

data_frame = np.hstack((hog_features,labels))

np.random.shuffle(data_frame)
#What percentage of data you want to keep for training

percentage = 80

partition = int(len(hog_features)*percentage/100)
x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]

y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()



clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

print('\n')

print(classification_report(y_test, y_pred))