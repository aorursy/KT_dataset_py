import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.cross_validation import train_test_split

from sklearn import svm

import numpy as np

%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')
### for trial run with smaller dataset ####



#np.random.seed = 28

#rand_image_index=np.random.randint(0,labeled_images.shape[0],5000)

#images = labeled_images.iloc[rand_image_index,1:]

#labels = labeled_images.iloc[rand_image_index,:1]





### for actual run with full dataset

images = labeled_images.iloc[:,1:]

labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

train_images /= 255

test_images /= 255



#from sklearn.preprocessing import Binarizer

#scaler = Binarizer()

#train_images = scaler.fit_transform(train_images)

#test_images = scaler.transform(test_images)

clf = svm.SVC(kernel='rbf', gamma=0.01, C=10)                                          



clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')

test_data /= 255

results=clf.predict(test_data)
df = pd.DataFrame({'ImageID': range(1, test_data.shape[0]+1, 1),

                  'Label': results})

df.to_csv('results.csv', index=False)