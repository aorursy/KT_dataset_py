# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

full=pd.read_csv('../input/train.csv')

images=full.iloc[:,1:]

labels=full.iloc[:,0]

from sklearn.model_selection import train_test_split

train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.3,random_state=0)

image=train_images.iloc[0].as_matrix()

image=image.reshape((28,28))

plt.imshow(image,cmap='gray')
plt.hist(train_images.iloc[0])

train_images[train_images>0]=1

test_images[test_images>0]=1

plt.imshow(image,cmap='gray')
plt.hist(train_images.iloc[0])

import tensorflow.contrib.learn as learn

feature_columns =learn.infer_real_valued_columns_from_input(train_images)

classifier = learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=10)#,feature_columns=feature_columns)

classifier.fit(train_images,train_labels, batch_size=35, steps=300)

predictions=list(classifier.predict(test_images))

from sklearn.metrics import classification_report

print(classification_report(test_labels,predictions))

test_images_final=pd.read_csv('../input/test.csv')

test_images_final[test_images_final>0]=1

predictions_final=np.array(list(classifier.predict(test_images_final)))

predictions_final
df=pd.DataFrame({"ImageId": list(range(1,len(predictions_final)+1)),

                         "Label": predictions_final})

df.to_csv("results.csv", index=False, header=True)

df