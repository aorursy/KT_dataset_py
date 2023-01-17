# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")

df_test.tail()
image=df_train.iloc[0:5000,1:]

predictions=df_train.iloc[0:5000,:1]

gg=image.iloc[1999].as_matrix()

gg=gg.reshape((28,28))

plt.imshow(gg,cmap='gray')

clf=svm.SVC()

clf.fit(image,predictions)

image[image>0]=1

gg=image.iloc[1].as_matrix().reshape((28,28))

plt.imshow(gg,cmap='binary')
clf=svm.SVC()

clf.fit(image,predictions.values.ravel())

clf.score(image,predictions)
df_test.head()

df_test[df_test>0]=1

result=clf.predict(df_test)
mnist_answer=pd.DataFrame()

mnist_answer['Label']=result

mnist_answer.index.name='ImageId'

mnist_answer.index-=1



mnist_answer.to_csv('mnist_a.csv')

mnist_answer.head()
