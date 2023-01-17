# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split 

from sklearn import svm
import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split



myCsvfile = pd.read_csv('../input/train.csv')

myImageO = myCsvfile.iloc[:500,1:]

myLableO = myCsvfile.iloc[:500,:1]

from sklearn.utils import shuffle



myImage, myLable = shuffle(myImageO,myLableO, random_state = 2)





#myImage = myImage.reshape(myImage,1,28,28)
myTrainImage, myTestImage, myTrainLabel, myTestLabel = train_test_split(myImage,myLable, train_size = 0.8 , random_state = 0)


import matplotlib.pyplot as plt, matplotlib.image as mpimg

#plt.imshow(myTemp, cmap = 'grey')

%matplotlib inline

myTrainImage, myTestImage, myTrainLabel, myTestLabel = train_test_split(myImage,myLable, train_size = 0.8 , random_state = 0)

#myTemp = myTrainImage.iloc[1]

#myTrainImage = myTrainImage.as_matrix((28,28))

#myTrainImage.head()

myTrainImage = myTrainImage.as_matrix()

myTemp1 = myTrainImage[0]

myTrainImage = myTrainImage.reshape(myTemp1,1,28,28)

#plt.imshow(myTemp, cmap = 'gray')

#plt.hist(myTemp)
myTrainImage[0].values.count()
print ("hi")






myTrainImage[myTrainImage>0]=1

myTestImage[myTestImage>0]=1

'''

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

'''









mySvm = svm.SVC()

mySvm.fit(myTrainImage,myTrainLabel.values.ravel())



myRealTestImages = pd.read_csv('../input/test.csv')

#myRealTestImages = myTestData[]

myRealTestImages[myRealTestImages>0]=1

myPredictedValues = mySvm.predict(myRealTestImages)



df = pd.DataFrame(myPredictedValues)

df.index.name = 'ImageId'

df.index += 1

df.columns =  ['Label']

df.head()

df.to_csv('results4.csv')

mySvm.score(myTestImage,myTestLabel)



myCnn = Sequential()

myCnn.add(Dense())