# This Python 3 environment comes with many helpful analytics libraries installed





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory





import os

trainfile = '../input/digit-recognizer/train.csv'

traindata = pd.read_csv(trainfile).to_numpy()

traindata
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
X = traindata[:,1:]

y = traindata[:,0]

classifier = DecisionTreeClassifier()
classifier.fit(X,y)
testfile = '../input/digit-recognizer/test.csv'

testdata = pd.read_csv(testfile).to_numpy()

testdata
#printing any random data



ptd = testdata[0,:]

ptd.shape = (28,28)

plt.imshow(255-ptd,cmap ='gray')

print(classifier.predict([testdata[0]]))

#preparation for the submission

res = classifier.predict(testdata)

df =pd.DataFrame(res)

df = df.rename(columns={0:'#label'})

df['ImageId'] = [i+1 for i in range(len(df)) ]

df.set_index('ImageId',inplace=True)

df

df.to_csv('submission.csv')