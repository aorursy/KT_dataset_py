# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing all the required libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier

import scikitplot as skplt
data= pd.read_csv('../input/iris/Iris.csv')

print(data.shape)

data.head()
#checking for missing values using heat map

sns.heatmap(data.isnull(),linecolor="white");
#correlation between data

data.corr()

#using heatmap to show data correlation along with tabular corelation

sns.heatmap(data.corr(),annot=True,linecolor='grey');
#splitting the data and storing it

trainSet, testSet = train_test_split(data, test_size = 0.33)#test set is chosen to be 1/3rd of dataset
trainData = pd.DataFrame(trainSet[['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]).values

trainTarget = pd.DataFrame(trainSet[['Species']]).values.ravel()

testData = pd.DataFrame(testSet[['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]).values

testTarget = pd.DataFrame(testSet[['Species']]).values.ravel()
classifier = GaussianNB()

classifier.fit(trainData, trainTarget)

predicted_value = classifier.predict(testData)



predictions = dict()

accuracy = accuracy_score(testTarget,predicted_value) 

predictions['Naive-Bayes']=accuracy*100

print("The accuracy of the model is {}".format(accuracy))

confusionmatrix = confusion_matrix(testTarget, predicted_value)

cm=pd.DataFrame(confusion_matrix(testTarget, predicted_value))

print("The confusion matrix of the model is \n{}".format(cm))
skplt.metrics.plot_confusion_matrix(testTarget, predicted_value, normalize=True,cmap=plt.cm.Greys)

plt.show()