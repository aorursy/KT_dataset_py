# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/data.csv")

#deleting unwanted column

del data['Unnamed: 32']



#creating new column cancer_type

data['cancer_type'] = data['diagnosis'].astype('category').cat.codes 

    



#deleting diagnosis as now it is of no use



del data['diagnosis']



#print(data.shape)

#print(data.columns.values)



#Seperating features and lable(target value)

features = data.iloc[:,1:31]

target = data.iloc[:,31]

#print(features.head())

#print(target.head())



#spliting dataset into train and test

X_train,X_test,y_train,y_test = train_test_split(features,target,train_size=0.7,random_state=3)



#creating an instance of RandomForestClassifier model

clf = RandomForestClassifier()



#fitting training data into model

clf.fit(X_train,y_train)



#calculating accuracy

pred = clf.predict(X_test)

print("Accuracy ",accuracy_score(y_test,pred))



#plt.plot(y_test,pred)

#plt.show()

# Any results you write to the current directory are saved as output.