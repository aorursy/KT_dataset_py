# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import csv

import matplotlib.pyplot as plt

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

%matplotlib inline



#read file

items = pd.read_csv("../input/creditcard.csv")



#get human readable class definitions

items['Class'] = items['Class'].map({1: "Fraud", 0 :  "Regular"})



#get our feature trying to calculate

fraudclass = items["Class"]



items = items.drop("Class", axis=1)



#get train and test data from df

X_train, X_test, y_train, y_test = train_test_split(items, fraudclass, test_size=0.1, random_state=0)



#create prediction model for our data set

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 



#function to classify a specific row based on our classifier

#NOT WORKING FOR SOME REASON -- MAYBE YOU CAN FIGURE IT OUT



#def classifyRow(rowNum):

    #row = items.iloc[rowNum]

    #print rfc.predict(row)





#classifyRow(2)





print(classification_report(y_pred,y_test))