# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')
#our Data File is Loaded into 'data'
#lets check our loaded Data By Following Command
data.head(4)
#To Get Full Information Of the data
data.info()
# In this data We can find some null data like '?'
#'veil-type' and 'stalk-root' has Some NaN values 
#so Delete these 2 columns from the data
data.drop('veil-type',axis=1,inplace=True)
data.drop('stalk-root',axis=1,inplace=True)
data.info()
# data purification is Done
#Now Lets Move On to the next step
#First Column Is Our Target
#It Has 2 Catagories ('p','e')
data['class'].unique()
#In Machine Learning , we Need To Transform Every Data Into Integer of float
#because In ML,Only Numbers Are The Valid ones
#Our Given Data Is In The Form Of Strings
#SO Lets Convert All Our data in 'numbers' form

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in data.columns:
    data[i] = le.fit_transform(data[i])
data.head(5)
# We need features and labels to perform Our Further Process
# Our Target Is 'class' Column
Y = data['class']
# Except 'class' column all are features 
data1 = data
data1.drop('class',axis=1,inplace=True)
data1.head(5)
# we done with droping 'class' column from data
# So Remained Data is belongs to 'features'
#now load data1 into 'X'
X = data1
X.head(5)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=42,test_size=0.33)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
# Let's Take another model
from sklearn import svm
model = svm.SVC(kernel='rbf',gamma=0.3,C=1)
model.fit(X_train,y_train)
model.score(X_test,y_test)