# In this study I have utilized a dataset that exist at: 

# https://moodle.telt.unsw.edu.au/mod/resource/view.php?id=2535204.

# AND I UPLOADED THE SAME IN MY KAGGLE FOLDER INPUT\Credit Card Balanced Data. 

# Please don't get confused by MY folder name, 

# the folder has UN-BALANCED DATA, WHICH WILL BE BALANCED as we PROCEED.



# Please note, this

# dataset used in my Research Report is not exactly the same as the one used in the paper

# by Wickramasinghe, R. I. P. (2017), but very close. 



import pandas as pd

creditcard = pd.read_csv("../input/credit-card-balanced-data/creditcard.csv")



# ****** THE PAPER by Wickramasinghe, R. I. P. (2017). 

# titled "Attribute Noise, Classification Technique, and Classification Accuracy". 

# In Data Analytics and Decision Support for Cybersecurity (pp. 201-220). Springer, Cham. 

# USES THE FOLLOWING DATASET 

# This secondary dataset has been modified from the initial dataset, which contains

# credit cards’ transactions by European credit cards holders within two days in

# September 2013. This dataset includes 29 features including time, amount, and the

# time duration of the transaction in seconds

# Copyright©: This dataset is made available under the Open Database License

# (http://opendatacommons.org/licenses/odbl/1.0/). The Open Database License

# (ODbL) is license agreement intended to allow users to freely share, modify,

# and use this Dataset while maintaining this freedom for others, provided that the

# original source and author(s) are credited.



# *** BUT I AM NOT USING THE SAME for my Research Report *** BUT THE ONE PICKED FROM  

# https://moodle.telt.unsw.edu.au/mod/resource/view.php?id=2535204

# AND UPLOADED IN MY KAGGLE FOLDER INPUT\Credit Card Balanced Data. Please don't get confused by folder name, 

# the folder has UN-BALANCED DATA, WHICH WILL BE BALANCED as we PROCEED.
# Importing necessary libraries

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Loading data

data = pd.read_csv('../input/credit-card-balanced-data/creditcard.csv')
# View DataFrame

data.head()
data.shape
data.columns
data.info()
data.describe()  #statistical inference
# Visualising every feature

data.hist(figsize=(20,20))

plt.show()
# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]

Valid = data[data['Class'] == 0]



outlier_fraction = len(Fraud)/(len(Valid))

print(outlier_fraction)



print('Fraud Cases : {}'.format(len(Fraud)))

print('Valid Cases : {}'.format(len(Valid)))
# Correlation

corr = data.corr()

figure = plt.figure(figsize=(12,10))

sns.heatmap(corr)
# Splitting data

x = data.iloc[:,:-1].values

y = data.iloc[:,-1].values



from sklearn.model_selection import train_test_split

xtr,xtest,ytr,ytest = train_test_split(x,y,test_size=0.3,random_state=0)
xtr.shape,ytr.shape
xtest.shape,ytest.shape
from xgboost import XGBClassifier

xg = XGBClassifier(random_state=0)

xg.fit(xtr,ytr)

xg.score(xtr,ytr)
pred = xg.predict(xtest)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(pred,ytest)

from sklearn.metrics import accuracy_score

accuracy_score(pred,ytest)