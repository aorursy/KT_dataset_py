# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
PATH = "../input"

# Let's try to understand how to work with kaggle kernel
# Any results you write to the current directory are saved as output.
#Let's first explore then application train and import the data
# It takes time to importing the data because the adta set is large
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
# This task is very big. Let's use some giants shoulders to get things around and understand how to operate
# One thing to notice: you should write your code in a more readable way
print("application_train - rows:", application_train.shape[0],"columns:", application_train.shape[1])
print("application_test - rows:", application_test.shape[0],"columns:", application_test.shape[1])
print("bureau - rows:", bureau.shape[0],"columns:", bureau.shape[1])
print("bureau_balance - rows:", bureau_balance.shape[0],"columns:", bureau_balance.shape[1])
print("credit_card_balance - rows:", credit_card_balance.shape[0],"columns:", credit_card_balance.shape[1])
print("installments_payments - rows:", installments_payments.shape[0],"columns:", installments_payments.shape[1])
print("previous_application - rows:", previous_application.shape[0],"columns:", previous_application.shape[1])
print("POS_CASH_balance - rows:", POS_CASH_balance.shape[0],"columns:", POS_CASH_balance.shape[1])
# In here, we could see our dataset is not following the 80% - 20% rule
print(float(307511)/(307511+48744)) 
application_train.head(10)
# Spliting the dataset
y = application_train['TARGET'] # this is the label we could use for making the prediction
# Drop one column in application train
application_train.drop('TARGET', axis=1, inplace = True)

X = application_train 

# I want to see the heatmap for those attributes. This couold take very long time to process
import seaborn as sns
matrix = X
corr = matrix.corr()
fig, ax = plt.subplots(figsize=(100,100))
sns.heatmap(corr, annot= True, square = True)

#Let's try it out - the random forest algorithm 
from sklearn.ensemble import RandomForestClassifier 
#Isolate Data, class labels and column values - it seems like not easy to do it. 




# With the consideration of the domain of loan industry, which features are the most important?






















