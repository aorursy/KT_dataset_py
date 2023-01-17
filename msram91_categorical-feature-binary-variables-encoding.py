# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the Train file 

X_train=pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

X_train.info()
# Reading Test file

X_test=pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

X_test.head()
# Identifying Missing Values perecntage of train data set

round((X_train.isnull().sum()/len(X_train)*100),2)
# Identifying Missing Values perecntage of test data set

round((X_test.isnull().sum()/len(X_test)*100),2)
# Get list of categorical variables

s = X_train.select_dtypes(include=['object'])

Cat_cols=s.columns.to_list()

#Imputing Missg Values with Mode() for categorical columns

for c in Cat_cols:

    X_train[c].fillna(X_train[c].mode()[0], inplace = True)  

round((X_train.isnull().sum()/len(X_train)*100),2)    
#Imputing Missg Values with Mode() for categorical columns-Test data set

for c in Cat_cols:

    X_test[c].fillna(X_test[c].mode()[0], inplace = True)  

round((X_test.isnull().sum()/len(X_test)*100),2)  
Num_cols =X_train.select_dtypes(include=['int64','float']).columns.to_list()

Num_cols.pop(Num_cols.index('id'))  #Removing ID as it not requried

Num_cols





for c in Num_cols:

    X_train[c].fillna(np.mean(X_train[c]), inplace = True)  

round((X_train.isnull().sum()/len(X_train)*100),2)
# Test Dataset

Num_cols =X_test.select_dtypes(include=['int64','float']).columns.to_list()

Num_cols.pop(Num_cols.index('id'))  #Removing ID as it not requried

Num_cols





for c in Num_cols:

    X_test[c].fillna(np.mean(X_test[c]), inplace = True)  

round((X_test.isnull().sum()/len(X_test)*100),2)


bin_cols = [col for col in X_train if col.startswith('bin')]



for i in bin_cols:

    print([i],X_train[i].unique())
bin_cols = [col for col in X_test if col.startswith('bin')]



for i in bin_cols:

    print([i],X_test[i].unique())
# Converting into Binary values

for i in bin_cols[:3]:

    X_train[i]=X_train[i].apply(lambda x : 1 if x>0.5 else 0)

    

for i in bin_cols[:3]:

    X_test[i]=X_test[i].apply(lambda x : 1 if x>0.5 else 0)    

     
X_train['bin_0'].unique()
import seaborn as sns

import matplotlib.pyplot as plt 
# Binary variable visualisation



for i in bin_cols:

    sns.countplot(x=i, hue="target", data=X_train)

    plt.title(i)

    plt.show()
# Bin_3 and Bin_4- We convert categorical into Binary values

X_train['bin_3']=X_train['bin_3'].apply(lambda x : 1 if 'F' else 'T')

X_train['bin_4']=X_train['bin_2'].apply(lambda x : 1 if 'N' else 'Y')



# Test set

X_test['bin_3']=X_test['bin_3'].apply(lambda x : 1 if 'F' else 'T')

X_test['bin_4']=X_test['bin_2'].apply(lambda x : 1 if 'N' else 'Y')



X_train[bin_cols].head()
ord_cols = [col for col in X_train if col.startswith('ord')]



for i in ord_cols:

    print([i],X_train[i].unique())
nominal_cols = [col for col in X_train if col.startswith('nom')]



for i in nominal_cols:

    print([i],X_train[i].unique())
X_train.info()
cat_cols_encoding= X_train.select_dtypes(include=['object']).columns.to_list()

X_test[cat_cols_encoding].head()

# X_train[cat_cols_encoding].head()

X_test.columns
# we will do Random Forest Method to Identify the target. for that we will do Label encoding.





cat_cols_encoding= X_train.select_dtypes(include=['object']).columns.to_list()



from sklearn.preprocessing import LabelEncoder



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()



for col in cat_cols_encoding:

    X_train[c]= label_encoder.fit_transform(X_train[c])

 



for c in cat_cols_encoding:

    X_test[c] = label_encoder.transform(X_test[c])