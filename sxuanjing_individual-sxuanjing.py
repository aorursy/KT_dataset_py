# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/learn-together/train.csv', index_col='Id')

test_data =  pd.read_csv('../input/learn-together/test.csv', index_col='Id')
df_train = train_data.copy()

df_test = test_data.copy()



print ('number of rows and columns', df_train.shape)

print ('number of rows and columns', df_test.shape)



#print (df_train.describe())

data_col = df_train.columns



data_col
#df_train.dropna(axis=0, subset=['Cover_Type'], inplace=True)

y = df_train.Cover_Type

df_train.drop(['Cover_Type'], axis=1, inplace=True)



print (df_train.shape)
total_row = df_train.shape[0]            

missing_val_count_by_column = (df_train.isnull().sum())

na_col = missing_val_count_by_column[missing_val_count_by_column > 0]/total_row



print(na_col)
col = df_train.columns.tolist()



numcol =col[0:10]

bincol = col[10:]



print (numcol)

print (bincol)


DIMS=(16, 15)



def drawdistplot(n, df, bins):

    fig = plt.figure(figsize=DIMS)

    drow = math.ceil(n/2)

    for i in range(n):

        fig.tight_layout()

        ax = fig.add_subplot(drow, 2,i+1)

        sns.distplot(df.iloc[:, i],kde=False, bins = bins)

        ax.set_title(df.columns[i])

    plt.show()
numx = len(numcol)

numdf = df_train[numcol]



drawdistplot(numx,numdf,20)


def drawbar(n, df):

    fig = plt.figure(figsize=DIMS)

    drow = math.ceil(n/2)

    for i in range(n):

        s = df.iloc[:, i].groupby(df.iloc[:, i]).size()

        #fig.tight_layout()

        ax = fig.add_subplot(drow, 2,i+1)

        s.plot.bar()

        ax.set_title(df.columns[i], fontsize = 12)

    plt.show()

    



        

    

binx = len(bincol)

bindf=df_train[bincol]

drawbar(binx, bindf)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_train, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_valid)



from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))
# Retrieve a list of columns that have more than 99% for 1 value



def showpro(n, df):

    col_list =[]

    for i in range(n):

        s = df.iloc[:, i].groupby(df.iloc[:, i]).size()/len(df)*100

        if (s>99.5).any():

            print (s)

            print ('-----------')

        else:

            col_list.append(s.name)

    return col_list
numcol.remove('Hillshade_9am')

numcol.remove('Hillshade_Noon')





rc = showpro(binx, bindf)

rc.extend(numcol)

print (rc)
X_train= X_train[rc]

X_valid= X_valid[rc]
print (X_train.shape)

print (X_valid.shape)



from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_valid)



from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))
for i in range(100, 700, 100):

    clf=RandomForestClassifier(n_estimators=i)



#Train the model using the training sets y_pred=clf.predict(X_test)

    clf.fit(X_train,y_train)



    y_pred=clf.predict(X_valid)





# Model Accuracy, how often is the classifier correct?

    print("Accuracy:", i ,metrics.accuracy_score(y_valid, y_pred))