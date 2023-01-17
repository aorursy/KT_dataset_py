# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# import the tabular data manipulation and analysis library: pandas

import pandas as pd 



# import thel linear algebra library: numpy

import numpy as np



# import machine learning model library: sklearn

import sklearn as sk



# import visualization libraries: matplotlib, seaborn

from matplotlib import pyplot as plt 

import seaborn as sns
# Read the variable into a pandas DataFrame

fulldata=pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
# Take a peek at top-10 rows of data using the head function of dataframe

fulldata.head(10)
pd.options.display.max_columns=50 # increase the number of columns displayed without truncation

pd.options.display.max_rows=999 # increase the number of rows displayed without truncation
# Take a peek at top-10 rows of data using the head function of the dataframe

fulldata.head(10)
# Use shape attribute of DataFrame object to obtain size

fulldata.shape
fulldata.loc[fulldata.EDUCATION==6,'EDUCATION']=5
# Check the current dtype

fulldata.dtypes
# Correct the dtypes for categorical variables

fulldata['ID']=fulldata['ID'].astype(object)

fulldata['SEX']=fulldata['SEX'].astype(object)

fulldata['EDUCATION']=fulldata['EDUCATION'].astype(object)

fulldata['MARRIAGE']=fulldata['MARRIAGE'].astype(object)
# Re-check the current dtype

fulldata.dtypes
# convert ID as the index for the dataframe

fulldata=fulldata.set_index('ID')

fulldata.head()
# Use describe() function of pandas DataFrame to get a summary of all numeric attributes. Use a .T at the end of the function call to make the output more readable

fulldata.describe().T
# Use KDE PLOT to get detailed distribution for each attribute

for aCol in fulldata.columns:

    if fulldata[aCol].dtype==object:

        continue

    print('Column:',aCol)

    sns.kdeplot(fulldata[aCol],shade=True)

    plt.show()
# Use _value_counts() to plot values using histogram

for aCol in fulldata.columns:

    if fulldata[aCol].dtype==object:

        if aCol=='ID':

            continue

        print(aCol)

        print('----------------------------')

#         plt.figure(figsize=(15,5))

        sns.barplot(fulldata[aCol].value_counts().index,fulldata[aCol].value_counts())

        plt.show()

        print(fulldata[aCol].value_counts())
rseed=11 # ensures reproducibility of results, detailed later
from sklearn.model_selection import train_test_split # helps split the data into multiple components
# split into X and y

fullX=fulldata.iloc[:,:-1]

fully=fulldata.iloc[:,-1]
# use train validation test split using command from sklearn to split into Train, Validation and Test

trainX,testX,trainy,testy=train_test_split(fullX,fully,random_state=rseed)
# convert categorical to one hot

catCols=[]

i=-1

for aCol in trainX.columns:

    i+=1

    if trainX[aCol].dtype != object:

        continue

    catCols.append(i)

    print(aCol)

print('Categorical Features:',catCols)

ohe=sk.preprocessing.OneHotEncoder(categorical_features=catCols)

ohe=ohe.fit(trainX)

trainX2=pd.DataFrame(ohe.transform(trainX).toarray())

testX2=pd.DataFrame(ohe.transform(testX).toarray())
# checking what trainX2 looks like

trainX2.head()
# values identified

ohe.categories_
# comparing with initial dataframe

trainX.head()
# import decision tree module from sklearn

from sklearn.tree import DecisionTreeClassifier



# create a DecisiopnTreeClassifier() object

model=DecisionTreeClassifier()
# use the fit function on the model to train the model using training data

model.fit(trainX2,trainy)
# use the predict function on training and test data to come up with training data predictions

trainp=model.predict(trainX2)

testp=model.predict(testX2)
print('training dataset accuracy:',sk.metrics.accuracy_score(trainy,trainp))

print('test dataset accuracy:',sk.metrics.accuracy_score(testy,testp))
print('TRAINING DATA')

plt.figure(figsize=(4,4))

sns.heatmap(sk.metrics.confusion_matrix(trainy,trainp),annot=True,fmt='d',linewidths=0.5,annot_kws={'size':20})

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
print('TESTING DATA')

plt.figure(figsize=(4,4))

sns.heatmap(sk.metrics.confusion_matrix(testy,testp),annot=True,fmt='d',linewidths=0.5,annot_kws={'size':20})

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()