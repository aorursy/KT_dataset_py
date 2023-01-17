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
#read data file

df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

print(df)

#read first 6 rows



df.head(6)
#read last 6 rows

df.tail(6)
##read structure of dataframe

#df in which 8124 rows,23 columns

#all not null values

#all datatype-object,object dtype is for all text data; other dtypes-int,float,date-time

df.info()

#t1-no null values

#t2-all data should be in numeric format only



#label encoding for single column

df["class"].value_counts()#eg-in column class all "e" and "p" are replaced with numeric format & total=8124

#Fetch features of type object

objfeatures = df.select_dtypes(include="object").columns

print(objfeatures)
#label encoding for entire dataset;



#itrate a loop for features of type object

from sklearn import preprocessing

le = preprocessing.LabelEncoder()#it will assign unique values for individual features i.e cols



for feat in objfeatures:#for loop will run for 23 cols i.e range objfeature

    df[feat] = le.fit_transform(df[feat].astype(str))#all "object(str) dtype" will replace by "int dtype"





#df["class"] = le.fit_transform(df["class"].astype(str)):-internally it will assign unique values for each col

#fit-it will understand internaL data & transform - replace values and it will be saved in same dataframe itself



df.info()
#x & y

X = df.drop(["class"],axis=1)#dropping class form level 1 i.e x= other col except classfeatures

y = df['class']#explicitly for class y=labels

X.info()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train.info()#70%values

X_test.info()#30%values
#model training

from sklearn.naive_bayes import GaussianNB #gaussian naive bayes



#gnb ="" #training empty brain from algo 



gnb = GaussianNB()#creating variable of algo and follow rules defined by algo



gnb.fit(X_train, y_train) #training empty brain i.e for given x these was y i.e understanding 70% data
#predicting Y value

y_prediction = gnb.predict(X_test)



print("Train score",gnb.score(X_train, y_train)*100)

print("Test score",gnb.score(X_test, y_test)*100)

print(y_prediction)
print(y_test)
for i in y_prediction:

    print(i)
temp_predictions =[] #emty array 



for i, val in enumerate(y_prediction):#preprocessing it make array 2d 

    if val == 0:

        temp_predictions.append('e')

    if val == 1:

        temp_predictions.append('p')

            

            

            
for i in temp_predictions:

    print(i)
for d, c in zip(temp_predictions,y_prediction):

    print(d,c)
print("Train score",clf_gnb.score(X_train, y_train)*100)

print("Test score",clf_gnb.score(X_test, y_test)*100)