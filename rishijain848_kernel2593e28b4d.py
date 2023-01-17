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
# Random Forest Classification



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

df=dataset

for i in range(891):

    df.iloc[i,3]=df.iloc[i,3].split(", ")[1].split(".")[0]

    

df['Name'] = df["Name"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle',

       'the Countess'], 'Rare')



for i in range(891):

    df.iloc[i,10]=str(df.iloc[i,10])

    if (df.iloc[i,10]!="nan"):

        df.iloc[i,10]=df.iloc[i,10].split(" ")[0]

        df.iloc[i,10]=df.iloc[i,10][0]

    if (df.iloc[i,10] == "nan"):

        df.iloc[i,10]="X"



for i in range(891):



    if (df.iloc[i,10]=="T"):

        df.iloc[i,10]="X"





df["Family"]=df["SibSp"]

df["Alone"]=df["Family"]

for i in range(891):

    df.iloc[i,12]=df.iloc[i,6]+df.iloc[i,7]+1

    if df.iloc[i,12]==1:

        df.iloc[i,13]=1

    if df.iloc[i,12] != 1 :

        df.iloc[i,13]=0



df["SibSp"]=df["Family"]

df["Parch"]=df["Alone"]

df=df.drop(["Family","Alone"],axis=1)





X = df.iloc[:, [2,3,4,5,6,7,9,10,11]].values

y = df.iloc[:,1 ].values





from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 3:4])

X[:, 3:4] = imputer.transform(X[:, 3:4])



from sklearn_pandas import CategoricalImputer

imputer_2 = CategoricalImputer()

imputer_2.fit(X[:,8:])

X[:,8:]=imputer_2.transform(X[:,8:])



# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,1]= labelencoder_X.fit_transform(X[:,1])

X[:,2]=labelencoder_X.fit_transform(X[:,2])

X[:,7]=labelencoder_X.fit_transform(X[:,7])

X[:,8]=labelencoder_X.fit_transform(X[:,8])





onehotencoder = OneHotEncoder(categorical_features=[1,2,7,8])

X= onehotencoder.fit_transform(X).toarray()



x1=pd.DataFrame(X)



# Splitting the dataset into the Training set and Test set

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

classifier.fit(X, y)



df2=pd.read_csv("/kaggle/input/titanic/test.csv")



for i in range(418):

    df2.iloc[i,2]=df2.iloc[i,2].split(", ")[1].split(".")[0]

    

df2["Name"] = df2["Name"].replace([  'Ms','Col', 'Dr', 'Rev', 'Dona'], 'Rare')





for i in range(418):

    df2.iloc[i,9]=str(df2.iloc[i,9])

    if (df2.iloc[i,9]!="nan"):

        df2.iloc[i,9]=df2.iloc[i,9].split(" ")[0]

        df2.iloc[i,9]=df2.iloc[i,9][0]

    if (df2.iloc[i,9] == "nan"):

        df2.iloc[i,9]="X"

        

        



df2["Family"]=df2["SibSp"]

df2["Alone"]=df2["Family"]





for i in range(418):

    df2.iloc[i,11]=df2.iloc[i,5]+df2.iloc[i,6]+1

    if df2.iloc[i,11]==1:

        df2.iloc[i,12]=1

    if df2.iloc[i,11] != 1 :

        df2.iloc[i,12]=0



df2["SibSp"]=df2["Family"]

df2["Parch"]=df2["Alone"]

df2=df2.drop(["Family","Alone"],axis=1)







X=df2.iloc[:,[1,2,3,4,5,6,8,9,10]].values



from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

X[:,3:4] = imputer.fit_transform(X[:, 3:4])

X[:,6:7] = imputer.fit_transform(X[:,6:7])



# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,1]= labelencoder_X.fit_transform(X[:,1])

X[:,2]=labelencoder_X.fit_transform(X[:,2])

X[:,7]=labelencoder_X.fit_transform(X[:,7])

X[:,8]=labelencoder_X.fit_transform(X[:,8])



onehotencoder = OneHotEncoder(categorical_features=[1,2,7,8])

X= onehotencoder.fit_transform(X).toarray()



x2=pd.DataFrame(X)







y_pred = classifier.predict(X)









# Making the Confusion Matrix

#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, y_pred)





    

    



y_pred
