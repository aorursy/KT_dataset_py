# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test_data.columns
print("TRAIN DATA - SHAPE" +str(train_data.shape))

print("TEST DATA - SHAPE" +str(test_data.shape))

print("_"*50)

print("COLUMNS TRAIN:")

print(train_data.columns.values)

print("_"*50)

print("COLUMNS TEST:")

print(test_data.columns.values)

print("_"*50)

print("COLUMNS GENDER:")

print(gender_data.columns.values)
print(train_data.info())

print("-"*50)

print(test_data.info())

print("-"*50)

print(gender_data.info())
print(train_data.describe())

print("-    "*20)

print(train_data.describe(include=['object']))

print("-"*100)

print(test_data.describe())

print("-"*100)

print(gender_data.describe())

print("TRAIN DATA")

print("Survived: " + str(train_data[train_data['Survived']==1]['Survived'].count()))

print("Died: " + str(train_data[train_data['Survived']==0]['Survived'].count()))

print('-------------------------------')

print("VALIDATION DATA")

print("Survived: " + str(gender_data[gender_data['Survived']==1]['Survived'].count()))

print("Died: " + str(gender_data[gender_data['Survived']==0]['Survived'].count()))



print(" ")

print("# of NULL")

for i in list(train_data.columns):

    print(i)

    print(train_data[i].isna().sum())

    print("---------------------------------")
train_data = pd.get_dummies(train_data, columns=['Sex','Pclass',"SibSp","Parch", 'Embarked'])

test_data = pd.get_dummies(test_data,columns=['Sex','Pclass',"SibSp","Parch",'Embarked'])

#train_data = train_data.drop('Sex_female', axis = 1)

#train_data= train_data.rename(columns={'Sex_male': 'Sex'})
print("SURVIVED - MEAN")

print(train_data[train_data["Survived"]==1].mean())

print("_ "*25)

print("NOT SURVIVED - MEAN")

print(train_data[train_data["Survived"]==0].mean())



train_data['Age_cat'] = pd.cut(train_data['Age'], bins = [0,12,24,36,48,60,72,84,96])
choice_column = ['Age','Fare',"Sex_female","Sex_male","Pclass_1","Pclass_2","Pclass_3"]

train_data.groupby(["Survived"])[choice_column].agg([np.mean,np.std, np.max,np.min]).T
train_data = train_data.fillna(train_data.mean())

test_data = test_data.fillna(test_data.mean())
def hist_columns (df):

    list_columns = list(df.columns)

    for i in list_columns:

        try:

            print(i)

            lenx = len(df[i].unique())

            if lenx >10:

                lenx = 10

            else:

                pass

            df_surv = df[df['Survived']==1]

            df_dief = df[df['Survived']==0]

            x = df_surv[i].plot.hist(bins=lenx, alpha = 0.5)

           # y = df_dief[i].plot.hist(bins=lenx,alpha = 0.5)

            plt.show()

            print("_"*50)

        except:

            print("------"*10)

            pass
print("SURVIVED")



hist_columns(train_data)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn import linear_model



def model_ml(x_model, columns):

    try:

        regx = eval(str(x_model))

        X_train = train_data[columns]

        X_test = test_data[columns]

        Y_train = train_data['Survived']

        regx.fit(X_train,Y_train)

        pred = regx.predict(X_test)

        accuracy = accuracy_score(gender_data["Survived"],pred.round())

        print(str(i) + ":  " + str(accuracy))

    except Exception as e:

        print(x_model,e)

        pass
models = ['linear_model.LinearRegression()','linear_model.Ridge()','linear_model.RidgeCV()',

          'linear_model.Lasso()','linear_model.MultiTaskLasso()','linear_model.ElasticNet()',

          'linear_model.MultiTaskElasticNet()','linear_model.Lars()','linear_model.LassoLars()',

          'linear_model.BayesianRidge()','linear_model.ARDRegression()','linear_model.LogisticRegression()',

          'linear_model.SGDRegressor()','linear_model.Perceptron()',"svm.SVC(kernel='linear', C=1)"]



cols = ['Age', 'Fare', 'Sex_female', 'Sex_male',

        'Pclass_1','Pclass_2', 'Pclass_3', 'SibSp_0','SibSp_1', 'SibSp_2',

        'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',

        'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5','Parch_6', 'Embarked_C',

        'Embarked_Q', 'Embarked_S']



for i in models:

    model_ml(i,cols)