import pandas as pd

import numpy as np

pd.set_option('max_rows', 1000)

pd.set_option('max_columns', 200)

from IPython.display import display

# reading data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

display(train.head())

display(train.describe())

train.info()
def data_cleaning(data):

    # encode sex

    data.Sex =  data.Sex.factorize()[0]

    # encoding miss age 

    data.Age = data.Age.apply(lambda x: x//10)

    data.Age.fillna(-1, inplace=True)

    data.Fare = data.Fare.apply(lambda x: x//50)

    # NAN becomes n

    data.Cabin = data.Cabin.astype(str).apply(lambda x: x[0])

    data.Cabin = data.Cabin.factorize()[0]

    data.Embarked = data.Embarked.astype(str).apply(lambda x: x[0])

    data.Embarked = data.Embarked.factorize()[0]

    return data



train_cleaned = data_cleaning(train)

test_cleaned = data_cleaning(test)
def trainVStest(train, test, *cols):

    for col in cols:

        display(test[col].value_counts(normalize=True))

        display(train[col].value_counts(normalize=True))



trainVStest(train_cleaned, test_cleaned, 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked')
# plot against target to extract information

import matplotlib.pyplot as plt 

def plotVStarget(data, target, *cols):

    for col in cols:

        table = data.groupby([col], as_index=False)[target].mean()

        plt.bar(table.iloc[:, 0], table.iloc[:, 1])

        plt.title(col)

        plt.show()

plotVStarget(train_cleaned, 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked')
# make sure train test encoding is the same before apply this, as we see from EDA, there are some classes not available in test

# from Age plot

def data_engineering(data):

    data['G_Age'] = data.Age.apply(lambda x: 1 if x==0 else 0)

    data['G_SibSp'] = data.SibSp.apply(lambda x:1 if (x==1 or x==2) else 0)

    data['B_SibSp'] = data.SibSp.apply(lambda x:1 if x>=4 else 0)

    data['G_Parch'] = data.Parch.apply(lambda x:1 if (x==1 or x==2 or x==3) else 0)

    data['B_Parch'] = data.Parch.apply(lambda x:1 if x>=5 else 0)

    data['B_Fare'] = data.Fare.apply(lambda x:1 if x==0 else 0)

    data['G_Cabin'] = data.Cabin.apply(lambda x:1 if (x==2 or x==4 or x==6) else 0)

    data['B_Cabin'] = data.Cabin.apply(lambda x:1 if x==0 else 0)

    data['G_Embarked'] = data.Embarked.apply(lambda x:1 if x==3 else 0)

    return data

train_cleaned = data_engineering(train_cleaned)

test_cleaned = data_engineering(test_cleaned)
# before stacking, make sure no NANs or non-numeric for the input columns 

display(train_cleaned.isnull().sum(axis=0))

display(test_cleaned.isnull().sum(axis=0))
# fill in missing fare with median for test_cleaned

test_cleaned.Fare.fillna(test_cleaned.Fare.median(), inplace=True)
# prepare the input columns to model fitting

train_col = train_cleaned.columns.tolist()

test_col = test_cleaned.columns.tolist()

print(train_col)

print(test_col)

input_col = set(test_col) - set(['PassengerId', 'Name', 'Ticket'])

print(input_col)
# 1st level: GBT_R, RF_R, SVR, LR. KNN_R 

# 2nd level: Linear model

import sklearn.ensemble as ensemble

import sklearn.neighbors as neighbors

import sklearn.linear_model as linear

import sklearn.svm as svm

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn import tree

from sklearn import neural_network

from sklearn.model_selection import train_test_split



models = [[ensemble.GradientBoostingRegressor(),

           ensemble.RandomForestRegressor(),

           svm.SVR(),

           linear.BayesianRidge(),

           GaussianProcessRegressor(),

           tree.DecisionTreeRegressor(),

           neural_network.MLPRegressor(),

           neighbors.KNeighborsRegressor()], 

         linear.BayesianRidge()]



# creates dataframe to store first level result

first_level_B = pd.DataFrame()

first_level_C = pd.DataFrame()

i = 0

for AB_C in [(train_cleaned[input_col], test_cleaned[input_col])]:

    AB, C = AB_C[0], AB_C[1]

    A_x, B_x, A_y, B_y = train_test_split(AB, train_cleaned.Survived, test_size=0.5, random_state=42)

    for model in models[0]:

        model.fit(A_x, A_y)

        first_level_B[type(model).__name__+'_'+str(i)] = model.predict(B_x)

        first_level_C[type(model).__name__+'_'+str(i)] = model.predict(C)

    i += 1

    

first_level_B['B_y'] = B_y.to_numpy()

display(first_level_B.head())

first_level_C.head()
# perform second level stacking

final_model = models[1]

final_model.fit(first_level_B.iloc[:,:-1], first_level_B.iloc[:,-1])

from sklearn.metrics import accuracy_score

training_prediction = final_model.predict(first_level_B.iloc[:,:-1])

training_prediction = training_prediction > 0.5

training_accuracy = accuracy_score(first_level_B.iloc[:,-1], training_prediction)

print(training_accuracy)



submission = final_model.predict(first_level_C.iloc[:,:]) > 0.5
# output submission.csv

submission_df = pd.DataFrame({'PassengerId':test_cleaned.PassengerId , 'Survived':submission.astype(np.int32)})

submission_df.to_csv('./submission.csv', index=False)

from IPython.display import FileLink

FileLink(r'./submission.csv')