import numpy as np

import pandas as pd

from pandas import read_csv

from pandas import set_option



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

%matplotlib inline 



# 1. Load dataset

train_df = read_csv("../input/train.csv")

test_df = read_csv("../input/test.csv")
# 2. Data preprocessing

# check shape (row and column count)

print (train_df.shape)
# check few head datas  

# we can see missing value (NaN)

# and Name, PassengerId, Ticket columns is object type.

train_df.head(10)
# check mssing value 

for col_name in train_df.columns.values:

    col_null_ct = train_df[col_name].isnull().sum()

    if col_null_ct.sum() > 0:

        print ("col_name:%s, count:%d" % (col_name, col_null_ct))
# check type

for col in train_df:

    print (col, train_df[col].dtypes)
# check complexity

train_df['Sex'].unique()
# check complexity

train_df['Embarked'].unique()
# We set a naive Hypotheses is that drop all column having object type. (not consider to transform object to numeric)

# Except Sex and Embarked column because they have less complexity

train_df = train_df.drop(['Name','PassengerId','Ticket'], axis=1)
# Drop the Cabin, Age column because they have many missing values.

train_df = train_df.drop(['Cabin','Age'], axis=1)
# Transform object to numeric

train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Check the Embarked values distribution

train_df.groupby('Embarked').size()
# Fill missing value for Embarked

train_df['Embarked'] = train_df['Embarked'].fillna('S')
# Transform object to numeric

train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# check table

train_df.head(10)
# 3. Data EDA

# Set floating point precision to 3

set_option( "precision" , 3)

# Check descriptive statis metrics

print(train_df.describe())
# Check the Data balance

train_df.groupby('Survived').size()
# Check density

train_df.plot(kind= "density" , subplots=True, layout=(3,3), sharex=False, legend=False, fontsize=1)
# Check middle and distribution

train_df.plot(kind= "box" , subplots=True, layout=(3,3), sharex=False, sharey=False, fontsize=1)
# Check correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(train_df.corr(), vmin=-1, vmax=1, interpolation= "none" )

fig.colorbar(cax)
# 4. Evaluate Algorithms: Baseline



# Test options and evaluation metric

num_folds = 10

seed = 7

scoring = "accuracy"



# Spot-Check Algorithms

models = []

models.append(( "LR" , LogisticRegression()))

models.append(( "LDA" , LinearDiscriminantAnalysis()))

models.append(( "KNN" , KNeighborsClassifier()))

models.append(( "CART" , DecisionTreeClassifier()))

models.append(( "NB" , GaussianNB()))

models.append(( "SVM" , SVC()))
# Split-out validation dataset

array = train_df.values



X = array[:,1:7].astype(float)

Y = array[:,0]



validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print (msg)

# We can see LogisticRegression is best performance
# We can transform all row data to have zero mean and unit variance

# Standardize the dataset

pipelines = []

pipelines.append(( "ScaledLR" , Pipeline([( "Scaler" , StandardScaler()),( "LR" , LogisticRegression())])))

pipelines.append(( "ScaledLDA" , Pipeline([( "Scaler" , StandardScaler()),( "LDA" , LinearDiscriminantAnalysis())])))

pipelines.append(( "ScaledKNN" , Pipeline([( "Scaler" , StandardScaler()),( "KNN" , KNeighborsClassifier())])))

pipelines.append(( "ScaledCART" , Pipeline([( "Scaler" , StandardScaler()),( "CART" , DecisionTreeClassifier())])))

pipelines.append(( "ScaledNB" , Pipeline([( "Scaler" , StandardScaler()),( "NB" , GaussianNB())])))

pipelines.append(( "ScaledSVM" , Pipeline([( "Scaler" , StandardScaler()),( "SVM" , SVC())])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# We can see SVM is best performance now
# prepare predict the model

test_df_bk = test_df

test_df = test_df.drop(['Cabin','Age'], axis=1)

test_df = test_df.drop(['Name','PassengerId','Ticket'], axis=1)

test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df['Embarked'] = test_df['Embarked'].fillna('S')

test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())

X_test = test_df.values
# 5. Prepare the model

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = SVC()

model.fit(rescaledX, Y_train)



# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model.predict(rescaledValidationX)



#create solution csv for submission

PassengerId = np.array(test_df_bk['PassengerId']).astype(int)

predictions = predictions.astype(int)

submission = pd.DataFrame(predictions, PassengerId, columns=['Survived'])

print(submission.shape)

#submission.to_csv('Titanic_solution', index_label = ['PassengerId'])