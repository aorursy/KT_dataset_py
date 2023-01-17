# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

%matplotlib inline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline 

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score,make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.







gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

#test = pd.read_csv("../input/titanic/test.csv",index_col = 'PassengerId')

test = pd.read_csv("../input/titanic/test.csv")

submi = test['PassengerId']

train = pd.read_csv("../input/titanic/train.csv",index_col = 'PassengerId')











train_set , test_set = train_test_split(train,test_size = 0.2, random_state = 42,)













train_set.head()



train_set.info()
missing_val_count_by_column = (train_set.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

train_set.describe()
train_set.hist(bins=10, figsize=(20,15))

plt.show()
correlation_matrix = train_set.corr()



plt.figure(figsize=(14,7))



plt.title("Correlation Matrix")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=correlation_matrix, annot=True)



train_set = train_set.drop(['Ticket', 'Cabin','Name'], axis=1)

test_set = test_set.drop(['Ticket', 'Cabin','Name'], axis=1)

test = test.drop(['Ticket', 'Cabin','Name','PassengerId'], axis=1)

dataset = [train_set, test_set,test]



for data in dataset: 

    data['Embarked'] = data['Embarked'].fillna('S')
missing_val_count_by_column = (dataset[0].isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
age_mean = train_set['Age'].mean()

print(age_mean)



for data in dataset: 

    data['Age'].fillna( age_mean,inplace = True)
def group_age(age):

    if (age < 13):

        return 0

    elif (age < 50):

        return 1

    else:

        return 2

    

for data in dataset:

    data['Age_groupe'] = data['Age'].apply(group_age)

sns.barplot(train_set['Age_groupe'], train_set['Survived'])
y_train = train_set['Survived']

y_test = test_set['Survived']



X_train_full = train_set.drop(['Survived'],axis=1)

X_valid_full = test_set.drop(['Survived'],axis=1)

s = (dataset[0].dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
label_X_train = X_train_full.copy()

label_X_valid = X_valid_full.copy()

label_X_test = test.copy()

# Apply label encoder to each column with categorical data





label_encoder = LabelEncoder()



for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train_full[col])

    label_X_valid[col] = label_encoder.transform(X_valid_full[col])

    label_X_test[col] = label_encoder.transform(test[col])

my_imputer = SimpleImputer()



imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(label_X_valid))

imputed_X_test = pd.DataFrame(my_imputer.transform(label_X_test))



# Imputation removed column names; put them back

imputed_X_train.columns = label_X_train.columns

imputed_X_valid.columns = label_X_valid.columns

imputed_X_test.columns = label_X_test.columns
sgd_model = SGDClassifier(random_state=42)

sgd_model.fit(imputed_X_train, y_train)

sgd_val_predictions = sgd_model.predict(imputed_X_valid)

sgd_conf_matrix = confusion_matrix(sgd_val_predictions,y_test)



plt.figure(figsize=(14,7))

plt.title("Confusion Matrix")

sns.heatmap(data=sgd_conf_matrix, annot=True)

print(sgd_model.score(imputed_X_valid, y_test))
dtc_model = DecisionTreeClassifier(random_state=42)

dtc_model.fit(imputed_X_train, y_train)

dtc_val_predictions = dtc_model.predict(imputed_X_valid)

dtc_conf_matrix = confusion_matrix(dtc_val_predictions,y_test)



plt.figure(figsize=(14,7))

plt.title("Confusion Matrix")

plt.ylabel('True label')

plt.xlabel('Predicted label')



sns.heatmap(data=dtc_conf_matrix, annot=True)

print(dtc_model.score(imputed_X_valid, y_test))




rfc_model = RandomForestClassifier(random_state = 42,n_estimators = 100)

rfc_model.fit(imputed_X_train, y_train)

rfc_val_predictions = rfc_model.predict(imputed_X_valid)

rfc_conf_matrix = confusion_matrix(rfc_val_predictions,y_test)



plt.figure(figsize=(14,7))

plt.title("Confusion Matrix")

plt.ylabel('True label')

plt.xlabel('Predicted label')



sns.heatmap(data=rfc_conf_matrix, annot=True)

print(rfc_model.score(imputed_X_valid, y_test))
knn_model = KNeighborsClassifier(n_neighbors = 3)

knn_model.fit(imputed_X_train, y_train)

knn_val_predictions = knn_model.predict(imputed_X_valid)

knn_conf_matrix = confusion_matrix(knn_val_predictions,y_test)



plt.figure(figsize=(14,7))

plt.title("Confusion Matrix")

plt.ylabel('True label')

plt.xlabel('Predicted label')



sns.heatmap(data=knn_conf_matrix, annot=True)

print(knn_model.score(imputed_X_valid, y_test))
f1_sdg = f1_score(sgd_val_predictions,y_test)

print("le score f1 du SGDClassier est de ",f1_sdg)

f1_dtc = f1_score(dtc_val_predictions,y_test)

print("le score f1 du DecisionTreeClassifier est de ",f1_dtc)

f1_knn = f1_score(knn_val_predictions,y_test)

print("le score f1 du KNN est de ",f1_knn)

f1_rfc = f1_score(rfc_val_predictions,y_test)

print("le score f1 du RandomForestClassifier est  de ",f1_rfc)
f1 = make_scorer((f1_score) , average='binary')
param_grid =[{

    #'max_depth': [i for i in range(0,2000,10)], 

    'min_samples_split': [i for i in range(0,10,1)]

    }]

grid_search = GridSearchCV(dtc_model,param_grid,cv=5,n_jobs=-1)

grid_search.fit(imputed_X_train,y_train)

print(grid_search.best_params_)








# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features





# make predictions which we will submit. 

test_preds = rfc_model.predict(imputed_X_test)

print(test_preds)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'PassengerId': submi,

                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)