# Import our libraries

import pandas as pd

import numpy as np



# Import sklearn libraries

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, make_scorer, confusion_matrix, f1_score, fbeta_score



# Import the Naive Bayes, logistic regression, Bagging, RandomForest, AdaBoost, GradientBoost, Decision Trees and SVM Classifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from xgboost import XGBClassifier

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#from matplotlib import style

#plt.style.use('bmh')

#plt.style.use('ggplot')

plt.style.use('seaborn-notebook')

from matplotlib.ticker import StrMethodFormatter

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer



from sklearn.metrics import classification_report, confusion_matrix
# load the datasets

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")



print("Train size: {0}\nTest size: {1}".format(train.shape,test.shape))
train.head()
test.isna().sum()

test.fillna(value=np.mean(test['Fare']), inplace=True)
child_age = 18

def get_person(passenger):

    age, sex = passenger

    if (age < child_age):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

train = pd.concat([train, pd.DataFrame(train[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)

train = pd.concat([train, pd.get_dummies(train['person'])], axis=1)
test = pd.concat([test, pd.DataFrame(test[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)

test = pd.concat([test, pd.get_dummies(test['person'])], axis=1)
train['surname'] = train["Name"].apply(lambda x: x.split(',')[0].lower())

test['surname'] = test["Name"].apply(lambda x: x.split(',')[0].lower())
train.head()
test.head()
### FEATURES BASED ON SURNAME    --------------------------------------------------------

table_surname = pd.DataFrame(train["surname"].value_counts())

table_surname.rename(columns={'surname':'Surname_Members'}, inplace=True)



table_surname['Surname_perishing_women'] = train.surname[(train.female_adult == 1.0) 

                                    & (train.Survived == 0.0) 

                                    & ((train.Parch > 0) | (train.SibSp > 0))].value_counts()

table_surname['Surname_perishing_women'] = table_surname['Surname_perishing_women'].fillna(0)

table_surname['Surname_perishing_women'][table_surname['Surname_perishing_women'] > 0] = 1.0 



table_surname['Surname_surviving_men'] = train.surname[(train.male_adult == 1.0) 

                                    & (train.Survived == 1.0) 

                                    & ((train.Parch > 0) | (train.SibSp > 0))].value_counts()

table_surname['Surname_surviving_men'] = table_surname['Surname_surviving_men'].fillna(0)

table_surname['Surname_surviving_men'][table_surname['Surname_surviving_men'] > 0] = 1.0 



####O PULO DO GATO FOI AQUI

table_surname['Surname_Id'] = pd.CategoricalIndex(table_surname.index).codes

table_surname["Surname_Id"][table_surname["Surname_Members"] < 3 ] = -1

table_surname["Surname_Members"] = pd.cut(table_surname["Surname_Members"], bins=[0,1,4,20], labels=[0,1,2])

train = pd.merge(train, table_surname, left_on="surname",right_index=True,how='left', sort=False)
### FEATURES BASED ON SURNAME    --------------------------------------------------------

table_surname = pd.DataFrame(test["surname"].value_counts())

table_surname.rename(columns={'surname':'Surname_Members'}, inplace=True)



table_surname['Surname_perishing_women'] = test.surname[(test.female_adult == 1.0) 

                                    & ((test.Parch > 0) | (test.SibSp > 0))].value_counts()

table_surname['Surname_perishing_women'] = table_surname['Surname_perishing_women'].fillna(0)

table_surname['Surname_perishing_women'][table_surname['Surname_perishing_women'] > 0] = 1.0 



table_surname['Surname_surviving_men'] = test.surname[(test.male_adult == 1.0) 

                                    & ((test.Parch > 0) | (test.SibSp > 0))].value_counts()

table_surname['Surname_surviving_men'] = table_surname['Surname_surviving_men'].fillna(0)

table_surname['Surname_surviving_men'][table_surname['Surname_surviving_men'] > 0] = 1.0 



####O PULO DO GATO FOI AQUI

table_surname['Surname_Id'] = pd.CategoricalIndex(table_surname.index).codes

table_surname["Surname_Id"][table_surname["Surname_Members"] < 3 ] = -1

table_surname["Surname_Members"] = pd.cut(table_surname["Surname_Members"], bins=[0,1,4,20], labels=[0,1,2])

test = pd.merge(test, table_surname, left_on="surname",right_index=True,how='left', sort=False)
labels = train['Survived']

labels
train.head()
train.columns
train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'surname'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'surname'], axis=1, inplace=True)
data = [train, test]

for dataset in data:

    mean = train["Age"].mean()

    std = test["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train["Age"].astype(int)
data
embarked_mode = train['Embarked'].mode()

data = [train, test]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)
data = [train, test]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'

    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'
train_numerical_features = list(train.select_dtypes(include=['int64', 'float64', 'int32']).columns)

ss_scaler = StandardScaler()

train_df_ss = pd.DataFrame(data = train)

train_df_ss[train_numerical_features] = ss_scaler.fit_transform(train_df_ss[train_numerical_features])
test_numerical_features = list(test.select_dtypes(include=['int64', 'float64', 'int32']).columns)

test_df_ss = pd.DataFrame(data = test)

test_df_ss[test_numerical_features] = ss_scaler.fit_transform(test_df_ss[test_numerical_features])

test_df_ss.head()
encode_col_list = list(train.select_dtypes(include=['object']).columns)

for i in encode_col_list:

    train_df_ss = pd.concat([train_df_ss,pd.get_dummies(train_df_ss[i], prefix=i, drop_first=True)],axis=1)

    train_df_ss.drop(i, axis = 1, inplace=True)
train_df_ss.head()
encode_col_list = list(test.select_dtypes(include=['object']).columns)



for i in encode_col_list:

    test_df_ss = pd.concat([test_df_ss,pd.get_dummies(test_df_ss[i], prefix=i, drop_first=True)],axis=1)

    test_df_ss.drop(i, axis = 1, inplace=True)
test_df_ss.head()
train_df_ss.columns
train_df_ss.drop(['female_adult', 'male_adult', 'Sex_male','Surname_perishing_women',

       'Surname_surviving_men'], axis=1, inplace=True)

test_df_ss.drop(['female_adult', 'male_adult', 'Sex_male', 'Surname_perishing_women',

       'Surname_surviving_men'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train_df_ss,

                                                    labels,

                                                    test_size=0.25,

                                                    stratify=labels,

                                                    random_state=12)
X_train.head()
test_df_ss
# Using Logistic Regression with [child, female, male_adult]



# Instantiate our model

logreg = LogisticRegression() #===>0,78468

# Fit our model to the training data

logreg.fit(X_train, y_train)

# Predict on the test data

logreg_predictions = logreg.predict(X_test)

print(classification_report(y_test, logreg_predictions))

#logreg_data = pd.read_csv('test.csv')

#logreg_data.insert((logreg_data.shape[1]),'Survived',logreg_predictions)

#logreg_data.to_csv('LogisticRegression_SS_OH_FE2.csv')
lr_pred = logreg.predict(test_df_ss)

lr_data = pd.read_csv('test.csv')

lr_data.insert((lr_data.shape[1]),'Survived',lr_pred)
lr_data.head()
lr_data.columns
lr_data.set_index('PassengerId')

lr_submit = lr_data[lr_data.columns.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked'])]

lr_submit.set_index('PassengerId', inplace=True)
lr_submit
lr_submit.to_csv('test_lr.csv', sep=',')
X_train.head()
test_df_ss.head()
X_train['Surname_Members'] = pd.to_numeric(X_train['Surname_Members'])

X_test['Surname_Members'] = pd.to_numeric(X_test['Surname_Members'])

test_df_ss['Surname_Members'] = pd.to_numeric(test_df_ss['Surname_Members'])
X_train.dtypes
# Instantiate our model => 0,73684

xg = XGBClassifier(learning_rate=0.02, n_estimators=750,

                   max_depth= 3, min_child_weight= 1, 

                   colsample_bytree= 0.6, gamma= 0.0, 

                   reg_alpha= 0.001, subsample= 0.8

                  )

xg.fit(X_train, y_train)



xg_predictions = xg.predict(X_test)

print(classification_report(y_test, xg_predictions))
xg_pred = xg.predict(test_df_ss)

xg_data = pd.read_csv('test.csv')

xg_data.insert((xg_data.shape[1]),'Survived',xg_pred)

xg_data
xg_submit = xg_data[xg_data.columns.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked'])]

xg_submit.set_index('PassengerId',inplace=True)

xg_submit.to_csv('test_xgb.csv', sep=',')
rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745,1:0.255})

rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)



print(classification_report(y_test, rfc_predictions))
rfc_predictions = rfc.predict(test_df_ss)
rfc_data = pd.read_csv('test.csv')

rfc_data.insert((rfc_data.shape[1]),'Survived',rfc_predictions)

rfc_data.set_index('PassengerId')

rfc_submit = rfc_data[rfc_data.columns.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked'])]

rfc_submit.set_index('PassengerId')

rfc_submit.to_csv('test_rfc.csv', index=False, sep=',')
train_df_ss.head()