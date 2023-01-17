import pandas as pd
df_train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

df_test = test.copy()

datasets = [df_train, df_test]

df_train.head()
df_train.describe()
for df in datasets:
    df.set_index('PassengerId',inplace=True)
df_train.Survived.value_counts(normalize=True)
df_train.Pclass.value_counts(normalize=True)
df_train.SibSp.value_counts(normalize=True)
df_train.Parch.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot(col_x, col_y,hue):
    x = df_train[col_x]
    hue = df_train[hue]
    y = df_train[col_y]
    ax = sns.boxplot(x = x, y = y, hue = hue, palette = 'husl')
    ax.set_title(f'{col_x} per {col_y}')  
    plt.show()
boxplot('Survived', 'Fare', 'Sex')
boxplot('Survived','Age','Pclass')
df_survived = df_train.query('Survived ==1')
df_survived.describe()
df_not_survived = df_train.query('Survived == 0')
df_not_survived.describe()
def survived_per_class(column):
    data = df_train.groupby(column)['Survived'].sum()/len(df_survived)
    plt.bar(data.index, data.values)
    plt.title('Survived by '+ column)
    plt.show()
    
survived_per_class('Sex')

survived_per_class('Pclass')

survived_per_class('Embarked')
def survived_per_class_relative(df, column):
    data = df.groupby(column)['Survived'].sum()*100/df.groupby(column)['Survived'].count()
    sns.barplot(x = data.index,y = data.values)
    plt.title('% survived by '+ column)
    plt.show()
    
survived_per_class_relative(df_train,'Sex')
survived_per_class_relative(df_train,'Pclass')
survived_per_class_relative(df_train,'Embarked')
survived_per_class_relative(df_train,'SibSp')
survived_per_class_relative(df_train,'Parch')
df_train.groupby('Embarked').Fare.mean()
def hist_age(df,title):
    plt.hist(df['Age'])
    plt.title(title)
    plt.show()

for i,j in zip([df_train,df_survived,df_not_survived], ['Total','Survived','Not Survived']):
    hist_age(i,j)
df_train.loc[(df_train.Sex == 'male') & (df_train.Age > 16)].Survived.value_counts(normalize=True)
df_train.loc[(df_train.Sex == 'male') & (df_train.Age < 16)].Survived.value_counts(normalize=True)
df_train.loc[(df_train.Sex == 'female') & (df_train.Age > 16)].Survived.value_counts(normalize=True)
df_train.loc[(df_train.Sex == 'female') & (df_train.Age < 16)].Survived.value_counts(normalize=True)
df_young = df_train.query('Age < 16')

sns.scatterplot(x = df_young.Age, y = df_young.Fare, hue = df_young.Survived)
#Obtain only the cabin level
import numpy as np


for df in datasets:
    df.Cabin.fillna('U', inplace=True)
    for i in range(len(df)):
        df.Cabin.iloc[i] = df.Cabin.iloc[i][0]
survived_per_class('Cabin')
survived_per_class_relative(df_train,'Cabin')
df_train.info()
df_train.info()
df_unknown = df_train.loc[np.isnan(df_train.Age) == True]
df_unknown.describe()
df_unknown.Survived.value_counts(normalize=True)
df_unknown.Pclass.value_counts(normalize=True)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

for df in datasets:
    mice_imputer = IterativeImputer()
    df['Age'] = mice_imputer.fit_transform(df[['Age']])
df_train.corr()
df_train.query("Cabin == 'T'") 
df_train.Cabin.replace('T','U',inplace=True)
df_train['Embarked'].fillna('S', inplace=True)

df_test['Fare'].fillna(df_test.Fare.median(), inplace=True)
df_train.info()
df_test.info()
#Feature Engineering

df_train
#separe features and result

y = df_train['Survived']
x = df_train.iloc[:,1:]

datasets = [x, df_test]
for df in datasets:
    df['Family'] = df['SibSp'] + df['Parch']
df_corr = pd.concat([x,y],axis=1)
df_corr.corr().Survived.sort_values(ascending=False)
survived_per_class_relative(df_corr,'Family')
for dataset in datasets:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

df_test
x
col_drop = ['Ticket', 'Name','Family','Parch','SibSp']

x.drop(columns = col_drop, inplace=True)
df_test.drop(columns = col_drop, inplace=True)
df_corr = pd.concat([x,y],axis=1)
df_corr.corr().Survived.sort_values(ascending=False)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def encode_features(cat, data):
    transformer = ColumnTransformer([('encoder', OneHotEncoder(), cat)], remainder='passthrough')
    data = transformer.fit_transform(data)
    return pd.DataFrame(data)

cat = ['Pclass','Sex','Cabin','Embarked','Title']

df_test = encode_features(cat, df_test)
x = encode_features(cat,x)
#split train and val data

from sklearn.model_selection import train_test_split

random_state = 42

x_train, x_val, y_train, y_val = train_test_split(x, y)
x_train.head()
x_val.head()
#Scale the data

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
df_test = scaler.transform(df_test)
#Choose model

from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
random_state = 42

scoring = ('f1', 'roc_auc', 'accuracy')

def best_model(estimator, x, y, scoring):
    crossval = cross_validate(estimator(),x,y, scoring=scoring, cv=10, return_train_score=True)
    print('_________________________________________')
    print(estimator)
    print('Test:')
    print(f"F1: {crossval['test_f1'].mean()}")
    print(f"Roc_auc: {crossval['test_roc_auc'].mean()}")
    print(f"Accuracy: {crossval['test_accuracy'].mean()}")
    print('Train:')
    print(f"F1: {crossval['train_f1'].mean()}")
    print(f"Roc_auc: {crossval['train_roc_auc'].mean()}")
    print(f"Accuracy: {crossval['train_accuracy'].mean()}")
    
estimators= [SVC, XGBClassifier, RandomForestClassifier, LogisticRegression]

for estimator in estimators:    
    best_model(estimator, x_train, y_train, scoring)
    
#Tuning
from sklearn.model_selection import GridSearchCV


random_state = 42

def tuning(estimator,param_grid, x, y):
    gridsearch = GridSearchCV(estimator, param_grid)
    gridsearch.fit(x,y)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)

estimator = XGBClassifier()
param_grid = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

tuning(estimator, param_grid, x_train, y_train)
#Tuning 2

estimator = XGBClassifier(max_depth = 5, min_child_weight = 3)

param_grid = {
 'gamma':[i/10.0 for i in range(0,5)]
}

tuning(estimator, param_grid, x_train, y_train)
#Tuning 3

estimator = XGBClassifier(max_depth = 5, min_child_weight = 5, gamma = 0)

param_grid = {
 'learning_rate':[0.01, 0.03, 0.1, 0.03, 1]
}

tuning(estimator, param_grid, x_train, y_train)
#train model

model_xgb = XGBClassifier(max_depth = 5, min_child_weight = 3, gamma = 0, learning_rate = 0.1)
model_xgb.fit(x_train,y_train)
y_predict_xgb = model_xgb.predict(x_val)
#Evaluate

def evaluate_model(y_val, y_predict):
    print(f"Accuracy: {accuracy_score(y_val, y_predict)}")
    print(f"F1: {f1_score(y_val, y_predict)}")
    print(f"Roc_auc: {roc_auc_score(y_val, y_predict)}")
    print(f"Confusion Matrix: {confusion_matrix(y_val, y_predict)}")
    

evaluate_model(y_val,y_predict_xgb)