import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')
# Vemos una muestra de como son los datos

train_df
test_df
# Comprobamos tanto en el training set como en el de test cuales son las columnas con mayor número de NaNs

train_df.isna().sum()
test_df.isna().sum()
cabin_nans = train_df['Cabin'].isna().sum()

age_nans = train_df['Age'].isna().sum()

total_rows = train_df['Cabin'].size

print('En la columna cabin hay {} filas nulas de un total de {}'.format(cabin_nans, total_rows))

print('En la columna age hay {} filas nulas de un total de {}'.format(age_nans, total_rows))
sns.pairplot(train_df[['Age','Survived']])
sns.distplot(train_df['Age'])
sns.barplot(x='Survived', y='Age', hue='Sex', data=train_df)
train_df
values = {'Age': train_df['Age'].mean()}

train_df = train_df.fillna(value=values)



values_test = {'Age': test_df['Age'].mean(),'Fare': test_df['Fare'].mean()}

test_df = test_df.fillna(value=values_test)
train_df = train_df.drop(columns=['Cabin','Name','Ticket'])

test_df = test_df.drop(columns=['Cabin','Name','Ticket'])
train_df = pd.get_dummies(train_df, columns=["Sex","Embarked"])

test_df = pd.get_dummies(test_df, columns=["Sex","Embarked"])
sns.pairplot(train_df[['SibSp','Parch','Survived']])
train_df['num_rel'] = train_df['SibSp']+train_df['Parch']

test_df['num_rel'] = test_df['SibSp']+test_df['Parch']
sns.pairplot(train_df[['num_rel','Survived']])
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(max_depth=2, random_state=0)

from sklearn.model_selection import train_test_split

X=train_df.drop(columns=['Survived'])

y=train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
rf_model.fit(X_train, y_train)
#Creamos una función para mostrar mediante una gráfica el resultado de las variables más significativas de un modelo

def plot_feature_importance(importance,names,model_type):



    feature_importance = np.array(importance)

    feature_names = np.array(names)



    data={'feature_names':feature_names,'feature_importance':feature_importance}

    fi_df = pd.DataFrame(data)



    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)



    plt.figure(figsize=(10,8))

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    plt.title(model_type + 'FEATURE IMPORTANCE')

    plt.xlabel('FEATURE IMPORTANCE')

    plt.ylabel('FEATURE NAMES')
plot_feature_importance(rf_model.feature_importances_,X.columns,'RANDOM FOREST')
rf_model.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=0,max_iter=1000).fit(X_train, y_train)
lr_model.score(X_test,y_test)
from sklearn.ensemble import GradientBoostingClassifier

xgb_model = GradientBoostingClassifier(random_state=123)

xgb_model.fit(X_train, y_train)
xgb_model.score(X_test, y_test)
xgb_model.fit(X, y)
predictions = xgb_model.predict(test_df)
test_df.insert(0, 'Survived', predictions, allow_duplicates=True)

result = test_df[['PassengerId','Survived']]
result.to_csv('result.csv',index=False)