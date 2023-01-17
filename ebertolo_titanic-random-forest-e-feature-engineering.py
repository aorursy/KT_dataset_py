import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Vamos iniciar o notebook importanto o Dataset
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# Podemos observar as primeiras linhas dele.
test_df.head()
print(test_df.shape, titanic_df.shape)
titanic_df.head()
titanic_df['Age'].isnull().any()
age_median = titanic_df['Age'].median()
print(age_median)
titanic_df['Age'] = titanic_df['Age'].fillna(age_median)
test_df['Age'] = test_df['Age'].fillna(age_median)
print('Existem nulos?', titanic_df['Age'].isnull().any())
import seaborn as sns
sns.countplot(titanic_df['Sex']);
from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()
sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))
sex_encoder.classes_
titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)
test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)
sns.countplot(titanic_df['Sex'], order=[1,0]);
titanic_df.head()
feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare']
from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(np.array(titanic_df[feature_names].values), 
                                                      np.array(titanic_df['Survived'].values),
                                                      test_size=0.2,
                                                      random_state=42)
                                                      
                                                      
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)
from sklearn.ensemble import RandomForestClassifier

#Hiperparametros
rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)

#Treino
rf_clf.fit(train_X, train_y)

print("Score Treino")
print(rf_clf.score(train_X, train_y))


print("Score Validação")
print(rf_clf.score(valid_X, valid_y))
import seaborn as sns

plt.title('Exibindo a importância de cada atributo do dataset')
sns.barplot(rf_clf.feature_importances_, feature_names);
seed = 42

feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex']

X = np.array(titanic_df[feature_names].values)
y = np.array(titanic_df['Survived'].values)

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.2,random_state=seed)
                                                                                                        
#print(train_X.shape)
#print(valid_X.shape)                                           
#print(train_y.shape)
#print(valid_y.shape)

rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=5)
rf_clf.fit(train_X, train_y)

print('Score de treino:',rf_clf.score(train_X, train_y))
print('Score de validação:',rf_clf.score(valid_X, valid_y))


plt.title('Com novas features a relação de importância ou correlação muda:')
sns.barplot(rf_clf.feature_importances_, feature_names);
titanic_df.head()['Name']
import re
def extract_title(name):
    x = re.search(', (.+?)\.', name)
    if x:
        return x.group(1)
    else:
        return ''
titanic_df['Name'].apply(extract_title).unique()
titanic_df['Title'] = titanic_df['Name'].apply(extract_title)
test_df['Title'] = test_df['Name'].apply(extract_title)
#imprimindo o novo dataset
titanic_df.head()
train_X.shape
titanic_df['Embarked']= titanic_df['Embarked'].fillna('Z')
#from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'Embarked']
dv = DictVectorizer()
dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))
dv.feature_names_
train_X, valid_X, test_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),
                                                     titanic_df['Survived'],
                                                     test_size=0.2,
                                                     random_state=42)
from sklearn.model_selection import train_test_split

rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5)
rf_clf.fit(train_X, train_y)

print('Novo Score de Treino:', rf_clf.score(train_X, train_y))
print('Novo Score de Validação:', rf_clf.score(valid_X, valid_y))

plt.title('Novo gráfico de correlação das features:')
sns.barplot(rf_clf.feature_importances_, dv.feature_names_);
test_df['Fare'] = test_df['Fare'].fillna(0)
test_df['Embarked']= test_df['Embarked'].fillna('Z')
test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))
print(test_X.shape)
y_pred = rf_clf.predict(test_X)
y_pred.shape
submission_df = pd.DataFrame()
submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = y_pred
submission_df
submission_df.to_csv('submit_final.csv', index=False)
