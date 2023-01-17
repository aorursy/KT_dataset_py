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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
df_train.describe()
df_train.info()
# df_train['Ticket']
df_train[df_train['Embarked'].isna() == True]
def cal_stat(df):
    # ratio of male adn female VS survaval
    survival = df[df['Survived'] == 1]
    male_sur = survival[survival['Sex'] == 'male']
    female_sur = survival[survival['Sex'] == 'female']
    male = df[df['Sex'] == 'male']
    female = df[df['Sex'] == 'female']

    ratio_male_sur_of_male = len(male_sur) / len(male)
    ratio_female_sur_of_female = len(female_sur) / len(female)
    ratio_male_sur_of_all = len(male_sur) / len(df)
    ratio_female_sur_of_all = len(female_sur) / len(df)

    print(f'ratio_male_sur_of_male = {ratio_male_sur_of_male:0.4f} , ratio_female_sur_of_female = {ratio_female_sur_of_female:0.4f}')
    print(f'male = {len(male)/len(df)}, female = {len(female)/len(df)}')
    print(f'ratio_male_sur_of_all = {ratio_male_sur_of_all:0.4f} , ratio_female_sur_of_all = {ratio_female_sur_of_all:0.4f}')
cal_stat(df_train)
df_drop_Age_index = df_train[df_train['Age'].isna() == True].index
df_drop_Embarked_index = df_train[df_train['Embarked'].isna() == True].index

df_train_clean = df_train.drop(df_drop_Age_index)
df_train_clean = df_train_clean.drop(df_drop_Embarked_index)
df_train_clean.info()
cal_stat(df_train_clean)
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer 
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
df_train_clean.head()
# set1
x_col = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# x_col = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
y_col = ['Survived']
non_num_col = ['Pclass', 'Sex', 'Embarked']
num_col = ['Age', 'SibSp', 'Parch', 'Fare']
# set2
x_col = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
# x_col = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
y_col = ['Survived']
non_num_col = ['Pclass', 'Sex']
num_col = ['Age', 'SibSp', 'Parch', 'Fare']
def print_stat(my_results):
    for result in my_results:
        print('Classifier: ', result[0])
        for i in range(len(result[1])):
            print(f'Folder {i}: Accuracy = {result[1][i]:0.5f}, Recall = {result[2][i]:0.5f}, Precision = {result[3][i]:0.5f}, F1 = {result[4][i]:0.5f}')
        print(f'Avg F1 = {np.mean(result[4]):0.5f}')
# df_train_clean = df_train_clean.copy().reset_index()
X = df_train_clean[x_col]
y = df_train_clean[y_col]

ct = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), non_num_col)
#         ,('num', Normalizer(), num_col)
], remainder='passthrough')

cv = KFold(n_splits=5,shuffle=False, random_state=42)

models = [
    ('Decision Tree', DecisionTreeClassifier()), 
    ('Complement Naïve Bayes', ComplementNB()), 
    ('Gaussian Naive Bayes', GaussianNB()),
    ('Neural Network', MLPClassifier(alpha=0.001, max_iter=1000, verbose=False))
]

# model = MLPClassifier(alpha=0.01, max_iter=1000, verbose=False)

results = []
for cur_model in models:
    model_name = cur_model[0]
    model = cur_model[1]
    
    accuracies = []
    recalls = []
    precisions = []
    FMeasures = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(ct.fit_transform(X_train), y_train.values.ravel())
        y_predict = model.predict(ct.fit_transform(X_test))
        accuracy = metrics.accuracy_score(y_test, y_predict)
        recall = metrics.recall_score(y_test, y_predict)
        precision = metrics.precision_score(y_test, y_predict)
        FMeasure = metrics.f1_score(y_test, y_predict)
        
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        FMeasures.append(FMeasure)
        
    results.append([model_name, accuracies, recalls, precisions, FMeasures])
    
print_stat(results)



