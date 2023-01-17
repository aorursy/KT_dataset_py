# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotnine import *

import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
training_data = '../input/adult-training.csv'

test_data = '../input/adult-test.csv'

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',\

           'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']



df_train_set = pd.read_csv(training_data, names = columns)

df_test_set = pd.read_csv(test_data, names = columns)



#fnlwgt列无实际意义，不重要可去除

df_train_set.drop('fnlwgt', axis = 1, inplace=True)

print('Training data shape: ', df_train_set.shape)

print('Tesing data shape: ', df_test_set.shape)

df_train_set.head()

all_data = [df_train_set, df_test_set]

#查看缺失值

def missing_values_table(df):

    # Total missing values

        mis_val = df.isnull().sum()

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns



print("df_train_set")

missing_values_table(df_train_set)

print("df_test_set")

missing_values_table(df_test_set)

df_test_set
#通过查看数据，发现存在异常数据如？,将?替换为Nan或Unknown

df_test_set.drop(df_test_set.index[0])

df_train_set.replace(" ?", np.nan, inplace=True)

df_test_set.replace(" ?", np.nan, inplace=True)

df_train_set.dropna(inplace=True)

df_test_set.dropna(inplace=True)

df_train_set.income_bracket.value_counts()
df_test_set.income_bracket.value_counts()
all_data = [df_train_set, df_test_set]

for data in all_data:

    data['target']=data['income_bracket'].apply(lambda x: x.replace('.', ''))

    data['target']=data['target'].apply(lambda x: x.strip())

    data['target'] = data['target'].apply(lambda x: 1 if x=='>50K' else 0)

    data.drop(['income_bracket'], axis=1, inplace=True)

df_train_set.target.value_counts()
df_train_set.dtypes.value_counts()
df_train_set.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
df_train_set.drop('native_country', axis=1, inplace=True)

df_test_set.drop('native_country', axis=1, inplace=True)

df_train_set.drop('education', axis=1, inplace=True)

df_test_set.drop('education', axis=1, inplace=True)
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in df_train_set:

    if df_train_set[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(df_train_set[col].unique())) <= 2:

            print(col + " were label encoded")

            # Train on the training data

            le.fit(df_train_set[col])

            # Transform both training and testing data

            df_train_set[col] = le.transform(df_train_set[col])

            df_test_set[col] = le.transform(df_test_set[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

df_train_set = pd.get_dummies(df_train_set)

df_test_set = pd.get_dummies(df_test_set)

print('Training Features shape: ', df_train_set.shape)

print('Testing Features shape: ', df_test_set.shape)
df_train_set, df_test_set = df_train_set.align(df_test_set, join = 'inner', axis = 1)

print('Training Features shape: ', df_train_set.shape)

print('Testing Features shape: ', df_test_set.shape)
df_train_set.columns
cols = list(df_train_set.columns)

cols.remove("target")



x_train, y_train = df_train_set[cols].values, df_train_set["target"].values

x_test, y_test = df_test_set[cols].values, df_test_set["target"].values

# 采用决策树算法

treeClassifier = DecisionTreeClassifier()

treeClassifier.fit(x_train, y_train)

treeClassifier.score(x_test, y_test)
import itertools

from sklearn.metrics import confusion_matrix

# 混淆矩阵

def plot_confusion_matrix(cm, classes, normalize=False):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    cmap = plt.cm.Blues

    title = "Confusion Matrix" 

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = np.around(cm, decimals=3)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# 决策树算法评估

y_pred = treeClassifier.predict(x_test)

cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(10,6))

plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)
# 采用随机森林算法

rclf = RandomForestClassifier(n_estimators=500)

rclf.fit(x_train, y_train)

rclf.score(x_test, y_test)
# 随机森林算法评估

y_pred = rclf.predict(x_test)

cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(10,6))

plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)
# 特征重要性

importances = rclf.feature_importances_

indices = np.argsort(importances)

cols = [cols[x] for x in indices]

plt.figure(figsize=(10,20))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), cols)

plt.xlabel('Relative Importance')


parameters = {

    'n_estimators':(100, 500, 1000),

    'max_depth':(None, 24, 16),

    'min_samples_split': (2, 4, 8),

    'min_samples_leaf': (16, 4, 12)

}



clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=8)

clf.fit(x_train, y_train)

clf.best_score_, clf.best_params_
rclf2 = RandomForestClassifier(n_estimators=1000,max_depth=24,min_samples_leaf=4,min_samples_split=8)

rclf2.fit(x_train, y_train)



y_pred = rclf2.predict(x_test)

cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(10,6))

plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)