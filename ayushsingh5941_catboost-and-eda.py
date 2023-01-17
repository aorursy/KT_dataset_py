import pandas as pd

import category_encoders as ce

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict

import numpy as np

from sklearn.preprocessing import RobustScaler, label_binarize

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve

import catboost as cb



%matplotlib inline
sns.set_style('darkgrid')
df = pd.read_csv('../input/terrabluext-intern-test-data/Test_Data.csv')
df.shape
df.head(20)
df.info()
df.isna().sum()
pd.set_option('display.max_columns', 90)
df.describe()
def describe_plot(label):

    print(df[label].describe())

    print(df.shape)

    sns.boxplot(df[label])
def remover_plot(label, outlier):

    df.drop(df[df[label] > outlier].index, inplace=True)

    print(df.shape)

    sns.boxplot(df[label])
# Class A

describe_plot('A')
# Class E

describe_plot("E")
remover_plot('E', 280)
# Class F

describe_plot('F')
remover_plot('F', 138)
# Class G

describe_plot('G')
remover_plot('G', 105)
# Class j

describe_plot('J')
remover_plot('J', 32)
# Class L

describe_plot('L')
remover_plot('L', 325)
# Class O

describe_plot('O')
remover_plot('O', 2.2)
# Class P

describe_plot('P')
remover_plot('P', 0.78)
#  Class

describe_plot('R')
remover_plot('R', 35000)
# Class X1

describe_plot('X1')
remover_plot('X1', 18000)
# Class

describe_plot('X2')
remover_plot('X2', 10.0)
# Class X

describe_plot('X9')
remover_plot('X9', 0.6)
# Class Y1

describe_plot('Y1')
remover_plot('Y1', 4.7)
# Class Y2

describe_plot('Y2')
df = df.drop(df[df['Y2'] < -3.480000e-05].index)

sns.boxplot(df['Y2'])
# Class Y4

describe_plot('Y4')
df = df.drop(df[df['Y4'] < 0.012].index)

sns.boxplot(df['Y4'])
# Class y5

describe_plot('Y5')
remover_plot('Y5', 0.021)
df = df.drop(df[df['Y5'] < -0.02].index)

sns.boxplot(df['Y5'])
# Class Y6

describe_plot('Y6')
remover_plot('Y6', 0.6)
# Class Y7

describe_plot('Y7')
df = df.drop(df[df['Y7'] < -0.35].index)

sns.boxplot(df['Y7'])
# Class Y8

describe_plot('Y8')
remover_plot('Y8', 0.020)
# Class Y9

describe_plot('Y9')
remover_plot('Y9', 0.25)
# Class Z2

describe_plot('Z2')
remover_plot('Z2', 0.035)
# Class Z3

describe_plot('Z4')
remover_plot('Z4', 0.08)
# Class Z5

describe_plot('Z6')
remover_plot('Z6', 0.0003)
corr_mat = df.corr(method='pearson')
fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corr_mat, vmax=0.99, vmin=0.5, ax=ax, square=True, cmap='Blues')
sns.catplot(x='Class', kind='count', data=df)
df.shape
Y = df['Class']
X = df.drop('Class', axis=1)
print('Size of label and target',X.shape, Y.shape)
X = np.log1p(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, stratify=Y)
sns.countplot(y_train.sort_values(ascending=True))
sns.countplot(y_test.sort_values(ascending=True))
Rs = RobustScaler()
x_train = Rs.fit_transform(x_train)
x_test = Rs.transform(x_test)
def grid_search(estimator, param):



    grid = GridSearchCV(estimator, param, n_jobs=-1, cv=5)

    grid.fit(x_train, y_train)

    print('Best parameter', grid.best_params_)

    model = grid.best_estimator_

    pred = model.predict(x_test)

    return model, pred
def metrics_evaluate(y_pred):

    con_mat = confusion_matrix(y_test, y_pred)

    print(con_mat)

    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(con_mat, ax=ax, square=True, vmax=500, vmin=60)

    

    # Classification report

    class_report = pd.DataFrame(data=classification_report(y_test, y_pred, output_dict=True))

    print(class_report.head())

    

    # Accuracy

    accuracy = accuracy_score(y_test, y_pred)    

    print('Accuracy Test', accuracy*100)
cat_boost = cb.CatBoostClassifier()
cat_param = {}
model_cat, pred_cat = grid_search(cat_boost, cat_param)
metrics_evaluate(pred_cat)