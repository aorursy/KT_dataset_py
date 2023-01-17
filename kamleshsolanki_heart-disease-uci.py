
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score,accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
heart_df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
heart_df.info()
plt.figure(figsize=(15, 7))
sns.heatmap(heart_df.isnull())
heart_df.head()
heart_df['target'].value_counts().plot(kind = 'bar')
plt.legend()
def drawKdePlot(df, col):
    plt.figure(figsize=(15, 7))
    df = pd.DataFrame({'positive':heart_df[heart_df['target'] == 1][col],'negative':heart_df[heart_df['target']==0][col]})
    sns.kdeplot(df['positive'], shade = True)
    sns.kdeplot(df['negative'], shade = True)    
def drawbarChart(df, col):
    df = df.groupby(col).agg({'target':['sum']})
    df.columns = ['target']
    df.plot(kind = 'bar', figsize = (15, 7))    
    plt.legend()
drawKdePlot(heart_df, 'age')
drawbarChart(heart_df, 'sex')
drawbarChart(heart_df, 'cp')
drawKdePlot(heart_df, 'trestbps')
drawKdePlot(heart_df, 'chol')
drawbarChart(heart_df, 'fbs')
drawbarChart(heart_df, 'restecg')
drawKdePlot(heart_df, 'thalach')
drawbarChart(heart_df, 'exang')
drawKdePlot(heart_df, 'oldpeak')
drawbarChart(heart_df, 'slope')
drawbarChart(heart_df, 'ca')
drawbarChart(heart_df, 'thal')
heart_df.columns
for col in heart_df:
    print('{} unique value: {}'.format(col, len(heart_df[col].unique())))
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
continous_columns   = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
def meanEncoding(df, col):
    df_1 = df.groupby(col).agg({'target' : ['mean']}).reset_index()
    mean_col = col + '_mean_encode'
    df_1.columns = [col, mean_col]
    df = df.merge(df_1, on = col, how = 'left')
    df.drop([col], inplace = True, axis = 1)
    return df
heart_df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
for col in categorical_columns:
       heart_df = meanEncoding(heart_df, col)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
train  = heart_df.drop(['target'], axis = 1)
target = heart_df['target']

X_train, X_test, Y_train, Y_test = train_test_split(train, target, test_size = 0.3, shuffle = True, random_state = 7)
svm = SVC()
cross_val_score(svm , train, target, cv = 3)
xgb = XGBClassifier(n_estimator = 1000)
cross_val_score(xgb , train, target, cv = 3)
catboost = CatBoostClassifier()
cross_val_score(catboost , train, target, cv = 3)
xgb.fit(X_train,Y_train, eval_set=((X_train,Y_train), (X_test, Y_test)))
catboost.fit(X_train,Y_train)
predict = catboost.predict(X_test)
actual = Y_test
predict_df = pd.DataFrame({'predict' : predict, 'actual' : actual})
predict_df['TP'] = (predict_df['predict'] == 1) * (predict_df['actual'] == 1 )
predict_df['TN'] = (predict_df['predict'] == 0) * (predict_df['actual'] == 0 )

predict_df['FP'] = (predict_df['predict'] == 1) * (predict_df['actual'] == 0 )
predict_df['FN'] = (predict_df['predict'] == 0) * (predict_df['actual'] == 1 )
predict_df.head()
accuracy = accuracy_score(actual, predict)
print('accuracy: {}'.format(accuracy))
accuracy = accuracy_score(actual, predict)
print('accuracy: {}'.format(accuracy))
