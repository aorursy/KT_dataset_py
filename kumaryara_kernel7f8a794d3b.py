# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv("../input/diabetes.csv")
df.info()
import numpy as np
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns:
    df[col].replace(0, np.NaN, inplace=True)
df.dropna(inplace=True)
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
cols = df.columns[:8]
for item in cols:
    plt.figure(figsize=(4, 2))
    plt.title(str(item) + ' With' + ' Outcome')
    sns.violinplot(x=df.Outcome, y=df[item], data=df)
    plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
x,y = df.loc[:,df.columns != 'Outcome'], df.loc[:,'Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
predictors=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': rf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 
import xgboost as xgb 
y = df['Outcome']
X = df.drop(['Outcome'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
print(f'X_train : {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_test : {y_test.shape}')
xgbModel = xgb.XGBClassifier()
mod = xgbModel.fit(X_train,y_train)
print(mod)
test_pred = xgbModel.predict(X_test)
print(test_pred)
cm = confusion_matrix(y_test,test_pred)
print('Confusion matrix: \n',cm)
Test_Actual_Pred = pd.DataFrame({ 'Actual' : y_test, 'Prediction': test_pred})
Test_Actual_Pred.head()
print (f'Train Accuracy - : {xgbModel.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {xgbModel.score(X_test,y_test):.3f}')