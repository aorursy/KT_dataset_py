# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/loan-eligible-dataset/loan-train.csv')
df_test = pd.read_csv('../input/loan-eligible-dataset/loan-test.csv')
df_train.head()
df_train.drop(columns='Loan_ID', inplace = True)
df_test.drop(columns='Loan_ID', inplace = True)
df_train.info()
df_train.describe().transpose()
sns.set(style = 'whitegrid')
plt.figure(figsize = (16,8))
plt.subplot(2,2,1)
sns.countplot(x = 'Gender' , data = df_train)
plt.subplot(2,2,2)
sns.countplot(x = 'Married', data= df_train)
plt.subplot(2,2,3)
sns.countplot(x = 'Gender', hue= 'Loan_Status', data = df_train)
plt.subplot(2,2,4)
sns.countplot(x = 'Married',hue = 'Loan_Status' , data= df_train)
plt.show()
sns.pairplot(df_train)
plt.show()
plt.figure(figsize = (16,8))
sns.scatterplot(x = 'LoanAmount' , y = 'ApplicantIncome' , hue = 'Loan_Status', data = df_train)
plt.show()
sns.set(style = 'whitegrid')
plt.figure(figsize = (16,8))
plt.subplot(2,2,1)
sns.countplot(x = 'Education' , data = df_train)
plt.subplot(2,2,2)
sns.countplot(x = 'Property_Area', data= df_train)
plt.subplot(2,2,3)
sns.countplot(x = 'Education', hue= 'Loan_Status', data = df_train)
plt.subplot(2,2,4)
sns.countplot(x = 'Property_Area',hue = 'Loan_Status' , data= df_train)
plt.show()
graduate = df_train['Education'].value_counts()['Graduate']
not_graduate = df_train['Education'].value_counts()['Not Graduate']
graduate_yes = df_train[df_train['Loan_Status'] == 'Y']['Education'].value_counts()['Graduate']
not_graduate_yes = df_train[df_train['Loan_Status'] == 'Y']['Education'].value_counts()['Not Graduate']
print(graduate_yes / graduate)
print(not_graduate_yes / not_graduate)
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.pie(df_train['Self_Employed'].value_counts(),explode = [0,0.2] , autopct='%.1f%%' , shadow = True , labels = ['No', 'Yes'])
plt.title('percent of values in Self employed column')
plt.subplot(1,2,2)
plt.pie(df_train['Loan_Status'].value_counts(), explode = [0,0.2], autopct = '%.1f%%' , labels = ['N', 'Y'])
plt.title('percent of acceptance')
plt.show()
plt.figure(figsize = (10,5))
sns.countplot('Self_Employed' , hue = 'Loan_Status' , data = df_train)
plt.show()
sns.boxplot(x = 'Gender', y='ApplicantIncome' , data = df_train)
plt.show()
plt.bar('mean of Applicant Income',df_train['ApplicantIncome'].mean())
plt.bar('mean of Loan Amount', df_train['LoanAmount'].mean())
plt.show()
print('Unique values in Gender :',df_train['Gender'].unique())
print('Unique values in Married :',df_train['Married'].unique())
print('Unique values in Dependents :',df_train['Dependents'].unique())
print('Unique values in Education :',df_train['Education'].unique())
print('Unique values in Self Employed :',df_train['Self_Employed'].unique())
print('Unique values in Property Area :',df_train['Property_Area'].unique())
print('Unique values in Loan Status :',df_train['Loan_Status'].unique())
df_train['Gender'] = df_train['Gender'].replace(['Male','Female'],[1,0])
df_train['Married'] = df_train['Married'].replace(['Yes','No'],[1,0])
df_train['Dependents'] = df_train['Dependents'].replace(['0','1','2'],[0,1,2])
df_train['Dependents'] = df_train['Dependents'].replace('3+' , 3)
df_train['Education'] = df_train['Education'].replace(['Graduate' , 'Not Graduate'],[1,0])
df_train['Self_Employed'] = df_train['Self_Employed'].replace(['Yes','No'],[1,0])
df_train['Property_Area'] = df_train['Property_Area'].replace(['Urban' ,'Rural' ,'Semiurban'],[0,1,2])
df_train['Loan_Status'] = df_train['Loan_Status'].replace(['Y','N'],[1,0])
df_train.describe()
df_train.isnull().sum()
df_test.isnull().sum()
df_temp = df_train.dropna()
df_temp.describe()
df_temp = pd.get_dummies(df_temp , columns = ['Property_Area'])
df_temp
corr = df_temp.corr()
plt.figure(figsize = (16,8))
sns.heatmap(corr , annot = True )
plt.show()
df_test.dropna(inplace = True)
df_test['Property_Area'] = df_test['Property_Area'].replace(['Urban' ,'Rural' ,'Semiurban'],[0,1,2])
df_test = pd.get_dummies(df_test , columns = ['Property_Area'])
df_test['Married'] = df_test['Married'].replace(['Yes', 'No'], [1,0])
df_tr_x = df_temp[['Married' , 'Credit_History' , 'Property_Area_1', 'Property_Area_2']].values
df_tr_y = df_temp['Loan_Status'].values
df_ts = df_test[['Married', 'Credit_History', 'Property_Area_1', 'Property_Area_2']].values
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
x_train, x_test, y_train, y_test = train_test_split(df_tr_x , df_tr_y , test_size = 0.25)
cls_lr = LogisticRegression()
cls_lr.fit(x_train, y_train)
y_pred = cls_lr.predict(x_test)
print(classification_report(y_test , y_pred))
plot_confusion_matrix(cls_lr , x_test, y_test)
plt.show()
cross_val = cross_val_score(cls_lr , x_train, y_train , cv=10)
print(cross_val.mean())
cls_rf = RandomForestClassifier()
cls_rf.fit(x_train, y_train)
y_pred = cls_rf.predict(x_test)
print(classification_report(y_test , y_pred))
plot_confusion_matrix(cls_rf, x_test, y_test)
plt.show()
cross_val = cross_val_score(cls_rf , x_train, y_train , cv=10)
print(cross_val.mean())
results = []
for i in range(1,25): 
    cls_knn_2 = KNeighborsClassifier(n_neighbors=i)
    cls_knn_2.fit(x_train, y_train)
    score = cls_knn_2.score(x_test ,y_test)
    results.append([score])
plt.plot(list(range(1,25)) , results)
plt.show()
cls_knn = KNeighborsClassifier(n_neighbors=11)
cls_knn.fit(x_train, y_train)
predicted = cls_knn.predict(df_ts)
predicted
