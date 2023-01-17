# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath='../input/health-insurance-cross-sell-prediction/train.csv'
data=pd.read_csv(filepath)
data.head()
Y_Train=data['Response']
X_Train=data.drop(['Response'],axis=1)
X_Test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
data.info()
data.describe()
sns.countplot(data.Response)

plt.subplot(121)
circle = plt.Circle((0, 0), 0.5, color = 'white')
data[data['Response']==0]['Gender'].value_counts().plot(kind='pie',figsize=(12, 12), rot=1, colors=['#1849CA', 'crimson','green'],autopct = '%.2f%%')
plots = plt.gcf()
plots.gca().add_artist(circle)
plt.title('Response=0')

plt.subplot(122)
circle = plt.Circle((0, 0), 0.5, color = 'white')
data[data['Response']==1]['Gender'].value_counts().plot(kind='pie',figsize=(12, 12), rot=1, colors=['#1849CA', 'crimson','green'],autopct = '%.2f%%')
plots = plt.gcf()
plots.gca().add_artist(circle)
plt.title('Response=1')
plt.subplot(121)
plt.title('Response=0')
circle = plt.Circle((0, 0), 0.5, color = 'white')
data[data['Response']==0]['Vehicle_Age'].value_counts().plot(kind='pie',figsize=(12, 12), rot=1, colors=['#1849CA', 'crimson','green'], autopct = '%.2f%%')
plots = plt.gcf()
plots.gca().add_artist(circle)
plt.legend()


plt.subplot(122)
plt.title('Response=1')
circle = plt.Circle((0, 0), 0.5, color = 'white')
data[data['Response']==1]['Vehicle_Age'].value_counts().plot(kind='pie',figsize=(12, 12), rot=1, colors=['#1849CA', 'crimson','green'], autopct = '%.2f%%')
plots = plt.gcf()
plots.gca().add_artist(circle)
df=data.groupby(['Vehicle_Damage','Gender'])['id'].count().to_frame()
df=df.rename(columns={'id':'count'}).reset_index()
df
g = sns.catplot(x="Vehicle_Damage", y="count",col="Gender",
                data=df, kind="bar",height=5,aspect=.7);
plt.subplots(figsize=(15,6))
plt.subplot(121)
sns.distplot(data['Age'],bins=100,color='r')
plt.subplot(122)
sns.distplot(data['Annual_Premium'],bins=50,color='b')
sns.countplot(data['Driving_License'])
plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr().abs(),annot=True)
X_Train.drop(['id'],axis=1,inplace=True)
X_Train['Gender'].replace(['Male','Female'],[1,0],inplace=True)
X_Train['Vehicle_Damage'].replace(['Yes','No'],[1,0],inplace=True)
X_Train['Vehicle_Age'].replace(['> 2 Years','1-2 Year','< 1 Year'],[2,1,0],inplace=True)
X_Train.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train= sc.fit_transform(X_Train)

print('Positive cases % in validation set: ', round(100 * len(y_test[y_test == 1]) / len(y_test), 3), '%')
print('Positive cases % in train set: ', round(100 * len(y_train[y_train == 1]) / len(y_train), 3), '%')
my_list=[]
c=[0.0001,.01,1,10]
for i in c:
    lr=LogisticRegression(penalty='l2',C=i,solver='lbfgs',random_state=0)
    lr.fit(X_train,y_train)
    pred=lr.predict(X_test)
    print(accuracy_score(pred,y_test))
    print(confusion_matrix(y_test,pred))
    print(f1_score(y_test,pred))
    
    