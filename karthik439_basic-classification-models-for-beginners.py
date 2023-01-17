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
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
df.info()
df.describe()
df.isna().sum()
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='Category',y='Survived',hue='Sex',data=df,palette='hls')
plt.title('Age vs Survived')
plt.show()
sns.countplot(x='Survived',hue='Sex',data=df,palette='magma')
plt.show()
from pandas_profiling import ProfileReport
ProfileReport(df)
color = plt.cm.plasma
sns.heatmap(df.corr(), annot=True, cmap=color)
plt.show()
values = list(df.Sex.value_counts().values)
gender = list(df.Sex.value_counts().index)
print(gender, '\n', values)
sns.countplot(x='Survived',hue='Sex',data=df)
plt.show()
labels = ['Total Men', 'Total Women']
values = df.Sex.value_counts().values
print(labels[0], ':', values[0])
print(labels[1], ':', values[1])
labels = ['Survived Men', 'Survived Women']
Survived_men = df['Sex'][(df['Sex']=='M') & (df['Survived']==1)].count()
Survived_women = df['Sex'][(df['Sex']=='F') & (df['Survived']==1)].count()
print(labels[0], ':', Survived_men)
print(labels[1], ':', Survived_women)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
data = [Survived_men, Survived_women]
ind = ['Male', 'Female']
plt.bar(ind,data,color='g')
plt.bar(ind,values,bottom=data,color='r')
plt.title('Dead vs Survived')
plt.show()
df = df. drop(['PassengerId','Country','Firstname', 'Lastname'],axis=1)
df.head()
df['Sex']=df['Sex'].map({'M':1,'F':0})
df['Category']=df['Category'].map({'P':1,'C':0})
df.head()
sns.heatmap(df.corr(), annot=True, cmap=color)
y=df.pop('Survived')
X=df
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50)
from sklearn.preprocessing import StandardScaler
msc = StandardScaler()
msc.fit_transform(X_train,y_train)
X_train.head()
from sklearn.metrics import confusion_matrix, classification_report
def model(name):
    name.fit(X_train, y_train)
    preds = name.predict(X_test)
    score = name.score(X_test,y_test)
    cm = confusion_matrix(preds, y_test)
    cf = classification_report(preds, y_test)
    print("Results for ",name, "\n")
    print("Score for ",name, ":", score, "\n")
    print("Confusion Matrix : \n",cm)
    print("Classification Matrix : \n",cf)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
model(LogisticRegression())
model(DecisionTreeClassifier())
model(RandomForestClassifier())
model(AdaBoostClassifier())
model(XGBClassifier())
