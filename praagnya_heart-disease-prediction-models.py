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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().value_counts()
df.drop_duplicates()
y = df["target"]



sns.countplot(y)





target_temp = df.target.value_counts()



print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))

print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))
df['sex'].value_counts()
sns.barplot(df["sex"],y)
ax = sns.countplot(x = "cp",hue = "sex", data = df)

plt.title('Heart Disease count according To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.ylabel('Count')

plt.show()
df['fbs'].describe()
df['fbs'].unique()
sns.barplot(df["fbs"],y)
df['restecg'].unique()
sns.countplot(df['restecg'])
sns.barplot(df['restecg'],y)
df['slope'].unique()
sns.countplot(df['slope'])
sns.barplot(df['slope'],y)
df['ca'].unique()
sns.countplot(df["ca"])
sns.barplot(df['ca'],y)
df['thal'].unique()
sns.countplot(df['thal'])
sns.barplot(df['thal'],y)
df['exang'].unique()
sns.countplot(df['exang'])
sns.barplot(df['exang'],y)
import warnings 

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (7,5)

sns.distplot(df['age'])

plt.title('Distribution of Age')

plt.show()
df['trestbps'].describe()
sns.boxplot(df['target'],df['trestbps'])

plt.title('Relation btw restbp and target')

plt.show()
df['chol'].describe()
plt.rcParams['figure.figsize'] = (10, 9)

sns.violinplot(df['target'], df['chol'])

plt.title('Relation of Cholestrol with Target')

plt.show()
df['thalach'].describe()
sns.boxplot(df['target'],df['thalach'])

plt.title('Relation btw max heart rate and target')

plt.show()
sns.violinplot(df['target'], df['thalach'])

plt.title('Relation btw max heart rate and target')

plt.show()
corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

g = sns.heatmap(df[top_corr_features].corr(),annot = True, cmap = "RdYlGn")
df = df.drop(columns=(['exang']))
df = pd.get_dummies(df, columns = ['sex','cp','restecg','slope','thal','ca','fbs'])
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = ['age','trestbps','chol','oldpeak']

df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
df.head(20)
#train-test split

y = df['target']

X = df.drop(['target'], axis = 1)

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 10, shuffle=True)
from sklearn import neighbors 

clf = neighbors.KNeighborsClassifier(n_neighbors=2, p=1)

clf.fit(X_train,y_train)
k_range = range(1,26)

scores={}



for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    predict_knn = knn.predict(X_test)

    scores[k]=accuracy_score(y_test,predict_knn)

scores
accuracy = clf.score(X_test,y_test)

accuracy
from xgboost import XGBClassifier

model_xgb = XGBClassifier()

model_xgb.fit(X_train,y_train)
# Predicting the model

y_predict_xgb = model_xgb.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_xgb))

print(classification_report(y_test,y_predict_xgb))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=3)

classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred_rf = classifier.predict(X_test)



print(accuracy_score(y_test,y_pred_rf))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3,min_samples_split=3,max_features=13)

dt.fit(X_train, y_train)
y_pred_df = dt.predict(X_test)



print(accuracy_score(y_test, y_pred_df))
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)
y_pred_svm = svm.predict(X_test)



print(accuracy_score(y_test, y_pred_svm))