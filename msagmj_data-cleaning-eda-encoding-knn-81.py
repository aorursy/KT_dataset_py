# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")
df.head()
df['income']=df['income'].map({'<=50K': 0, '>50K': 1})
df.head()
df.describe(include="all")
from numpy import nan
df=df.replace("?",nan)
null_values=df.isnull().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
j=1
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot
round(null_values*100,3).sort_values('percent',ascending=False)
df["occupation"].unique()
df["workclass"].unique()
df["native-country"].unique()
df['native-country'].fillna(df['native-country'].mode()[0], inplace=True)
df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
df['workclass'].fillna(df['workclass'].mode()[0], inplace=True)
null_values=df.isnull().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
j=1
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot
round(null_values*100,3).sort_values('percent',ascending=False)
sns.pairplot(df)
df.info()
c=df.columns
for i in c:
    print(df[i].value_counts())
fig = plt.figure(figsize=(10,10)) 
sns.boxplot(x="income", y="age", data=df)
plt.show()
fig = plt.figure(figsize=(12,12)) 
ax = sns.countplot(x="workclass", hue="income", data=df).set_title("workclass vs count")
plt.figure(figsize=(10,7))
sns.boxplot(x="income", y="fnlwgt", data=df)
plt.show()
sns.catplot(y="education", hue="income", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);
g = sns.catplot(y="marital-status", hue="gender", col="income",

                data=df, kind="count",

                height=4, aspect=.7);
sns.countplot(y="occupation", hue="income",
            data=df);
plt.figure(figsize=(10,7))
sns.countplot(x="relationship", hue="income",
            data=df);
plt.figure(figsize=(20,7))
sns.catplot(y="race", hue="income", kind="count",col="gender", data=df);
df["capital-gain"].hist(figsize=(8,8))
plt.show()
df["capital-loss"].hist(figsize=(8,8))
plt.show()
df['hours-per-week'].hist(figsize=(8,8))
plt.show()
plt.figure(figsize=(10,20))
sns.countplot(y="native-country", hue="income",
            data=df);
sns.heatmap(df.corr())
df.describe()
#educational-num seems to be not so important
df=df.drop(["educational-num"],axis=1)
#also fnlwgt is not so important
df=df.drop(["fnlwgt"],axis=1)
df=df[(df["hours-per-week"] < 80)]
df=df[(df["hours-per-week"] > 20)]
print(df["hours-per-week"].min(),
df["hours-per-week"].max(),
df["hours-per-week"].mean())
df=df[(df["age"] < 60)]
df=df[(df["age"] > 20)]
print(df["age"].min(),
df["age"].max(),
df["age"].mean())
from sklearn import preprocessing
import pandas as pd
le = preprocessing.LabelEncoder()
df.columns
df[['age', 'workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
       'hours-per-week', 'native-country']]=df[['age', 'workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
       'hours-per-week', 'native-country']].apply(le.fit_transform)
X=df.drop(["income"],axis=1)
y=df["income"]
df["income"].value_counts()
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_sample(X,y)
X_res.shape,y_res.shape
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,df['income'],cv=10)
    accuracy_rate.append(score.mean())


plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('accuracy_rate vs. K Value')
plt.xlabel('K')
plt.ylabel('accuracy_rate')


# NOW WITH K=18
knn = KNeighborsClassifier(n_neighbors=18)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=18')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
