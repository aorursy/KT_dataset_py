import pandas as pd

import numpy as np

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")
df = pd.read_csv("../input/student-grade-prediction/student-mat.csv")

df.head()
d = sns.countplot(df['school'])

d.axes.set_title("Count of School (GP/MS)")

d.axes.set_xlabel("Schools")

d.axes.set_ylabel("Number of Schools")
d = sns.countplot(df['sex'])

d.axes.set_title("Distribution of Gender")

d.axes.set_xlabel("Gender")

d.axes.set_ylabel("Total Student")
d = sns.countplot(df['address'],hue=df['health'])

d.axes.set_title("Distrution of Heath in Urban/Rural Students",fontsize=20)

d.axes.set_xlabel("Urban/Rural Students",fontsize=20)

d.axes.set_ylabel("Total Students",fontsize=20)
sns.swarmplot(df['age'],df['G3'])
plt.figure(figsize=(15,7))

d = sns.countplot(df['Medu'],hue=df['G3'])

d.set_title("Distribution Student Grade with Mother Education")

d.set_xlabel("Mother Education Level")

d.set_ylabel("Number of Grades")
plt.figure(figsize=(15,7))

d = sns.countplot(df['Mjob'],hue=df['G3'])

d.set_title("Distribution Student Grade with Mother Job")

d.set_xlabel("Mother Job")

d.set_ylabel("Number of Grades")
plt.figure(figsize=(15,7))

d = sns.countplot(df['Fedu'],hue=df['G3'])

d.set_title("Distribution Student Grade with Father Education")

d.set_xlabel("Father Education Level")

d.set_ylabel("Number of Grades")
plt.figure(figsize=(15,7))

d = sns.countplot(df['Fjob'],hue=df['G3'])

d.set_title("Distribution Student Grade with Father Job")

d.set_xlabel("Father Job Level")

d.set_ylabel("Number of Grades")
plt.figure(figsize=(17,5))

b = sns.boxplot(df['absences'],df['G3'])

b.axes.set_title("Distrution of Grade and Absenes")

b.axes.set_xlabel("Number of Absenes")

b.axes.set_ylabel("Grade")
d = sns.swarmplot(df['freetime'],y=df['G3'])

d.axes.set_title("Distribution of Freetime and Final Grade")

d.axes.set_xlabel("Free Time")

d.axes.set_ylabel("Grades")
fig, axs = plt.subplots(1,2)



plt.figure(figsize=(15,7))

d = sns.boxplot(df['Pstatus'],df['G3'],ax=axs[0])

d.set_title("Distribution of Parent Living Status with Final Grade")

d.set_xlabel("Parent Living Status")

d.set_ylabel("Final Score")





d = sns.swarmplot(df['Pstatus'],df['G3'],ax=axs[1])

d.set_title("Distribution of Parent Living Status with Final Grade")

d.set_xlabel("Parent Living Status")

d.set_ylabel("Final Score")
family_ed = df['Fedu'] + df['Medu'] 

b = sns.boxplot(x=family_ed,y=df['G3'])

plt.show()
# sns.countplot(df['school'],hue=df['G3'])
gpSchool = df[df['school']=='GP'][:40]

msSchool = df[df['school']=='MS'][:40]

twoSchool = pd.concat([gpSchool,msSchool])
d = sns.countplot(twoSchool['school'],hue=twoSchool['G3'])

d.axes.set_title("Distribution Grade between Two School")

d.axes.set_xlabel("Schools")

d.axes.set_ylabel("Grade")
b = sns.swarmplot(x=df['failures'],y=df['G3'])

b.axes.set_title('Students with less failures score higher', fontsize = 30)

b.set_xlabel('Number of failures', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.boxplot(x = df['higher'], y=df['G3'])

b.axes.set_title('Students who wish to go for higher studies score more', fontsize = 30)

b.set_xlabel('Higher education (1 = Yes)', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.boxplot(x = df['activities'], y=df['G3'])

b.axes.set_title('Students who do Extra Activites')

b.set_xlabel('Extra Activites')

b.set_ylabel('Final Grade')

plt.show()
plt.figure(figsize=(17,7))

b = sns.countplot(df['goout'],hue=df['G3'])

b.axes.set_title('How often do students go out with friends')

b.set_xlabel('Go out')

b.set_ylabel('Count')

plt.show()
df.corr()['G3'].sort_values()
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.metrics import accuracy_score

from random import randint
enc = LabelEncoder()

category_colums = df.select_dtypes('object').columns

for i in category_colums:

    df[i] = enc.fit_transform(df[i])
df.head()
X = df.drop(['school', 'G1', 'G2'], axis=1)

y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
cross_val_score(DecisionTreeClassifier(),X,y)[0]
log = DecisionTreeClassifier()

log.fit(X_train,y_train)

y_predict = log.predict(X_test)

accuracy_score(y_predict,y_test)
param_dist = {"max_depth": [3, None],

              "max_features": range(1,7),

              "min_samples_leaf": range(1,7),

              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(DecisionTreeClassifier(),param_grid=param_dist,scoring="accuracy",n_jobs=-1)

grid_search.fit(X_train,y_train)