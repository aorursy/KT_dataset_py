import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/heart.csv")

df.head()
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr")

plt.show()
sns.countplot(x='sex', data=df, palette="bwr")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
df.groupby('target').mean()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Without Disease", "With Disease"])

plt.ylabel('Frequency')

plt.show()
corr = df.corr()

sns.heatmap(corr,  cmap="YlGnBu", square=True);
a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]

df = pd.concat(frames, axis = 1)

df.head()
y = df.target.values

x_data = df.drop(['target'], axis = 1)



# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
lr = LogisticRegression(random_state=1)

lr.fit(x_train,y_train)

acc = lr.score(x_test,y_test)*100
print(acc)
y_head_lr = lr.predict(x_test)

from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test,y_head_lr)

plt.suptitle("Confusion Matrix",fontsize=24)

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
max_iter=[100,110,120,130,140]

penalty=['l2']

class_weight=['balanced']

solver=['saga','newton-cg','sag','lbfgs','liblinear']

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]



param_grid=dict(max_iter=max_iter,penalty=penalty,class_weight=class_weight,solver=solver,C=C)
from sklearn.model_selection import GridSearchCV



grid=GridSearchCV(estimator=lr,param_grid=param_grid,cv=5,n_jobs=-1)

grid.fit(x_train,y_train)
tuned_model=grid.best_estimator_

tuned_model
acc = tuned_model.score(x_test,y_test)*100

print('The tuned model accuracy ',acc,'%')