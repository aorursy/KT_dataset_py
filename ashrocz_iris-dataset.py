#Objective: Classification using SVM on the Iris Dataset
import pandas as pd

import numpy as np



df=pd.read_csv("../input/Iris.csv")
print(df)
import seaborn as sns

import matplotlib.pyplot as plt
df.head()
df.describe()
df.info()
df['Species'].value_counts()
#Describing the data

df.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")

plt.xlabel('Statistics',)

plt.ylabel('Value')

plt.title("General Statistics of Iris Dataset")
#Frequency of the three species in the dataset

ax = sns.countplot(x="Species", data=df)

f,ax=plt.subplots(1,2,figsize=(18,8))

df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Iris Species Count')

ax[0].set_ylabel('Count')

sns.countplot('Species',data=df,ax=ax[1])

ax[1].set_title('Iris Species Count')

plt.show()
#Train Test Split

from sklearn.model_selection import train_test_split

X=df.drop(['Species', 'Id'], axis=1)

y=df['Species']

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.5, shuffle=True,random_state=100)
from sklearn.svm import SVC

model=SVC(C=1, kernel='rbf', tol=0.001)

model.fit(X_train, y_train)
#Model Evaluation

pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, pred))

print('\n')

print(classification_report(y_test, pred))

print('\n')

print('Accuracy score is: ', accuracy_score(y_test, pred))