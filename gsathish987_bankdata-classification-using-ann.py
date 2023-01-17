import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
bankdata=pd.read_csv("../input/bankdata.csv")
bankdata.head()
bankdata.info()
bankdata.isnull().sum()
numericfeatures=bankdata.select_dtypes(include=[np.number])
catfeatures=bankdata.select_dtypes(include=[object])
numericfeatures.count()
catfeatures.count()
pd.value_counts(bankdata["job"])
counts=pd.value_counts(bankdata["job"])

plt.figure(figsize=(20,12))
sns.countplot(x="job",data=bankdata,palette="muted")
catfeatures.count()
sns.countplot(x="job",hue="marital",data=bankdata,palette="muted")
pd.crosstab(bankdata.job,bankdata.marital)
sns.barplot(x="job",y="balance",hue="education",data=bankdata,palette="muted")
plt.figure(figsize=(26,18))
sns.barplot(x="education",y="balance",data=bankdata,palette="muted")
sns.barplot(x="education",y="age",data=bankdata,palette="muted")
sns.barplot(x="marital",y="age",data=bankdata,palette="muted")
catfeatures.count()
sns.countplot(x="loan",data=bankdata,palette="muted")
pd.crosstab(bankdata.marital,bankdata.loan)
sns.countplot(x="marital",hue="loan",data=bankdata,palette="muted")
sns.barplot(x="loan",y="age",data=bankdata,palette="muted")
sns.heatmap(data=bankdata.corr(),annot=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
bankdata.info()
bankdata.job=le.fit_transform(bankdata.job.values)
bankdata.marital=le.fit_transform(bankdata.marital.values)
bankdata.education=le.fit_transform(bankdata.education.values)
bankdata.loan=le.fit_transform(bankdata.loan.values)
bankdata.default=le.fit_transform(bankdata.default.values)
bankdata.housing=le.fit_transform(bankdata.housing.values)
bankdata.contact=le.fit_transform(bankdata.contact.values)
bankdata.month=le.fit_transform(bankdata.month.values)
bankdata.poutcome=le.fit_transform(bankdata.poutcome.values)
bankdata.y=le.fit_transform(bankdata.y.values)
bankdata.head()
y=bankdata.y
y
x=bankdata.drop("y",axis=1)
x
import keras
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
x
bankdata.info()
classifier.add(Dense(output_dim=8,init="uniform",activation="relu",input_dim=16))
classifier.add(Dense(output_dim=8,init="uniform",activation="relu",))
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(x_train,y_train,batch_size=32,nb_epoch=100)
y_predict=classifier.predict(x_test)
y_predict=(y_predict>0.5)
from sklearn.metrics import confusion_matrix
y_predict
cm=confusion_matrix(y_test,y_predict)
cm
(7729+373)/(7729+251+690+373)
