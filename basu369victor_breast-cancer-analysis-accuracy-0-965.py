import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv("../input/data.csv")

Data.head()
Data.isnull().sum()
Data.drop(["id","Unnamed: 32"], axis=1, inplace=True)
Data.dtypes
#Data.diagnosis[Data["diagnosis"]=="B"]=1.0

#Data.diagnosis[Data["diagnosis"]=="M"]=2.0

#Data['diagnosis'] = Data['diagnosis'].astype('int')
import matplotlib.pyplot as plt

import seaborn as sns
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(Data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax,cmap="YlGnBu")

plt.show()
plt.figure(figsize=(14,14))

k=sns.scatterplot(Data["radius_mean"],Data["radius_worst"], hue=Data["area_worst"],palette="rainbow",edgecolor='yellow')

plt.xlabel("radius_mean",fontsize=14)

plt.ylabel("radius_worst",fontsize=14)

plt.title("radius_mean, radius_worst and area_worst are positively corelated to one another",fontsize=14)

plt.show()
plt.figure(figsize=(14,14))

l=sns.scatterplot(Data["radius_mean"],Data["fractal_dimension_mean"], hue=Data["area_worst"],palette="magma_r",edgecolor='red')

plt.xlabel("radius_mean",fontsize=14)

plt.ylabel("fractal_dimension_mean",fontsize=14)

plt.title("radius_mean and fractal_dimension_mean are negatively corelated to one another",fontsize=14)

plt.show()
plt.figure(figsize=(14,10))

l2=sns.lmplot("texture_mean","area_mean", hue="diagnosis", data=Data)#palette="magma_r",edgecolor='red')

plt.xlabel("texture_mean",fontsize=14)

plt.ylabel("area_mean",fontsize=14)

#plt.title("1-Benign, 2-Malignant",fontsize=14)

plt.show()
plt.figure(figsize=(20,14))

l=sns.lineplot(Data["smoothness_worst"],Data["concavity_mean"], hue=Data["diagnosis"],palette="magma",)#palette="magma_r",edgecolor='red')

plt.xlabel("smoothness_worst",fontsize=18)

plt.ylabel("concavity_mean",fontsize=18)

plt.xlim([0.08,0.20])

plt.title("Conclusion: For Smoothness_worse greater than 0.15 higher concavity means\n higher chances of Malignant Tumor but below 0.15 both the chances of Malignant and Benign are 50%",fontsize=18)

plt.show()
plt.figure(figsize=(20,14))

l=sns.scatterplot(Data["texture_mean"],Data["area_mean"], hue=Data["diagnosis"],palette="magma_r",edgecolor='black')#palette="magma_r",edgecolor='red')

plt.xlabel("texture_mean",fontsize=18)

plt.ylabel("area_mean",fontsize=18)

plt.xlim([8,30])

plt.title("Conclusion: area_mean less than or equal to 752 are mostly Benign\n but for texture mean between 15.0 - 25.5 there are cances of Malignant Tumor",fontsize=18)

plt.show()
plt.figure(figsize=(20,14))

l=sns.scatterplot(Data["concave points_mean"],Data["concavity_mean"], hue=Data["diagnosis"],palette="magma_r",edgecolor='black')#palette="magma_r",edgecolor='red')

plt.ylabel("concavity_mean",fontsize=18)

plt.xlabel("concave points_mean",fontsize=18)

#plt.xlim([8,30])

plt.title("Conclusion: Higher Concavity_mean and higher number of concave points_mean\n indicate higher chances of Malignant Tumor",fontsize=18)

plt.show()
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score,auc,roc_curve
Data.diagnosis[Data["diagnosis"]=="B"]=0.0

Data.diagnosis[Data["diagnosis"]=="M"]=1.0

Data['diagnosis'] = Data['diagnosis'].astype('int')

y = Data['diagnosis']

Data.drop("diagnosis", axis=1, inplace=True)

X = Data
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
Model = GradientBoostingClassifier(verbose=1, learning_rate=0.4,warm_start=True)

Model.fit(x_train, y_train)
y_pred = Model.predict(x_test)
#acc= accuracy_score(y_test,y_pred)

print("Accuracy\t:"+str(accuracy_score(y_test,y_pred)))

print("Precision\t:"+str(precision_score(y_test,y_pred)))

print("Recall\t:"+str(recall_score(y_test,y_pred)))
prob=Model.predict_proba(x_test)

prob = prob[:,1]
fpr,tpr,_ = roc_curve(y_test, prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(14,12))

plt.title('Receiver Operating Characteristic')

sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
