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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#import 'output_notebook()

from bokeh.plotting import figure, output_notebook, show

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import seaborn as sns

#from IPython.display import HTML



#https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
data=pd.read_csv(r"/kaggle/input/heart-disease-uci/heart.csv")

data.head(5)
data.describe()
#check the dimensions

data.shape



#check the total size

data.size

#check null values

data.isnull().sum()



#check null values using heatmap

sns.heatmap(data.drop(['sex','target','fbs',"restecg","exang","slope","ca"],axis=1))
##Check which age group has more heartattacks

b=data.groupby(['age','target']).size()

#bf=pd.DataFrame(b,index="target")

plt.figure(figsize=(17,6))

sns.countplot('age',data=data)

sns.despine()

##dataset contains people with age 
f,ax=plt.subplots(figsize=(5,6))

sns.barplot('target','age',data=data)

#sns.barplot('age','target',data=data)
plt.figure(figsize=(17, 6))

#sns.catplot('age','chol',hue='target',data=data)

plt.scatter('age','chol',c='cp',data=data)

plt.title("Age and cholestrol with chestpain")

plt.xlabel=("Age")

sns.set(font_scale=1.5)

pal= {0:'green',1:'yellow',2:'violet',3:'red'}

g = sns.FacetGrid(data, hue="cp",palette=pal,  height=6,aspect=2)##or palette==sns.color_palette('Set1') Set2 faltui

g.map(plt.scatter, "age", "chol", s=160, alpha=.7 ,linewidth=.5, edgecolor="black")

plt.grid(None)

plt.title("Chest pain according to the agee and cholestrol",fontsize=26,color='black')

g.add_legend(fontsize=25);

#plt.xlabel=('Age')

plt.figure(figsize=(14,15))

sns.catplot('sex','age',data=data,hue='target',palette=('red','green'))  ### we can see that 
#reg plot

f,ax=plt.subplots(figsize=(15,5))

sns.regplot('age','chol',data)

plt.title("Age and Cholestrol")

plt.xlabel=("Sex")

plt.ylabel=("Cholestrol")





##relationship betwenn cholestrol level and age
import matplotlib.pyplot

plt.figure(figsize=(45,10))



v=sns.FacetGrid(data,col='cp',row="target",height=6)

v.map(sns.barplot,"trestbps","chol",color="orange")

bins = np.arange(0, 65, 5)



g=sns.FacetGrid(data,col="sex",hue="target",height=5)

g.map(plt.scatter,"age","oldpeak")

g.add_legend()
#fig,ax=plt.subplot(figsize=(5,6))

#sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(6,6))



sns.boxplot("cp",'chol',data=data).set(xlabel=("Chest pain on a rate of 0-3"),ylabel=("Cholestrol"),title="Chestpain for different cholestrol level")
#Check the correlations between features 

fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

c=data.corr()

sns.heatmap(c,annot=True)

#We can see that there is no high correlations between features i.e no multicoliinearity.As Multicollinearitybad
## 

train=data.drop('target',axis=1)

test=data[['target']]

test.head()

train.head()
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.2,random_state=4)





##logistic

model=linear_model.LogisticRegression(solver="newton-cg") ###solvers:'liblinear',lgfbs','newton-cg','sag','saga'

model.fit(x_train,y_train) 

model.predict(x_test)

m=model.predict(x_test)





###support vector

from sklearn.svm import SVC

model1=SVC()

model1.fit(x_train,y_train)

f=model.predict(x_test)

accuracy_score(f,y_test)



##Naive Bayes

from sklearn.naive_bayes import GaussianNB,BernoulliNB

bayes=GaussianNB().fit(x_train,y_train)

nb=bayes.predict(x_test)

accuracy_score(nb,y_test)





##accuracy score and other performance metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac=accuracy_score(m,y_test)

ac

##confusion matrix

print(confusion_matrix(y_test,m,labels=None))



#classification report

print(classification_report(m,y_test))
###predicting probabilities

pb=model.predict_proba(x_test)

#pb[:,1]  ##positive outcomes

#pb[:,0]  ##negative outcomes



pb_table=pd.DataFrame(pb,index=None,columns=("Prob of 0","prob of 1"))

pb_table.head()



from sklearn.metrics import roc_auc_score,roc_curve

print(roc_curve(m,y_test))

#ns_auc = roc_auc_score(testy, ns_probs)

lr_auc = roc_auc_score(y_test,pb[:,1])

print('Logistic: ROC AUC=%.3f' % (lr_auc))



norm=[0 for i in range(len(y_test))]

norm_auc=roc_auc_score(y_test,norm)

#norm_auc

fp,tp,thr=roc_curve(y_test,norm)





sup_auc=roc_auc_score(y_test,f)

fs,ts,th=roc_curve(y_test,f)





bay=bayes.predict_proba(x_test)

nb_auc=roc_auc_score(y_test,bay[:,1])

fn,tn,thn=roc_curve(y_test,bay[:,1])



##plotting roc curves

fpr,tpr,thresholds=roc_curve(y_test,pb[:,1])   ##roc_curve

plt.plot(fp,tp,marker="_",label="Logistic: AUROC=%.3f"%lr_auc)

plt.plot(fpr,tpr,marker=".",label="AUROC =%.2f" %norm_auc)

plt.plot(fs,ts,marker="*",label="Support Vector:AUROC=%.4f" %sup_auc)

plt.plot(fn,tn,marker="*",label="Naives Bayes:AUROC=%.4f" %nb_auc)

plt.xlim([0.0,1.0])

plt.ylim([0,1.0])

plt.xlabel=('False Positives')

plt.ylabel=("True Positives")

plt.plot()

plt.legend()

plt.show()