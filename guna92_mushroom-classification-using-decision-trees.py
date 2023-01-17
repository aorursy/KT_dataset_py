# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

raw_data=pd.read_csv("../input/mushrooms.csv")





# Data analysis and cleaning:

raw_data.head(10)

df=pd.DataFrame(raw_data)

# To check if there are any NUll's in the dataset

df[df.isnull().any(axis=1)]

df.isnull().sum()
# Plots and EDA

sns.countplot(x=df['class'],hue=df['population'],data=df);

sns.countplot(y=df['class'],hue=df['cap-shape'],data=df);
#Convert categories to numbers using one hot encoder

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

for col in df.columns:

    df[col]=lb.fit_transform(df[col])

df.head(5)
df['class'].unique()
df.groupby('class').size()
# Create a figure instance

fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))



# Create an axes instance and the boxplot

bp1 = axes[0,0].boxplot(df['stalk-color-above-ring'],patch_artist=True)



bp2 = axes[0,1].boxplot(df['stalk-color-below-ring'],patch_artist=True)



bp3 = axes[1,0].boxplot(df['stalk-surface-below-ring'],patch_artist=True)



bp4 = axes[1,1].boxplot(df['stalk-surface-above-ring'],patch_artist=True)



ax = sns.boxplot(x='class', y='stalk-color-above-ring', 

                data=df)

ax = sns.stripplot(x="class", y='stalk-color-above-ring',

                   data=df, jitter=True,

                   edgecolor="gray")

#sns.title("Class w.r.t stalkcolor above ring",fontsize=12)
#Seperating the label from the data

x=df.iloc[:,1:23]

y=df.iloc[:,0]

x.head(5)

x.describe()
df.groupby('veil-type').size()
df.corr()
#Standradise the data1;

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

newx=scaler.fit_transform(x)

print(newx)

from sklearn.decomposition import PCA

pca=PCA()

pca.fit_transform(newx)
cov=pca.get_covariance()

eigenvalues=pca.explained_variance_
print(eigenvalues)
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    

    plt.bar(range(22), eigenvalues, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
# take first 2 components and visualize using K-means

N=df.values

pca=PCA(n_components=2)

x1=pca.fit_transform(N)

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2,random_state=5)

x_clustered=kmeans.fit_predict(N)

#Plotting the Kmeans

a=x1[:,0]

b=x1[:,1]

plt.figure(figsize=(8,8))

sns.set()

ax=sns.scatterplot(a,b,hue=x_clustered)
#Split train test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=4)
from sklearn.tree import DecisionTreeClassifier

model_tree=DecisionTreeClassifier()
x_train.shape,y_train.shape
model_tree.fit(x_train,y_train)
y_prob = model_tree.predict_proba(x_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_tree.score(x_test, y_pred)
from sklearn import metrics

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
from sklearn.metrics import roc_curve,auc

fp,tp,thresholds=roc_curve(y_test,y_prob)

roc_auc=auc(fp,tp)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(fp,tp, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
# Tuned model

from sklearn.tree import DecisionTreeClassifier

model_DD = DecisionTreeClassifier()

tuned_parameters= {'criterion': ['gini','entropy'], 'max_features': ["auto","sqrt","log2"],

                   'min_samples_leaf': range(1,100,1) , 'max_depth': range(1,50,1)

                  }

           
from sklearn.model_selection import RandomizedSearchCV

DD_model= RandomizedSearchCV(model_DD, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1,random_state=5)
DD_model.fit(x_train, y_train)
print(DD_model.cv_results_)
print(DD_model.best_score_)

print(DD_model.best_params_)
y_prob = DD_model.predict_proba(x_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

DD_model.score(x_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')