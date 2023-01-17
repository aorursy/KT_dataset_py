import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import classification_report,roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from mpl_toolkits.mplot3d import Axes3D 

from sklearn.linear_model import LogisticRegression

import scipy

from scipy.spatial.distance import pdist,cdist

from scipy.cluster.hierarchy import dendrogram,linkage

from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import cophenet

from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, log_loss, classification_report,f1_score,confusion_matrix)

import xgboost

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder

import xgboost as xgb

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

import scipy

from sklearn.preprocessing import StandardScaler

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('/kaggle/input/onlinenewspop/OnlineNewsPopularity.csv')
df.head()
df.isnull().sum()
df.info()
df.describe()
#droping url

df.drop(columns=['url'],inplace=True)
#correlation plot

plt.figure(figsize=(40,40))

sns.heatmap(data=df.corr(),annot=True,cmap='BuPu')
df.columns
#box plot to check outliers

for i in df.columns:

    sns.boxplot(df[i])

    plt.show()
#removing outliers

Q1 = df.quantile(q=0.25) 



Q3 = df.quantile(q=0.75)



IQR = Q3-Q1

print('IQR for each column:- ')

print(IQR)


sorted_shares = df.sort_values(' shares') 



median = sorted_shares[' shares'].median() 



q1 = sorted_shares[' shares'].quantile(q=0.25) 



q3 = sorted_shares[' shares'].quantile(q=0.75) 



iqr = q3-q1
Inner_bound1 = q1-(iqr*1.5) 

print(f'Inner Boundary 1 = {Inner_bound1}')

Inner_bound2 = q3+(iqr*1.5)  

print(f'Inner Boundary 2 = {Inner_bound2}')

Outer_bound1 = q1-(iqr*3)    

print(f'Outer Boundary 1 = {Outer_bound1}')

Outer_bound2 = q3+(iqr*3)   

print(f'Outer Boundary 2 = {Outer_bound2}')
Df = df[df[' shares']<=Outer_bound2]
print(f'Data before Removing Outliers = {df.shape}')

print(f'Data after Removing Outliers = {Df.shape}')

print(f'Number of Outliers = {df.shape[0] - Df.shape[0]}')
Df.hist(figsize=(30,30))

plt.show()
#EDA

a,b = Df[' shares'].mean(),Df[' shares'].median()

print(f'Mean article shares = {a}')

print(f'Median article share = {b}')
Wd = Df.columns.values[30:37]

Wd
Unpop=Df[Df[' shares']<a]

Pop=Df[Df[' shares']>=a]

Unpop_day = Unpop[Wd].sum().values

Pop_day = Pop[Wd].sum().values



fig = plt.figure(figsize = (13,5))

plt.title("Count of popular/unpopular news over different day of week (Mean)", fontsize = 16)

plt.bar(np.arange(len(Wd)), Pop_day, width = 0.3, align="center", color = 'r', \

          label = "popular")

plt.bar(np.arange(len(Wd)) - 0.3, Unpop_day, width = 0.3, align = "center", color = 'b', \

          label = "unpopular")

plt.xticks(np.arange(len(Wd)), Wd)

plt.ylabel("Count", fontsize = 12)

plt.xlabel("Days of week", fontsize = 12)

    

plt.legend(loc = 'upper right')

plt.tight_layout()

plt.show()
Unpop2=Df[Df[' shares']<b]

Pop2=Df[Df[' shares']>=b]

Unpop_day2 = Unpop2[Wd].sum().values

Pop_day2 = Pop2[Wd].sum().values

fig = plt.figure(figsize = (13,5))

plt.title("Count of popular/unpopular news over different day of week (Median)", fontsize = 16)

plt.bar(np.arange(len(Wd)), Pop_day2, width = 0.3, align="center", color = 'r', \

          label = "popular")

plt.bar(np.arange(len(Wd)) - 0.3, Unpop_day2, width = 0.3, align = "center", color = 'b', \

          label = "unpopular")

plt.xticks(np.arange(len(Wd)), Wd)

plt.ylabel("Count", fontsize = 12)

plt.xlabel("Days of week", fontsize = 12)

    

plt.legend(loc = 'upper right')

plt.tight_layout()

plt.show()
Dc = Df.columns.values[12:18]
Unpop3=Df[Df[' shares']<a]

Pop3=Df[Df[' shares']>=a]

Unpop_day3 = Unpop3[Dc].sum().values

Pop_day3 = Pop3[Dc].sum().values

fig = plt.figure(figsize = (13,5))

plt.title("Count of popular/unpopular news over different data channel (Mean)", fontsize = 16)

plt.bar(np.arange(len(Dc)), Pop_day3, width = 0.3, align="center", color = 'r', \

          label = "popular")

plt.bar(np.arange(len(Dc)) - 0.3, Unpop_day3, width = 0.3, align = "center", color = 'b', \

          label = "unpopular")

plt.xticks(np.arange(len(Dc)), Dc)

plt.ylabel("Count", fontsize = 12)

plt.xlabel("Days of week", fontsize = 12)

    

plt.legend(loc = 'upper right')

plt.tight_layout()

plt.show()
Unpop4=Df[Df[' shares']<b]

Pop4=Df[Df[' shares']>=b]

Unpop_day4 = Unpop4[Dc].sum().values

Pop_day4 = Pop4[Dc].sum().values

fig = plt.figure(figsize = (13,5))

plt.title("Count of popular/unpopular news over different data channel (Median)", fontsize = 16)

plt.bar(np.arange(len(Dc)), Pop_day4, width = 0.3, align="center", color = 'r', \

          label = "popular")

plt.bar(np.arange(len(Dc)) - 0.3, Unpop_day4, width = 0.3, align = "center", color = 'b', \

          label = "unpopular")

plt.xticks(np.arange(len(Dc)), Dc)

plt.ylabel("Count", fontsize = 12)

plt.xlabel("Days of week", fontsize = 12)

    

plt.legend(loc = 'upper right')

plt.tight_layout()

plt.show()
Df.head()
mean = Df[' shares'].mean()
#Converting output columns to 0 and 1

Df[' shares'] = Df[' shares'].apply(lambda x: 0 if x <mean  else 1)
Df[' shares'].value_counts()
#Scaling and Doing SMOTE 

X = Df.drop(' shares',axis=1)

y = Df[' shares']



scaler=StandardScaler()

X=scaler.fit_transform(X)

from imblearn.over_sampling import SMOTE

SMOTE().fit_resample(X, y)

X,y = SMOTE().fit_resample(X, y)
print(y)
def calculateScore(confMat):

    TP = confMat[0][0]

    TN = confMat[1][1]

    FP = confMat[0][1]

    FN = confMat[1][0]

    Sen.append(TP / (TP + FN))

    Spe.append(TN / (FP + TN))

    FPR.append(FP / (FP + TN))

    FNR.append(FN / (FN + TP))
train, test, target_train, target_val = train_test_split(X, 

                                                         y, 

                                                         train_size= 0.80,

                                                         random_state=0);
#Using multiple classifiers

Model = []

Accuracy= []

F1Score = []

Sen = []

Spe = []

FPR = []

FNR = []
LR = LogisticRegression(multi_class='auto')

LR.fit(train,target_train)

lr_pred = LR.predict(test)

Model.append('Logistic Regression')

Accuracy.append(accuracy_score(target_val,lr_pred))

F1Score.append(f1_score(target_val,lr_pred,average=None))
data = confusion_matrix(target_val,lr_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
seed = 0

params = {

    'n_estimators':range(10,100,10),

    'criterion':['gini','entropy'],

}

rf = RandomForestClassifier()

rs = RandomizedSearchCV(rf, param_distributions=params, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)

rs.fit(X,y)
rs.best_params_
rf = RandomForestClassifier(**rs.best_params_)

rf.fit(train, target_train)

rf_pred = rf.predict(test)
features = Df.columns

importance = rf.feature_importances_

indices = np.argsort(importance)

plt.figure(1,figsize=(10,20))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importance[indices], color='lightblue', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.xlabel('Relative Importance')
Model.append('Random Forrest')

Accuracy.append(accuracy_score(target_val,rf_pred))

F1Score.append(f1_score(target_val,rf_pred,average=None))
data = confusion_matrix(target_val,rf_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
params = {

    

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'max_depth':range(1,10,1),

    'max_leaf_nodes':range(2,10,1),

}

dt = DecisionTreeClassifier()

rs = RandomizedSearchCV(dt, param_distributions=params, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)

rs.fit(X,y)
rs.best_params_
dt = DecisionTreeClassifier()

dt.fit(train, target_train)

dt_pred = dt.predict(test)

features = Df.columns

importance = dt.feature_importances_

indices = np.argsort(importance)

plt.figure(1,figsize=(10,20))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importance[indices], color='lightblue', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.xlabel('Relative Importance')
Model.append('Decision Tree')

Accuracy.append(accuracy_score(target_val,dt_pred))

F1Score.append(f1_score(target_val,dt_pred,average=None))
data = confusion_matrix(target_val,dt_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)

gb_Boost.fit(train, target_train)

y_pred = rf.predict(test)
Model.append('Gradient Boosting')

Accuracy.append(accuracy_score(target_val,y_pred))

F1Score.append(f1_score(target_val,dt_pred,average=None))
data = confusion_matrix(target_val,y_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
naiveClassifier=GaussianNB()

naiveClassifier.fit(train, target_train)

naiveClassifier_pred = naiveClassifier.predict(test)
Model.append('Naive')

Accuracy.append(accuracy_score(target_val,naiveClassifier_pred))

F1Score.append(f1_score(target_val,naiveClassifier_pred,average=None))
data = confusion_matrix(target_val,naiveClassifier_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
# knn = KNeighborsClassifier(n_neighbors=8)

# knn.fit(train, target_train)

# knn_pred = knn.predict(test)
# Model.append('KNN')

# Accuracy.append(accuracy_score(target_val,knn_pred))

# F1Score.append(f1_score(target_val,knn_pred,average=None))
# data = confusion_matrix(target_val,knn_pred)

# calculateScore(data)

# df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

# df_cm.index.name = 'Actual'

# df_cm.columns.name = 'Predicted'

# plt.figure(figsize = (10,7))

# sns.set(font_scale=1.4)

# sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
result = pd.DataFrame({'Model':Model,'Accuracy':Accuracy,'F1Score':F1Score,'Sensitivity':Sen,'Specificity':Spe,'FPR':FPR,'FNR':FNR})

result