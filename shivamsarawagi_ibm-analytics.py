import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

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
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.columns
df.head()
df.info()
df.describe()
#count of categories in categorial columns

def printCategoryCounts():

    for col, value in df.iteritems():

        if value.dtype == 'object':

            print(df[col].value_counts())

            print("=========")

            

printCategoryCounts()
#distribution plot for numerical features

fig,ax = plt.subplots(8,3, figsize=(20,35))

i = 0

j = 0

for col, value in df.iteritems():

        if value.dtype != 'object' and col != 'EmployeeCount' and col != 'StandardHours' :

            sns.distplot(df[col], ax = ax[i,j],color='orange')

            j = j +1

            if j==3:

                j = 0

                i = i + 1



plt.show()
sns.set(style="darkgrid")

plt.figure(figsize=(10,8))

total = float(len(df)) 

ax = sns.countplot(x="JobLevel", hue="Attrition", data=df) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format((height/total)*100),

            ha="center") 

plt.show()
fig,ax = plt.subplots(5,2, figsize=(25,35))

i = 0

j = 0

for col, value in df.iteritems():

        if value.dtype == 'object':

            ax1 = sns.countplot(data=df,x= col,hue="Attrition", ax = ax[i,j])

            for p in ax1.patches:

                height = p.get_height()

                ax1.text(p.get_x()+p.get_width()/2.,height + 3,

                '{:1.2f}%'.format((height/total)*100),

                ha="center") 

            j = j +1

            if j==2:

                j = 0

                i = i + 1

i = 0

for ax in fig.axes:

    if i == 2 or i==3 or i==5:

        plt.sca(ax)

        plt.xticks(rotation=90)

    i = i +1

plt.subplots_adjust(bottom=-0.2)

plt.show()
#Pair Plot

cont_col= ['Attrition','Age','MonthlyIncome', 'JobLevel','DistanceFromHome']

sns.pairplot(df[cont_col],  kind="reg", diag_kind = "kde"  , hue = 'Attrition' )

plt.show()
#box plot

fig,ax = plt.subplots(2,2, figsize=(10,10))                       

sns.boxplot(df['Attrition'], df['MonthlyIncome'], ax = ax[0,0]) 

sns.boxplot(df['Gender'], df['MonthlyIncome'], ax = ax[0,1])

plt.xticks( rotation=90)

sns.boxplot(df['Department'], df['MonthlyIncome'], ax = ax[1,0]) 

plt.xticks( rotation=90)

sns.boxplot(df['JobRole'], df['MonthlyIncome'], ax = ax[1,1])

plt.show() 
plt.figure(figsize=(12,8))

sns.heatmap(data=df.corr(),cmap="YlGnBu")
#convert object columns with hotencoding

categorical = []

for col, value in df.iteritems():

    if value.dtype == 'object':

        categorical.append(col)

numerical = df.columns.difference(categorical)

attrition_cat = df[categorical]

attrition_cat = attrition_cat.drop(['Attrition'], axis=1)

attrition_cat = pd.get_dummies(attrition_cat)

attrition_num = df[numerical]

df_final = pd.concat([attrition_num, attrition_cat], axis=1)
#Check for outlier in numerical

Q1 = attrition_num.quantile(0.25)

Q3 = attrition_num.quantile(0.75)

IQR = Q3 - Q1

((attrition_num < (Q1 - 1.5 * IQR)) | (attrition_num > (Q3 + 1.5 * IQR))).sum()
#Encode target

target_map = {'Yes':1, 'No':0}

target = df["Attrition"].apply(lambda x: target_map[x])

target.head(3)
#dropping columns which are not very significant

df_final.drop(columns=['StandardHours','Over18_Y','EmployeeCount'],inplace=True)
#Scaling

from imblearn.over_sampling import SMOTE

scaler=StandardScaler()

scaled_df=scaler.fit_transform(df_final)

X=scaled_df

Y=target

SMOTE().fit_resample(X, Y)

X,Y = SMOTE().fit_resample(X, Y)

#split data

train, test, target_train, target_val = train_test_split(X, 

                                                         Y, 

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
def calculateScore(confMat):

    TP = confMat[0][0]

    TN = confMat[1][1]

    FP = confMat[0][1]

    FN = confMat[1][0]

    Sen.append(TP / (TP + FN))

    Spe.append(TN / (FP + TN))

    FPR.append(FP / (FP + TN))

    FNR.append(FN / (FN + TP))
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
LR.coef_
seed = 0

params = {

    'n_estimators':range(10,100,10),

    'criterion':['gini','entropy'],

    'max_depth':range(2,10,1),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2'],

    'verbose':[0]

}

rf = RandomForestClassifier()

rs = RandomizedSearchCV(rf, param_distributions=params, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)

rs.fit(X,Y)
rs.best_params_
rf = RandomForestClassifier(**rs.best_params_)

rf.fit(train, target_train)

rf_pred = rf.predict(test)
features = df_final.columns

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

    'max_features':['auto','log2']

    

}

dt = DecisionTreeClassifier()

rs = RandomizedSearchCV(dt, param_distributions=params, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)

rs.fit(X,Y)
rs.best_params_
dt = DecisionTreeClassifier()

dt.fit(train, target_train)

dt_pred = dt.predict(test)
features = df_final.columns

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
# Export our trained model as a .dot file

with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(dt,

                              out_file=f,

                              max_depth = 4,

                              impurity = False,

                              feature_names = df_final.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png", height=2000, width=1900)
gb_params ={

    'n_estimators': 1500,

    'max_features': 0.9,

    'learning_rate' : 0.25,

    'max_depth': 4,

    'min_samples_leaf': 2,

    'subsample': 1,

    'max_features' : 'sqrt',

    'verbose': 0

}
gb = GradientBoostingClassifier(**gb_params)

gb.fit(train, target_train)

gb_pred = gb.predict(test)
features = df_final.columns

importance = gb.feature_importances_

indices = np.argsort(importance)

plt.figure(1,figsize=(10,20))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importance[indices], color='lightblue', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.xlabel('Relative Importance')
Model.append('Gradient Boosting')

Accuracy.append(accuracy_score(target_val,gb_pred))

F1Score.append(f1_score(target_val,gb_pred,average=None))
data = confusion_matrix(target_val,gb_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
xgb_cfl = xgb.XGBClassifier(n_jobs = -1)





params = {

        'n_estimators' : [100, 200, 500],

        'learning_rate' : [0.05, 0.1],

        'min_child_weight': [1, 5, 7],

        'gamma': [1, 1.5, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



rs = RandomizedSearchCV(xgb_cfl, param_distributions=params, scoring='accuracy', n_jobs=-1)

rs.fit(X,Y)
rs.best_params_
xgcl = xgb.XGBClassifier(**rs.best_params_)

xgcl.fit(train, target_train)

xg_pred = xgcl.predict(test)
features = df_final.columns

importance = xgcl.feature_importances_

indices = np.argsort(importance)

plt.figure(1,figsize=(10,20))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importance[indices], color='lightblue', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.xlabel('Relative Importance')
Model.append('XG Boost')

Accuracy.append(accuracy_score(target_val,xg_pred))

F1Score.append(f1_score(target_val,xg_pred,average=None))
data = confusion_matrix(target_val,xg_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
params = {

    

    'n_neighbors': range(1,25),

    'weights': ['uniform','distance'],

    'algorithm': ['ball_tree','kd_tree','brute','auto'],

    'p': [1,2,3]

}



knn = KNeighborsClassifier()



gs = GridSearchCV(estimator=knn,n_jobs=-1,cv=5,param_grid=params)

gs.fit(X,Y)
gs.best_params_
knn = KNeighborsClassifier(**gs.best_params_)

knn.fit(train, target_train)

knn_pred = knn.predict(test)
Model.append('KNN')

Accuracy.append(accuracy_score(target_val,knn_pred))

F1Score.append(f1_score(target_val,knn_pred,average=None))
data = confusion_matrix(target_val,knn_pred)

calculateScore(data)

df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
# Create ROC Graph

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(target_val, LR.predict_proba(test)[:,1])

rf_fpr, rf_tpr, rf_thresholds = roc_curve(target_val, rf.predict_proba(test)[:,1])

dt_fpr, dt_tpr, dt_thresholds = roc_curve(target_val, dt.predict_proba(test)[:,1])

gb_fpr, gb_tpr, gb_thresholds = roc_curve(target_val, gb.predict_proba(test)[:,1])

xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(target_val, xgcl.predict_proba(test)[:,1])

knn_fpr, knn_tpr, knn_thresholds = roc_curve(target_val, knn.predict_proba(test)[:,1])

plt.figure(figsize=(10,8))

plt.plot(fpr, tpr, label='Logistic Regression')

plt.plot(rf_fpr, rf_tpr, label='Random Forest)')

plt.plot(dt_fpr, dt_tpr, label='Decision Tree')

plt.plot(gb_fpr, gb_tpr, label='Gradient boosting')

plt.plot(xgb_fpr, xgb_tpr, label='XGBoost')

plt.plot(knn_fpr, knn_tpr, label='KNN')

plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Graph')

plt.legend(loc="lower right")

plt.show()
print('Classification report for LinearRegression')

print(classification_report(target_val, lr_pred))
print('Classification report for Random Forrest')

print(classification_report(target_val, rf_pred))
print('Classification report for Decsion Tree')

print(classification_report(target_val, dt_pred))
print('Classification report for GradientBoosting')

print(classification_report(target_val, gb_pred))
print('Classification report for XGB')

print(classification_report(target_val, xg_pred))
print('Classification report for KNN')

print(classification_report(target_val, knn_pred))
result = pd.DataFrame({'Model':Model,'Accuracy':Accuracy,'F1Score':F1Score,'Sensitivity':Sen,'Specificity':Spe,'FPR':FPR,'FNR':FNR})

result