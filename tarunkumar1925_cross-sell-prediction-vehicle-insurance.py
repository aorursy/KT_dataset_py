import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/janatahack-crosssell-prediction/train.csv')

df.head()
del df['id']
df.shape
df.isnull().sum()
df_test=pd.read_csv('../input/janatahack-crosssell-prediction/test.csv')

df_test.head()
df_test.shape
df.info()
df_test.info()
sns.countplot(df['Gender']);
sns.countplot(df['Gender'],hue=df['Driving_License']);
plt.figure(figsize=(10,10))

sns.catplot(x="Vehicle_Age", y="Response", hue="Gender", kind="bar", data=df);
sns.catplot(x="Vehicle_Age", y="Response", hue="Previously_Insured", kind="bar", data=df);
sns.catplot(x="Vehicle_Age", y="Response", hue="Driving_License", kind="bar", data=df);
sns.catplot(x="Gender", y="Response", hue="Previously_Insured", kind="bar", data=df.query("Vehicle_Age == '> 2 Years'"))
sns.catplot(x="Gender", y="Response", hue="Previously_Insured", kind="bar", data=df.query("Vehicle_Age == '1-2 Year'"))
sns.lineplot(df['Policy_Sales_Channel'],df['Vintage'],hue=df['Gender'])
sns.boxplot(df['Age'])
df[df['Age']>=80]['Response'].value_counts()
# index=df[df['Age']>=80].index
# df.drop(labels=index,inplace=True)
df['Annual_Premium'].max(),df['Annual_Premium'].min()


def outliers(df,features):

  for c in features:

    Q1=np.percentile(df[c],25)

    Q3=np.percentile(df[c],75)

    IQR=Q3-Q1

    outliers=df[(df[c] < (Q1-1.5 * IQR)) | (df[c] > (Q3 + 1.5 * IQR))]

    return outliers.index
outliers(df,['Annual_Premium'])
df.drop(labels=outliers(df,['Annual_Premium','Age','Vintage']),inplace=True)
sns.countplot(df['Response']);
df['Gender']=df['Gender'].replace(['Male','Female'],[1,0])

df['Vehicle_Age']=df['Vehicle_Age'].replace(['< 1 Year','1-2 Year','> 2 Years'],[1,2,3])

df['Vehicle_Damage']=df['Vehicle_Damage'].replace(['Yes','No'],[1,0])





df_test['Gender']=df_test['Gender'].replace(['Male','Female'],[1,0])

df_test['Vehicle_Age']=df_test['Vehicle_Age'].replace(['< 1 Year','1-2 Year','> 2 Years'],[1,2,3])

df_test['Vehicle_Damage']=df_test['Vehicle_Damage'].replace(['Yes','No'],[1,0])
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True);
X=df.drop(['Response'],axis=1)

y=df.Response.values
X.columns
y

from imblearn.over_sampling import SMOTE

sm=SMOTE()
X_res,y_res=sm.fit_sample(X,y)
X_res.shape,y_res.shape
from collections import Counter

print("Orginal Dataset Shape {}".format(Counter(y)))

print("Applying Smote dataset shape {}".format(Counter(y_res)))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.3,random_state=23,stratify=y_res,shuffle=True)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

sc=StandardScaler()

X_train_res=sc.fit_transform(X_train)

X_test_res=sc.transform(X_test)
X_train_res
# !pip install xgboost
# from xgboost import  XGBClassifier

# xg=XGBClassifier()
# model=xg.fit(X_train_res,y_train)
# y_pred=model.predict(X_test_res)
!pip install catboost
from catboost import CatBoostClassifier

cb=CatBoostClassifier(task_type='GPU',loss_function='Logloss',iterations=9500,l2_leaf_reg=8,depth=8)
model_cb=cb.fit(X_train_res,y_train)
y_pred=model_cb.predict(X_test_res)
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_score,f1_score,classification_report,confusion_matrix,plot_confusion_matrix
print("Accuarcy on training", accuracy_score(y_train,model_cb.predict(X_train_res)))

print("Accuarcy on testing",accuracy_score(y_test,model_cb.predict(X_test_res)))
precision_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
print(plot_confusion_matrix(model_cb,X_test_res,y_test,values_format='.1f',cmap='Blues'))
cb_probs=model_cb.predict_proba(X_test_res)[:,1]

cb_probs
cb_auc=roc_auc_score(y_test,cb_probs)

print("cb_area",cb_auc)
fpr,tpr,th=roc_curve(y_test,cb_probs)
plt.plot([0,1],[0,1],linestyle='--',color='red')

plt.plot(fpr,tpr,marker='*',label='RF auc {}'.format(cb_auc.round(2)))

plt.legend()
feat_importances = pd.Series(model_cb.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')

#feat_importances.nsmallest(20).plot(kind='barh')

plt.show()
# parameters={"learning_rate"    : [0.10, 0.15, 0.20,0.30] ,

#  "max_depth"        : [ 3,5,8,10, 12, 15],

#  "min_child_weight" : [ 1, 3, 5, 7 ],

#  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3],

#  "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

#  }



# parameters={'depth':[4,6,7,8],

#             'learning_rate':[0.14,0.16,0.18,0.2],

#             'iterations':[5000,5500,6000,7000],

#             'l2_leaf_reg': [2,6,10,14]

           

#             }
# from sklearn.model_selection import RandomizedSearchCV

# import time
# rd_obj=RandomizedSearchCV(model_cb,parameters,scoring='accuracy',cv=20)
# rd_obj
# start=time.time()

# rd_obj.fit(X_train_res,y_train)

# end=time.time()

# print("Total time taken is {}".format(end-start))
# rd_obj.best_params_
# best_split=rd_obj.best_estimator_

# best_split
#  model_cv=best_split.fit(X_train_res,y_train)
# from sklearn.model_selection import cross_val_score

# print(cross_val_score(model_cv,X_train_res,y_train,cv=10,scoring='accuracy').mean())
df_test.head()
df_test_copy=df_test.copy()
df_test_copy.drop('id',axis=1,inplace=True)
df_test_copy.columns
df_test_copy=sc.transform(df_test_copy)
df_test_copy
predictions=model_cb.predict_proba(df_test_copy)[:,1]
predictions
final=pd.DataFrame()

final['id']=df_test['id']

final['Response']=predictions
final
final.to_csv('final_cb.csv',index=False)
#if you like my work ,Please upvote it.