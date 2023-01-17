import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")

df=df.iloc[:,1:]

df.head()
sns.set(font_scale=1,style="whitegrid")

fig,ax=plt.subplots(ncols=2,nrows=3,figsize=(16,12))

cat_list=["Age","Credit amount","Duration"]

count=0

for i in range(3):

    sns.distplot(df[cat_list[count]],ax=ax[i][0],kde=False,color="#F43EEC")

    sns.kdeplot(df[cat_list[count]],ax=ax[i][1],shade=True,color="#359F4B")

    count+=1
fig,ax=plt.subplots(ncols=2,nrows=3,figsize=(16,12))

cat_list=["Age","Duration","Credit amount"]

count=0

for i in range(3):

    sns.distplot(df[df["Sex"]=="male"][cat_list[count]],ax=ax[i][0],kde=False,color="#2758FF")

    sns.distplot(df[df["Sex"]=="female"][cat_list[count]],ax=ax[i][0],kde=False,color="#FF62AE")

    sns.kdeplot(df[df["Sex"]=="male"][cat_list[count]],ax=ax[i][1],shade=True,color="#2758FF")

    sns.kdeplot(df[df["Sex"]=="female"][cat_list[count]],ax=ax[i][1],shade=True,color="#FF62AE")

    ax[i][0].legend(labels=['male','female'])

    ax[i][1].legend(labels=['male','female'])

    count+=1
fig,ax=plt.subplots(ncols=2,nrows=3,figsize=(16,12))

cat_list=["Age","Duration","Credit amount"]

count=0

for i in range(3):

    sns.distplot(df[df["Risk"]=="good"][cat_list[count]],ax=ax[i][0],kde=False,color="#00FF7F")

    sns.distplot(df[df["Risk"]=="bad"][cat_list[count]],ax=ax[i][0],kde=False,color="#FF2424")

    sns.kdeplot(df[df["Risk"]=="good"][cat_list[count]],ax=ax[i][1],shade=True,color="#00FF7F")

    sns.kdeplot(df[df["Risk"]=="bad"][cat_list[count]],ax=ax[i][1],shade=True,color="#FF2424")

    ax[i][0].legend(labels=['good','bad'])

    ax[i][1].legend(labels=['good','bad'])

    count+=1
df.insert(1,"Cat Age",np.NaN)

df.loc[df["Age"]<25,"Cat Age"]="0-25"

df.loc[((df["Age"]>=25) & (df["Age"]<30)),"Cat Age"]="25-30"

df.loc[((df["Age"]>=30) & (df["Age"]<35)),"Cat Age"]="30-35"

df.loc[((df["Age"]>=35) & (df["Age"]<40)),"Cat Age"]="35-40"

df.loc[((df["Age"]>=40) & (df["Age"]<50)),"Cat Age"]="40-50"

df.loc[((df["Age"]>=50) & (df["Age"]<76)),"Cat Age"]="50-75"
df.insert(9,"Cat Duration",df["Duration"])

for i in df["Cat Duration"]:

    if i<12:

        df["Cat Duration"]=df["Cat Duration"].replace(i,"0-12")

    elif (i>=12) and (i<24):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"12-24")

    elif (i>=24) and (i<36):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"24-36")

    elif (i>=36) and (i<48):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"36-48")

    elif (i>=48) and (i<60):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"48-60")

    elif (i>=60) and (i<=72):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"60-72")
df.insert(4,"Cat Job",df["Job"])

df["Cat Job"]=df["Cat Job"].astype("category")

df["Cat Job"]=df["Cat Job"].replace(0,"unskilled")

df["Cat Job"]=df["Cat Job"].replace(1,"resident")

df["Cat Job"]=df["Cat Job"].replace(2,"skilled")

df["Cat Job"]=df["Cat Job"].replace(3,"highly skilled")
df["Job"]=pd.Categorical(df["Job"],categories=[0,1,2,3],ordered=True)

df["Cat Age"]=pd.Categorical(df["Cat Age"],categories=['0-25','25-30', '30-35','35-40','40-50','50-75'],ordered=True)

df["Cat Duration"]=pd.Categorical(df["Cat Duration"],categories=['0-12','12-24', '24-36','36-48','48-60','60-72'],ordered=True)
df.head()
import missingno as msno
msno.bar(df,sort='descending')
msno.matrix(df)
msno.heatmap(df)
df["Saving accounts"].fillna(df["Saving accounts"].mode()[0],inplace=True)

df["Checking account"].fillna(df["Checking account"].mode()[0],inplace=True)
msno.matrix(df,sort='descending')
df["Saving accounts"]=pd.Categorical(df["Saving accounts"],ordered=True,categories=['little','moderate','rich','quite rich'])

df["Checking account"]=pd.Categorical(df["Checking account"],ordered=True,categories=['little','moderate','rich'])
df.head()
fig,ax=plt.subplots(ncols=2,figsize=(16,5))

df["Risk"].value_counts().plot.pie(autopct="%.2f%%",colors=['#00FF7F','#FF2424'],explode = (0.1, 0.1),ax=ax[0])

sns.countplot(df["Risk"],ax=ax[1],palette=['#00FF7F','#FF2424'])
fig,ax=plt.subplots(ncols=2,nrows=3,figsize=(16,18))

cat_list=["Cat Age","Sex","Cat Job","Housing","Cat Duration","Purpose"]

palette=["red","blue","purple","green","yellow","cyan"]

count=0

for i in range(3):

    for j in range(2):

        sns.countplot(df[cat_list[count]],ax=ax[i][j],palette=sns.dark_palette(palette[count],reverse=True))

        ax[i][j].set_xticklabels(ax[i][j].get_xticklabels(),rotation=30)

        count+=1
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.stripplot(x="Sex",y="Age",data=df,ax=ax[0],palette="bwr")

sns.stripplot(x="Risk",y="Age",data=df,ax=ax[1],palette="rainbow_r")
plt.figure(figsize=(16,5))

sns.barplot(data=df,x="Sex",y="Age",hue="Cat Job",palette="hsv_r")
g=sns.FacetGrid(data=df,col="Risk",aspect=1.68,height=4).map(sns.pointplot,"Purpose","Age","Sex",palette="Set2_r",ci=None).add_legend();

g.set_xticklabels(rotation=60)
sns.FacetGrid(data=df,col="Risk",aspect=1.68,height=4).map(sns.pointplot,"Cat Age","Credit amount","Sex",palette=["#FF7659","#30AB55"],ci=None).add_legend();
fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(16,10))

sns.boxplot(x="Saving accounts",y="Age",data=df,ax=ax[0][0],hue="Sex",palette=["#008080","#E74C3C"]).legend(loc="upper right")

sns.boxplot(x="Saving accounts",y="Age",data=df,ax=ax[0][1],hue="Risk",palette=["#00FFFF","#FFB600"]).legend(loc="upper right")

sns.boxplot(x="Checking account",y="Age",data=df,ax=ax[1][0],hue="Sex",palette=["#008080","#E74C3C"]).legend(loc="upper right")

sns.boxplot(x="Checking account",y="Age",data=df,ax=ax[1][1],hue="Risk",palette=["#00FFFF","#FFB600"]).legend(loc="upper right")
plt.figure(figsize=(16,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=sns.cubehelix_palette(8),linecolor='black',linewidths=10)

plt.show()
df["Age"],df["Duration"],df["Job"]=df["Cat Age"],df["Cat Duration"],df["Cat Job"]

df=df.drop(["Cat Age","Cat Duration","Cat Job"],axis=1)
liste_columns=list(df.columns)

liste_columns.remove("Sex")

liste_columns.remove("Risk")

liste_columns.remove("Credit amount")
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df["Sex"]=label.fit_transform(df["Sex"])

df["Risk"]=label.fit_transform(df["Risk"])

df=pd.get_dummies(df,columns=liste_columns,prefix=liste_columns)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df["Credit amount"]=scaler.fit_transform(df[["Credit amount"]])
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,roc_curve,roc_auc_score,auc
X=df.drop(["Risk"],axis=1)

Y=df["Risk"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
xgb_model=XGBClassifier().fit(X_train,Y_train)

lgbm_model=LGBMClassifier().fit(X_train,Y_train)

cat_model=CatBoostClassifier().fit(X_train,Y_train)
print("XGBoost Model")

print("-"*15)

print("Train R2 Score : ",r2_score(Y_train,xgb_model.predict(X_train)))

print("Train Accuracy Score : ",accuracy_score(Y_train,xgb_model.predict(X_train)))

print("-"*50)

print("Test R2 Score : ",r2_score(Y_test,xgb_model.predict(X_test)))

print("Test Accuracy Score : ",accuracy_score(Y_test,xgb_model.predict(X_test)))



plt.figure(figsize=(16,5))

logit_roc_auc=roc_auc_score(Y_test,xgb_model.predict(X_test))

fpr,tpr,thresholds = roc_curve(Y_test,xgb_model.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="AUC = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(loc="lower right")

plt.title("ROC")

plt.show()
print("LightGBM Model")

print("-"*15)

print("Train R2 Score : ",r2_score(Y_train,lgbm_model.predict(X_train)))

print("Train Accuracy Score : ",accuracy_score(Y_train,lgbm_model.predict(X_train)))

print("-"*50)

print("Test R2 Score : ",r2_score(Y_test,lgbm_model.predict(X_test)))

print("Test Accuracy Score : ",accuracy_score(Y_test,lgbm_model.predict(X_test)))



plt.figure(figsize=(16,5))

logit_roc_auc=roc_auc_score(Y_test,lgbm_model.predict(X_test))

fpr,tpr,thresholds = roc_curve(Y_test,lgbm_model.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="AUC = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(loc="lower right")

plt.title("ROC")

plt.show()
print("Cat Model")

print("-"*15)

print("Train R2 Score : ",r2_score(Y_train,cat_model.predict(X_train)))

print("Train Accuracy Score : ",accuracy_score(Y_train,cat_model.predict(X_train)))

print("-"*50)

print("Test R2 Score : ",r2_score(Y_test,cat_model.predict(X_test)))

print("Test Accuracy Score : ",accuracy_score(Y_test,cat_model.predict(X_test)))



plt.figure(figsize=(16,5))

logit_roc_auc=roc_auc_score(Y_test,cat_model.predict(X_test))

fpr,tpr,thresholds = roc_curve(Y_test,cat_model.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="AUC = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(loc="lower right")

plt.title("ROC")

plt.show()
before_model=pd.DataFrame({"Model":["XGBoost","LightGBM","CatBoost"],

                   "Train Accuracy":[accuracy_score(Y_train,xgb_model.predict(X_train)),accuracy_score(Y_train,lgbm_model.predict(X_train)),accuracy_score(Y_train,cat_model.predict(X_train))],

                   "Test Accuracy":[accuracy_score(Y_test,xgb_model.predict(X_test)),accuracy_score(Y_test,lgbm_model.predict(X_test)),accuracy_score(Y_test,cat_model.predict(X_test))]})
fig,ax=plt.subplots(ncols=2,figsize=(16,5))

sns.barplot(x="Model",y="Train Accuracy",data=before_model,ax=ax[0],palette="tab20c_r")

sns.barplot(x="Model",y="Test Accuracy",data=before_model,ax=ax[1],palette="tab20c_r")
from sklearn.model_selection import GridSearchCV
xgb_params={"max_depth":[3,4,5,6],

            "subsample":[0.6,0.8,1],

            "n_estimators":[100,200,500,1000],

            "learning_rate":[0.1,0.01,0.2,0.5,0.05]}
lgbm_params={"max_depth":[3,4,5,6],

              "subsample":[0.6,0.8,1.0],

              "n_estimators":[100,200,500,1000],

              "learning_rate":[0.1,0.01,0.02,0.05],

              "min_child_samples":[5,10,20]}
cat_params={"iterations":[100,200,500,1000,2000],

           "learning_rate":[0.1,0.01,0.2,0.5,1],

           "depth":[3,4,5,6]}
xgb_params_tuned=GridSearchCV(xgb_model,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,Y_train)
lgbm_params_tuned=GridSearchCV(lgbm_model,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,Y_train)
cat_params_tuned=GridSearchCV(cat_model,cat_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,Y_train)
print(xgb_params_tuned.best_params_)

print(lgbm_params_tuned.best_params_)

print(cat_params_tuned.best_params_)
xgb_model_tuned = XGBClassifier(learning_rate=0.01,max_depth=3,n_estimators=100,subsample=0.8).fit(X_train,Y_train)

lgbm_model_tuned = LGBMClassifier(learning_rate=0.02,max_depth=3,min_child_samples=10,n_estimators=200,subsample=0.6).fit(X_train,Y_train)

cat_model_tuned = CatBoostClassifier(depth=4,iterations=100,learning_rate=0.01).fit(X_train,Y_train)
print("XGBoost Model")

print("-"*15)

print("Train R2 Score : ",r2_score(Y_train,xgb_model_tuned.predict(X_train)))

print("Train Accuracy Score : ",accuracy_score(Y_train,xgb_model_tuned.predict(X_train)))

print("-"*50)

print("Test R2 Score : ",r2_score(Y_test,xgb_model_tuned.predict(X_test)))

print("Test Accuracy Score : ",accuracy_score(Y_test,xgb_model_tuned.predict(X_test)))



plt.figure(figsize=(16,5))

logit_roc_auc=roc_auc_score(Y_test,xgb_model_tuned.predict(X_test))

fpr,tpr,thresholds = roc_curve(Y_test,xgb_model_tuned.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="AUC = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(loc="lower right")

plt.title("ROC")

plt.show()
print("LightGBM Model")

print("-"*15)

print("Train R2 Score : ",r2_score(Y_train,lgbm_model_tuned.predict(X_train)))

print("Train Accuracy Score : ",accuracy_score(Y_train,lgbm_model_tuned.predict(X_train)))

print("-"*50)

print("Test R2 Score : ",r2_score(Y_test,lgbm_model_tuned.predict(X_test)))

print("Test Accuracy Score : ",accuracy_score(Y_test,lgbm_model_tuned.predict(X_test)))



plt.figure(figsize=(16,5))

logit_roc_auc=roc_auc_score(Y_test,lgbm_model_tuned.predict(X_test))

fpr,tpr,thresholds = roc_curve(Y_test,lgbm_model_tuned.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="AUC = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(loc="lower right")

plt.title("ROC")

plt.show()
print("Cat Model")

print("-"*15)

print("Train R2 Score : ",r2_score(Y_train,cat_model_tuned.predict(X_train)))

print("Train Accuracy Score : ",accuracy_score(Y_train,cat_model_tuned.predict(X_train)))

print("-"*50)

print("Test R2 Score : ",r2_score(Y_test,cat_model_tuned.predict(X_test)))

print("Test Accuracy Score : ",accuracy_score(Y_test,cat_model_tuned.predict(X_test)))



plt.figure(figsize=(16,5))

logit_roc_auc=roc_auc_score(Y_test,cat_model_tuned.predict(X_test))

fpr,tpr,thresholds = roc_curve(Y_test,cat_model_tuned.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="AUC = %0.2f)"%logit_roc_auc)

plt.plot([0,1],[0,1],"r--")

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(loc="lower right")

plt.title("ROC")

plt.show()
after_model=pd.DataFrame({"Model":["XGBoost","LightGBM","CatBoost"],

                   "Train Accuracy":[accuracy_score(Y_train,xgb_model_tuned.predict(X_train)),accuracy_score(Y_train,lgbm_model_tuned.predict(X_train)),accuracy_score(Y_train,cat_model_tuned.predict(X_train))],

                   "Test Accuracy":[accuracy_score(Y_test,xgb_model_tuned.predict(X_test)),accuracy_score(Y_test,lgbm_model_tuned.predict(X_test)),accuracy_score(Y_test,cat_model_tuned.predict(X_test))]})
fig,ax=plt.subplots(ncols=2,figsize=(16,5))

sns.barplot(x="Model",y="Train Accuracy",data=after_model,ax=ax[0],palette="tab20c_r")

sns.barplot(x="Model",y="Test Accuracy",data=after_model,ax=ax[1],palette="tab20c_r")