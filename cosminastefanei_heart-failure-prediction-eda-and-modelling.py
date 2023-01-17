#import necessary modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#load data

df=pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
#check the beginning

df.head()
#check for missing values

df.isnull().sum()    



# we notice that there are no missing values
# investigate data types

df.info()
#change data type for the indicated columns



cat_cols=["anaemia","diabetes","high_blood_pressure","sex","smoking","DEATH_EVENT"]



df[cat_cols]=df[cat_cols].astype("category")

#rename DEATH_EVENT column

df.rename({"DEATH_EVENT":"death"},axis=1,inplace=True)
#check statistical measures of the data

df.describe()
#import matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns



#create correlation matrix

cor_mat=df.corr()



# set figure size

plt.figure(figsize=(9,7))



#create the heatmap

ax=sns.heatmap(cor_mat,cmap="Blues",linewidths=2, linecolor='black',annot=True)

ax.set_ylim([0,7])

plt.show()
#set the style of the plots

sns.set_style("darkgrid")
#plot age

plt.figure(figsize=(6,6))

sns.distplot(df.age,bins=10)

plt.xlabel("Age of the patients")

plt.ylabel("Number of patients")
#plot creatinine_phosphokinase

plt.figure(figsize=(6,6))

sns.distplot(df.creatinine_phosphokinase,bins=10)

plt.xlabel("Level of creatinine phosphokinase")

plt.ylabel("Number of patients")
#plot ejection_fraction

plt.figure(figsize=(6,6))

sns.distplot(df.ejection_fraction,bins=10)

plt.xlabel("Ejection fraction")

plt.ylabel("Number of patients")
#plot platelets

plt.figure(figsize=(6,6))

sns.distplot(df.platelets,bins=10)

plt.xlabel("Concentration of platelets")

plt.ylabel("Number of patients")
#plot serum_creatinine

plt.figure(figsize=(6,6))

sns.distplot(df.serum_creatinine,bins=10)

plt.xlabel("Level of creatinine")

plt.ylabel("Number of patients")
#plot serum_sodium

plt.figure(figsize=(6,6))

sns.distplot(df.serum_sodium,bins=10)

plt.xlabel("Level of sodium")

plt.ylabel("Number of patients")
#plot anaemia for each sex

plt.figure(figsize=(6,6))

sns.set_style("darkgrid")

sns.catplot(x="anaemia",hue="death",data=df,kind="count",col="sex",palette="colorblind")

plt.show()
#plot diabetes for each sex

plt.figure(figsize=(6,6))

sns.set_style("darkgrid")

sns.catplot(x="diabetes",hue="death",data=df,kind="count",col="sex",palette="colorblind")

plt.show()
#plot smoking for each sex

plt.figure(figsize=(6,6))

sns.set_style("darkgrid")

sns.catplot(x="smoking",hue="death",data=df,kind="count",col="sex",palette="colorblind")

plt.show()
#plot high_blood_pressure for each sex

plt.figure(figsize=(6,6))

sns.set_style("darkgrid")

sns.catplot(x="high_blood_pressure",hue="death",data=df,kind="count",col="sex",palette="colorblind")

plt.show()
#the graphs in this section will be produced using plotly

import plotly.express as px
#plot platelets vs creatinine_phosphokinase

fig=px.scatter(df,x="platelets",y="creatinine_phosphokinase",color="death",template="plotly_dark",width=1000,height=500)

fig.update_traces(marker=dict(size=12, line=dict(width=1,color='LightBlue')),selector=dict(mode='markers'))
#plot serum_creatinine vs serum_sodium

fig=px.scatter(df,x="serum_creatinine",y="serum_sodium",color="death",template="plotly_dark",width=1000,height=500)

fig.update_traces(marker=dict(size=12, line=dict(width=1,color='LightBlue')),selector=dict(mode='markers'))
#plot serum_creatinine vs creatinine_phosphokinase 

fig=px.scatter(df,x="serum_creatinine",y="creatinine_phosphokinase",color="death",template="plotly_dark",width=1000,height=500)

fig.update_traces(marker=dict(size=12, line=dict(width=1,color='LightBlue')),selector=dict(mode='markers'))
# boxplot of the age variable 

px.box(df, x="smoking", y="age",width=1000,height=500,facet_col="high_blood_pressure",color_discrete_sequence=['darkorchid']

)
# boxplot of the ejection_fraction variable 

px.box(df, x="smoking", y="ejection_fraction",width=1000,height=500,color="high_blood_pressure",

       color_discrete_sequence=['crimson','yellow']

)
#boxplot of ejection_fraction variable|

px.box(df, x="smoking", y="creatinine_phosphokinase",width=1000,height=500,color="high_blood_pressure",

       color_discrete_sequence=['darkgreen','blue']

)
# boxplot of the platelets variable 

px.box(df, x="smoking", y="platelets", color="sex",width=1000,height=500)
#scaling our data



from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()



columns=["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]



for column in columns:

    df[column]=scaler.fit_transform(df[column].values.reshape(-1,1))
#split into train and test



from sklearn.model_selection import train_test_split



X=df.drop("death",axis=1)

y=df.death



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.3,stratify=y)
#import modules needed for hyperparameter tunning

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,cross_val_score



#import modules needed for performance check

from sklearn.metrics import confusion_matrix, classification_report
#import the model

from sklearn.neighbors import KNeighborsClassifier



#instantiate the classifier

knn=KNeighborsClassifier()



#define parameter range

param_grid_knn={"n_neighbors":range(1,15)}



#run gridsearch 

cv_knn=GridSearchCV(knn,param_grid_knn,cv=10)

cv_knn.fit(X_train,y_train)
#get the best estimator

best_knn=cv_knn.best_estimator_



#get the predicted classes

y_pred_knn=best_knn.predict(X_test)



#get the score for the test set

print("The score for the tuned KNN model is {}".format(best_knn.score(X_test,y_test)))
#get the metrics for the two classes

print(pd.DataFrame(classification_report(y_test,y_pred_knn,output_dict=True)))
#plot the heatmap corresponding to the confusion matrix

ax=sns.heatmap(confusion_matrix(y_test,y_pred_knn),annot=True,cmap="GnBu")

ax.set_ylim([0,2])
#import the model

from sklearn.linear_model import LogisticRegression



#instantiate the classifier

logreg=LogisticRegression(random_state=10,solver='liblinear')



#define parameter range

param_grid_logreg={"C":np.logspace(-4, 4, 20),'penalty' : ['l1', 'l2']}



#run gridsearch 

cv_logreg=GridSearchCV(logreg,param_grid_logreg,cv=10)

cv_logreg.fit(X_train,y_train)
#get the best estimator

best_logreg=cv_logreg.best_estimator_



#get the predicted classes

y_pred_logreg=best_logreg.predict(X_test)



#get the score for the test set

print("The score for the tuned Logistic Regression is {}".format(best_logreg.score(X_test,y_test)))
#get the metrics for the two classes

print(pd.DataFrame(classification_report(y_test,y_pred_logreg,output_dict=True)))
#plot the heatmap corresponding to the confusion matrix

ax=sns.heatmap(confusion_matrix(y_test,y_pred_logreg),annot=True,cmap="GnBu")

ax.set_ylim([0,2])
#import the model

from sklearn.ensemble import RandomForestClassifier



#instantiate the classifier

rf=RandomForestClassifier(random_state=10)



#define parameter range

param_grid_rf={"n_estimators":range(100,401,50),"criterion":["gini","entropy"],"max_depth":range(2,10),

               "min_samples_leaf":np.arange(0.1,0.51,0.1),"max_features":["auto","sqrt","log2"]

              }



#run randomized search 

cv_rf=RandomizedSearchCV(rf,param_grid_rf,cv=10)

cv_rf.fit(X_train,y_train)
#get the best estimator

best_rf=cv_rf.best_estimator_



#get the predicted classes

y_pred_rf=best_rf.predict(X_test)



#get the score for the test set

print("The score for the tuned Random Forest is {}".format(best_rf.score(X_test,y_test)))
#get the metrics for the two classes

print(pd.DataFrame(classification_report(y_test,y_pred_rf,output_dict=True)))
#plot the heatmap corresponding to the confusion matrix

ax=sns.heatmap(confusion_matrix(y_test,y_pred_rf),annot=True,cmap="GnBu")

ax.set_ylim([0,2])
#import the model

from sklearn.ensemble import AdaBoostClassifier



#instantiate the classifier

ada=AdaBoostClassifier(random_state=10)



#range for number of estimators

param_grid_ada={"n_estimators":range(50,401,50)}



#run the gridsearch

cv_ada=GridSearchCV(ada,param_grid_ada,cv=3)

cv_ada.fit(X_train,y_train)
#get the best estimator

best_ada=cv_ada.best_estimator_

best_ada
#get the predicted classes

y_pred_ada=best_ada.predict(X_test)



#get the score for the test set

print("The score for the tuned AdaBoost model is {}".format(best_ada.score(X_test,y_test)))
#get the metrics for the two classes

print(pd.DataFrame(classification_report(y_test,y_pred_ada,output_dict=True)))
#plot the heatmap corresponding to the confusion matrix

ax=sns.heatmap(confusion_matrix(y_test,y_pred_ada),annot=True,cmap="GnBu")

ax.set_ylim([0,2])