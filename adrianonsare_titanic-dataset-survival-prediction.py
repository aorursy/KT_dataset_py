# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
os.getcwd()
os.chdir('/kaggle/input/titanic')#change working directory to datasource
#Read both training and testing datasets into notebook



df=pd.read_csv('train.csv')

df_test=pd.read_csv('test.csv')
os.chdir('/kaggle/working')#Change to working Dir
df.head()
df_test.head()#Test dataset first few rows
df.describe().T


df.isnull().sum(axis = 0)#return the total of "NaN" values in each column ofr training dataset

df_test.isnull().sum(axis=0)#return the total of "NaN" values in each column of test dataset

#The "Cabin" column is dropped from the training and test datasets because of a very high percentage of missing values. Any imputation would introduce sigificant errors

drop_column = ['Cabin']

df.drop(drop_column, axis=1, inplace = True)

df_test.drop(drop_column,axis=1,inplace=True)



df.head(5)#Confirm dropped columns
df_test.head()#Confirm dropped columns
#Impute missing age values with median since the missing vlaues in this case are small enough, and the median offers best representation of distribution

df['Age'] = df['Age'].replace(0, np.NaN)#Replace all zero values with "Nan"

df['Age'] = df['Age'].fillna((df['Age'].median()))



#Only two values missing for "Embarked" impute using mode()[0] 

#which is the first index mode in the series of modes in a multimodal distribution

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
#impute missing Fare and Age values with medians for test dataframe

df_test['Fare'] = df_test['Fare'].replace(0, np.NaN)#Replace all zero values with "Nan"

df_test['Age'] = df_test['Age'].replace(0, np.NaN)#Replace all zero values with "Nan"



df_test['Age'] = df_test['Age'].fillna((df_test['Age'].median()))

df_test['Fare'] = df_test['Fare'].fillna((df_test['Fare'].median()))

df.describe().T
df_test.describe().T
df_test.isnull().sum()
df.isnull().sum()
#Plotting distributions of training dataset

plt.style.use('ggplot')

pd.DataFrame.hist(df, figsize=(16,16));
#Plotting Distributions for Test dataset

plt.style.use('ggplot')

pd.DataFrame.hist(df_test, figsize=(16,16));
#How many passenger ssurvived,vs those who died

fig=plt.figure(figsize=(12,8))

rg=df['Survived'].value_counts().plot(kind="bar",alpha=0.5,color=['r','b'])



for i in rg.patches:

       rg.annotate('{:.0f}'.format(i.get_height()), (i.get_x()+0.3, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')

plt.show()
#Plotting survival vs other features

fig,axes=plt.subplots(2,2,figsize=(12,8))





#axes=es.flatten()

sns.boxplot(x=df['Survived'],y=df['Age'],ax=axes[0,0],data=df)

sns.barplot(x=df['Survived'],y=df['Pclass'],ax=axes[0,1],data=df)



estimator=lambda x: len(x) / len(df) * 100

sns.barplot(x=df['Pclass'],y=df['Pclass'],ax=axes[1,0],estimator=estimator)





sns.scatterplot(x=df['Age'],y=df['Pclass'],hue=df['Survived'],ax=axes[1,1])





plt.suptitle("Survival Vs Features")
#scatterplot of age vs fare against survival

sns.scatterplot(x=df['Age'],y=df['Fare'],hue=df['Survived'])
#KDE plot of Age distribution per class

for x in [1,2,3]:

    df.Age[df.Pclass==x].plot(kind="kde",figsize=(10,6))

    plt.title("KDE Age Class Distribution")

    plt.legend(("1st Class","2nd Class","3rd Class"))

#comparing Sex and survival



fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6))





sns.countplot(x=df['Sex'],ax=ax1,data=df)





estimator=lambda x: len(x) / len(df) * 100

sns.barplot(x=df['Survived'],y=df['Survived'],hue=df['Sex'],ax=ax2,estimator=estimator)


#The set of plots below further seek to explore the relationship between sex,passenger class and survival





#Plot subplots

fig= plt.figure(figsize=(12,10))



plt.subplot2grid((3,4),(0,0))

df.Survived[df.Sex=="male"].value_counts(normalize=True).plot(kind="bar",color='g')

plt.title("Survival Rate Males")





plt.subplot2grid((3,4),(0,1))

df.Survived[df.Sex=="female"].value_counts(normalize=True).plot(kind="bar",color='b')

plt.title("Survival Rate Females")



plt.subplot2grid((3,4),(0,2))

df.Sex[df.Survived==1].value_counts(normalize=True).plot(kind="bar",color=['g','b'])

plt.title("Survival Per Gender")







plt.subplot2grid((3,4),(2,0),colspan=4)

for x in [1,2,3]:

    df.Survived[df.Pclass==x].plot(kind="kde",figsize=(16,11.5))

    plt.title("Class wrt survival")

    plt.legend(("1st Class","2nd Class","3rd Class"))

    

plt.subplot2grid((3,4),(1,0))

df.Survived[(df.Sex=="male")&(df.Pclass==1)].value_counts(normalize=True).plot(kind="bar",color=['r','b'])

plt.title("Class 1 Male Survival")



plt.subplot2grid((3,4),(1,1))

df.Survived[(df.Sex=="male")&(df.Pclass==3)].value_counts(normalize=True).plot(kind="bar",color=['r','b'])

plt.title("Class 3 male Survival")



plt.subplot2grid((3,4),(1,2))

df.Survived[(df.Sex=="female")&(df.Pclass==1)].value_counts(normalize=True).plot(kind="bar",color=['y','g'])

plt.title("Class 1 female Survival")



plt.subplot2grid((3,4),(1,3))

df.Survived[(df.Sex=="female")&(df.Pclass==3)].value_counts(normalize=True).plot(kind="bar",color=['y','g'])

plt.title("Class 3 female Survival")





plt.show()
#Extract titles



df["Title"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #Creating new column name Title

df_test["Title"] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #Creating new column name Title
df.Title.unique() #Printout result of extraction of titles
#function to create new columns to cluster the titles along common descriptions



def title_change(data):

    data["Title"] = data["Title"].replace('Master', 'Master')

    data["Title"] = data["Title"].replace('Mlle', 'Miss')

    data["Title"] = data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')

    data["Title"] = data["Title"].replace(['Don','Mr'],'Mr')

    data["Title"] = data["Title"].replace(['Capt','Rev','Major', 'Col','Dr'], 'Honorary')

    data["Title"] = data["Title"].replace(['Lady', 'Countess','Sir','Jonkheer','Don'], 'nobility')

    return data
title_change(df)#call function on training dataset

title_change(df_test)#call function on test dataset
df=df.drop(['Name'],axis=1)

df_test=df_test.drop(['Name'],axis=1)
#select columns with string datatypes for training dataset

cat_feat= df.dtypes==object

cat_cols = df.columns[cat_feat].tolist()

cat_cols

#select columns with string datatypes for test dataset

cat_feat_test = df_test.dtypes==object

cat_cols_test = df_test.columns[cat_feat_test].tolist()

cat_cols_test
# import labelencoder



from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()
# apply encoding to test dataset

df[cat_cols] = df[cat_cols].apply(lambda col: le.fit_transform(col))

df[cat_cols].head(10)



# apply encoding to test dataset

df_test[cat_cols_test] = df_test[cat_cols_test].apply(lambda col: le.fit_transform(col))

df_test[cat_cols_test].head(10)
predictors=df.drop(['Survived'],axis=1)

target=df['Survived'].values

predictors[:10]
target[:20]
#obtain train-test split for training dataset

X_train, X_test, y_train, y_test = train_test_split(predictors, target,random_state=42)
len(X_train)
#Import sklearn modules for the ML tasks



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from xgboost import XGBClassifier

from sklearn.metrics import classification_report,confusion_matrix
#Train and test model





lr = LogisticRegression(max_iter=2000)

lr.fit(X_train,y_train)



print("Training score: "+str(lr.score(X_train,y_train)))

print("Test score:     "+str(lr.score(X_test,y_test)))



kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts

result_lr=cross_val_score(lr,predictors, target,cv=10,scoring='accuracy')



#Cross Validated Score

print('The cross validated score for Logistic REgression is:',round(result_lr.mean()*100,2))







#predictions & Confusion matrix



lr_predictions=lr.predict(X_test) 

df_cm=confusion_matrix(y_test,lr_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
rf =RandomForestClassifier(criterion='gini', n_estimators=700,

                             min_samples_split=10,min_samples_leaf=1,oob_score=True,

                             random_state=1,n_jobs=-1)

rf.fit(X_train,y_train)



print("Training score: "+str(rf.score(X_train,y_train)))

print("Test score:     "+str(rf.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_RF=cross_val_score(rf,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for Random Classifier is:',round(result_RF.mean()*100,2))









#predictions & Confusion matrix



rf_predictions=rf.predict(X_test) 

df_cm=confusion_matrix(y_test,rf_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
mlp =MLPClassifier()

mlp.fit(X_train,y_train)



print("Training score: "+str(mlp.score(X_train,y_train)))

print("Test score:     "+str(mlp.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_mlp=cross_val_score(mlp,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for MLPC Classifier is:',round(result_mlp.mean()*100,2))





#predictions & Confusion matrix



mlp_predictions=mlp.predict(X_test) 

df_cm=confusion_matrix(y_test,mlp_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
dt =DecisionTreeClassifier(max_depth=20)

dt.fit(X_train,y_train)



print("Training score: "+str(dt.score(X_train,y_train)))

print("Test score:     "+str(dt.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_dt=cross_val_score(dt,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for MLPC Classifier is:',round(result_dt.mean()*100,2))





#predictions & Confusion matrix



dt_predictions=dt.predict(X_test) 

df_cm=confusion_matrix(y_test,dt_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
#ADABoost Classifier

ada =AdaBoostClassifier(n_estimators=100, random_state=0)

ada.fit(X_train,y_train)



print("Training score: "+str(ada.score(X_train,y_train)))

print("Test score:     "+str(ada.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_ada=cross_val_score(ada,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for MLPC Classifier is:',round(result_ada.mean()*100,2))



#predictions & Confusion matrix



ada_predictions=ada.predict(X_test) 

df_cm=confusion_matrix(y_test,ada_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
gb =GradientBoostingClassifier(n_estimators=7,learning_rate=1.1)

gb.fit(X_train,y_train)



print("Training score: "+str(gb.score(X_train,y_train)))

print("Test score:     "+str(gb.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_gb=cross_val_score(gb,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for Gradient Boost Classifier is:',round(result_gb.mean()*100,2))





#predictions & Confusion matrix



gb_predictions=gb.predict(X_test) 

df_cm=confusion_matrix(y_test,gb_predictions)



plt.figure(figsize=(8,7))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()


kn =KNeighborsClassifier(n_neighbors = 6)

kn.fit(X_train,y_train)

print("Training score: "+str(kn.score(X_train,y_train)))

print("Test score:     "+str(kn.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_kn=cross_val_score(kn,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for K_Nearest Neighbour is:',round(result_kn.mean()*100,2))





#predictions & Confusion matrix



kn_predictions=kn.predict(X_test) 

df_cm=confusion_matrix(y_test,kn_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()



sv=svm.SVC()

sv.fit(X_train,y_train)

print("Training score: "+str(sv.score(X_train,y_train)))

print("Test score:     "+str(sv.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_sv=cross_val_score(sv,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for Support VectorMachines  is:',round(result_sv.mean()*100,2))





#predictions & Confusion matrix



sv_predictions=sv.predict(X_test) 

df_cm=confusion_matrix(y_test,sv_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
lda=LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

print("Training score: "+str(lda.score(X_train,y_train)))

print("Test score:     "+str(lda.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_lda=cross_val_score(lda,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for Linear Discriminant Analysis  is:',round(result_lda.mean()*100,2))





#predictions & Confusion matrix



lda_predictions=lda.predict(X_test) 

df_cm=confusion_matrix(y_test,lda_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
xg= XGBClassifier(base_score=0.8, booster="gbtree", colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.300000012, max_delta_step=0, max_depth=8,

              n_estimators=1000, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

              validate_parameters=False, verbosity=None)





eval_set = [(X_test, y_test)]

xg.fit(X_train,y_train,early_stopping_rounds=3, eval_metric="logloss",eval_set=eval_set)

print("Training score: "+str(xg.score(X_train,y_train)))

print("Test score:     "+str(xg.score(X_test,y_test)))



kfold = KFold(n_splits=10) #setting k=10

result_xg=cross_val_score(xg,predictors,target,cv=10,scoring='accuracy')

print('The cross validation score for XGBOOST  is:',round(result_xg.mean()*100,2))





#predictions & Confusion matrix



xg_predictions=xg.predict(X_test) 

df_cm=confusion_matrix(y_test,xg_predictions)



plt.figure(figsize=(5,4))



sns.set(font_scale=1.4) # for label size

sns.heatmap(df_cm, annot=True,cbar=False, annot_kws={"size": 16}) # font size



plt.title("Confusion Matrix for Survival Prediction",fontsize=18)

plt.show()
       

        

rand_seed=1     

        

models = []

models.append(('Logistic Regression', LogisticRegression(max_iter=1000)))

models.append(('RandomForest', RandomForestClassifier(criterion='gini', n_estimators=700,

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto',oob_score=True,

                             random_state=1,n_jobs=-1)))



models.append(('ADA Boost', AdaBoostClassifier()))



models.append(('GradientBoost', GradientBoostingClassifier()))

models.append(('Linear Discriminant', LinearDiscriminantAnalysis()))

models.append(('KNN Classifier', KNeighborsClassifier(n_neighbors = 4)))

models.append(('Decision Tree', DecisionTreeClassifier()))

models.append(('MLPC Classifier', MLPClassifier()))

models.append(('Support Vector', SVC()))

models.append(('XGB Classifier',  XGBClassifier()))





# evaluate each model in turn

results = []

names = []

scoring_method = 'accuracy'



for name, model in models:

    kfold = KFold(n_splits=10, random_state=rand_seed,shuffle=True)

    cv_results = cross_val_score(model,predictors,target, cv=kfold, scoring=scoring_method)

    results.append(cv_results)

    names.append(name)

    outputs = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(outputs)

# boxplot algorithm comparison

fig = plt.figure(figsize=(14,8))

fig.suptitle('Classifier Comparison')

ax = fig.add_subplot(111)

plt.style.use('ggplot')



#sns.boxplot(y=mod_list,x=m,data=mod_list)

plt.boxplot(results,patch_artist=True,showmeans=True,meanline=True)

ax.set_xticklabels(names,rotation='vertical')

plt.show()
# XGBOOST Classifier Parameters tunning 



from sklearn.model_selection import GridSearchCV

XG =XGBClassifier()

n_estim=range(10,800,20)#range and steps for parameter tuning



## obtain 

param_grid = {"n_estimators" :n_estim,'max_depth':range(3,10,2)}





XG_t = GridSearchCV(XG,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 7, verbose = 1)



XG_t.fit(X_train,y_train)







# Best score

print(XG_t.best_score_)



#print out best estimator

XG_t.best_estimator_
# Random Forest Classifier Parameters tunning 



from sklearn.model_selection import GridSearchCV

RF = RandomForestClassifier()

n_estim=range(10,1000,20)#range and steps for parameter tuning





## obtain 

param_grid = {"n_estimators" :n_estim}





RF_t = GridSearchCV(RF,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



RF_t.fit(X_train,y_train)







# Best score

print(RF_t.best_score_)



#print out best estimator

RF_t.best_estimator_
# # Gradient boosting tunning





# import xgboost as xgb

# gb = GradientBoostingClassifier()

# param_grid = {'loss' : ["deviance"],

#               'n_estimators' : [100,200,300,400],

#               'learning_rate': [0.1, 0.05, 0.01,0.001],

#               'max_depth': [4, 8],

#               'min_samples_leaf': [100,150],

#               'max_features': [0.3, 0.2,0.1] 

#               }



# gb_tun = GridSearchCV(gb,param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



# gb_tun.fit(X_train,y_train)



# # Best score

# print(gb_tun.best_score_)



# # Best Estimator

# print(gb_tun.best_estimator_)
RF_t = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=None, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=900,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

RF_t.fit(predictors, target)







prediction = RF_t.predict(df_test)

passenger_id = df_test.PassengerId

output = pd.DataFrame({'PassengerId':passenger_id, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")