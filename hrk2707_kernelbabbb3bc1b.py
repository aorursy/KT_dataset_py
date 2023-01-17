# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Lodind the data

df=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

print(df.shape)

print(df.columns)

df.head()

# Plotting the distribution of data in different class

positive=0

neg=0

import seaborn as sns

import matplotlib.pyplot as plt

for i in range(df.shape[0]):

    if df.loc[i,"Class"] == 1:

        positive=positive+1

    else:

        neg=neg+1

        

print("Positive class size : " , positive)

print("Negative class size : ",neg)





sns.catplot(x='Class',kind='count', data=df);





# Histogram of different features to whether they are normally distributed or not 

df.hist(bins=50,grid=False,figsize=(200,200),xlabelsize=100,ylabelsize=100)



## Since Amount and time are not normalized and they are not normally distributed so i used robustscaler() instead of standardscaler()

## as robustscaler takes IQR range for normalization and doesn't assume data to be normally distributed

from sklearn import preprocessing

scaler = preprocessing.RobustScaler()

df["Scaled_Amount"]=scaler.fit_transform(df["Amount"].values.reshape(-1,1))

df["Scaled_Time"]=scaler.fit_transform(df["Time"].values.reshape(-1,1))

df=df.drop(["Amount","Time"],axis=1)

# Data after normalization 

df
## Splitting the data in training and testing

from sklearn.model_selection import train_test_split

y=df["Class"]

X=df.drop(["Class"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# training and testing on the imbalanced data without tuning,undersampling,oversampling etc 

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score



log_reg_default=LogisticRegression()

log_reg_default.fit(X_train, y_train)

y_pred=log_reg_default.predict(X_test)



print("Logistic Regression before any tuning of parameter : ")

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))







# Undersampling the training data i.e creating a new_df in which the number of abundant class data 

# is reduced to the rare class data after shuffling

new_df=pd.concat([X_train,y_train],axis=1)

new_df=new_df.sample(frac=1)

undersample_data_positive=new_df.loc[new_df["Class"]==1]

undersample_data_negative=new_df.loc[new_df["Class"]==0][:undersample_data_positive.shape[0]]

new_df=pd.concat([undersample_data_positive,undersample_data_negative])

new_df=new_df.sample(frac=1)

new_df
# Now finding the correlation matrix for the new_df because after undersampling the feature that are highly correlated to rare class is more effectively described by corr. matrix 

corr=new_df.corr()

f, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap(corr, center=0)

# boxplot of highly negatively correlated features wrt Class. This also shows the density of outliers present

f,axes = plt.subplots(ncols=5,figsize=(50,20))

sns.boxplot(x="Class",y="V17",data=new_df,ax=axes[0])

axes[0].set_title('V17 vs Class negative correlation')

sns.boxplot(x="Class",y="V14",data=new_df,ax=axes[1])

axes[1].set_title('V14 vs Class negative correlation')

sns.boxplot(x="Class",y="V12",data=new_df,ax=axes[2])

axes[2].set_title('V12 vs Class negative correlation')

sns.boxplot(x="Class",y="V10",data=new_df,ax=axes[3])

axes[3].set_title('V10 vs Class negative correlation')

sns.boxplot(x="Class",y="V16",data=new_df,ax=axes[4])

axes[4].set_title('V16 vs Class negative correlation')
# boxplot of highly positively correlated features wrt Class. This also shows the density of outliers present

f,axes = plt.subplots(ncols=4,figsize=(50,20))

sns.boxplot(x="Class",y="V11",data=new_df,ax=axes[0])

axes[0].set_title('V11 vs Class negative correlation')

sns.boxplot(x="Class",y="V4",data=new_df,ax=axes[1])

axes[0].set_title('V4 vs Class negative correlation')

sns.boxplot(x="Class",y="V2",data=new_df,ax=axes[2])

axes[0].set_title('V2 vs Class negative correlation')

sns.boxplot(x="Class",y="V19",data=new_df,ax=axes[3])

axes[0].set_title('V19 vs Class negative correlation')
# Removing the outliers from new_df by removing those which are very far i.e <(q25-iqr*1.5) and >(q75+iqr*1.5) for each of these highly correlated features

v17_values = new_df["V17"].values

q25,q75 = np.percentile(v17_values,25),np.percentile(v17_values,75)

iqr=q75-q25

v17_cutoff = iqr*1.5

v17_upper=q75+v17_cutoff

v17_lower=q25-v17_cutoff

new_df=new_df.drop(new_df[(new_df["V17"] > v17_upper) | (new_df["V17"] < v17_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V17_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v17_upper)

print('lower : ',v17_lower)



v14_values = new_df["V14"].values

q25,q75 = np.percentile(v14_values,25),np.percentile(v14_values,75)

iqr=q75-q25

v14_cutoff = iqr*1.5

v14_upper=q75+v14_cutoff

v14_lower=q25-v14_cutoff

new_df=new_df.drop(new_df[(new_df["V14"] > v14_upper) | (new_df["V14"] < v14_lower)].index)



print('V14_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v14_upper)

print('lower : ',v14_lower)



v12_values = new_df["V12"].values

q25,q75 = np.percentile(v12_values,25),np.percentile(v12_values,75)

iqr=q75-q25

v12_cutoff = iqr*1.5

v12_upper=q75+v12_cutoff

v12_lower=q25-v12_cutoff

new_df=new_df.drop(new_df[(new_df["V12"] > v12_upper) | (new_df["V12"] < v12_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V12_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v12_upper)

print('lower : ',v12_lower)



v10_values = new_df["V10"].values

q25,q75 = np.percentile(v10_values,25),np.percentile(v10_values,75)

iqr=q75-q25

v10_cutoff = iqr*1.5

v10_upper=q75+v10_cutoff

v10_lower=q25-v10_cutoff

new_df=new_df.drop(new_df[(new_df["V10"] > v10_upper) | (new_df["V10"] < v10_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V10_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v10_upper)

print('lower : ',v10_lower)



v16_values = new_df["V16"].values

q25,q75 = np.percentile(v16_values,25),np.percentile(v16_values,75)

iqr=q75-q25

v16_cutoff = iqr*1.5

v16_upper=q75+v16_cutoff

v16_lower=q25-v16_cutoff

new_df=new_df.drop(new_df[(new_df["V16"] > v16_upper) | (new_df["V16"] < v16_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V16_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v16_upper)

print('lower : ',v16_lower)
# Same is done for highly negatively correlated feautres

v11_values = new_df["V11"].values

q25,q75 = np.percentile(v11_values,25),np.percentile(v11_values,75)

iqr=q75-q25

v11_cutoff = iqr*1.5

v11_upper=q75+v11_cutoff

v11_lower=q25-v11_cutoff

new_df=new_df.drop(new_df[(new_df["V11"] > v11_upper) | (new_df["V11"] < v11_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V11_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v11_upper)

print('lower : ',v11_lower)



v4_values = new_df["V4"].values

q25,q75 = np.percentile(v4_values,25),np.percentile(v4_values,75)

iqr=q75-q25

v4_cutoff = iqr*1.5

v4_upper=q75+v4_cutoff

v4_lower=q25-v4_cutoff

new_df=new_df.drop(new_df[(new_df["V4"] > v4_upper) | (new_df["V4"] < v4_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V4_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v4_upper)

print('lower : ',v4_lower)



v2_values = new_df["V2"].values

q25,q75 = np.percentile(v2_values,25),np.percentile(v2_values,75)

iqr=q75-q25

v2_cutoff = iqr*1.5

v2_upper=q75+v2_cutoff

v2_lower=q25-v2_cutoff

new_df=new_df.drop(new_df[(new_df["V2"] > v2_upper) | (new_df["V2"] < v2_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V2_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v2_upper)

print('lower : ',v2_lower)





v19_values = new_df["V19"].values

q25,q75 = np.percentile(v19_values,25),np.percentile(v19_values,75)

iqr=q75-q25

v19_cutoff = iqr*1.5

v19_upper=q75+v19_cutoff

v19_lower=q25-v19_cutoff

new_df=new_df.drop(new_df[(new_df["V19"] > v19_upper) | (new_df["V19"] < v19_lower)].index)

print('Number of rows after outliers are removed : ',new_df.shape[0])

print('V19_values_stats')

print('q25 : ',q25)

print('q75 : ',q75)

print('iqr : ',iqr)

print('upper : ',v19_upper)

print('lower : ',v19_lower)



# After removing the outliers from the training set new X_train and y_train

y_train=new_df["Class"]

X_train=new_df.drop(["Class"],axis=1)
# Now Applying Log reg, svm, KNearesr Neigh on this reduced training sample and testing the model 

# on the orignal imbalanced test set without tuning the parameters for the model  

log_reg_default=LogisticRegression()

log_reg_default.fit(X_train, y_train)

y_pred=log_reg_default.predict(X_test)



print("Logistic Regression before any tuning of parameter : ")

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))



KNeigh = KNeighborsClassifier()

KNeigh.fit(X_train,y_train)

y_pred=KNeigh.predict(X_test)

print("KNearest Neighbours classifier before any tuning of parameter : ")

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))





svc = SVC()

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print("Support Vector classifier before any tuning of parameter : ")

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

# Now using the GridsearchCV for finding the best parameter using cross validation and 

# again testing the model on orignal imbalanced test data

#Logistic regression

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)



log_reg = grid_log_reg.best_estimator_

print(log_reg)



log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')



y_pred=log_reg.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))



# Knearest neigbour



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_

print(knears_neighbors)

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



y_pred=knears_neighbors.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))





# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_

print(svc)

y_pred=svc.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,6,1,)), 

              "min_samples_leaf": list(range(5,20,2))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_

print(tree_clf)

y_pred=tree_clf.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))



#learning curves  of the three models trained on the best parameter obtained.

# However the plots doesn't convey much. But it shows that the models are overfitting as there are high gaps in training and validation scores

# This is due to the reduced sampling to create balanced dataset

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

cv=ShuffleSplit(n_splits=5,random_state=None,test_size=0.25,train_size=None)

print(cv)

f, axes = plt.subplots(ncols=2,figsize=(30,20))

axes[0].set_title("Logistics regression learning curves")

train_size,train_score,cross_val_score = learning_curve(log_reg,X_train,y_train,cv=cv,n_jobs=4,train_sizes=np.linspace(.1, 1.0, 5))

print(train_size)

print(train_score)

print(cross_val_score)

train_scores_mean = np.mean(train_score, axis=1)

train_scores_std = np.std(train_score, axis=1)

cross_val_scores_mean = np.mean(cross_val_score, axis=1)

cross_val_scores_std = np.std(cross_val_score, axis=1)

axes[0].plot(train_size,train_scores_mean)

axes[0].plot(train_size,cross_val_scores_mean)





axes[1].set_title("Support Vector machine learning curves")

train_size,train_score,cross_val_score = learning_curve(svc,X_train,y_train,cv=cv,n_jobs=4,train_sizes=np.linspace(.1, 1.0, 5))

print(train_size)

print(train_score)

print(cross_val_score)

train_scores_mean = np.mean(train_score, axis=1)

train_scores_std = np.std(train_score, axis=1)

cross_val_scores_mean = np.mean(cross_val_score, axis=1)

cross_val_scores_std = np.std(cross_val_score, axis=1)

axes[1].plot(train_size,train_scores_mean)

axes[1].plot(train_size,cross_val_scores_mean)



f, axes = plt.subplots(ncols=2,figsize=(30,20))



axes[0].set_title("KNearest Neighbors classifier learning curves")

train_size,train_score,cross_val_score = learning_curve(knears_neighbors,X_train,y_train,cv=cv,n_jobs=4,train_sizes=np.linspace(.1, 1.0, 5))

print(train_size)

print(train_score)

print(cross_val_score)

train_scores_mean = np.mean(train_score, axis=1)

train_scores_std = np.std(train_score, axis=1)

cross_val_scores_mean = np.mean(cross_val_score, axis=1)

cross_val_scores_std = np.std(cross_val_score, axis=1)

axes[0].plot(train_size,train_scores_mean)

axes[0].plot(train_size,cross_val_scores_mean)



axes[1].set_title("decision tree learning curves")

train_size,train_score,cross_val_score = learning_curve(tree_clf,X_train,y_train,cv=cv,n_jobs=4,train_sizes=np.linspace(.1, 1.0, 5))

print(train_size)

print(train_score)

print(cross_val_score)

train_scores_mean = np.mean(train_score, axis=1)

train_scores_std = np.std(train_score, axis=1)

cross_val_scores_mean = np.mean(cross_val_score, axis=1)

cross_val_scores_std = np.std(cross_val_score, axis=1)

axes[1].plot(train_size,train_scores_mean)

axes[1].plot(train_size,cross_val_scores_mean)

# Now trying different approach(oversampling) by adding multiple entries the rare class data to increase the rare class size and make it near the abundant class size 

# Then training(Logistic Regression) on this oversampled data and testing on the orignal imbalanced test data which is seperated before oversampling

# This is done without tuning parameters

y=df["Class"]

X=df.drop(["Class"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)

new_df=pd.concat([X_train,y_train],axis=1)



fraud_df=new_df[new_df["Class"]==1]

non_fraud_df=new_df[new_df["Class"]==0]

fraud_upsampled_df=fraud_df.sample(frac=int(len(non_fraud_df)*1.0/len(fraud_df)),replace=True,random_state=None)

print(fraud_upsampled_df.shape)

print(non_fraud_df.shape)

new_df=pd.concat([fraud_upsampled_df,non_fraud_df])

new_df=new_df.sample(frac=1)

y_upsampled_train=new_df["Class"]

X_upsampled_train=new_df.drop(["Class"],axis=1)

log_reg=LogisticRegression()

log_reg.fit(X_upsampled_train,y_upsampled_train)

y_pred=log_reg.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

# Now applying same technique but tuning the parameters by KFold cross validation   

# Note - During crossvalidation, splittng of the training data into sub_train and validation is done before doing oversampling on sub_train data

# This is done to prevent overfitting 



y=df["Class"]

X=df.drop(["Class"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)



from sklearn.model_selection import StratifiedKFold



print(X_train.shape[0])

skf= StratifiedKFold(n_splits=5,shuffle=True,random_state=None)

for (penalty,solver) in [('l1','liblinear'),('l2','liblinear'),('l2','lbfgs')]:

    for C in [0.001,0.01,0.1,1]:

        log_reg=LogisticRegression(C=C,penalty=penalty,solver=solver)

        print("Parameters penalty,solver and C is ",penalty,solver,C)

        avg_acc_score=0

        avg_prec_score=0

        avg_recall_score=0

        avg_f1_score=0

        avg_confusion_mat=np.array([[0,0],[0,0]])

        for train_index,cross_val_index in skf.split(X_train,y_train):

            X_cross_val, y_cross_val = X_train.iloc[cross_val_index],y_train.iloc[cross_val_index]

            X_sub_train , y_sub_train = X_train.iloc[train_index] , y_train.iloc[train_index]

            new_df=pd.concat([X_sub_train,y_sub_train],axis=1)

            fraud_df=new_df[new_df["Class"]==1]

            non_fraud_df=new_df[new_df["Class"]==0]

            fraud_upsampled_df=fraud_df.sample(frac=int(len(non_fraud_df)*1.0/len(fraud_df)),replace=True,random_state=None)

            new_df=pd.concat([fraud_upsampled_df,non_fraud_df])

            new_df=new_df.sample(frac=1)

            y_upsampled_train=new_df["Class"]

            X_upsampled_train=new_df.drop(["Class"],axis=1)

            log_reg.fit(X_upsampled_train,y_upsampled_train)

            y_pred=log_reg.predict(X_cross_val)

            

            avg_acc_score=avg_acc_score+accuracy_score(y_cross_val,y_pred)

            avg_prec_score=avg_prec_score+precision_score(y_cross_val,y_pred)

            avg_recall_score=avg_recall_score+recall_score(y_cross_val,y_pred)

            avg_f1_score=avg_f1_score+f1_score(y_cross_val,y_pred)

            

            from sklearn.metrics import confusion_matrix

#             print('Confusion Matrix : \n' + str(confusion_matrix(y_cross_val,y_pred)))

            avg_confusion_mat=np.add(avg_confusion_mat,np.array(confusion_matrix(y_cross_val,y_pred)))

            

        avg_acc_score=avg_acc_score/5

        avg_prec_score=avg_prec_score/5

        avg_recall_score=avg_recall_score/5

        avg_f1_score=avg_f1_score/5

        avg_confusion_mat=avg_confusion_mat/5



        print('Accuracy Score : ' + str(avg_acc_score))

        print('Precision Score : ' + str(avg_prec_score))

        print('Recall Score : ' + str(avg_recall_score))

        print('F1 Score : ' + str(avg_f1_score))

        print('Average confusion matrix : \n'+ str(avg_confusion_mat))



        

        





    
# Best parameters obtained from above after tuning based on F1 score and recall score is - (l2,lbfgs,0.001)

# So taking those parameter and again getting the metrics on the test data(compare it to the results without tuning any parameters for Logistic regression)

y=df["Class"]

X=df.drop(["Class"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)

new_df=pd.concat([X_train,y_train],axis=1)



fraud_df=new_df[new_df["Class"]==1]

non_fraud_df=new_df[new_df["Class"]==0]

fraud_upsampled_df=fraud_df.sample(frac=int(len(non_fraud_df)*1.0/len(fraud_df)),replace=True,random_state=None)

print(fraud_upsampled_df.shape)

print(non_fraud_df.shape)

new_df=pd.concat([fraud_upsampled_df,non_fraud_df])

new_df=new_df.sample(frac=1)

y_upsampled_train=new_df["Class"]

X_upsampled_train=new_df.drop(["Class"],axis=1)



log_reg=LogisticRegression(penalty='l2',solver='lbfgs',C=0.001)

print(log_reg)



log_reg.fit(X_upsampled_train,y_upsampled_train)

y_pred=log_reg.predict(X_test)



print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

# Now trying third technique in which i used MiniBatchKMeans(As KMeans was quite slow due to large size)

# In this technique, instead of randomly choosing data of size equal to rare class size(say r) from abundant class 

# I find r number of clusters of the abundant class and used their centroids along with the rare class as the training data 

# After training on these reduced data, i tested on the original imbalanced test dat which was seperated beforehand

# This is done without tuning the parameters of kmeans like batch_size, n_cluster etc

from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

y=df["Class"]

X=df.drop(["Class"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)

new_df=pd.concat([X_train,y_train],axis=1)

new_df=new_df.sample(frac=1)

fraud_df=new_df[new_df["Class"]==1]

non_fraud_df=new_df[new_df["Class"]==0]

non_fraud_X=non_fraud_df.drop(["Class"],axis=1)

r=len(fraud_df)

print(r)

print(non_fraud_df.shape[0])

import time

start=time.time()

kmeans = MiniBatchKMeans(n_clusters=r,batch_size=100).fit(non_fraud_X)

end=time.time()

print("Time taken : ",end-start)

non_fraud_centroids = kmeans.cluster_centers_

print(non_fraud_centroids.shape)



non_fraud_centroids_df_X=pd.DataFrame(data=non_fraud_centroids,columns=df.drop(["Class"],axis=1).columns)

non_fraud_centroids_df=pd.concat([non_fraud_centroids_df_X,non_fraud_df["Class"][:len(non_fraud_centroids_df_X)].reset_index()],axis=1)

non_fraud_centroids_df=non_fraud_centroids_df.drop(["index"],axis=1)

new_df=pd.concat([fraud_df,non_fraud_centroids_df])



new_df=new_df.sample(frac=1)



y_train=new_df["Class"]

X_train=new_df.drop(["Class"],axis=1)



log_reg_default=LogisticRegression()

log_reg_default.fit(X_train, y_train)

y_pred=log_reg_default.predict(X_test)



print("Performance Metrics without tuning : ")

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))



from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))
