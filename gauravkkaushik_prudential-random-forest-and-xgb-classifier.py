# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing necessary files

#import files

#load package

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#from math import sqrt

import seaborn as sns

import pandas_profiling as pf

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE



from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn import ensemble
# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



#Shape of dataset and sample

print("Shape of Dataset {}".format(X_full.shape))



X_full.head()
#print unique values of target variable

print("Unique values in Target Variable: {}".format(X_full.Response.dtype))

print("Unique values in Target Variable: {}".format(X_full.Response.unique()))

print("Total Number of unique values : {}".format(len(X_full.Response.unique())))



#distribution plot for target classes

sns.countplot(x=X_full.Response).set_title('Distribution of rows by response categories')
#create a funtion to createa  new target variable based on conditions



def new_target(row):

    if (row['Response']<=7) & (row['Response']>=0):

        val=0

    elif (row['Response']==8):

        val=1

    else:

        val=-1

    return val





#create a copy of original dataset

new_data=X_full.copy()



#create a new column

new_data['Final_Response']=new_data.apply(new_target,axis=1)





#print unique values of target variable

print("Unique values in Target Variable: {}".format(new_data.Final_Response.dtype))

print("Unique values in Target Variable: {}".format(new_data.Final_Response.unique()))

print("Total Number of unique values : {}".format(len(new_data.Final_Response.unique())))



#distribution plot for target classes

sns.countplot(x=new_data.Final_Response).set_title('Distribution of rows by response categories')
#drop the actual response column

new_data.drop(axis=1,labels=['Response'],inplace=True)



#rename the "Final_Response" to "Response"

new_data.rename(columns={"Final_Response":"Response"},inplace=True)

#print the nw column names

print("New columns: ", new_data.columns)

X_original=X_full

X_full=new_data
#Are there any columns with missing values?



fig, ax = plt.subplots(figsize=(20,5))  

sns.heatmap(X_full.isnull(), cbar=False)
missing_val_count_by_column = (X_full.isnull().sum()/len(X_full))

print(missing_val_count_by_column[missing_val_count_by_column>0.3].sort_values(ascending=False))
# What are the diffrent datatypes available in datasource



columns_df=pd.DataFrame({'column_names':X_full.columns,'datatypes':X_full.dtypes},index=None)

x=columns_df.groupby(by=['datatypes']).count()

x.reset_index(inplace=True)

x.rename(columns={"column_names":"Number_of_columns"},inplace=True)

lst=[]

for data_type in x.datatypes:

    v=list(X_full.select_dtypes(include=data_type).columns)

    lst.append(v)

    x['Column_Names']=pd.Series(lst)

    



x
#Lets look at only object column : [Product_Info_2]



#check the values in Product_info_ column

print("Total Unique Values: ", len(X_full['Product_Info_2'].unique()))

print("Unique values in 'Product_Info_2':", X_full['Product_Info_2'].unique())
# Exploring Numerical variables

misc_cols=["Ins_Age","Ht","Wt","BMI"]



sns.boxplot(data=X_full[misc_cols])
##Import libraries for classifiers



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier 

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier



from sklearn import ensemble

from sklearn import metrics

from sklearn.metrics import classification_report,recall_score,accuracy_score,precision_score

from sklearn.model_selection import train_test_split
y = X_full.Response

X = X_full.drop(labels=['Response'],axis=1)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=.30,random_state=1)



#create train and test dataset after dropping columns with null values and categorical column



#drop categorical column

X_dropped_train=X_train.drop(axis=1,labels=["Product_Info_2"]).copy()

X_dropped_valid=X_valid.drop(axis=1,labels=["Product_Info_2"]).copy()



#drop columns with any null

X_dropped_train.dropna(axis=1,inplace=True)

X_dropped_valid.dropna(axis=1,inplace=True)



# print shape of dataset

print("Shape of X_train dataset {}".format(X_dropped_train.shape))

print("Shape of X_test dataset {}".format(X_dropped_valid.shape))



print("Shape of y_train dataset {}".format(y_train.shape))

print("Shape of y_valid dataset {}".format(y_valid.shape))
#set seed for same results everytime

seed=0



#declare the models

dt=DecisionTreeClassifier(random_state=seed)

rf=RandomForestClassifier(random_state=seed)

lr=LogisticRegression(random_state=seed)

adb=ensemble.AdaBoostClassifier()

bgc=ensemble.BaggingClassifier()

gbc=ensemble.GradientBoostingClassifier()

xgb=XGBClassifier(random_state=seed)

#sgdc=SGDClassifier(random_state=seed)

svc=SVC(random_state=seed)

#knn=KNeighborsClassifier()

#nb=GaussianNB()



#create a list of models

models=[dt,rf,lr,adb,bgc,gbc,svc,xgb]



def score_model(X_train,y_train,X_valid,y_valid):

    df_columns=[]

    df=pd.DataFrame(columns=df_columns)

    i=0

    #read model one by one

    for model in models:

        model.fit(X_train,y_train)

        y_pred=model.predict(X_valid)

        

        #compute metrics

        train_accuracy=model.score(X_train,y_train)

        test_accuracy=model.score(X_valid,y_valid)

        

        p_score=metrics.precision_score(y_valid,y_pred)

        r_score=metrics.recall_score(y_valid,y_pred)

        f1_score=metrics.f1_score(y_valid,y_pred)

        fp, tp, th = metrics.roc_curve(y_valid, y_pred)

        

        #insert in dataframe

        df.loc[i,"Model_Name"]=model.__class__.__name__

        df.loc[i,"Precision"]=round(p_score,2)

        df.loc[i,"Recall"]=round(r_score,2)

        df.loc[i,"Train_Accuracy"]=round(train_accuracy,2)

        df.loc[i,"Test_Accuracy"]=round(test_accuracy,2)

        df.loc[i,"F1_Score"]=round(f1_score,2)

        df.loc[i,'AUC'] = metrics.auc(fp, tp)

        

        i+=1

    

    #sort values by accuracy

    df.sort_values(by=['F1_Score'],ascending=False,inplace=True)

    return(df)
report_no_null=score_model(X_dropped_train,y_train,X_dropped_valid,y_valid)

report_no_null
sns.scatterplot(data=X_full,x='BMI',y='Wt',hue='Response',alpha=1)
sns.scatterplot(data=X_full,x='BMI',y='Ins_Age',hue='Response',alpha=1)
# Exploring Numerical variables for outliers

misc_cols=["Ins_Age","Ht","Wt","BMI"]



sns.boxplot(data=X_full[misc_cols])
#function to remove outliers

def remove_outlier(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    return df_out





dev=remove_outlier(X_full,'BMI')

dev=remove_outlier(dev,'Wt')

dev=remove_outlier(dev,'Ht')
sns.boxplot(data=dev[misc_cols])
sns.scatterplot(data=dev,x='BMI',y='Wt',hue='Response',alpha=1)
#prepare 3rd dataset

#identifying columns with more than 30% missing values and dropping them



y = dev.Response

X = dev.drop(labels=['Response'],axis=1)





# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=.30,random_state=1)


#dropping the columns with more than 30% missing values

missing_val_count_by_column = (dev.isnull().sum()/len(dev))

print(missing_val_count_by_column[missing_val_count_by_column > 0.3])

cols_to_drop=missing_val_count_by_column[missing_val_count_by_column > 0.3].index.values



# Make copy to avoid changing original data 

out_enc_X_train = X_train.drop(labels=cols_to_drop,axis=1).copy()

out_enc_X_valid = X_valid.drop(labels=cols_to_drop,axis=1).copy()
#identify all cols with medical keywords

medical_keyword_cols=[col for col in out_enc_X_train.columns if str(col).startswith("Medical_Keyword")]



#identify all cols with medical keywords

medical_cols=[col for col in out_enc_X_train.columns if str(col).startswith("Medical_History")]



out_enc_X_train['Total_MedKwrds']=out_enc_X_train[medical_keyword_cols].sum(axis=1)

out_enc_X_train['Total_MedHist']=out_enc_X_train[medical_cols].sum(axis=1)



out_enc_X_valid['Total_MedKwrds']=out_enc_X_valid[medical_keyword_cols].sum(axis=1)

out_enc_X_valid['Total_MedHist']=out_enc_X_valid[medical_cols].sum(axis=1)
#label encoding 

le=LabelEncoder()

out_enc_X_train['Product_Info_2_en'] = le.fit_transform(out_enc_X_train['Product_Info_2'])

out_enc_X_valid['Product_Info_2_en'] = le.fit_transform(out_enc_X_valid['Product_Info_2'])



out_enc_X_train.drop(axis=1,labels=['Product_Info_2'],inplace=True)

out_enc_X_valid.drop(axis=1,labels=['Product_Info_2'],inplace=True)



# imputing missing values

imputer=SimpleImputer()



out_enc_X_train= imputer.fit_transform(out_enc_X_train)

out_enc_X_valid= imputer.transform(out_enc_X_valid)
#Rename the datasets



X_train=out_enc_X_train

y_train=y_train

X_valid=out_enc_X_valid

y_valid=y_valid



#instantiate, fit and make preditions

model=RandomForestClassifier(random_state=seed)

model.fit(X_train,y_train)

y_pred=model.predict(X_valid)



#compute metrics

train_accuracy=model.score(X_train,y_train)

test_accuracy=model.score(X_valid,y_valid)

p_score=metrics.precision_score(y_valid,y_pred)

r_score=metrics.recall_score(y_valid,y_pred)

f1_score=metrics.f1_score(y_valid,y_pred)

fp, tp, th = metrics.roc_curve(y_valid, y_pred)

auc = metrics.auc(fp, tp)
print("Train Accuracy: {}".format(round(train_accuracy,3)))

print("Test Accuracy: {}".format(round(test_accuracy,3)))

print("Precision Score: {}".format(round(p_score,3)))

print("Recall Score: {}".format(round(r_score,3)))

print("F1 Score: {}".format(round(f1_score,3)))

print("AUC: {}".format(round(auc,3)))



print("==============Classification Report=============================")

print(metrics.classification_report(y_valid,y_pred))





print("==============Confusion Matrix=============================")

print(metrics.confusion_matrix(y_valid,y_pred))
# import seaborn as sns

# import matplotlib.pyplot as plt     



# ax= plt.subplot()

# sns.heatmap(metrics.confusion_matrix(y_valid,y_pred),annot=True, ax = ax,); #annot=True to annotate cells



# # labels, title and ticks

# ax.set_xlabel('Predicted labels')

# ax.set_ylabel('True labels')

# ax.set_title('Confusion Matrix')

# ax.xaxis.set_ticklabels(['Declined', 'Approved'])

# ax.yaxis.set_ticklabels(['Declined', 'Approved']);
# Look at parameters used by our current forest

from pprint import pprint

pprint('Parameters currently in use:\n')

pprint(model.get_params())


np.arange(10,120,10)
# Lets create the parameter grid for tunning Random forest



# Number of trees in random forest

n_estimators = np.arange(100,1200,200)

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = np.arange(10,120,10)

# Minimum number of samples required to split a node

min_samples_split = [20,30,50]

# Minimum number of samples required at each leaf node

min_samples_leaf = [10,20,30,40]

# Method of selecting samples for training each tree

bootstrap = [True, False]





# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 50 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf,

                               param_distributions = random_grid,

                               n_iter = 50, cv = 3, 

                               verbose=2,

                               scoring='precision',

                               random_state=42,

                               n_jobs = -1)# Fit the random search model

rf_random.fit(X_train,y_train)
rf_random.best_params_
#instantiate, fit and make preditions

rf_model=RandomForestClassifier(n_estimators=700,min_samples_split=30,min_samples_leaf=10,max_features='sqrt',

                                max_depth=80,bootstrap=False)

rf_model.fit(X_train,y_train)

y_pred=rf_model.predict(X_valid)



#compute metrics

train_accuracy_rf=rf_model.score(X_train,y_train)

test_accuracy_rf=rf_model.score(X_valid,y_valid)

p_score_rf=metrics.precision_score(y_valid,y_pred)

r_score_rf=metrics.recall_score(y_valid,y_pred)

f1_score_rf=metrics.f1_score(y_valid,y_pred)

fp_rf, tp_rf, th_rf = metrics.roc_curve(y_valid, y_pred)

auc_rf = metrics.auc(fp_rf, tp_rf)





print("Train Accuracy: {}".format(round(train_accuracy_rf,3)))

print("Test Accuracy: {}".format(round(test_accuracy_rf,3)))

print("Precision Score: {}".format(round(p_score_rf,3)))

print("Recall Score: {}".format(round(r_score_rf,3)))

print("F1 Score: {}".format(round(f1_score_rf,3)))

print("AUC: {}".format(round(auc_rf,3)))



print("==============Classification Report=============================")

print(metrics.classification_report(y_valid,y_pred))





print("==============Confusion Matrix=============================")

print(metrics.confusion_matrix(y_valid,y_pred))
#evaluate models



def evaluate(model,X_train,y_train,X_valid,y_valid):

    y_pred=model.predict(X_valid)

    #compute metrics

    train_accuracy_rf=model.score(X_train,y_train)

    test_accuracy_rf=model.score(X_valid,y_valid)

    p_score_rf=metrics.precision_score(y_valid,y_pred)

    r_score_rf=metrics.recall_score(y_valid,y_pred)

    f1_score_rf=metrics.f1_score(y_valid,y_pred)

    fp_rf, tp_rf, th_rf = metrics.roc_curve(y_valid, y_pred)

    auc_rf = metrics.auc(fp_rf, tp_rf)

    

    print("Train Accuracy: {}".format(round(train_accuracy_rf,3)))

    print("Test Accuracy: {}".format(round(test_accuracy_rf,3)))

    print("Precision Score: {}".format(round(p_score_rf,3)))

    print("Recall Score: {}".format(round(r_score_rf,3)))

    print("F1 Score: {}".format(round(f1_score_rf,3)))

    print("AUC: {}".format(round(auc_rf,3)))

    

    print("==============Classification Report=============================")

    print(metrics.classification_report(y_valid,y_pred))

    

    print("==============Confusion Matrix=============================")

    print(metrics.confusion_matrix(y_valid,y_pred))

    

    return (r_score_rf,p_score,f1_score,auc,train_accuracy_rf,test_accuracy_rf)
print("\n ==========================Base Model==========================")

model.fit(X_train,y_train)

base_recall,base_precison,base_f1,base_auc,base_train_accuracy,base_test_accuracy= evaluate(model,X_train,y_train,X_valid,y_valid)



print("\n ==========================Tuned Model==========================")

best_random = rf_random.best_estimator_

randomcv_recall,randomcv_precison,randomcv_f1,randomcv_auc,randomcv_train_accuracy,randomcv_test_accuracy = evaluate(best_random, X_train,y_train, X_valid,y_valid)





print('RandomSearchCV Improvement in Recall of {:0.2f}%.'.format( 100 * (randomcv_recall - base_recall) / base_recall))

print('RandomSearchCV Improvement in Precision of {:0.2f}%.'.format( 100 * (randomcv_precison - base_precison) / base_precison))

print('RandomSearchCV Improvement in F1 Score of {:0.2f}%.'.format( 100 * (randomcv_f1 - base_f1) / base_f1))

print('RandomSearchCV Improvement in AUC of {:0.2f}%.'.format( 100 * (randomcv_auc - base_auc) / base_auc))

print('RandomSearchCV Improvement in Train Accuracy of {:0.2f}%.'.format( 100 * (randomcv_train_accuracy - base_train_accuracy) / base_train_accuracy))

print('RandomSearchCV Improvement in Test Accuracy of {:0.2f}%.'.format( 100 * (randomcv_test_accuracy - base_test_accuracy) / base_test_accuracy))
names=list(X.drop(labels=cols_to_drop,axis=1).columns.values)

names.append('Total_MedKwrds')

names.append('Total_MedHist')



data={'Feature_Name':names,

      'Feature_Importance': rf_model.feature_importances_

     }



feature_df=pd.DataFrame(data)



feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)



fig, ax = plt.subplots(figsize=(15,25))

sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance',)
fpr,tpr,thresholds=metrics.roc_curve(y_valid,y_pred)



def plot_roc_curve(fpr,tpr,label=None):

    plt.plot(fpr,tpr,linewidth=2,label=label)

    plt.plot([0,1],[0,1],'k--')

    plt.axis([0,1,0,1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    

plot_roc_curve(fpr,tpr)

plt.show()
# plot precision vs recall



from sklearn.model_selection import cross_val_predict



y_scores=cross_val_predict(rf_model,X_train,y_train,cv=3)



precisions,recalls,thresholds=metrics.precision_recall_curve(y_train,y_scores)



def plot_precision_recall_vs_threshold(precisions,recalls,threshold):

    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")

    plt.plot(thresholds,recalls[:-1],"g--",label="Recalls")

    plt.xlabel("Threshold")

    plt.legend(loc="upper left")

    plt.ylim([0,1])

    

plot_precision_recall_vs_threshold(precisions,recalls,thresholds)

plt.show()
estimator=rf_model.estimators_[5]



from sklearn.tree import export_graphviz



export_graphviz(estimator, out_file='tree.dot', 

                feature_names =names,

                class_names = 'Response',

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
# Lets create the parameter grid for tunning Random forest



# Create the parameter grid based on the results of random search 

param_grid = { 'n_estimators': [300,500,700,900,1000],

               'max_features': ['sqrt'],

               'max_depth': [80, 100, 110],

               'min_samples_split': [20,30],

               'min_samples_leaf': [10,20,30],

               'bootstrap': ['False']

}# Create a based model

rf = RandomForestClassifier()# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,scoring='precision', cv = 3, n_jobs = -1, verbose = 2)



grid_search.fit(X_train, y_train)

grid_search.best_params_
print("\n ==========================Tuned Model==========================")

best_grid = grid_search.best_params_

gridcv_recall,gridcv_precison,gridcv_f1,gridcv_auc,gridcv_train_accuracy,gridcv_test_accuracy = evaluate(best_random, X_train,y_train, X_valid,y_valid)





print('GridSearchCV Improvement in Recall of {:0.2f}%.'.format( 100 * (gridcv_recall - base_recall) / base_recall))

print('GridSearchCV in Precision of {:0.2f}%.'.format( 100 * (gridcv_precison - base_precison) / base_precison))

print('GridSearchCV in F1 Score of {:0.2f}%.'.format( 100 * (gridcv_f1 - base_f1) / base_f1))

print('GridSearchCV in AUC of {:0.2f}%.'.format( 100 * (gridcv_auc - base_auc) / base_auc))

print('GridSearchCV in Train Accuracy of {:0.2f}%.'.format( 100 * (gridcv_train_accuracy - base_train_accuracy) / base_train_accuracy))

print('GridSearchCV in Test Accuracy of {:0.2f}%.'.format( 100 * (gridcv_test_accuracy - base_test_accuracy) / base_test_accuracy))