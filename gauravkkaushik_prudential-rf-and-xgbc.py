# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing necessary files

#import files



#load package

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



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



import pandas_profiling as pp
# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



#Shape of dataset and sample

print("Shape of Dataset {}".format(X_full.shape))



#print few rows

X_full.head(10)
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
# Compute the correlation matrix

sns.set(style="white")

corr = X_full.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(60, 40))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#sns.pairplot(X_full,hue='Response',diag_kind='kde')
#pp.ProfileReport(X_full)
# Exploring Numerical variables

misc_cols=["Ins_Age","Ht","Wt","BMI"]



sns.boxplot(data=X_full[misc_cols])
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

sgdc=SGDClassifier(random_state=seed)

svc=SVC(random_state=seed)

knn=KNeighborsClassifier()

nb=GaussianNB()



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
#shuffle the dataset



shuffled_df=X_full.sample(frac=1,random_state=1)



#put all the approved applications in another datset(i.e. 19489)

approved_df=shuffled_df[shuffled_df.Response==1]



#randomly select same number of unapproved applications as approved

non_approved_df=shuffled_df[shuffled_df.Response==0].sample(n=len(approved_df),random_state=1)



#concatenate both dataframes

normalized_df=pd.concat([approved_df,non_approved_df])



#plot the dataset after sampling



plt.figure(figsize=(8,8))

sns.countplot(data=normalized_df,x='Response')

plt.title("Balanced Class after - Random undersampling")

plt.show()





#Split the normalized_dataset into train test.



norm_y = normalized_df.Response

norm_X = normalized_df.drop(labels=['Response'],axis=1)



# Break off validation set from training data

norm_X_train,norm_X_valid, norm_y_train, norm_y_valid = train_test_split(norm_X,norm_y,test_size=.20,random_state=1)



#create train and test dataset after dropping columns with null values and categorical column



#drop categorical column

norm_X_dropped_train=norm_X_train.drop(axis=1,labels=["Product_Info_2"]).copy()

norm_X_dropped_valid=X_valid.drop(axis=1,labels=["Product_Info_2"]).copy()



#drop columns with any null

norm_X_dropped_train.dropna(axis=1,inplace=True)

norm_X_dropped_valid.dropna(axis=1,inplace=True)



# print shape of dataset

print("Shape of X_train dataset {}".format(norm_X_dropped_train.shape))

print("Shape of X_test dataset {}".format(norm_X_dropped_valid.shape))



print("Shape of y_train dataset {}".format(norm_y_train.shape))

print("Shape of y_valid dataset {}".format(y_valid.shape))
# Undersampled Dataset report

undersample_report=score_model(norm_X_dropped_train,

                               norm_y_train,

                               norm_X_dropped_valid,

                               y_valid)



undersample_report
from imblearn.over_sampling import SMOTE





#take original X_train,y_train, X_valid and y_valid and then perform set of operation on X_train 



sm=SMOTE(sampling_strategy='minority',random_state=1)



over_X_train=X_train.drop(axis=1,labels=["Product_Info_2"]).copy()

over_X_valid=X_valid.drop(axis=1,labels=["Product_Info_2"]).copy()



#drop columns with any null

over_X_train.dropna(axis=1,inplace=True)

over_X_valid.dropna(axis=1,inplace=True)



names=over_X_train.columns



#fit the model and generate the dataset

oversample_X_train,oversample_y_train=sm.fit_sample(over_X_train,y_train)



oversample_X_train=pd.DataFrame(oversample_X_train,columns=names)

oversample_X_train.head()



# print shape of dataset

print("Shape of X_train dataset {}".format(oversample_X_train.shape))

print("Shape of X_test dataset {}".format(over_X_valid.shape))



print("Shape of y_train dataset {}".format(oversample_y_train.shape))

print("Shape of y_valid dataset {}".format(y_valid.shape))

# Undersampled Dataset report

oversample_report=score_model(oversample_X_train,

                               oversample_y_train,

                               over_X_valid,

                               y_valid)



oversample_report
print("=========================================No_null_random_sampling====================================")

report_no_null.sort_values(by=['F1_Score'],ascending=False,inplace=True)

report_no_null
print("=========================================random_undersampling====================================")

undersample_report.sort_values(by=['F1_Score'],ascending=False,inplace=True)

undersample_report
print("=========================================SMOTE_upsampling====================================")

oversample_report.sort_values(by=['F1_Score'],ascending=False,inplace=True)

oversample_report
# sns.scatterplot(data=X_full,x='BMI',y='Wt',hue='Response',alpha=1)

# normalized_df

sns.scatterplot(data=X_full,x='BMI',y='Wt',hue='Response',alpha=1)
misc_cols=["Ins_Age","Ht","Wt","BMI"]



sns.boxplot(data=X_full[misc_cols])
def remove_outlier(df, col_names):

    df_in=df

    for col_name in col_names:

        q1 = df_in[col_name].quantile(0.25)

        q3 = df_in[col_name].quantile(0.75)

        iqr = q3-q1 #Interquartile range

        fence_low  = q1-1.5*iqr

        fence_high = q3+1.5*iqr

        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

        df_in=df_out

    return df_in
dev=remove_outlier(X_full,['BMI','Wt','Ht'])

sns.boxplot(data=dev[misc_cols])
sns.scatterplot(data=dev,x='BMI',y='Wt',hue='Response',alpha=1)
from imblearn.over_sampling import SMOTE



#use actual  train test split and perform below steps:

#Label encoding

OS_X_train=X_train.copy()

OS_X_valid=X_valid.copy()

OS_y_train=y_train.copy()

OS_y_valid=y_valid.copy()





# Step 1: Drop columns with more than 30% missing values



missing_val_count_by_column = (OS_X_train.isnull().sum()/len(OS_X_train))

print(missing_val_count_by_column[missing_val_count_by_column > 0.3])

cols_to_drop=missing_val_count_by_column[missing_val_count_by_column > 0.3].index.values

print("\n\nTotal columns to be dropped: {}".format(len(cols_to_drop)),"\n\nAnd they are: {}".format(cols_to_drop))



# drop the identified columns from train and test dataset

OS_X_train.drop(labels=cols_to_drop,axis=1,inplace=True)

OS_X_valid.drop(labels=cols_to_drop,axis=1,inplace=True)



# Step 2: For less than 30% missing columns perform most frequent missing value treatment

names=OS_X_train.columns.values

imputer=SimpleImputer(strategy="most_frequent")



OS_X_train_clean=pd.DataFrame(data=imputer.fit_transform(OS_X_train),columns=names)

OS_X_valid_clean=pd.DataFrame(data=imputer.fit_transform(OS_X_valid),columns=names)



#Step 3: Create features



#identify all cols with medical keywords

medical_keyword_cols=[col for col in OS_X_train_clean.columns if str(col).startswith("Medical_Keyword")]



#identify all cols with medical history

medical_cols=[col for col in OS_X_train_clean.columns if str(col).startswith("Medical_History")]



OS_X_train_clean['Total_MedKwrds']=OS_X_train_clean[medical_keyword_cols].sum(axis=1)

OS_X_valid_clean['Total_MedKwrds']=OS_X_train_clean[medical_keyword_cols].sum(axis=1)



# Step 4: Create label encoding

le=LabelEncoder()

OS_X_train_clean['Product_Info_2_en'] = le.fit_transform(OS_X_train_clean['Product_Info_2'])

OS_X_valid_clean['Product_Info_2_en'] = le.transform(OS_X_valid_clean['Product_Info_2'])



OS_X_train_clean.drop(axis=1,labels=['Product_Info_2'],inplace=True)

OS_X_valid_clean.drop(axis=1,labels=['Product_Info_2'],inplace=True)





#Step 5: Perform SMOTE



names=OS_X_train_clean.columns

sm=SMOTE(sampling_strategy='minority',random_state=1)

#fit the model and generate the dataset

OS_Final_X_train,OS_Final_y_train=sm.fit_sample(OS_X_train_clean,OS_y_train)


X_train=OS_Final_X_train

y_train=OS_Final_y_train

X_valid=OS_X_valid_clean

y_valid=OS_y_valid
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

    

    return (r_score_rf,p_score_rf,f1_score_rf,auc_rf,train_accuracy_rf,test_accuracy_rf)
rf=RandomForestClassifier(random_state=1)

rf.fit(X_train,y_train)
evaluate(rf,X_train,y_train,X_valid,y_valid)
rf
# Lets create the parameter grid for tunning Random forest

from pprint import pprint

# Number of trees in random forest

n_estimators = np.arange(20,150,20)

# Maximum number of levels in tree

max_depth = np.arange(5,20,3)

# Minimum number of samples required to split a node

min_samples_split = [20,30,50]

# Minimum number of samples required at each leaf node

min_samples_leaf = [40,50]

# Method of selecting samples for training each tree

bootstrap = [True]





# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': [True]

              }



pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()



# Random search of parameters, using 3 fold cross validation, 

# search across 50 different combinations, and use all available cores

rf_random = GridSearchCV(estimator = rf,

                         param_grid=random_grid,

                         #n_iter = 50,

                         cv = 3,

                         verbose=2,

                         scoring='precision',

                         #random_state=42,

                         n_jobs = -1)# Fit the random search model



rf_random.fit(X_train,y_train)
rf_random.best_estimator_
evaluate(rf_random.best_estimator_, X_train,y_train, X_valid,y_valid)
print("\n ==========================Base Model==========================")

model=RandomForestClassifier(random_state=1)

model.fit(X_train,y_train)

base_recall,base_precison,base_f1,base_auc,base_train_accuracy,base_test_accuracy= evaluate(model,X_train,y_train,X_valid,y_valid)





print("\n ==========================Tuned Model==========================")

best_random = rf_random.best_estimator_

randomcv_recall,randomcv_precison,randomcv_f1,randomcv_auc,randomcv_train_accuracy,randomcv_test_accuracy = evaluate(best_random, X_train,y_train, X_valid,y_valid)





print('Gridsearch Improvement in Recall of {:0.2f}%.'.format( 100 * (randomcv_recall - base_recall) / base_recall))

print('Gridsearch Improvement in Precision of {:0.2f}%.'.format( 100 * (randomcv_precison - base_precison) / base_precison))

print('Gridsearch Improvement in F1 Score of {:0.2f}%.'.format( 100 * (randomcv_f1 - base_f1) / base_f1))

print('Gridsearch Improvement in AUC of {:0.2f}%.'.format( 100 * (randomcv_auc - base_auc) / base_auc))

print('Gridsearch Improvement in Train Accuracy of {:0.2f}%.'.format( 100 * (randomcv_train_accuracy - base_train_accuracy) / base_train_accuracy))

print('Gridsearch Improvement in Test Accuracy of {:0.2f}%.'.format( 100 * (randomcv_test_accuracy - base_test_accuracy) / base_test_accuracy))
from sklearn.metrics import precision_recall_curve



# getting the probabilities of our predictions

random_forest=rf_random.best_estimator_

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
def plot_precision_vs_recall(precision, recall):

    plt.plot(recall, precision, "g--", linewidth=2.5)

    plt.ylabel("recall", fontsize=19)

    plt.xlabel("precision", fontsize=19)

    plt.axis([0, 1.5, 0, 1.5])



plt.figure(figsize=(14, 7))

plot_precision_vs_recall(precision, recall)

plt.show()
from sklearn.metrics import roc_curve

# compute true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)



# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
y_train_probabilities = random_forest.predict_proba(X_train)

y_test_probabilities = random_forest.predict_proba(X_train)

# y_test_probabilities has shape = [n_samples, n_classes]



y_train_predictions_high_precision = y_train_probabilities[:,1] > 0.55

y_test_predictions_high_precision = y_test_probabilities[:,1] > 0.55



#y_test_predictions_high_recall = y_test_probabilities[:,1] > 0.1



high_P_X_train=X_train[y_train_predictions_high_precision==True]

high_P_y_train=y_train[y_train_predictions_high_precision==True]



high_P_X_valid=X_train[y_train_predictions_high_precision==True]

high_P_y_valid=y_train[y_test_predictions_high_precision==True]
random_forest_high_p=random_forest.fit(high_P_X_train,high_P_y_train)

evaluate(random_forest_high_p,high_P_X_train,high_P_y_train,high_P_X_valid,high_P_y_valid)
print("\n ==========================Base Model==========================")

model=RandomForestClassifier(random_state=1)

model.fit(X_train,y_train)

base_recall,base_precison,base_f1,base_auc,base_train_accuracy,base_test_accuracy= evaluate(model,X_train,y_train,X_valid,y_valid)





print("\n ==========================Tuned Model==========================")

high_prcn_recall,high_prcn_precison,high_prcn_f1,high_prcn_auc,high_prcn_train_accuracy,high_prcn_test_accuracy = evaluate(random_forest_high_p,X_train,y_train, X_valid,y_valid)





print('Improvement in Recall @ threshold of 0.55{:0.2f} is %.'.format( 100 * (high_prcn_recall - base_recall) / base_recall))

print('Gridsearch Improvement in Precision @ threshold of 0.55: {:0.2f} %.'.format( 100 * (high_prcn_precison - base_precison) / base_precison))

print('Gridsearch Improvement in F1 Score @ threshold of 0.55: {:0.2f}%.'.format( 100 * (high_prcn_f1 - base_f1) / base_f1))

print('Gridsearch Improvement in AUC of @ threshold of 0.55: {:0.2f}%.'.format( 100 * (high_prcn_auc - base_auc) / base_auc))

print('Gridsearch Improvement in Train @ threshold of 0.55: {:0.2f}%.'.format( 100 * (high_prcn_train_accuracy - base_train_accuracy) / base_train_accuracy))

print('Gridsearch Improvement in Test @ threshold of 0.55: {:0.2f}%.'.format( 100 * (high_prcn_test_accuracy - base_test_accuracy) / base_test_accuracy))
# Train

# importances = rf_random.best_estimator_.feature_importances_

importances = random_forest_high_p.feature_importances_



data={'Feature_Name':names,

      'Feature_Importance': importances

     }



feature_df=pd.DataFrame(data)



feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)



fig, ax = plt.subplots(figsize=(15,25))

sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance',)
estimator=random_forest_high_p

# estimator.fit(X_train,y_train)

rf_estimator=estimator.estimators_[5]



from sklearn.tree import export_graphviz



export_graphviz(rf_estimator, out_file='tree.dot', 

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
xgbc_model= XGBClassifier()

xgbc_model.fit(X_train,y_train)

xgbc_model
#evalutate the base model

xgbc_base_recall,xgbc_base_precison,xgbc_base_f1,xgbc_base_auc,xgbc_base_train_accuracy,xgbc_base_test_accuracy= evaluate(xgbc_model,X_train,y_train,np.array(X_valid),np.array(y_valid))
#list of parameters that can be tuned in xgbc model

pprint(xgbc_model.get_params())
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

#               colsample_bynode=1, colsample_bytree=1, gamma=0,

#               learning_rate=0.1, max_delta_step=0, max_depth=3,

#               min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

#               nthread=None, objective='binary:logistic', random_state=0,

#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

#               silent=None, subsample=1, verbosity=1)





# ### Parameters to tune based on above results:

# 1. n_estimators: max_depth is limited to 3. As we will control the depth , we could increase the the number of estimators to see any enhancment

# 2. max_depth: Currently 3, we can try to increase it

# 3. colsample_bytree:  currently tree will be build till there 1 value in leaf.Condition for overfitting

# 4. min_child_weight: Instead of building RF on whole dataset, we will take random sets and train indidvidual trees

# 4. Min_samples_split: To avoid overfitting we will only split if certain number of records are present in node

    

# Lets create the parameter grid for tunning XGBClassifier



xgbc_grid = {'n_estimators': np.arange(110,200,40),

              'learning_rate': np.arange(0.01, 0.1,0.03),

              'subsample': [0.6],

              'max_depth': np.arange(4,10,3),

              'colsample_bytree': np.arange(0.3, 0.6,0.3),

              'min_child_weight': [1, 2, 3],

             'n_jobs':[-1]

             }

pprint(xgbc_grid)

random_xgbc=GridSearchCV(estimator = XGBClassifier(),

                         param_grid=xgbc_grid,

                         #n_iter = 50,

                         cv = 2,

                         verbose=2,

                         scoring='precision',

                         #random_state=42,

                         n_jobs = -1)# Fit the random search model



random_xgbc.fit(X_train, y_train)
random_xgbc.best_estimator_
print("\n ==========================Base Model==========================")

#model=RandomForestClassifier(random_state=1)

xgbc_model.fit(X_train,y_train)

xgbc_base_recall,xgbc_base_precison,xgbc_base_f1,xgbc_base_auc,xgbc_base_train_accuracy,xgbc_base_test_accuracy= evaluate(xgbc_model,X_train,y_train,np.array(X_valid),np.array(y_valid))





print("\n ==========================Tuned Model==========================")

best_random_xgbc = random_xgbc.best_estimator_

xgbc_randomcv_recall,xgbc_randomcv_precison,xgbc_randomcv_f1,xgbc_randomcv_auc,xgbc_randomcv_train_accuracy,xgbc_randomcv_test_accuracy = evaluate(best_random_xgbc,X_train,y_train,np.array(X_valid),np.array(y_valid))

                                                                                                                                                   

print('RandomSearchCV Improvement for XGBC in Recall of {:0.2f}%.'.format( 100 * (xgbc_randomcv_recall - xgbc_base_recall) / xgbc_base_recall))

print('RandomSearchCV Improvement for XGBC in Precision of {:0.2f}%.'.format( 100 * (xgbc_randomcv_precison - xgbc_base_precison) / xgbc_base_precison))

print('RandomSearchCV Improvement for XGBC in F1 Score of {:0.2f}%.'.format( 100 * (xgbc_randomcv_f1 - xgbc_base_f1) / xgbc_base_f1))

print('RandomSearchCV Improvement in for XGBC AUC of {:0.2f}%.'.format( 100 * (xgbc_randomcv_auc - xgbc_base_auc) / xgbc_base_auc))

print('RandomSearchCV Improvement in for XGBC Train Accuracy of {:0.2f}%.'.format( 100 * (xgbc_randomcv_train_accuracy - xgbc_base_train_accuracy) / xgbc_base_train_accuracy))

print('RandomSearchCV Improvement in for XGBC Test Accuracy of {:0.2f}%.'.format( 100 * (xgbc_randomcv_test_accuracy - xgbc_base_test_accuracy) / xgbc_base_test_accuracy))
# Train

importances = random_xgbc.best_estimator_.feature_importances_



data={'Feature_Name':names,

      'Feature_Importance': importances

     }



feature_df=pd.DataFrame(data)



feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)



fig, ax = plt.subplots(figsize=(15,25))

sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance',)