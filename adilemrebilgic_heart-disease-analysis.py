# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:,.3f}'.format



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def Numeric_data_eda(feature_name,feature_name_long,target):

    '''

    This function can be used to get general information about a feature and its relationship with a target variable

    The feature must be numeric and the target must be  binary.

    The relationship is only visualized, no statistical test are done to examine the relationship.

    '''

    #print the descriptive statistics

    print('Descriptive Statistics of {} \n'.format(feature_name_long))

    print(data[feature_name].describe())

    

    #print the descriptive statistics by target

    print('Descriptive Statistics of {} by target\n'.format(feature_name_long))

    print(data.groupby(target)[feature_name].describe())



    #Visualize 

    fig=plt.figure(figsize=(21,6))

    ### first plot-distribution plot

    ax1=fig.add_subplot(1,1,1)

    sns.distplot(data[feature_name],kde=False,ax=ax1)

    plt.xlabel(feature_name_long)

    plt.title(str(feature_name_long+' Distribution'))



    #Visualize the relationship

    ##create a filter for histogram

    filter_target=data[target]==0

    ##create a figure

    fig=plt.figure(figsize=(21,6))

    ### first plot

    ax1=fig.add_subplot(1,2,1)

    filter_target=data[target]==0

    sns.distplot(data[filter_target][feature_name],color='b',label='Healthy',ax=ax1,kde=False)

    sns.distplot(data[~filter_target][feature_name],color='r',label='Unhealthy',ax=ax1,kde=False)

    plt.title('{} Distribution by Target'.format(feature_name_long))

    plt.xlabel(feature_name_long)

    plt.legend()

    ### second plot

    ax2=fig.add_subplot(1,2,2)

    sns.violinplot(x=target,y=feature_name,data=data,ax=ax2)

    plt.title('{} Distribution by Target'.format(feature_name_long))

    plt.xlabel(feature_name_long)

    

def categoric_data_eda(feature_name,feature_name_long,target):

    '''

    This function can be used to get general information about a feature and its relationship with a target variable

    The feature must be categoric and the target must be  binary.

    The relationship is only visualized, no statistical test are done to examine the relationship.

    '''

    #print the number of patients by feature name

    print(feature_name_long)

    print(data[feature_name].value_counts())

    

    #Plot the data and visualize the relationship between target and feature 

    ##Create a figure

    fig=plt.figure(figsize=(18,6))

    ## Add a subplot for feature distribution

    ax1=fig.add_subplot(1,2,1)

    sns.countplot(x=feature_name,data=data,ax=ax1)

    plt.title('{} Distribution'.format(feature_name_long))

    plt.xlabel(feature_name_long)

    

    ## Add another subplot for target ratio and feature 

    ### Create a temp table for visualization

    temp_data=data.groupby(feature_name,as_index=False).agg({'target':['sum','count']})

    temp_data['ratio']=temp_data['target']['sum']/temp_data['target']['count']



    ax2=fig.add_subplot(1,2,2)

    sns.barplot(x=feature_name,y='ratio',data=temp_data,ax=ax2) 

    plt.ylim((0,1))

    plt.title('Heart Disease Ratio by {}'.format(feature_name_long))

    plt.xlabel(feature_name_long)

    plt.ylabel('Heart Disease Ratio')

    

def dataFrameTrim(dt,col,perc=0.75):

    '''

    This function replaces values in column in dataframe with perc. 

    '''

    upper_limit=dt[col].quantile(q=perc)

    dt[col][dt[col]>upper_limit]=upper_limit

    data_frame_standardize(dt=dt,col=col)



def data_frame_standardize(dt,col):

    '''

    This function standardizes a column in a dataframe

    '''

    from sklearn.preprocessing import StandardScaler

    data_Standardize=dt[col].values.reshape(-1,1)

    dt[col]=StandardScaler(copy=False).fit_transform(data_Standardize)

    

def impute_outliers(df,col_name,upper_limit,lower_limit):

    '''

    This function replace the values, which are outside of the lower and upper limit boundiries, with the lower and upper limit values.

    The values between lower and upper limit boundiries stay the same.

    '''

    df[col_name][df[col_name]>upper_lmt]=upper_lmt

    df[col_name][lower_lmt>df[col_name]]=lower_lmt
data=pd.read_csv('../input/heart.csv')
print('Number of observations is {}.'.format(data.shape[0]))

print('Number of features is {}.'.format(data.shape[1]-1)) #one of them is target variable
data.head()
data.dtypes
print('Number of features with missing values: {}'.format(data.isnull().any().sum()))
#Age

Numeric_data_eda(feature_name='age',feature_name_long='Age',target='target')

# looks like older people gets less heart disease
#extreme values

upper_lmt=data['age'].mean()+data['age'].std()*2

lower_lmt=data['age'].mean()-data['age'].std()*2

outlierfilter=np.logical_or(data['age']>upper_lmt,(lower_lmt>data['age']))



fig=plt.figure(figsize=(15,10))

ax1=fig.add_subplot(1,2,1)



plt.plot(data['age'][outlierfilter],

         linestyle='', 

         marker='o',

         color='r',

         alpha=0.8)

plt.plot(data['age'][~outlierfilter],

         linestyle='', 

         marker='o',

         color='b',

         alpha=0.1)

plt.title('Age Distribution-Before Outlier Imputation')

plt.ylabel('Age')

plt.xlabel('Patients')

plt.axhline(upper_lmt, color='r', linestyle='--')

plt.axhline(lower_lmt, color='r', linestyle='--')

plt.axhline(data['age'].mean(), color='b', linestyle='--')



#impute outliers   

print(data['age'].describe())

impute_outliers(df=data,col_name='age',upper_limit=upper_lmt,lower_limit=lower_lmt)

print(data['age'].describe())



ax2=fig.add_subplot(1,2,2)

plt.plot(data['age'][outlierfilter],

         linestyle='', 

         marker='o',

         color='r',

         alpha=0.8)

plt.plot(data['age'][~outlierfilter],

         linestyle='', 

         marker='o',

         color='b',

         alpha=0.1)

plt.title('Age Distribution-After Outlier Imputation')

plt.ylabel('Age')

plt.xlabel('Patients')

plt.axhline(upper_lmt, color='r', linestyle='--')

plt.axhline(lower_lmt, color='r', linestyle='--')

plt.axhline(data['age'].mean(), color='b', linestyle='--')
#The Resting Blood Pressure

Numeric_data_eda(feature_name='trestbps',feature_name_long='Resting Blood Pressure',target='target')

# looks like there is no relationship between the resting blood pressure and heart disease
#Cholestoral

Numeric_data_eda(feature_name='chol',feature_name_long='Cholestoral',target='target')

#looks like there is no relationship between the Cholestoral and the heart disease.

#there are some observations with heartdisease and high cholestoral values
#thalach-maximum heart rate achieved

Numeric_data_eda(feature_name='thalach',feature_name_long='Maximum Heart Rate Achieved',target='target')

#Looks like there is a relationship between Maximum Heart Rate Achieved and heart disease.

#Patients with higher Maximum Heart Rate Achieved are higher rate of heart disease.
#oldpeak-ST depression induced by exercise relative to rest

Numeric_data_eda(feature_name='oldpeak',feature_name_long='ST depression induced by exercise relative to rest',target='target')

#Looks like there is a relationship between oldpeak and heart disease.

#Patient with lower oldpeak values have higher portion of heart disease.!!
#sex-(1 = male; 0 = female)

categoric_data_eda(feature_name='sex',feature_name_long='Gender',target='target')
#cp-chest pain type

categoric_data_eda(feature_name='cp',feature_name_long='Chest Pain Type',target='target')
#fbs-(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

categoric_data_eda(feature_name='fbs',feature_name_long='Fasting Blood Sugar Flag',target='target')
#restecg-resting electrocardiographic results

categoric_data_eda(feature_name='restecg',feature_name_long='Resting Electrocardiographic Results',target='target')
#exang-exercise induced angina (1 = yes; 0 = no)

categoric_data_eda(feature_name='exang',feature_name_long='Exercise Induced Angina ',target='target')
#slope-the slope of the peak exercise ST segment

categoric_data_eda(feature_name='slope',feature_name_long='The Slope of The Peak Exercise ST Segment',target='target')
#ca-number of major vessels (0-3) colored by flourosopy

categoric_data_eda(feature_name='ca',feature_name_long='Number of Major Vessels (0-3) Colored by Flourosopy',target='target')
#thal-3 = normal; 6 = fixed defect; 7 = reversable defect

categoric_data_eda(feature_name='thal',feature_name_long='Thal',target='target')
data.columns
sns.pairplot(data=data,vars=[ 'age', 'trestbps',  'chol'],hue='target');
sns.pairplot(data=data,vars=['thalach', 'oldpeak',  'slope',  'ca'],hue='target');
#trim and standardize the numeric values

num_features=['age', 'trestbps',  'chol','thalach', 'oldpeak']

for feature in num_features: 

    dataFrameTrim(dt=data,col=feature,perc=0.9)
#create dummy variables

data=pd.get_dummies(data=data,columns=['cp','restecg','slope','ca','thal'])
#create a decision tree model

## split data as x and y

X=data.drop(columns=['target']).values

y=data['target'].values

Feature_Names=data.drop(columns=['target']).columns.values

##decision tree

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score,precision_score,accuracy_score

dt1_cv_scores=[]

dt1_precision=[]

dt1_recall=[]

dt1_accuracy=[]

for depth in range(2,20):

    dt_1=DecisionTreeClassifier(max_depth=depth)#min_samples_split =0.05,min_samples_leaf=0.02)

    dt_1_cv=cross_val_score(estimator=dt_1,X=X,y=y,cv=5)

    dt1_cv_scores.append(dt_1_cv)

    dt1_pred=dt_1.fit(X=X,y=y).predict(X=X)

    dt1_precision.append(precision_score(y_pred=dt1_pred,y_true=y))

    dt1_recall.append(recall_score(y_pred=dt1_pred,y_true=y))

    dt1_accuracy.append(accuracy_score(y_pred=dt1_pred,y_true=y))

#plot the results

fig=plt.figure(figsize=(30,6))

ax1=fig.add_subplot(1,2,1)

sns.boxplot(x=np.arange(2,20),y=dt1_cv_scores,ax=ax1)

plt.title('Decision Tree Cross Validation Accuracy Results with Different Depth Parameters')

plt.ylabel('Accuracy')

plt.ylim(top=0.9,bottom=0.6)

plt.xlabel('Decision Tree Depth Parameter')



ax2=fig.add_subplot(1,2,2)

sns.lineplot(x=np.arange(2,20),y=[i.mean() for i in dt1_cv_scores])

plt.title('Decision Tree Accuracy Results with Different Depth Parameters')

plt.ylabel('Average Accuracy')

plt.ylim(top=0.9,bottom=0.6)

plt.xlabel('Decision Tree Depth Parameter')

plt.xticks(ticks=np.arange(2,20),labels=np.arange(2,20));

#according to the accuracy and accuracy mean plots, decision tree with max depth parameter 5 has better results



#But before deciding the parameter, accuracy, precision and recall scores should be review together

fig=plt.figure(figsize=(12,6))

plt.figure(figsize=(15,6))

plt.plot(dt1_precision,'r',label='Precision')

plt.plot(dt1_precision,'ro')

plt.plot(dt1_recall,'b',label='Recall')

plt.plot(dt1_recall,'bo')

plt.plot(dt1_accuracy,'g', alpha=0.5,label='Accuracy')

plt.xticks(ticks=np.arange(0,20),labels=np.arange(2,20))

plt.title('Decision Tree Model Results with Different Depth Parameters')

plt.xlabel('Depth Parameter')

plt.ylabel('Score')

plt.legend();

#looks like a decision tree with max depth parameter 6 is a better than a decision tree with max depth parameter 5.
#Create a decision tree with max depth parameter, 6.

X=data.drop(columns=['target']).values

y=data['target'].values

dt=DecisionTreeClassifier(max_depth=6)

dt.fit(X=X,y=y)

dt_pred=dt.predict(X=X)

from sklearn.metrics import classification_report, precision_recall_fscore_support



print('Classification Report-Decision Tree')

print(classification_report(y_true=y, y_pred=dt_pred))



classifier_report_decision_tree=precision_recall_fscore_support(y_true=y, y_pred=dt_pred) #0:precision, 1: recall
#logistic regression

from sklearn.linear_model import LogisticRegression

lr_1=LogisticRegression()

lr_1_cv=cross_val_score(estimator=lr_1,X=X,y=y,cv=5)

print('Logistic Regression Cross Validation Accuracy Score: {:,.2f}'.format(lr_1_cv.mean()))

plt.figure(figsize=(12,4))

sns.barplot(y=lr_1_cv,x=np.arange(1,6))

plt.title('Logistic Regression Cross Validation Accuracy Score Distribution')

plt.xlabel('Cross Validation Trials')

plt.ylabel('Accuracy Score');



lr=LogisticRegression()

lr.fit(X=X,y=y)

lr_pred=lr.predict(X=X)

print('Classification Report-Logistic Regression')

print(classification_report(y_true=y, y_pred=lr_pred))



classifier_report_logistic_regression=precision_recall_fscore_support(y_true=y, 

                                                                      y_pred=lr_pred) #0:precision, 1: recall
#knn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

knn_array=[]

for n in range(2,20):

    knn=KNeighborsClassifier(n_neighbors=n)

    knn_1_cv=cross_val_score(estimator=knn,X=X,y=y,cv=5)

    #print('{} neighbors:{:,.2f}'.format(n,knn_1_cv.mean()))

    knn_array.append(knn_1_cv.mean())



plt.figure(figsize=(18,6))

plt.title('Cross Validation Trial Scores and Number of Neighbors')

plt.ylim(top=1,bottom=0.5)

sns.barplot(y=knn_array,

            x=np.arange(2,20));

#looks like the optimal neighbor parameter is 5 and 3 is another good option.
#First try 5 as a neighbor parameter

knn=KNeighborsClassifier(n_neighbors=5)



knn.fit(X=X,y=y)

knn_pred=knn.predict(X=X)

print('Classification Report-K-Neighbor Classifier')

print(classification_report(y_true=y, y_pred=knn_pred))



classifier_report_knn=precision_recall_fscore_support(y_true=y,y_pred=knn_pred) #0:precision, 1: recall
#Now try a KNN classifier with neighbor parameter 3.

#knn-with standardized variables

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler,RobustScaler

knn_array=[]

#standardize X 

robustStandardize=RobustScaler(quantile_range=(10,90))

robustStandardize.fit(X)

X_R_Std=robustStandardize.transform(X)



knn=KNeighborsClassifier(n_neighbors=3)



knn.fit(X=X_R_Std,y=y)

knn_pred=knn.predict(X=X_R_Std)

print('Classification Report-K-Neighbour Classifier')

print(classification_report(y_true=y, y_pred=knn_pred))



classifier_report_knn=precision_recall_fscore_support(y_true=y,y_pred=knn_pred) #0:precision, 1: recall
#Visualization of the features and their effect on target variable

Feature_Names=data.drop(columns=['target']).columns.values

plt.figure(figsize=(16,12))

plt.barh(y=Feature_Names,width=dt.feature_importances_)

plt.title('Decision Tree-Feature Importance Distribution')

plt.ylabel('Feature Importance');