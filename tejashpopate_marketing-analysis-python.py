# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from fancyimpute import KNN

import seaborn as sns

from scipy.stats import chi2_contingency

from random import randrange, uniform







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# setting working directory

#os.chdir("")

os.getcwd()
#loading data

marketing_train=pd.read_csv("../input/marketing_tr.csv")

marketing_train.shape
marketing_train.head()
marketing_train.columns
#Exploratory Data Analysis

marketing_train['schooling'] = marketing_train['schooling'].replace("illiterate", "unknown")

marketing_train['schooling'] = marketing_train['schooling'].replace(["basic.4y","basic.6y","basic.9y","high.school","professional.course"], "high.school")

marketing_train['default'] = marketing_train['default'].replace("yes", "unknown")

marketing_train['marital'] = marketing_train['marital'].replace("unknown", "married")

marketing_train['month'] = marketing_train['month'].replace(["sep","oct","mar","dec"], "dec")

marketing_train['month'] = marketing_train['month'].replace(["aug","jul","jun","may","nov"], "jun")

marketing_train['loan'] = marketing_train['loan'].replace("unknown", "no")

marketing_train['profession'] = marketing_train['profession'].replace(["management","unknown","unemployed","admin."], "admin.")

marketing_train['profession'] = marketing_train['profession'].replace(["blue-collar","housemaid","services","self-employed","entrepreneur","technician"], "blue-collar")
# checking missing values

missing_values=pd.DataFrame(marketing_train.isnull().sum())

missing_values

#reseting index

missing_values=missing_values.reset_index()

missing_values

#renaming the column names of the dataframes

missing_values= missing_values.rename(columns={'index':'Variables',0:'missing percentage'})

missing_values
#calcualting % of missing values

missing_values['missing percentage']=(missing_values['missing percentage']/len(marketing_train))*100

missing_values
# sorting data

missing_values=missing_values.sort_values('missing percentage',ascending=False).reset_index(drop=False)

missing_values.to_csv("missing_perc.csv",index=False)
# experiment 

marketing_train['custAge'].loc[70]=np.nan

marketing_train['custAge'].loc[70]
# impute with mean

#marketing_train['custAge']=marketing_train['custAge'].fillna(marketing_train['custAge'].mean())

#marketing_train['custAge'].loc[70]
#impute with median

#marketing_train['custAge']=marketing_train['custAge'].fillna(marketing_train['custAge'].median())

#marketing_train['custAge'].loc[70]
#impute with KNN imputation method

#assigning levels to each categories:

lis=[]

for i in range(0, marketing_train.shape[1]):

    if(marketing_train.iloc[:,i].dtype=='object'):

        marketing_train.iloc[:,i]=pd.Categorical(marketing_train.iloc[:,i])

        marketing_train.iloc[:,i]=marketing_train.iloc[:,i].cat.codes

        marketing_train.iloc[:,i]=marketing_train.iloc[:,i].astype('object')

        lis.append(marketing_train.columns[i])

        
#replace -1 with NA to impute

for i in range(0, marketing_train.shape[1]):

    marketing_train.iloc[:,i] = marketing_train.iloc[:,i].replace(-1, np.nan) 
# Apply KNN imputation algo

marketing_train=pd.DataFrame(KNN(k=3).fit_transform(marketing_train),columns=marketing_train.columns)

marketing_train['custAge'].loc[70]
# Covert the data in appropriate data type

for i in lis:

    marketing_train.loc[:,i]=marketing_train.loc[:,i].round()

    marketing_train.loc[:,i]=marketing_train.loc[:,i].astype('object')
marketing_train.to_csv("marketing_campaign_cleaned.csv",index=False)
marketing_train.head()
# import required library

#import ggplot as glt
# bar plot

#ggplot(marketing_train,aes(x='profession',y='campaign'))+\

#geom_bar(fill="blue")+theme_bw()+xlab("Profession")+ylab("Campaign")+\

#ggtitle("Marketing Campaign Analysis")+theme(text=element_text(size=20))
#copying the data for expts

#df=marketing_train.copy()

#marketing_train.info()
# visualization using box plots

#plt.boxplot(marketing_train['custAge'])
# separating continuous variables from the datasets

cont_names=['custAge','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'

            ,'pmonths','pastEmail']

#detect and delete the outlier from the data

#for i in cont_names:

#    q75,q25=np.percentile(marketing_train.loc[:,i],[75,25])

#    iqr=q75-q25 #inter quartile range

#    min=q25-(iqr*1.5)

#    max=q75+(iqr*1.5)

#    marketing_train=marketing_train.drop(marketing_train[marketing_train.loc[:,i]<min].index)

#    marketing_train=marketing_train.drop(marketing_train[marketing_train.loc[:,i]>max].index)

#marketing_train.shape

    
#detect and replace outliers with NAs

#marketing_train=df.copy()

#extract outliers

#q75,q25=np.percentile(marketing_train['custAge'],[75,25])

#calculate IQR

#iqr=q75-q25

#calculate inner and outer fence

#minimum=q25-(iqr*1.5)

#maximum=q75+(iqr*1.5)

#replacing the outiers

#marketing_train.loc[marketing_train['custAge']<minimum]=np.nan

#marketing_train.loc[marketing_train['custAge']>maximum]=np.nan

#calculate the missing values

#print('missing values=',marketing_train['custAge'].isna().sum())

# impute wiht KNN

#marketing_train=pd.DataFrame(KNN(3).fit_transform(marketing_train),columns=marketing_train.columns)

#marketing_train.isna().sum()
##Correlation analysis

#Correlation plot

df_corr = marketing_train.loc[:,cont_names]
#Set the width and hieght of the plot

f, ax = plt.subplots(figsize=(7, 5))



#Generate correlation matrix

corr = df_corr.corr()



#Plot using seaborn library

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
#Chisquare test of independence

#Save categorical variables

cat_names = ["profession", "marital", "schooling", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
#loop for getting chi square values

for i in cat_names:

    print(i)

    chi2, p, dof, ex = chi2_contingency(pd.crosstab(marketing_train['responded'], marketing_train[i]))

    print(p)
# deleting unwanted data from the dataframe

marketing_train = marketing_train.drop(['pdays', 'emp.var.rate', 'day_of_week', 'loan', 'housing'], axis=1)
df = marketing_train.copy()

#marketing_train = df.copy()
#Normality check

%matplotlib inline  

plt.hist(marketing_train['campaign'], bins='auto')
# remaining continuous variables after removing unwanted variables

cont_names = ["custAge","campaign","previous","cons.price.idx","cons.conf.idx","euribor3m","nr.employed",

           "pmonths","pastEmail"]
#Nomalisation

for i in cont_names:

    print(i)

    marketing_train[i] = (marketing_train[i] - min(marketing_train[i]))/(max(marketing_train[i]) - min(marketing_train[i]))
# #Standarisation

# for i in cnames:

#     print(i)

#     marketing_train[i] = (marketing_train[i] - marketing_train[i].mean())/marketing_train[i].std()
##Simple random sampling

#Sim_Sampling = marketing_train.sample(5000)
# ##Systematic Sampling

# #Calculate the K value

# k = len(marketing_train)/3500



# # Generate a random number using simple random sampling

# RandNum = randrange(0, 5)



# #select Kth observation starting from RandNum

# Sys_Sampling = marketing_train.iloc[RandNum::k, :]
# #Stratified sampling

# from sklearn.cross_validation import train_test_split



# #Select categorical variable

# y = marketing_train['profession']



#select subset using stratified Sampling

#Rest, Sample = train_test_split(marketing_train, test_size = 0.6, stratify = y)
#marketing_train = pd.read_csv("marketing_train_Model.csv")
#Import Libraries for decision tree

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
#replace target categories with Yes or No

marketing_train['responded'] = marketing_train['responded'].replace(0, 'No')

marketing_train['responded'] = marketing_train['responded'].replace(1, 'Yes')
#Divide data into train and test

X = marketing_train.values[:, 0:16]

Y = marketing_train.values[:,16]



X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)
#Decision Tree

C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)



#predict new test cases

y_predict = C50_model.predict(X_test)



#Create dot file to visualise tree  #http://webgraphviz.com/

# dotfile = open("pt.dot", 'w')

#df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names = marketing_train.columns)
# Model Evaluation

#import the required library and module

from sklearn.metrics import confusion_matrix

#Building confusion matrix

Conf_mat=confusion_matrix(y_test,y_predict)
Conf_mat=pd.crosstab(y_test,y_predict)

Conf_mat
# defining the proper parameters

TN=Conf_mat.iloc[0,0]

FP=Conf_mat.iloc[0,1]

FN=Conf_mat.iloc[1,0]

TP=Conf_mat.iloc[1,1]
# checking accuracy of the model

accuracy_score(y_test,y_predict)*100
# accuracy in another way

# accuracy=(correctly predicted values)*100/(total observation)

(TP+TN)*100/(TP+TN+FP+FN)
# false negative rate 

FNR=(FN)*100/(FN+TP)

FNR

# Recall

recall=(TP*100)/(TP+FN)

recall
#import required library and apply random forest for it

from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
#get the predictions

RF_predictions=RF_model.predict(X_test)
# build confusion matrix

#CM=confusion_matrix(y_test,RF_predictions)

CM=pd.crosstab(y_test,RF_predictions)

CM

# Set the parameters TP,TN,FP,FN

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]

# calculate the accuracy of the RF model

accuracy=(TN+TP)*100/(TN+TP+FN+FP)

accuracy

# false negative rate

FNR=(FN*100)/(FN+TP)

FNR
# Data Preparation

# replace target variable categories "Yes" and "No" with 1 and 0

marketing_train['responded'] = marketing_train['responded'].replace('No',0)

marketing_train['responded'] = marketing_train['responded'].replace('Yes',1)



#Prepare logistic data

# save target variable temperorily

marketing_train_logit=pd.DataFrame(marketing_train["responded"])

#add continuous variables here

marketing_train_logit=marketing_train_logit.join(marketing_train[cont_names])

# Dummification: Creating dummy variables for categorical variables

cat_names=["profession","marital","schooling","default","contact","month","poutcome"]

for i in cat_names:

    temp=pd.get_dummies(marketing_train[i],prefix=i)

    marketing_train_logit=marketing_train_logit.join(temp)

marketing_train_logit.shape
# separate the dataset into train and test

sample_index=np.random.rand(len(marketing_train_logit))<0.8 # this will generate random values ie. random sampling for training dataset 

train=marketing_train_logit[sample_index]

test=marketing_train_logit[~sample_index]
#save column indices for independent variables

train_cols=train.columns[1:30]
train.head()
# Build/ Train the Logistic Regression Algo

import statsmodels.api as sm

logit_model=sm.Logit(train["responded"],train[train_cols]).fit()
# check the summary of the model

logit_model.summary()
# Now make predictions 

test["actual_prob"]=logit_model.predict(test[train_cols])

test["actual_val"]=1

#test.loc[test["actual_prob"] < 0.5,"actual_val"]=0

test.loc[test.actual_prob < 0.5,"actual_val"]=0
# model evaluation

# make a confusion matrix

CM=pd.crosstab(test["responded"],test["actual_val"])

CM
# set the parameters

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]

# accuracy

accuracy=(TP+TN)*100/(TP+TN+FP+FN)

#FNR

FNR=(FN*100)/(FN+TP)

accuracy,FNR
# required libraries

from sklearn.naive_bayes import GaussianNB
# train the model

NB_model=GaussianNB().fit(X_train,y_train)
#make predictions

NB_predictions=NB_model.predict(X_test)
#model evaluation

CM=pd.crosstab(y_test,NB_predictions)

CM
# set the parameters

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]
#accuracy

accuracy=(TP+TN)*100/(TP+TN+FP+FN)

print(accuracy)

#FNR

FNR=(FN)*100/(FN+TP)

print(FNR)