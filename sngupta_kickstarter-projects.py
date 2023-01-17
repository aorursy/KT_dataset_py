from collections import Counter
from string import punctuation

import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
%matplotlib inline
data= pd.read_csv('../input/ks-projects-201801.csv', encoding= 'latin1')
data.shape
data.head()
data.info()
data.isnull().sum()
data.drop(['usd pledged', 'usd_pledged_real'], axis= 1, inplace= True)
data.dropna(inplace= True)
data['state'].value_counts(normalize= True)
#basically I have to deal with only binary classification is it failed or successful
#so, I remove all categories other than failed and successful
data= data[(data['state']=='successful') | (data['state']=='failed')]
data.shape
data['state']= data['state'].map({'successful': 1, 'failed': 0})
#description of the data
data.describe()
#create function to detect the outliers
def detect_outliers(df, cols, n):
    outlier_indices= []
    
    for col in cols:
        #find the first quantile and third quantile
        Q1= np.percentile(df[col], 25)
        Q3= np.percentile(df[col], 75)
        #interquatile range
        IQR= Q3-Q1
        #steps
        step= 1.5*IQR
        #check if dataset contains any outliers
        outlier_col_list= df[(df[col]<Q1-IQR) | (df[col]>Q3+IQR)].index
        #extend or append the outlier_indices
        outlier_indices.extend(outlier_col_list)
        
    #select observation containing more than n outliers
    outlier_indices= Counter(outlier_indices)
    multiple_outliers= list(k for k, v in outlier_indices.items() if v>n)
    
    return multiple_outliers
outlier_to_drop= detect_outliers(data, ['usd_goal_real', 'pledged', 'goal', 'backers'], 2)
len(outlier_to_drop)
#drop the outliers present in the dataset
data.drop(index= outlier_to_drop, axis= 1, inplace= True)
data.shape
#figure out the relation between the usd_pledged_real and usd_goal_real
plt.scatter(data['pledged'], data['goal'], cmap= 'Blues')
plt.title('Pledged vs Goal')
plt.xlabel('Pledged Amount')
plt.ylabel('Goal Amount')
plt.xticks(rotation= 60)
plt.show()
np.corrcoef(data['pledged'], data['goal'])
#there is no realation between the pledged and goal amount it seems like negligible.
#goal and usd_goal_real
plt.scatter(data['usd_goal_real'], data['goal'])
plt.title('Goal vs USD Goal real')
plt.xlabel('USD_GOAL_REAL AMOUNT')
plt.ylabel('GOAL AMOUNT')
plt.xticks(rotation= 60)
plt.show()
np.corrcoef(data['goal'], data['usd_goal_real'])
#highly correlated with each other
#pledged vs usd goal real
plt.scatter(data['usd_goal_real'], data['pledged'])
plt.title('Pledged vs USD Goal real')
plt.xlabel('USD_GOAL_REAL AMOUNT')
plt.ylabel('PLEDGED_AMOUNT')
plt.xticks(rotation= 60)
plt.show()
#not correlated with each other
(mu, sigma)= stats.norm.fit(data['usd_goal_real'])
skew= stats.skew(data['usd_goal_real'])
plt.figure(figsize= (10, 5))

plt.subplot(121)
sns.distplot(data['usd_goal_real'], bins= 50, fit= stats.norm)
plt.title('USD_GOAL_REAL')
plt.legend(['mu: {:.2f}, sigma: {:.2f}, skew: {:.2f}' .format(mu, sigma, skew)], loc= 4)

#take the log of that for the elimination of skewness
(mu_, sigma_)= stats.norm.fit(np.log(data['usd_goal_real']).replace([np.inf, -np.inf], 0))
skew_= stats.skew(np.log(data['usd_goal_real']))

plt.subplot(122)
sns.distplot(np.log(data['usd_goal_real']), bins= 50, fit= stats.norm)
plt.title('USD_GOAL_REAL (Log)')
plt.legend(['mu: {:.2f}, sigma: {:.2f}, skew: {:.2f}' .format(mu_, sigma_, skew_)], loc= 4)
plt.show()
#Taking log approximately drop the skewness of the data
#that is quite significant and thus I have to change it
data['usd_goal_real']= np.log(data['usd_goal_real'])
(mu, sigma)= stats.norm.fit(data['goal'])
skew= stats.skew(data['goal'])
plt.figure(figsize= (10, 5))

plt.subplot(121)
sns.distplot(data['goal'], bins= 50, fit= stats.norm)
plt.title('GOAL AMOUNT')
plt.legend(['mu: {:.2f}, sigma: {:.2f}, skew: {:.2f}' .format(mu, sigma, skew)], loc= 4)

#take the log of that for the elimination of skewness
(mu_, sigma_)= stats.norm.fit(np.log(data['goal']).replace([np.inf, -np.inf], 0))
skew_= stats.skew(np.log(data['goal']))

plt.subplot(122)
sns.distplot(np.log(data['goal']), bins= 50, fit= stats.norm)
plt.title('GOAL AMOUNT (Log)')
plt.legend(['mu: {:.2f}, sigma: {:.2f}, skew: {:.2f}' .format(mu_, sigma_, skew_)], loc= 4)
plt.show()
data['goal']= np.log(data['goal'])
(mu, sigma)= stats.norm.fit(data['pledged'])
skew= stats.skew(data['pledged'])
plt.figure(figsize= (10, 5))

plt.subplot(121)
sns.distplot(data['pledged'], bins= 50, fit= stats.norm)
plt.title('PLEDGED AMOUNT')
plt.legend(['mu: {:.2f}, sigma: {:.2f}, skew: {:.2f}' .format(mu, sigma, skew)], loc= 4)

#take the log of that for the elimination of skewness
(mu_, sigma_)= stats.norm.fit(np.log(data['pledged']).replace([np.inf, -np.inf], 0))
skew_= stats.skew(np.log(data['pledged']).replace([np.inf, -np.inf], 0))

plt.subplot(122)
sns.distplot(np.log(data['pledged']).replace([np.inf, -np.inf], 0), bins= 50, fit= stats.norm)
plt.title('PLEDGED AMOUNT (Log)')
plt.legend(['mu: {:.2f}, sigma: {:.2f}, skew: {:.2f}' .format(mu_, sigma_, skew_)], loc= 4)
plt.show()
data['pledged']= np.log(data['pledged'])
data.isnull().sum()
sns.boxplot(x= 'state', y= 'backers', data= data)
plt.title('BACKERS VS STATE')
plt.show()
data.currency.value_counts().plot(kind= 'bar', title= 'CURRENCY')
plt.figure(figsize= (25, 5))
data.category.value_counts().plot(kind= 'bar', title= 'CATEGORY')
data.main_category.value_counts().plot(kind= 'bar', title= 'MAIN CATEGORY')
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

for var in ['category', 'main_category', 'currency', 'country']:
    data[var]= le.fit_transform(data[var].astype(str))
#conver the date into datetime format
data['launched']= pd.to_datetime(data['launched'])
data['deadline']= pd.to_datetime(data['deadline'])
#find day required to complete the project
day_diff= data['deadline']-data['launched']
day_diff= day_diff.dt.days
data['day_diff']= day_diff
corr= data.corr()
plt.figure(figsize= (10, 10))
sns.heatmap(corr, cmap= 'Blues', annot= True, fmt= '.3f', square= True, linewidths= 0.2)
#from correlation I found that the state of project successful and failed is -vely correlate with the goal amount.
#Thus, I can say that the most dominating factor for the completion of project is goal amount.
#let check the diff in goal amount and the usd goal real amount.
data['diff_goal_real']= data['usd_goal_real']-data['goal']
#here, my intution is if the difference in goal and usd goal amount is greater than 0, then there is more chance
#of the completion of the project.
#let check
print('If the difference >= 0')
print(data[data['diff_goal_real']>=0]['state'].value_counts(normalize= True)*100)
print('If the difference < 0')
print(data[data['diff_goal_real']<0]['state'].value_counts(normalize= True)*100)
#let check the difference between the pledged and goal amount then check the ration of completion of project
data['diff_pledged_goal']= data['pledged']-data['goal']
print('If the difference >= 0')
print(data[data['diff_pledged_goal']>=0]['state'].value_counts(normalize= True)*100)
print('If the difference < 0')
print(data[data['diff_pledged_goal']<0]['state'].value_counts(normalize= True)*100)
#I am not found the difference in amount is so important. So drop it
data.drop(['launched', 'deadline', 'diff_pledged_goal', 'diff_goal_real'], axis= 1, inplace= True)
#replace -np.inf and np.inf to 0
data.replace([-np.inf, np.inf], 0, inplace= True)
name= data['name']
#here, remove all the punctuation and and the stopword from the name columns and then count the length of the name
len_name= []
for n in name:
    count= 0
    n= n.lower().strip().split(' ')
    
    for w in n:
        if w not in punctuation and w not in ENGLISH_STOP_WORDS:
            count= count+1
    len_name.append(count)
data['len_name']= len_name
data.drop(['name'], axis= 1, inplace= True)
data['goal']= round(data['goal']).astype(int)
data['pledged']= round(data['pledged']).astype(int)
data['usd_goal_real']= round(data['usd_goal_real']).astype(int)
data.head()
data.drop(['ID'], axis= 1, inplace= True)
#here we check out how the different machine learning model work on the given dataset
#import libraries

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from mlxtend.classifier import StackingClassifier
import xgboost as xgb
import lightgbm as lgbm 
import catboost as ct
#split data into train and test set
X= data.loc[:, data.columns!= 'state'].values
y= data.loc[:, ['state']].values
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state= 42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
clf_log= LogisticRegression(random_state= 101)
clf_dt= DecisionTreeClassifier(random_state= 101)
clf_knn= KNeighborsClassifier()
clf_rf= RandomForestClassifier(random_state= 101)
clf_gbc= GradientBoostingClassifier(random_state= 101)
clf_extra= ExtraTreesClassifier(random_state= 101)
clf_lin_dis= LinearDiscriminantAnalysis()
classifier= [('LogisticRegression', clf_log), ('LinearDiscrim', clf_lin_dis), ('DecisionTree', clf_dt), 
             ('RandomForest', clf_rf), ('GradientBoosting', clf_gbc), ('KNeighbors', clf_knn),
             ('ExtraTreeClf', clf_extra)]
acc_train= []
acc_test= []
for model in classifier:
    #fit the data into the model
    model[1].fit(X_train, y_train)
    #predict
    prediction= model[1].predict(X_test)
    pred_prob= model[1].predict_proba(X_test)[::, -1]
    #accuracy scores
    acc_train_= model[1].score(X_train, y_train)
    acc_test_= accuracy_score(y_test, prediction)
    acc_train.append(acc_train_)
    acc_test.append(acc_test_)
    
    fpr, tpr, threshold= roc_curve(y_test, pred_prob)
    auc_= auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw= 3.5, label= 'AUC({}): {}' .format(model[0], auc_))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.title('ROC CURVES')
plt.show()

#create df of the accuracy score of different algorithms
df_result= pd.DataFrame({
    'Name': ['LogisticRegression', 'LinearDiscrim', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'KNeighbors',
             'ExtraTreeClf'],
    'Accuracy_train': acc_train,
    'Accuracy_test': acc_test
})
df_result

