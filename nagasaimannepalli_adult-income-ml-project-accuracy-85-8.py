# 1.Importing the basic working libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# ignore warnings

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Working with os module - os is a module in Python 3.

# Its main purpose is to interact with the operating system. 

# It provides functionalities to manipulate files and folders.



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print('# File sizes')

for f in os.listdir('../input'):

    print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
# 2.Importing the dataset using pandas library

income = pd.read_csv("../input/income.csv")

income_c = pd.read_csv("../input/income.csv")
income

# 3.By looking into the dataset, there may not be required to remove any columns in this present scenario. After preprocessing we may check if any is needed.

# education_num, fnlwgt may not be useful for model building

# education_num is for indicating number of years of education

# fnlwgt - The fnlwgt which is the final weight determined by the Census Organization

# is of no use in any of the analysis that we are doing henceforth and is removed.

#The educationnum if a repetitive variable which recodes the categorical variable education as a 

#numeric variable but will be used in the analysis for decision trees, hence is not being removed.

# 4.Basic Analysis: identifying the datatypes, finding basicstats for each column.

income.info()

# There are 9 object datatypes and 6 int datatypes
income.columns

# names of all the varaibles.
# finding no of missing values for each variable

income.isnull().sum()

# there is no missing values in this dataset
import pandas_profiling as pp

pp.ProfileReport(income)
income.corr()
# Data Visualization - univariate and bivariate analysis 

# Univariate analysis

#age

#sns.distplot(income.Age,kde=False,rug=True)

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})

sns.distplot(

    income['Age'], norm_hist=False, kde=True, bins=20, hist_kws={"alpha": 1}

).set(xlabel='Age', ylabel='Count');

# kde = kernel density estimate

# rugplot = At every observation, it will draw a small vertical stick.

# from the below graph,concluded that age between late twenties and early thirties are present more 
income.dtypes
numerical_d = income.select_dtypes(include='int64')

numerical_d
categorical_d = income.select_dtypes(include ='object')

categorical_d
numerical_d.hist(bins=15,figsize=(15,10),layout=(3,2))

# From the below graphs on numerical data, Age is +vely skewed, kurtosis

# More no of people aged late 30's are present

# Most no of people worked for 35 - 40 hrs in the dataset

# More people educated for 9-10 years

# Capital_gain and Capital_loss have more no of zeros
fig, ax = plt.subplots(2, 5, figsize=(50, 30))

for variable, subplot in zip(categorical_d, ax.flatten()):

    sns.countplot(income[variable], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

# From the beolw graphs :

# 1.More no of people working for private sector

# 2.More no of people educated upto HS-grad

# 3.More no of people are Married-civ-spouse

# 4.More no of people are in occupations - prof-speciality, craft repair, Exec-managerial

# 5.More no of people are white, male and belong to united states of america
fig, ax = plt.subplots(3, 2, figsize=(30, 30))

for variable, subplot in zip(numerical_d, ax.flatten()):

    sns.boxplot(income[variable],income['Income'], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

# More no of people earning >50k are aged between 35-50 and working for 35-40 hours

# More no of people earning <=50k are aged between 25-45 and working for 40-50 hours 

# finalwgt, capital_gain, capital_loss doesn't gave any information required
fig, ax = plt.subplots(3, 2, figsize=(30, 30))

for variable, subplot in zip(categorical_d, ax.flatten()):

    sns.boxplot(income['Age' ],income[variable], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

        
sns.relplot(x ='Hours_per_week', y='Occupation',hue='Income',style='Sex',data=income)

# Transport-moving and Craft-repair occupations are ones in which some males are earning more than 50k working for less than 12 hours

# In Priv-house-serv occupation more no of females are present and are earning less than 50k

# In Protective-serv, tech-support,Machine-op-inspect,Farming-fishing,Craft-repair,Transport-moving,Adm-clerical

# occupation more no of males are present and are earning more than 50k

# In Exec-managerial occupation males working for 40-60 hours are earning more than 50k

# In Prof-speciality some of the females earning more than 50k while working for lesser hours
sns.relplot(x='Age',y='Race',hue='Income',style='Sex',data=income)

# In race of Asian-Pac-Islander aged between 45-55 earning more than 50k
sns.relplot(x='Age',y='Education',hue='Sex',style='Income',data=income)

# More no of males who are educated upto Prof-school, Doctorate and bachelors are earning more than 50k
sns.relplot(x='Age',y='Relationship',hue='Income',style='Sex',data=income)
sns.relplot(x='Age',y='Work_class',hue='Sex',style='Income',data=income)

# Self-emp-inc and Self-emp-not-inc work_class contain more no of males earning more than 50k

# Local-gov has males and females earning more than 50k
income.dtypes
# Age, Education_num, Capital_gain, Capital_loss, Hours_per_week - numerical dtypes

# data cleaning - no missing values, treating outliers

# Age



count_age = income.Age.value_counts()

print('count: ',count_age)

q_age1 = income.Age.quantile(0.25)

q_age2 = income.Age.quantile(0.75)

iqr_age = q_age2 -q_age1

lower_age = q_age1 - (1.5*iqr_age)

upper_age = q_age2 + (1.5*iqr_age)

income.Age[income.Age < lower_age] = lower_age

income.Age[income.Age > upper_age] = upper_age
income.Age.value_counts().sort_index()



income.Age.unique()#17,78,#16-30,25-4,40-60,60-80

income.Age[(income.Age > 16) & (income.Age <= 20)] = 0

income.Age[(income.Age > 20) & (income.Age <= 25)] = 1

income.Age[(income.Age > 25) & (income.Age <= 30)] = 2

income.Age[(income.Age > 30) & (income.Age <= 35)] = 3

income.Age[(income.Age > 35) & (income.Age <= 40)] = 4

income.Age[(income.Age > 40) & (income.Age <= 45)] = 5

income.Age[(income.Age > 45) & (income.Age <= 50)] = 6

income.Age[(income.Age > 50) & (income.Age <= 55)] = 7

income.Age[(income.Age > 55) & (income.Age <= 60)] = 8

income.Age[(income.Age > 60) & (income.Age <= 65)] = 9

income.Age[(income.Age > 65) & (income.Age <= 70)] = 10

income.Age[(income.Age > 70) & (income.Age <= 80)] = 11







# Education_num is total number of years they are educated, it is already mentioned in education

#del income['Education_num']



# Education_num

count_edu = income.Education_num.value_counts()

print('count: ',count_edu)

q_edu1 = income.Education_num.quantile(0.25)

q_edu2 = income.Education_num.quantile(0.75)

iqr_edu = q_edu2 -q_edu1

lower_edu = q_edu1 - (1.5*iqr_edu)

upper_edu = q_edu2 + (1.5*iqr_edu)

#income.Education_num[income.Education_num < lower_edu] = lower_edu

#income.Education_num[income.Education_num > upper_edu] = upper_edu

print('lower: ',lower_edu,'upper: ',upper_edu)





#del income['Education']



#income['Education_num'].value_counts()


# Capital_gain

count_capg = income.Capital_gain.value_counts()

print('count: ',count_capg)

q_capg1 = income.Capital_gain.quantile(0.25)

q_capg2 = income.Capital_gain.quantile(0.75)

iqr_capg = q_capg2 -q_capg1

lower_capg = q_capg1 - (1.5*iqr_capg)

upper_capg = q_capg2 + (1.5*iqr_capg)

unique = income.Capital_gain.unique()

print('unique: ',unique)

unique_med = pd.DataFrame(unique).median()

unique_med

income.Capital_gain[income.Capital_gain < lower_capg] = unique_med[0]

#income.Capital_gain[income.Capital_gain > upper_capg] = unique_med[0]

income.Capital_gain.value_counts()


# Capital_loss

count_capl = income.Capital_loss.value_counts()

print('count: ',count_capl)

q_capl1 = income.Capital_loss.quantile(0.25)

q_capl2 = income.Capital_loss.quantile(0.75)

iqr_capl = q_capl2 -q_capl1

lower_capl = q_capl1 - (1.5*iqr_capl)

upper_capl = q_capl2 + (1.5*iqr_capl)

unique_l = income.Capital_loss.unique()

print('unique: ',unique_l)

unique_medl = pd.DataFrame(unique_l).median()

print('unique_medl:',unique_medl)

income.Capital_loss[income.Capital_loss < lower_capl] = unique_medl[0]

#income.Capital_loss[income.Capital_loss > upper_capl] = unique_medl[0]

income.Capital_loss.value_counts()


#Hours_per_week

count_hour = income.Hours_per_week.value_counts()

print('count: ',count_hour)

q_hour1 = income.Hours_per_week.quantile(0.25)

q_hour2 = income.Hours_per_week.quantile(0.75)

print('quantile1: ',q_hour1,'quantile2: ',q_hour2)

iqr_hour = q_hour2 -q_hour1

print('iqr:',iqr_hour)

lower_hour = q_hour1 - (1.5*iqr_hour)

upper_hour = q_hour2 + (1.5*iqr_hour)

print('lower: ',lower_hour ,'upper: ',upper_hour)

income.Hours_per_week[income.Hours_per_week < lower_hour] = lower_hour

income.Hours_per_week[income.Hours_per_week > upper_hour] = upper_hour

income.Hours_per_week.value_counts()
# dealing with numerical data is completed

# next dealing with categorical_data

categorical_d.dtypes
# dealing with categorical data

# the columns Race and Sex are very less important when compared to other variables in dataset

# removing both columns -> accuracy - 85.3% - test_size=0.35

# removing Race column  -> accuracy - 85.88%

# removing Sex column   -> accuracy - 85.08%

# After these columns, Capital_loss and Education are less important, followed by Work_class

#print(income_c.Race.value_counts())

#print(income.Race.value_counts())

#sns.boxplot(income.Race)

#del income['Race']

#del income['Sex']

#del income['Education']

#del income['Work_class']

#income_c.Native_country.value_counts().sort_index()

#income_c.Native_country[income_c.Native_country.value_counts() < 100] 

#del income['Fnlwgt']

# removing fnlwgt is reducing the accuracy of model, may be useful for model building

# no column is removed in this dataset

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

income['Work_class']=le.fit_transform(income['Work_class'].astype('str'))

income['Education']=le.fit_transform(income['Education'].astype('str'))

income['Maritial_Status']=le.fit_transform(income['Maritial_Status'].astype('str'))

income['Occupation']=le.fit_transform(income['Occupation'].astype('str'))

income['Relationship']=le.fit_transform(income['Relationship'].astype('str'))

income['Race']=le.fit_transform(income['Race'].astype('str'))

income['Sex']=le.fit_transform(income['Sex'].astype('str'))

income['Native_country']=le.fit_transform(income['Native_country'].astype('str'))

income['Income']=le.fit_transform(income['Income'].astype('str'))
income['Education_num'] = income['Education_num'].astype(int)

income['Capital_loss'] = income['Capital_loss'].astype(int)

income['Hours_per_week'] = income['Hours_per_week'].astype(int)
income.dtypes
# After conversions, scaling should be appllied on input varaibles

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
input = income.iloc[:,:14]

output = income.iloc[:,14:]

output
input
income_n = pd.DataFrame(scaler.fit_transform(input),columns = input.columns)
income_n
income_n['Income'] = output['Income']
income_n

# 0 -   <=50k

# 1 -   >50k

x_labels = income_n.iloc[:,:14]

y_labels = income_n.iloc[:,14:]

y_labels
x_labels
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_labels,y_labels,test_size = 0.25)

#x_train
#x_test

#y_train

#y_test
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_labels,y_labels)

lr_pred = lr.predict(x_labels)

lr_pred
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix_lr = confusion_matrix(y_labels,lr_pred)

print(confusion_matrix_lr)

accuracy_score_lr = accuracy_score(y_labels,lr_pred)

print(accuracy_score_lr)
# using splitting data

lr.fit(x_train,y_train)

lr_pred1 = lr.predict(x_test)

lr_pred1
confusion_matrix_lr1 = confusion_matrix(y_test,lr_pred1)

print('confusion_matrix of linear regression is:',confusion_matrix_lr1)

print('\n')

accuracy_score_lr1 = accuracy_score(y_test,lr_pred1)

print('accuracy score of linear regression is:',accuracy_score_lr1)
# Applying supervised algorithms 

# Knn

from sklearn.neighbors import KNeighborsClassifier

classifier_k = KNeighborsClassifier(n_neighbors=3)

classifier_k.fit(x_train,y_train)

knn_pred = classifier_k.predict(x_test)

print(knn_pred)

confusion_matrix_knn = confusion_matrix(y_test,knn_pred)

print('confusion_matrix for knn is:  ' ,confusion_matrix_lr1)

print('')

accuracy_score_knn = accuracy_score(y_test,knn_pred)

print('accuracy_score for knn is: ',accuracy_score_knn)

# naive-bayes

from sklearn.naive_bayes import GaussianNB

classifier_n = GaussianNB()

classifier_n.fit(x_train,y_train)

nb_pred = classifier_n.predict(x_test)

print(nb_pred)

confusion_matrix_nb = confusion_matrix(y_test,nb_pred)

print('confusion_matrix for naive bayes is:  ' ,confusion_matrix_nb)

print('')

accuracy_score_nb = accuracy_score(y_test,nb_pred)

print('accuracy_score for naive bayes is: ',accuracy_score_nb)

# support vector machine

from sklearn.svm import SVC

classifier_s = SVC(kernel = 'linear')

classifier_s.fit(x_train,y_train)

svm_pred = classifier_s.predict(x_test)

print(svm_pred)

confusion_matrix_svm = confusion_matrix(y_test,svm_pred)

print('confusion_matrix for support vector machine is:  ' ,confusion_matrix_svm)

print('')

accuracy_score_svm = accuracy_score(y_test,svm_pred)

print('accuracy_score for support vector machine is: ',accuracy_score_svm)

# svm - rbf

from sklearn.svm import SVC

classifier_sr = SVC(kernel = 'rbf')

classifier_sr.fit(x_train,y_train)

svmr_pred = classifier_sr.predict(x_test)

print(svmr_pred)

confusion_matrix_svmr = confusion_matrix(y_test,svmr_pred)

print('confusion_matrix for support vector machine is:  ' ,confusion_matrix_svmr)

print('')

accuracy_score_svmr= accuracy_score(y_test,svmr_pred)

print('accuracy_score for support vector machine is: ',accuracy_score_svmr)

# svm - poly

from sklearn.svm import SVC

classifier_sp = SVC(kernel = 'poly')

classifier_sp.fit(x_train,y_train)

svmp_pred = classifier_sp.predict(x_test)

print(svmp_pred)

confusion_matrix_svmp = confusion_matrix(y_test,svmp_pred)

print('confusion_matrix for support vector machine is:  ' ,confusion_matrix_svmp)

print('')

accuracy_score_svmp= accuracy_score(y_test,svmp_pred)

print('accuracy_score for support vector machine is: ',accuracy_score_svmp)

# svm - sigmoid

# svm - rbf

from sklearn.svm import SVC

classifier_ss = SVC(kernel = 'sigmoid')

classifier_ss.fit(x_train,y_train)

svms_pred = classifier_ss.predict(x_test)

print(svms_pred)

confusion_matrix_svms = confusion_matrix(y_test,svms_pred)

print('confusion_matrix for support vector machine is:  ' ,confusion_matrix_svms)

print('')

accuracy_score_svms= accuracy_score(y_test,svms_pred)

print('accuracy_score for support vector machine is: ',accuracy_score_svms)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

classifier_d = DecisionTreeClassifier(criterion ='entropy',random_state = 0)

classifier_d.fit(x_train,y_train)

deci_pred = classifier_d.predict(x_test)

print(deci_pred)

confusion_matrix_deci = confusion_matrix(y_test,deci_pred)

print('confusion_matrix for Decision Tree is:  ' ,confusion_matrix_deci)

print('')

accuracy_score_deci= accuracy_score(y_test,deci_pred)

print('accuracy_score for Decision Tree is: ',accuracy_score_deci)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier_r = RandomForestClassifier(n_estimators=80,criterion ='entropy',random_state = 42)

classifier_r.fit(x_train,y_train)

rand_pred = classifier_r.predict(x_test)

print(rand_pred)

confusion_matrix_rand = confusion_matrix(y_test,rand_pred)

print('confusion_matrix for Random Forest is:  ' ,confusion_matrix_rand)

print('')

accuracy_score_rand= accuracy_score(y_test,rand_pred)

print('accuracy_score for Random Forest is: ',accuracy_score_rand)



# Application of PCA

from sklearn.decomposition import PCA

pca1 = PCA(n_components =10 )

pca = pca1.fit_transform(x_labels)

pca

income_pca = pd.DataFrame(pca)

income_pca





# The amount of variance that each PCA explains is 

var = pca1.explained_variance_ratio_

var

var1 = np.cumsum(np.round(var,decimals = 4)*100)

var1





income_pca['Income'] = income_n['Income']

income_pca

pca_x = income_pca.iloc[:,:10]

pca_y = income_pca.iloc[:,10:]

pca_y

pca_xtrain,pca_xtest,pca_ytrain,pca_ytest = train_test_split(pca_x,pca_y,test_size=0.25)

#pca_ytrain

#pca_ytest

#pca_xtrain

#pca_xtest



# Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier_rp = RandomForestClassifier(n_estimators=100,criterion ='entropy',random_state = 0)

classifier_rp.fit(pca_xtrain,pca_ytrain)

randp_pred = classifier_rp.predict(pca_xtest)

print(randp_pred)

confusion_matrix_randp = confusion_matrix(pca_ytest,randp_pred)

print('confusion_matrix for Random Forest is:  ' ,confusion_matrix_randp)

print('')

accuracy_score_randp= accuracy_score(pca_ytest,randp_pred)

print('accuracy_score for Random Forest is: ',accuracy_score_randp)

# May be applying pca is a bad idea, using pca is not useful

imp_var = pd.DataFrame(classifier_r.feature_importances_,[input.columns])
imp_var
income_n
income_c
print('accuracy of random forest: ',accuracy_score_rand)

print('accuracy of decision tree: ',accuracy_score_deci)

print('accuracy of linear regression: ',accuracy_score_lr)

print('accuracy of knn: ',accuracy_score_knn)

print('accuracy of naive bayes: ',accuracy_score_nb)

print('accuracy of svm: ',accuracy_score_svm)

print('accuracy of svm-rbf: ',accuracy_score_svmr)

print('accuracy of svm-poly : ',accuracy_score_svmp)

print('accuracy of svm-sigmoid: ',accuracy_score_svms)

# Best model -> random forest

# May be increasing the test_size will increase accuracy

# May be by removing columns like Race, Sex, Education, Capital_loss, Capital_gain, accuracy may be increased, 
