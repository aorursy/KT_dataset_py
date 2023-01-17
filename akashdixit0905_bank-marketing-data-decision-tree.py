# Author       : AKASH DIXIT

# E-Mail       : akashdixit453@gmail.com

# Contact      : +91-7415770162

# Designation  : Robotics Engineer

# Decision Tree for Financial Loam EMI default detection

# Data : bank.csv
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn import datasets

from io import StringIO

from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn import metrics

%matplotlib inline
#Load the dataset

data = pd.read_csv('/kaggle/input/bankcsv/bank.csv')

data.head()
#check if the data set contains any null value

data[data.isnull().any(axis=1)].count()
data.describe()
#Box plot of 'age'

g = sns.boxplot(data['age'])
#Distribution plot of 'age'

sns.distplot(data['age'], bins=100)
#Box plot of 'duration'

g = sns.boxplot(data['duration'])
#Distribution plot of 'duration'

sns.distplot(data['duration'], bins = 100)
#Make a copy of data

data = data.copy()
#Different types of job categories and their counts

data.job.value_counts()
#Explore People who made a deposit Vs Job category

jobs = ['management','blue-collar','technician','admin','services',

       'retired','self-employed','student','unemployed','entrepreneur',

       'housemaid','unknown']



for j in jobs:

    print("{:15} : {:5}". format(j, len(data[(data.deposit == "yes") & (data.job ==j)])))
#combine similar jobs into categiroes

data['job'] = data['job'].replace(['management','admin'],'white-collar')

data['job'] = data['job'].replace(['services','housemaid'],'pink-collar')

data['job'] = data['job'].replace(['retired','student','unemployed',

                                  'unknown'],'other')
#New Value counts

data.job.value_counts()
# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'

data['poutcome'] = data['poutcome'].replace(['other'],'unknown')



data.poutcome.value_counts()
# Drop 'contact', as every participant has been contacted.

data.drop('contact', axis = 1, inplace= True)
#values for "default" :yes/no

data['default']

data['default_cat'] = data['default'].map({'yes':1 , 'no':0})

data.drop('default', axis=1,inplace = True)
#values for "housing": yes/no

data['housing_cat'] = data['housing'].map({'yes':1, 'no':0})

data.drop('housing', axis=1, inplace = True)
#values for "loan": yes/no

data['loan_cat'] = data['loan'].map({'yes':1, 'no':0})

data.drop('loan', axis=1, inplace = True)
#day: last contact day of the month

#moth: last contact month of the year

#Drop 'month' and 'day' as they don't have any intrinsic meaning



data.drop('month', axis=1, inplace = True)

data.drop('day', axis=1, inplace = True)
#values for "deposit" : yes/no

data["deposit_cat"] = data['deposit'].map({'yes':1, 'no':0})

data.drop('deposit', axis=1, inplace =True)
# pdays: number of days that passed by after the client was last contacted from a previous campaign

# -1 means client was not previously contacted



print("Customers that have not been contacted before:", len(data[data.pdays==-1]))

print("Maximum values on pdays:", data['pdays'].max())
# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect

data.loc[data['pdays']==-1, 'pdays'] = 10000
#create a new column : recent_pdays

data['recent_pdays'] = np.where(data['pdays'],1/data.pdays,1/data.pdays)

print(data['recent_pdays'])



#Drop 'pdays'

data.drop('pdays', axis=1, inplace = True)
data.tail()
# Convert categorical variables to dummies

data_with_dummies = pd.get_dummies(data=data, columns=['job','marital','education',

                                                      'poutcome'],

                                  prefix=['job','marital','education','poutcome'])



data_with_dummies.head()
data_with_dummies.describe()
#Scatterplot showing age and balance

data_with_dummies.plot(kind='scatter', x='age', y='balance')



#Across all ages, majority of peoples have savings of less than 20000
data_with_dummies.plot(kind='hist', x='poutcome_success', y='duration')
#People who sign up to a term deposit

data_with_dummies[data.deposit_cat==1].describe()
#People signed up to a term deposit having a personal loan (loan_cat) and housing loan(housing_cat)

len(data_with_dummies[(data_with_dummies.deposit_cat == 1) & (data_with_dummies.loan_cat) & (data_with_dummies.housing_cat)])
#People Signed up to a term deposit with a credit default

len(data_with_dummies[(data_with_dummies.deposit_cat == 1) & (data_with_dummies.default_cat==1)])
#Bar chart of Job Vs deposit

plt.figure(figsize=(10,6))

sns.barplot(x='job',y='deposit_cat', data= data)
# Bar chart of "previous outcome" Vs "call duration"



plt.figure(figsize=(10,6))

sns.barplot(x='poutcome', y = 'duration', data = data)
#make a copy

data_cl = data_with_dummies
#The correlation matrix

corr = data_cl.corr()

corr
#Heatmap

plt.figure(figsize=(10,10))

cmap = sns.diverging_palette(300,80, as_cmap=True)

sns.heatmap(corr, xticklabels = corr.columns.values,

           yticklabels=corr.columns.values, cmap = cmap, vmax = 1,

           center = 0, square = True, linewidths=.8, cbar_kws={"shrink":.82})

plt.title("Heatmap of correlation matrix")
#Extract the deposit_cat column (the dependent variable)

corr_deposit = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))

corr_deposit.sort_values(by = 'deposit_cat', ascending = False)
# Train-Test split: 20% test data

data_drop_deposit = data_cl.drop('deposit_cat', 1)

label = data_cl.deposit_cat

data_train, data_test, label_train, label_test = train_test_split(data_drop_deposit, label, test_size = 0.2, random_state = 50)
# Decision tree with depth = 2

dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

dt2.fit(data_train, label_train)



dt2_score_train = dt2.score(data_train, label_train)

print("Training score: ",dt2_score_train)



dt2_score_test = dt2.score(data_test, label_test)

print("Testing score: ",dt2_score_test)
#Decision Tree with Depth = 3

dt3 = tree.DecisionTreeClassifier(random_state=1, max_depth=3)

dt3.fit(data_train, label_train)



dt3_score_train = dt3.score(data_train, label_train)

print("Training score: ",dt3_score_train)



dt3_score_test = dt3.score(data_test, label_test)

print("Testing score: ",dt3_score_test)
# Decision tree with depth = 4

dt4 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)

dt4.fit(data_train, label_train)



dt4_score_train = dt4.score(data_train, label_train)

print("Training score: ",dt4_score_train)



dt4_score_test = dt4.score(data_test, label_test)

print("Testing score: ",dt4_score_test)
# Decision tree with depth = 5

dt5 = tree.DecisionTreeClassifier(random_state=1, max_depth=5)

dt5.fit(data_train, label_train)

dt5_score_train = dt5.score(data_train, label_train)

print("Training score: ",dt5_score_train)

dt5_score_test = dt5.score(data_test, label_test)

print("Testing score: ",dt5_score_test)
# Decision tree with depth = 6

dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)

dt6.fit(data_train, label_train)



dt6_score_train = dt6.score(data_train, label_train)

print("Training score: ",dt6_score_train)



dt6_score_test = dt6.score(data_test, label_test)

print("Testing score: ",dt6_score_test)
#Decision Tree: TO full depth

dt1 = tree.DecisionTreeClassifier()

dt1.fit(data_train, label_train)



dt1_score_train = dt1.score(data_train, label_train)

print("Training score :", dt1_score_train)



dt1_score_test = dt1.score(data_test, label_test)

print("Testing score: ", dt1_score_test)
print('{:10} {:20} {:20}'.format('depth','Training score','Testing score'))

print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))

print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))

print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))

print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))

print('{:1} {:>25} {:>20}'.format(5, dt5_score_train, dt5_score_test))

print('{:1} {:>25} {:>20}'.format(6, dt6_score_train, dt6_score_test))

print('{:1} {:>23} {:>20}'.format("max", dt1_score_train, dt1_score_test))
#Let's generate the decision tree for depth = 2

#create a feature vector



features = data_cl.columns.tolist()

#Two classes:0 = not signed up, 1 = signed up

dt2.classes_
#create a feature vector

features = data_drop_deposit.columns.tolist()



features
#Investigate most important features with depth = 2



dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)



#Fit the decision tree classifier

dt2.fit(data_train, label_train)



fi = dt2.feature_importances_



l = len(features)

for i in range(0,l):

    print('{:.<20} {:3}'.format(features[i],fi[i]))
# According to feature importance results, most importtant feature is the "Duration"

# Let's calculte statistics on Duration



print("Mean duration: ", data_drop_deposit.duration.mean())

print("Maximum duration: ", data_drop_deposit.duration.max())

print("minimum duration: ", data_drop_deposit.duration.min())
# Get a row with poutcome_success = 1

#data_with_dummies[(bank_with_dummies.poutcome_success == 1)]

data_drop_deposit.iloc[985]
#make prediction in the test set



preds = dt2.predict(data_test)



#calculate accuracy

print("\nAccuracy score: \n{}".format(metrics.accuracy_score(label_test,preds)))



#make preidiction on the test set using predict_proba

probs = dt2.predict_proba(data_test)[:,1]



#calculate the AUC metric

print("\nArea Under Curve: \n{}".format(metrics.roc_auc_score(label_test,probs)))