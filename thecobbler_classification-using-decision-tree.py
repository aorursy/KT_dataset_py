import pandas as pd

import numpy as np

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

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

#Load Data File

bank=pd.read_csv('../input/bank.csv')

bank.head()
# Check if the data set contains any null values - Nothing found!

bank.isnull().sum()

bank.describe()
# Boxplot for 'age'

g = sns.boxplot(x=bank["age"])
# Distribution of Age

sns.distplot(bank.age, bins=100)
# Boxplot for 'duration'

g = sns.boxplot(x=bank["duration"])
sns.distplot(bank.duration, bins=100)
bank_data = bank.copy()
#Explore People who made a deposit vs Job Category

job = bank['job'].unique().tolist()



for i in job:

    print("{:15} : {:5}".format(i,len(bank[(bank['deposit'] == 'yes') & (bank['job'] == i)])))
# Different types of job categories and their counts

bank_data.job.value_counts()
# Combine similar jobs into categiroes

bank['job'] = bank['job'].replace(['management','admin.'],'white-collar')

bank['job'] = bank['job'].replace(['housemaid','services'],'pink-collar')

bank['job'] = bank['job'].replace(['retired','student','unemployed','unknown'],'other')
# New Value Counts

bank.job.value_counts()
bank.poutcome.value_counts()
#Combining Unknown and Other as Other

bank['poutcome'] = bank['poutcome'].replace('other','unknown')

bank.poutcome.value_counts()
#dropping contact feature since everyone had been contacted

bank.drop('contact',axis=1,inplace = True)
# values for "default" : yes/no

bank['default'].value_counts()

bank['default_cat'] = bank['default'].map({'yes':1,'no':0})

bank.drop('default',axis = 1, inplace = True)
#values for 'housing' : yes/no

bank['housing_cat'] = bank['housing'].map({'yes':1,'no':0})

bank.drop('housing',axis = 1, inplace = True)
#values for 'loan' : yes/no

bank['loan_cat'] = bank['loan'].map({'yes':1,'no':0})

bank.drop('loan',axis = 1, inplace = True)
# day  : last contact day of the month

# month: last contact month of year

# Drop 'month' and 'day' as they don't have any intrinsic meaning

bank.drop(['day','month'],axis = 1,inplace = True)
#Values for deposit : yes/no

bank['deposit_cat'] = bank['deposit'].map({'yes':1,'no':0})

bank.drop('deposit',axis = 1, inplace = True)
bank.head(2)
#Number of days passed by since the client was last contacted from a previous campaign

# -1 means client was not contacted previously

print("Number of Customers who were not contacted as part of any previous campaign:  ",len(bank[bank['pdays'] == -1]))

print("Maximum values on pdays: ",bank['pdays'].max())
#Mapping pdays = -1 with a value 10000, a value so large 

bank.loc[bank['pdays'] == -1,'pdays'] = 10000
bank1 = bank
#Create a new column recent_pdays

bank['recent_pdays'] = np.where(bank['pdays'],1/bank.pdays,1/bank.pdays)

bank.drop('pdays',axis=1,inplace = True)
bank.tail()
#Convert To Dummies

bank_with_dummies = pd.get_dummies(data = bank, columns=['job','marital','education','poutcome'],prefix = ['job','marital','education','poutcome'])
bank_with_dummies.head(3)
bank_with_dummies.shape
bank_with_dummies.describe()
#Observations on whole population

#Scatter Plot showing age and balance

sns.scatterplot(data = bank_with_dummies,x ='age',y = 'balance')
#poutcome vs duration

bank_with_dummies.plot(x='poutcome_success',y='duration',kind='hist')
# People who sign up to a term deposite

bank_with_dummies[bank.deposit_cat == 1].describe()
#People signed up to a term deposit having a personal loan and a housing loan

len(bank_with_dummies[(bank_with_dummies['deposit_cat'] == 1) & (bank_with_dummies['loan_cat'] == 1) & (bank_with_dummies['housing_cat'] == 1)])
# People signed up to a term deposite with a credit default

len(bank_with_dummies[(bank_with_dummies['deposit_cat'] == 1)&(bank_with_dummies['default_cat'] == 1)])
#Number of People with White Collared jobs who opted for term deposit

len(bank_with_dummies[(bank_with_dummies['deposit_cat'] == 1)&(bank_with_dummies['job_white-collar'] == 1)])
#Number of People with White Collared jobs who opted for term deposit

len(bank_with_dummies[(bank_with_dummies['deposit_cat'] == 0)&(bank_with_dummies['job_white-collar'] == 1)])
bank['job'].value_counts()
# Bar chart of job Vs deposite

plt.figure(figsize = (10,6))

sns.barplot(x='job_technician', y = 'deposit_cat', data = bank_with_dummies)
# Bar chart of job Vs deposite

plt.figure(figsize = (10,6))

sns.barplot(x='job_other', y = 'deposit_cat', data = bank_with_dummies)
# Bar chart of job Vs deposite

plt.figure(figsize = (10,6))

sns.barplot(x='job_white-collar', y = 'deposit_cat', data = bank_with_dummies)
# Bar chart of job Vs deposite

plt.figure(figsize = (10,6))

sns.barplot(x='job', y = 'deposit_cat', data = bank)
#Need to find out how to use SNS to fetch the data for FALSE conditions as well

#Bar Chart of "Previous Outcome" and duration

plt.figure(figsize = (10,6))

sns.barplot(x='poutcome', y = 'duration', data = bank)
# make a copy

bankcl = bank_with_dummies
# The Correltion matrix

corr = bankcl.corr()

corr
# Heatmap

plt.figure(figsize = (10,10))

cmap = sns.diverging_palette(220,10,as_cmap = True)

#Deep dive into diverging_pattern

sns.heatmap(corr,xticklabels=corr.columns.values,

           yticklabels=corr.columns.values,cmap=cmap,vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})

plt.title('Heatmap of Correlation Matrix')
# Extract the deposte_cat column (the dependent variable) to understand the correlation with other features

corr_deposit = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))
corr_deposit.sort_values(by = 'deposit_cat',ascending = False)
# Train-Test split: 20% test data

data_drop_deposite = bankcl.drop('deposit_cat', 1)

label = bankcl.deposit_cat

data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)
# Decision tree with depth = 2

dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

dt2.fit(data_train, label_train)

dt2_score_train = dt2.score(data_train, label_train)

print("Training score: ",dt2_score_train)

dt2_score_test = dt2.score(data_test, label_test)

print("Testing score: ",dt2_score_test)
# Decision tree with depth = 3

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
# Decision tree with depth = 6

dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)

dt6.fit(data_train, label_train)

dt6_score_train = dt6.score(data_train, label_train)

print("Training score: ",dt6_score_train)

dt6_score_test = dt6.score(data_test, label_test)

print("Testing score: ",dt6_score_test)
# Decision tree: To the full depth

dt1 = tree.DecisionTreeClassifier()

dt1.fit(data_train, label_train)

dt1_score_train = dt1.score(data_train, label_train)

print("Training score: ", dt1_score_train)

dt1_score_test = dt1.score(data_test, label_test)

print("Testing score: ", dt1_score_test)
print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))

print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))

print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))

print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))

print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))

print('{:1} {:>25} {:>20}'.format(6, dt6_score_train, dt6_score_test))

print('{:1} {:>23} {:>20}'.format("max", dt1_score_train, dt1_score_test))
# Let's generate the decision tree for depth = 2

# Create a feature vector

features = data_drop_deposite.columns.tolist()
len(features)
# Uncomment below to generate the digraph Tree.

tree.export_graphviz(dt2, out_file='tree_depth_2.dot', feature_names=features)
#Feature Importance Metrics

dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

# Fit the decision tree classifier

dt2.fit(data_train, label_train)



fi = dt2.feature_importances_

l = len(features)

for i in range(0,len(features)):

    print('{:.<20} {:3}'.format(features[i],fi[i]))
# According to feature importance results, most importtant feature is the "Duration"

# Let's calculte statistics on Duration

print("Mean duration   : ", data_drop_deposite.duration.mean())

print("Maximun duration: ", data_drop_deposite.duration.max())

print("Minimum duration: ", data_drop_deposite.duration.min())
# Predict: Successful deposite with a call duration = 371 sec



print(dt2.predict_proba(np.array([0, 0, 371, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))

print(dt2.predict(np.array([0, 0, 371, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))
# Predict: Successful deposite with a maximun call duration = 3881 sec



print(dt2.predict_proba(np.array([0, 0, 3881, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))

print(dt2.predict(np.array([0, 0, 3881, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))
#All rows with poutcome == 1

bank_with_dummies[(bank_with_dummies.poutcome_success == 1)]
data_drop_deposite.iloc[985]
# Predict: Probability for above



print(dt2.predict_proba(np.array([46,3354,522,1,1,0,1,0,0.005747,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0]).reshape(1, -1)))

#print(ctree.predict(np.array([46,3354,522,1,1,0,1,0,0.005747,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0]).reshape(1, -1)))
# Make predictions on the test set

preds = dt2.predict(data_test)



# Calculate accuracy

print("\nAccuracy score: \n{}".format(metrics.accuracy_score(label_test, preds)))



# Make predictions on the test set using predict_proba

probs = dt2.predict_proba(data_test)[:,1]



# Calculate the AUC metric

print("\nArea Under Curve: \n{}".format(metrics.roc_auc_score(label_test, probs)))