#very good at graphs and visuals.
import matplotlib.pyplot as mplt
#used to work with dataframes/datasets
import pandas as pd
#change directory
import os
#used with matplotlib to make more dynamic visuals
import seaborn as sns
#manipulating dataframe
from sklearn.preprocessing import StandardScaler
#numpy
import numpy as np
#visualization of high dimension to two dimension
from sklearn.manifold import TSNE
#????
import matplotlib.patches as mpatches
#used to ignore warnings
import warnings
#split datasets intro random train and test subset
from sklearn.model_selection import train_test_split

#Changed directory
try:
    os.chdir('../input/creditcardfraud')
except OSError:
    print("!!!wrong directory!!!")
print("Changed our directory to where CSV is saved: "+os.getcwd())

#Accessing our CSV file through pandas as pd
df=pd.read_csv('creditcard.csv')

#showing shape of file rows and columns
##Cannot concatenate str with int types in python
print("Our Dataframe as nxm is{}: ".format(df.shape))

#get sample of 5 dataframe records to get a glimpse
print(df.sample(5))

#first five values
print(df.head())

#prints concise summary of data by column giving dtype as well
df.info()

#set the decimal to 3 places in the dataframe
pd.set_option('precision', 3)

#setting columns to display... .describe gives us the 5 number summary
print(df.loc[:, ['Time', 'Amount']].describe())

#sets the size of matplot visuals
mplt.figure(figsize=(10,8))

#sets visual title
mplt.title('Time Feature')

#visual of df....
sns.distplot(df.Time)
mplt.show()

#repeat for other column
mplt.figure(figsize=(10,8))
mplt.title("Amount Feature")
sns.distplot(df.Amount)
mplt.show()

#representing normal from fraudulent transactions
#value_counts returns count of unique values in a class
counts = df.Class.value_counts()
normal=counts[0]
fraudulent=counts[1]
perc_normal=(normal/(normal+fraudulent))*100
perc_fraudulent=(fraudulent/(normal+fraudulent))*100
print("There are {} non-fraudulent transactions at {}% and {} fraudulent transactions at {}%".format(normal, perc_normal, fraudulent, perc_normal))

#bar plot representation of normal, fraudulent transaction
mplt.figure(figsize=(8,6))
mplt.title("Fraudulent vs Non-Fraudulent")
sns.barplot(x=counts.index, y=counts)
mplt.ylabel('Count')
mplt.xlabel('Class: Non-Fraudulent=0______Fraudulent=1')
mplt.show()

#getting the correlation between the two variables
corr=df.corr()
print(corr)

#Displaying heatmap of correlation *visual
mplt.figure(figsize=(12,10))
mplt.title('Heatmap of correlation')
heat_map=sns.heatmap(data=corr)
mplt.show()

#skewness of each column
skew=df.skew()
print(skew)

#*scaling* our anonymized values are centered around 0
#and if compared to non-anonymized features(class,amount,time)
#our algorithms will perform poorly.

"""Scaling features"""

#standardscaler alters the data(scales) that the data has a mean of 0 and stdev of 1
scaler = StandardScaler()
scaler2=StandardScaler()

#_.fit computes the mean and std
#_.transform perform standardization by centering and scaling
#_.fit_transform fit to data then transform.
###scaling the time"""Why is time written like this?"""////The outer brackets are pandas' typical selector brackets, telling pandas to select a column from the dataframe. The inner brackets indicate a list. You're passing a list to the pandas selector. If you just use single brackets - with one column name followed by another, separated by a comma - pandas interprets this as if you're trying to select a column from a dataframe with multi-level columns (a MultiIndex) and will throw a keyerror. –
scaled_time = scaler.fit_transform(df[['Time']])
print('\n Scaled Time')
print(scaled_time)
#putting nested lists in flatlist
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
flattened_scaled_time = pd.Series(flat_list1)
print(flattened_scaled_time)

#scaling amount
scaled_amount = scaler2.fit_transform(df[['Amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
flattened_scaled_amount = pd.Series(flat_list2)

#concat two columns: axis =0 appends to row, axis=1 appends to column
df= pd.concat([df, flattened_scaled_amount.rename('scaled amount'), flattened_scaled_time.rename('scaled time')], axis=1)
print(df.sample(5))

#drops the old columns per row and sets the new dataframe
df.drop(['Amount', 'Time'], axis=1, inplace=True)

"""Splitting Data into Training and Testing data"""

#creates uniform rand array of n=(len(df)) true or false booleans
#assigns true or false in the dataset
#~ flips true to false
mask = np.random.rand(len(df)) < 0.9
train = df[mask]
test = df[~mask]
print('Train Shape: \n {}\n Test Shape: \n{}'.format(train.shape, test.shape))
#drops the current index column and replaces with new sequential index
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print( df.shape)

"""Creating a subsample data set with balanced class distributions"""

#number of fraudulent transactions
no_of_frauds = train.Class.value_counts()[1]
print('There are {} fraudulent transactions in the train data.'.format(no_of_frauds))
print('\n Null values in Dataframe \n', df.isnull().sum())

#seperated fraudulent from non fraudelent sets
# fraud and selected_to_match_frauds ARE PAIRED
non_fraud = train[train['Class']==0]
fraud = train[train['Class']==1]
selected_to_match_frauds=non_fraud.sample(no_of_frauds)
selected_to_match_frauds.reset_index(drop=True, inplace=True)
fraud.reset_index(drop=True, inplace=True)

#concatenating both fraud and selected_to_match_frauds
subsample = pd.concat([selected_to_match_frauds, fraud])
print(len(subsample))

#(frac=1) selects the fractional number to return. Since its one, not fractional but randomized w/ reset.
subsample = subsample.sample(frac=1).reset_index(drop=True)
print(subsample.head())

#visual of subsample
new_counts = subsample.Class.value_counts()
mplt.figure(figsize=(8,6))
sns.barplot(x = new_counts.index, y = new_counts)
mplt.title('Count of Fraudulent vs. Non-Fraudulent Transactions In Subsample')
mplt.ylabel('Count')
mplt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent')
mplt.show()

#!!! look into this code !!!
corr=subsample.corr()
corr = corr[['Class']]
print('Subsample correlation \n', corr)
#negative correlation smaller than -0.5
print('negative correlation less than -0.5 \n', corr[corr.Class <  -0.5])
#positive correlation greater than 0.5
print('Positive correlation \n',corr[corr.Class > 0.5])

#negative correlation box plot
f, axes = mplt.subplots(nrows=2, ncols=4, figsize=(26,16))

f.suptitle('Features With High Negative Correlation', size=35)
sns.boxplot(x="Class", y="V3", data=subsample, ax=axes[0,0])
sns.boxplot(x="Class", y="V9", data=subsample, ax=axes[0,1])
sns.boxplot(x="Class", y="V10", data=subsample, ax=axes[0,2])
sns.boxplot(x="Class", y="V12", data=subsample, ax=axes[0,3])
sns.boxplot(x="Class", y="V14", data=subsample, ax=axes[1,0])
sns.boxplot(x="Class", y="V16", data=subsample, ax=axes[1,1])
sns.boxplot(x="Class", y="V17", data=subsample, ax=axes[1,2])
f.delaxes(axes[1,3])
mplt.show()

#positive box plot
f, axes = mplt.subplots(nrows=1, ncols=2, figsize=(18, 9))

f.suptitle('Features With High Positive Correlation', size=20)
sns.boxplot(x="Class", y="V4", data=subsample, ax=axes[0])
sns.boxplot(x="Class", y="V11", data=subsample, ax=axes[1])
mplt.show()

""" Extreme Outlier Removal"""

#.25 below median of all class datapoints / .75 above median of all class datapoints
Q1 = subsample.quantile(0.25)
Q3 = subsample.quantile(0.75)
print('qunatile: ', Q1,Q3)
#Inter Quartile Range: Box of the box plot
IQR = Q3 - Q1
print(IQR)
df2 = subsample[~((subsample < (Q1 - 2.5 * IQR)) | (subsample > (Q3 + 2.5 * IQR))).any(axis = 1)]
print('df2!!!!!!!!!!!!! \n',df2.sample(15))
len_after = len(df2)
len_before = len(subsample)
len_difference = len(subsample) - len(df2)
print('Length after removing outliers: ',len_after, '\n Length before removing outliers: ', len_before, '\n The difference removed:', len_difference)

"""Dimensionality Reduction"""

#T.. stochaistic neiborhood embedding
X = df2.drop('Class', axis=1)
Y = df2['Class']
#
X_reduced_tsne = TSNE(n_components=2,perplexity=30, n_iter=5000, n_iter_without_progress=200, random_state=1).fit_transform(X.values)
#
f, ax = mplt.subplots(figsize=(24,16))
#
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
#
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(Y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(Y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax.set_title('t-SNE', fontsize=14)

ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])

mplt.show()

"""CLASSIFICATION ALGORITHM"""

#pass warnings in code
def warn(*args, **kwargs):
    pass
warnings.warn = warn

#X-dependent var...Y-dependent var...........split x,y into two sets train test.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
#seperating train, test to validation
X_train = X_train.values
X_validation = X_test.values
Y_train = Y_train.values
Y_validation = Y_test.values
print('X Shapes \n', 'X train', ' X validation\n', X_train.shape, X_validation.shape)
print('Y shapes \n', 'Y train', ' Y validation\n', Y_train.shape, Y_validation.shape )

#sets kfold classification
from sklearn.model_selection import KFold
#executes kfold classification
from sklearn.model_selection import cross_val_score
#
from sklearn.metrics import roc_auc_score
#details how well or poorly our prediction
from sklearn.metrics import classification_report
#details how well or poorly our prediction
from sklearn.metrics import confusion_matrix
#categorical binary (1)(0) model / s or curve shape regression line
from sklearn.linear_model import LogisticRegression
#categorical classification model 2+
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#find k nearest neighbors of datapoints. if even target classes then we need odd k
from sklearn.neighbors import KNeighborsClassifier
#classification algorithm for less seperated datasets
from sklearn.tree import DecisionTreeClassifier
#line seperating value???
from sklearn.svm import SVC
#all in one algorithm / decision tree based / gridient boosting framework
from xgboost import XGBClassifier
#regression and classification / handles missing val / wont overfit/ handle large data with higher dim.
from sklearn.ensemble import RandomForestClassifier

models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier()))
models.append(('RF', RandomForestClassifier()))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)


#Compare Algorithms

fig = mplt.figure(figsize=(12,10))
mplt.title('Comparison of Classification Algorithms')
mplt.xlabel('Algorithm')
mplt.ylabel('ROC-AUC Score')
mplt.boxplot(results)
ax = fig.add_subplot(111)
ax.set_xticklabels(names)
mplt.show()
