import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import collections



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier





# Other Libraries

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import make_scorer, precision_score, recall_score, classification_report, confusion_matrix

from collections import Counter

from sklearn.preprocessing import RobustScaler

import warnings

warnings.filterwarnings("ignore")





data = pd.read_csv('../input/creditcard.csv',sep=',')

data.head()
print(data.columns)
data.shape
data.info()
#Lets start looking the difference by Normal and Fraud transactions

print("Distribuition of Normal(0) and Frauds(1): ")

print(data["Class"].value_counts())

print('')



# The classes are heavily skewed we need to solve this issue later.

print('Non-Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')



plt.figure(figsize=(7,5))

sns.countplot(data['Class'])

plt.title("Class Count", fontsize=18)

plt.xlabel("Is fraud?", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
plt.figure(figsize=(16,4))

data.iloc[:,:-1].boxplot()

plt.title('(Raw) Distribution of Features', fontsize=17)

plt.show()



plt.figure(figsize=(16,4))

np.log(data.iloc[:,:-1]).boxplot()

plt.title('(Log) Distribution of Features', fontsize=17)

plt.show()
#Now look at Fraud Amounts

plt.figure(figsize=(16,5))

sns.boxplot(x=data.Amount[data.Class == 1])

plt.title('Distribution of (Fraud) Amounts',fontsize=17)

plt.show()

#Now look at Non-Fraud Amounts

plt.figure(figsize=(16,5))

sns.boxplot(x=data.Amount[data.Class == 0])

plt.title('Distribution of (Non-Fraud) Amounts',fontsize=17)

plt.show()
print('Top 85% of transaction amounts:', round(data.Amount.quantile(.85),2))

print('Top 1% of transaction amounts:', round(data.Amount.quantile(.99),2))

print('Largest transaction amount:', round(data.Amount.quantile(1),2))

print('80% of Frauds are less than:', round(data.Amount[data.Class==1].quantile(.80),2))
#First look at Time

plt.figure(figsize=(11,6))

sns.distplot(data.Time,kde=False)

plt.title('Distribution of Time', fontsize=17)

plt.show()
# Create a EDA dataframe for the time units and visualizations

eda = pd.DataFrame(data.copy())



# Tell timedelta to interpret the Time as second units

timedelta = pd.to_timedelta(eda['Time'], unit='s')



# Create a hours feature from timedelta

eda['Time_hour'] = (timedelta.dt.components.hours).astype(int)
#Exploring the distribuition by Class types through seconds

plt.figure(figsize=(12,5))

sns.distplot(eda[eda['Class'] == 0]["Time"], 

             color='g')

sns.distplot(eda[eda['Class'] == 1]["Time"], 

             color='r')

plt.title('(Density Histogram) Fraud VS Normal Transactions by Second', fontsize=17)

plt.xlim([-2000,175000])

plt.show()
#Exploring the distribuition by Class types through hours

plt.figure(figsize=(12,5))

sns.distplot(eda[eda['Class'] == 0]["Time_hour"], 

             color='g')

sns.distplot(eda[eda['Class'] == 1]["Time_hour"], 

             color='r')

plt.title('(Density Histogram) Fraud VS Normal Transactions by Hour', fontsize=17)

plt.xlim([-1,25])

plt.show()
# Define outcome and predictors to split into train and test groups

y = data['Class']

X = data.drop('Class', 1)



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42)



# Class balance in test group

print("TEST GROUP")

print('Size:',y_test.count())

print("Frauds percentage:",

      y_test.value_counts()[1]/y_test.count())

print("Nonfrauds percentage:",

      y_test.value_counts()[0]/y_test.count())



# Class balance in train group

print("\nTRAIN GROUP")

print('Size:',y_train.count())

print("Frauds percentage:",

      y_train.value_counts()[1]/y_train.count())

print("Nonfrauds percentage:",

      y_train.value_counts()[0]/y_train.count())
# Invoke classifier

clf = LogisticRegression()



# Cross-validate on the train data

train_cv = cross_val_score(X=X_train,y=y_train,estimator=clf,cv=3)

print("TRAIN GROUP")

print("\nCross-validation accuracy scores:",train_cv)

print("Mean score:",train_cv.mean())



# Now predict on the test group

print("\nTEST GROUP")

y_pred = clf.fit(X_train, y_train).predict(X_test)

print("\nAccuracy score:",clf.score(X_test,y_test))



# Classification report

print('\nClassification report:\n')

print(classification_report(y_test, y_pred))



# Confusion matrix

conf_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(conf_matrix, annot=True,fmt='d', cmap=plt.cm.copper)

plt.show()

features = pd.DataFrame()
plt.figure(figsize=(12,6))



# Visualize where Time is less than 100,000

plt.subplot(1,2,1)

plt.title("Time < 100,000")

data[data['Time']<100000]['Time'].hist()



# Visualize where Time is more than 100,000

plt.subplot(1,2,2)

plt.title("Time >= 100,000")

data[data['Time']>=100000]['Time'].hist()



plt.tight_layout()

plt.show()
# Create a feature from normal distributions above

features['100k_time'] = np.where(data.Time<100000, 1,0)
plt.figure(figsize=(12,6))



plt.subplot(1,2,1)

plt.title("Non-Frauds, Hour <= 4")

eda.Time_hour[(eda.Class == 0) & (eda.Time_hour <= 4)].plot(kind='hist',bins=15)



plt.subplot(1,2,2)

plt.title("Non-Frauds, Hour > 4")

eda.Time_hour[(eda.Class == 0) & (eda.Time_hour > 4)].plot(kind='hist',bins=15)



plt.tight_layout()

plt.show()
# Create a feature from distributions above

features['4_hour'] = np.where((eda.Class == 0) & (eda.Time_hour > 4), 1,0)
# how many frauds are actually 0 dollars?

print("Non-Fraud Zero dollar Transactions:")

display(data[(data.Amount == 0) & (data.Class == 0)]['Class'].count())

print("Fraudulent Zero dollar Transactions:")

display(data[(data.Amount == 0) & (data.Class == 1)]['Class'].count())
# Capture where transactions have a $0 amount

features['amount0'] = np.where(data.Amount == 0,1,0)
rob_scaler = RobustScaler()



features['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

features['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
# Add the PCA components to our features DataFrame.

features = features.join(data.iloc[:,1:-1].drop('Amount',axis=1))



# Add 'Class' to our features DataFrame.

features = features.join(data.Class)



# Nice! These are the final features I'll settle for.

features.head()
# Define outcome and predictors USE FEATURE-ENGINEERED DATA

y = features['Class']

X = features.drop('Class', 1)



# Split X and y into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42)



# Class balance in test group

print("TEST GROUP")

print('Size:',y_test.count())

print("Frauds percentage:",

      y_test.value_counts()[1]/y_test.count())

print("Nonfrauds percentage:",

      y_test.value_counts()[0]/y_test.count())



# Class balance in train group

print("\nTRAIN GROUP")

print('Size:',y_train.count())

print("Frauds percentage:",

      y_train.value_counts()[1]/y_train.count())

print("Nonfrauds percentage:",

      y_train.value_counts()[0]/y_train.count())
# Invoke classifier

clf = LogisticRegression()



# Make a scoring callable from recall_score

recall = make_scorer(recall_score)



# Cross-validate on the train data

train_cv = cross_val_score(X=X_train,y=y_train,estimator=clf,scoring=recall,cv=3)

print("TRAIN GROUP")

print("\nCross-validation recall scores:",train_cv)

print("Mean recall score:",train_cv.mean())



# Now predict on the test group

print("\nTEST GROUP")

y_pred = clf.fit(X_train, y_train).predict(X_test)

print("\nRecall:",recall_score(y_test,y_pred))



# Classification report

print('\nClassification report:\n')

print(classification_report(y_test, y_pred))



# Confusion matrix

conf_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(conf_matrix, annot=True,fmt='d', cmap=plt.cm.copper)

plt.show()
# Balancing Classes before checking for correlation



# Join the train data

train = X_train.join(y_train)



print('Data shape before balancing:',train.shape)

print('\nCounts of frauds VS non-frauds in previous data:')

print(train.Class.value_counts())

print('-'*40)



# Oversample frauds. Imblearn's ADASYN was built for class-imbalanced datasets

X_bal, y_bal = ADASYN(sampling_strategy='minority',random_state=0).fit_resample(

    X_train,

    y_train)



# Join X and y

X_bal = pd.DataFrame(X_bal,columns=X_train.columns)

y_bal = pd.DataFrame(y_bal,columns=['Class'])

balanced = X_bal.join(y_bal)





print('-'*40)

print('Data shape after balancing:',balanced.shape)

print('\nCounts of frauds VS non-frauds in new data:')

print(balanced.Class.value_counts())
print('Distribution of the Classes in the subsample dataset')

print(balanced.Class.value_counts()/len(train))



sns.countplot('Class', data=balanced)

plt.title('Class Distribution', fontsize=14)

plt.show()

# Compare correlation of raw train data VS balanced train data



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))



# Imbalanced DataFrame

corr = train.corr()

sns.heatmap(corr, annot_kws={'size':20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (Biased)", fontsize=14)



# Balanced DataFrame

bal_corr = balanced.corr()

sns.heatmap(bal_corr, annot_kws={'size':20}, ax=ax2)

ax2.set_title('Balanced Correlation Matrix', fontsize=14)

plt.show()
# Each feature's correlation with Class

bal_corr.Class
no_outliers=pd.DataFrame(balanced.copy())
# Removing Outliers from high-correlation features



cols = bal_corr.Class.index[:-1]



# For each feature correlated with Class...

for col in cols:

    # If absolute correlation value is more than X percent...

    correlation = bal_corr.loc['Class',col]

    if np.absolute(correlation) > 0.1:

        

        # Separate the classes of the high-correlation column

        nonfrauds = no_outliers.loc[no_outliers.Class==0,col]

        frauds = no_outliers.loc[no_outliers.Class==1,col]



        # Identify the 25th and 75th quartiles

        all_values = no_outliers.loc[:,col]

        q25, q75 = np.percentile(all_values, 25), np.percentile(all_values, 75)

        # Get the inter quartile range

        iqr = q75 - q25

        # Smaller cutoffs will remove more outliers

        cutoff = iqr * 7

        # Set the bounds of the desired portion to keep

        lower, upper = q25 - cutoff, q75 + cutoff

        

        # If positively correlated...

        # Drop nonfrauds above upper bound, and frauds below lower bound

        if correlation > 0: 

            no_outliers.drop(index=nonfrauds[nonfrauds>upper].index,inplace=True)

            no_outliers.drop(index=frauds[frauds<lower].index,inplace=True)

        

        # If negatively correlated...

        # Drop nonfrauds below lower bound, and frauds above upper bound

        elif correlation < 0: 

            no_outliers.drop(index=nonfrauds[nonfrauds<lower].index,inplace=True)

            no_outliers.drop(index=frauds[frauds>upper].index,inplace=True)

        

print('\nData shape before removing outliers:', balanced.shape)

print('\nCounts of frauds VS non-frauds in previous data:')

print(balanced.Class.value_counts())

print('-'*40)

print('-'*40)

print('\nData shape after removing outliers:', no_outliers.shape)

print('\nCounts of frauds VS non-frauds in new data:')

print(no_outliers.Class.value_counts())
no_outliers.iloc[:,:-1].boxplot(rot=90,figsize=(16,4))

plt.title('Distributions with Less Outliers', fontsize=17)

plt.show()
feat_sel =pd.DataFrame(no_outliers.copy())
# Make a dataframe with the class-correlations before removing outliers

corr_change = pd.DataFrame()

corr_change['correlation']= bal_corr.Class

corr_change['origin']= 'w/outliers'



# Make a dataframe with class-correlations after removing outliers 

corr_other = pd.DataFrame()

corr_other['correlation']= feat_sel.corr().Class

corr_other['origin']= 'no_outliers'



# Join them

corr_change = corr_change.append(corr_other)



plt.figure(figsize=(14,6))

plt.xticks(rotation=90)



# Plot them

sns.set_style('darkgrid')

plt.title('Class Correlation per Feature. With VS W/out Outliers', fontsize=17)

sns.barplot(data=corr_change,x=corr_change.index,y='correlation',hue='origin')

plt.show()
# Feature Selection based on correlation with Class



print('\nData shape before feature selection:', feat_sel.shape)

print('\nCounts of frauds VS non-frauds before feature selection:')

print(feat_sel.Class.value_counts())

print('-'*40)



# Correlation matrix after removing outliers

new_corr = feat_sel.corr()



for col in new_corr.Class.index[:-1]:

    # Pick desired cutoff for dropping features. In absolute-value terms.

    if np.absolute(new_corr.loc['Class',col]) < 0.1:

        # Drop the feature if correlation is below cutoff

        feat_sel.drop(columns=col,inplace=True)



print('-'*40)

print('\nData shape after feature selection:', feat_sel.shape)

print('\nCounts of frauds VS non-frauds in new data:')

print(feat_sel.Class.value_counts())
feat_sel.iloc[:,:-1].boxplot(rot=90,figsize=(16,4))

plt.title('Distribution of Features Selected', fontsize=17)

plt.show()
# Undersample model for efficiency and balance classes.



X_train = feat_sel.drop('Class',1)

y_train = feat_sel.Class



# After feature-selection, X_test needs to include only the same features as X_train

cols = X_train.columns

X_test = X_test[cols]



# Undersample and balance classes

X_train, y_train = RandomUnderSampler(sampling_strategy={1:5000,0:5000}).fit_resample(X_train,y_train)



print('\nX_train shape after reduction:', X_train.shape)

print('\nCounts of frauds VS non-frauds in y_train:')

print(np.unique(y_train, return_counts=True))
# DataFrame to store classifier performance

performance = pd.DataFrame(columns=['Train_Recall','Test_Recall','Test_Specificity'])
# Load simple classifiers

classifiers = [SVC(max_iter=1000),LogisticRegression(),

               DecisionTreeClassifier(),KNeighborsClassifier()]



# Get a classification report from each algorithm

for clf in classifiers:    

    

    # Heading

    print('\n','-'*40,'\n',clf.__class__.__name__,'\n','-'*40)

    

    # Cross-validate on the train data

    print("TRAIN GROUP")

    train_cv = cross_val_score(X=X_train, y=y_train, 

                               estimator=clf, scoring=recall,cv=3)

    print("\nCross-validation recall scores:",train_cv)

    print("Mean recall score:",train_cv.mean())



    # Now predict on the test group

    print("\nTEST GROUP")

    y_pred = clf.fit(X_train, y_train).predict(X_test)

    print("\nRecall:",recall_score(y_test,y_pred))

    

    # Print confusion matrix

    conf_matrix = confusion_matrix(y_test,y_pred)

    sns.heatmap(conf_matrix, annot=True,fmt='d', cmap=plt.cm.copper)

    plt.show()

    

    # Store results

    performance.loc[clf.__class__.__name__+'_default',

                    ['Train_Recall','Test_Recall','Test_Specificity']] = [

        train_cv.mean(),

        recall_score(y_test,y_pred),

        conf_matrix[0,0]/conf_matrix[0,:].sum()

    ]
# Scores obtained

performance
# Parameters to optimize

params = [{

    'solver': ['newton-cg', 'lbfgs', 'sag'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l2']

    },{

    'solver': ['liblinear','saga'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l1','l2']

}]



clf = LogisticRegression(

    n_jobs=-1, # Use all CPU

    class_weight={0:0.1,1:1} # Prioritize frauds

)



# Load GridSearchCV

search = GridSearchCV(

    estimator=clf,

    param_grid=params,

    n_jobs=-1,

    scoring=recall

)



# Train search object

search.fit(X_train, y_train)



# Heading

print('\n','-'*40,'\n',clf.__class__.__name__,'\n','-'*40)



# Extract best estimator

best = search.best_estimator_

print('Best parameters: \n\n',search.best_params_,'\n')



# Cross-validate on the train data

print("TRAIN GROUP")

train_cv = cross_val_score(X=X_train, y=y_train, 

                           estimator=best, scoring=recall,cv=3)

print("\nCross-validation recall scores:",train_cv)

print("Mean recall score:",train_cv.mean())



# Now predict on the test group

print("\nTEST GROUP")

y_pred = best.fit(X_train, y_train).predict(X_test)

print("\nRecall:",recall_score(y_test,y_pred))



# Get classification report

print(classification_report(y_test, y_pred))



# Print confusion matrix

conf_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.copper)

plt.show()

    

# Store results

performance.loc[clf.__class__.__name__+'_search',

                ['Train_Recall','Test_Recall','Test_Specificity']] = [

    train_cv.mean(),

    recall_score(y_test,y_pred),

    conf_matrix[0,0]/conf_matrix[0,:].sum()

]
performance
pd.DataFrame(search.cv_results_).iloc[:,4:].sort_values(by='rank_test_score').head()
# Make a scoring function that improves specificity while identifying all frauds

def recall_optim(y_true, y_pred):

    

    conf_matrix = confusion_matrix(y_true, y_pred)

    

    # Recall will be worth a greater value than specificity

    rec = recall_score(y_true, y_pred) * 0.8 

    spe = conf_matrix[0,0]/conf_matrix[0,:].sum() * 0.2 

    

    # Imperfect recalls will lose a penalty

    # This means the best results will have perfect recalls and compete for specificity

    if rec < 0.8:

        rec -= 0.2

    return rec + spe 

    

# Create a scoring callable based on the scoring function

optimize = make_scorer(recall_optim)
scores = []

for rec, spe in performance[['Test_Recall','Test_Specificity']].values:

    rec = rec * 0.8

    spe = spe * 0.2

    if rec < 0.8:

        rec -= 0.20

    scores.append(rec + spe)

performance['Optimize'] = scores

display(performance)
def score_optimization(params,clf):

    # Load GridSearchCV

    search = GridSearchCV(

        estimator=clf,

        param_grid=params,

        n_jobs=-1,

        scoring=optimize

    )



    # Train search object

    search.fit(X_train, y_train)



    # Heading

    print('\n','-'*40,'\n',clf.__class__.__name__,'\n','-'*40)



    # Extract best estimator

    best = search.best_estimator_

    print('Best parameters: \n\n',search.best_params_,'\n')



    # Cross-validate on the train data

    print("TRAIN GROUP")

    train_cv = cross_val_score(X=X_train, y=y_train, 

                               estimator=best, scoring=recall,cv=3)

    print("\nCross-validation recall scores:",train_cv)

    print("Mean recall score:",train_cv.mean())



    # Now predict on the test group

    print("\nTEST GROUP")

    y_pred = best.fit(X_train, y_train).predict(X_test)

    print("\nRecall:",recall_score(y_test,y_pred))



    # Get classification report

    print(classification_report(y_test, y_pred))



    # Print confusion matrix

    conf_matrix = confusion_matrix(y_test,y_pred)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.copper)

    plt.show()



    # Store results

    performance.loc[clf.__class__.__name__+'_optimize',:] = [

        train_cv.mean(),

        recall_score(y_test,y_pred),

        conf_matrix[0,0]/conf_matrix[0,:].sum(),

        recall_optim(y_test,y_pred)

    ]

    # Look at the parameters for the top best scores

    display(pd.DataFrame(search.cv_results_).iloc[:,4:].sort_values(by='rank_test_score').head())

    display(performance)
# Parameters to optimize

params = [{

    'solver': ['newton-cg', 'lbfgs', 'sag'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l2'],

    'class_weight':[{1:1,0:0.3},{1:1,0:0.5},{1:1,0:0.7}]

    },{

    'solver': ['liblinear','saga'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l1','l2'],

    'class_weight':[{1:1,0:0.3},{1:1,0:0.5},{1:1,0:0.7}]

}]



clf = LogisticRegression(

    n_jobs=-1 # Use all CPU

)



score_optimization(clf=clf,params=params)
# Parameters to optimize

params = {

    'criterion':['gini','entropy'],

    'max_features':[None,'sqrt'],

    'class_weight':[{1:1,0:0.3},{1:1,0:0.5},{1:1,0:0.7}]

    }



clf = DecisionTreeClassifier(

)



score_optimization(clf=clf,params=params)
# Parameters to optimize

params = {

    'kernel':['rbf','linear'],

    'C': [0.3,0.5,0.7,1],

    'gamma':['auto','scale'],

    'class_weight':[{1:1,0:0.3},{1:1,0:0.5},{1:1,0:0.7}]

    }



# Load classifier

clf = SVC(

    cache_size=3000,

    max_iter=1000, # Limit processing time

)

score_optimization(clf=clf,params=params)
# Parameters to compare

params = {

    "n_neighbors": list(range(2,6,1)), 

    'leaf_size': list(range(20,41,10)),

    'algorithm': ['ball_tree','auto'],

    'p': [1,2] # Regularization parameter. Equivalent to 'l1' or 'l2'

}



# Load classifier

clf = KNeighborsClassifier(

    n_jobs=-1

)

score_optimization(clf=clf,params=params)
# Parameters to compare

params = {

    'class_weight':[{1:1,0:0.3},{1:1,0:0.4},{1:1,0:0.5},{1:1,0:0.6},{1:1,0:7}],

    'sampling_strategy':['all','not majority','not minority']

}



# Implement the classifier

clf = BalancedRandomForestClassifier(

    criterion='entropy',

    max_features=None,

    n_jobs=-1

)

score_optimization(clf=clf,params=params)
# Parameters to compare

params = {

    'criterion':['entropy','gini'],

    'class_weight':[{1:1,0:0.3},{1:1,0:0.4},{1:1,0:0.5},{1:1,0:0.6},{1:1,0:7}]

}



# Implement the classifier

clf = RandomForestClassifier(

    n_estimators=100,

    max_features=None,

    n_jobs=-1,

)



score_optimization(clf=clf,params=params)
# Let's get the mean between test recall and test specificity

performance['Mean_RecSpe'] = (performance.Test_Recall+performance.Test_Specificity)/2

performance