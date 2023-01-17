# data analysis
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Configuring plotting visual and sizes
sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

# tools libraries
import random
import math

# Scientific packages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
# Load the datasets
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Check datasets dimensions
print(train_df.shape)
print(test_df.shape)
# Take a look to the first few rows of training dataset
train_df.head()
# Filling Survived column with 0s as a placeholder in test dataset
test_df['Survived'] = 0

# merge two dataset in order to construct full dataset
full = pd.concat([train_df, test_df], 
                 axis=0, # merge on rows
                 ignore_index=True, 
                 sort=False)
# Check dataset again
full.head()
# Renaming dataset columns to increase readbility
full.rename(columns={'Pclass':'TicketClass',
                     'SibSp':'Sibling_Spouse',
                     'Parch':'Parent_Children',
                     'Fare':'TicketFare'}, 
            inplace=True)  # Apply changes on the dataset
# Checking all columns for missed values and types
full.info()
# Checking numerical columns
full.describe()
# Checking categorical columns
full.describe(include=['O'])
full[full.Name.isin(['Connolly, Miss. Kate','Kelly, Mr. James'])].sort_values(by='Name')
# Make a plot instance 
fig, axes = plt.subplots(2,2,figsize=(15,15))

# Drawing plots
sns.countplot(data=train_df, x='Survived', ax=axes[0][0])
sns.countplot(data=full, x='TicketClass', ax=axes[0][1])
sns.countplot(data=full, x='Sex', ax=axes[1][0])
sns.countplot(data=full, x='Embarked', ax=axes[1][1])

# showing frequency percentage on top of each column
for ax in axes.flatten():
    total = len(full) if ax.get_xlabel() != 'Survived' else len(train_df)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('Freq: {:.1f}%'.format(100.*y/total), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
full.loc[:891,['TicketClass', 'Survived']]\
.groupby(['TicketClass'], as_index=False)\
.mean().sort_values(by='Survived', 
                    ascending=False)
full.loc[:891,['Sex', 'Survived']]\
.groupby(['Sex'], as_index=False).mean()\
.sort_values(by='Survived', 
             ascending=False)
full.loc[:891,['Embarked', 'Survived']]\
.groupby(['Embarked'], as_index=False).mean()\
.sort_values(by='Survived', 
             ascending=False)
grid = sns.FacetGrid(full.loc[:891], 
                     col='Survived', 
                     row='TicketClass',
                     aspect=2)
grid.map(plt.hist, 'TicketFare', alpha=0.8)
grid = sns.FacetGrid(full.loc[:891], 
                     col='Survived', 
                     row='Sex',
                     aspect=2)

grid.map(plt.hist, 'Age', alpha=.8)
# Getting rid of useless columns
full.drop(['Cabin','Ticket','PassengerId'], 
          axis=1, # Drop column
          inplace=True)
# Extracting titles out of names
full['Title'] = full['Name'].str.extract('([A-Za-z]+)\.', 
                                         expand=False)
    
# Making a cross-table of Titles and their gender
title_cross = pd.crosstab(full['Title'], 
                          full['Sex'])

title_cross
# Transforming less regular titles to more regulars.
full.Title.replace(['Col','Rev','Sir'],
                   'Mr',
                   inplace=True)

full.Title.replace(['Ms','Mlle'],
                   'Miss',
                   inplace=True)

full.Title.replace('Mme',
                   'Mrs',
                   inplace=True)


# Convert remaining titles as Rare
full.Title.replace([x for x in list(np.unique(full.Title)) \
                    if x not in ['Mr','Miss','Mrs','Master']] ,
                   'Rare',
                   inplace=True)
    
title_report = full.loc[:891,['Title','Survived']]\
.groupby('Title',
         as_index=False).mean()

# Visualization
plt.bar(x=title_report['Title'], 
        height=title_report['Survived'],
        alpha=0.8,color='grkmb')

plt.title('Survival rate based on Titles',
          fontsize=20)
plt.ylabel('Survived Percentage %',
           fontsize=15)

# showing frequency percentage on top of each column
ax = plt.gca()
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2f}%'.format(y), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
# Getting rid of useless columns
full.drop(['Name'], axis=1, inplace=True)
# Checking for columns with NAs
full.isnull().sum()
# Filling NAs in Embarked with the most frequent value.
full.loc[full['Embarked'].isnull(),'Embarked'] = \
full['Embarked'].dropna().mode()[0]
# Filling missed value for ticket fare by mean of the column
full.loc[full['TicketFare'].isnull(),'TicketFare'] = \
full['TicketFare'].dropna().mean()
# First see how many missed value we have in Age column
full.Age.isnull().sum()
# Making correlation matrix of all columns except Survived
corrMatt = full[full.columns.difference(['Survived'])].corr()
mask = np.array(corrMatt)

# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False

# Making the heatmap of correlations
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=.8, 
            square=True,
            annot=True,
            ax=ax)
# Finding mean and standard deviation for age of passengers 
age_estimator = full[['Age','TicketClass','Sibling_Spouse']]\
.groupby(['TicketClass','Sibling_Spouse']).agg(['mean','std'])

# Filling the NAs by making random numbers around their group mean
age_nulls = full.loc[full.Age.isnull(),:]

for idx,rec in age_nulls.iterrows():
    # For each null age calculating a random age based on correlated Ticketclass and Sibling_Spouse columns
    mean = age_estimator.loc[(rec['TicketClass'],rec['Sibling_Spouse']),('Age','mean')]
    std = age_estimator.loc[(rec['TicketClass'],rec['Sibling_Spouse']),('Age','std')]
    gen_age = random.uniform(mean-std, mean+std)
    
    # Convert negative ages to 1
    full.loc[idx,'Age'] = gen_age if gen_age >= 1 else 1

# Transform ages value to upper integer value
full['Age'] = full['Age'].apply(math.ceil)
# Just to make sure there is no more NAs in Age column
full.Age.isnull().sum()
full[['Age', 'TicketFare']].describe()
# Making subplots axes
fig, axes = plt.subplots(2,2,figsize=(12,7))

sns.boxplot(data=full, 
            x='Age', 
            ax=axes[0][0])

sns.countplot(data=full, 
              x='Age', 
              ax=axes[0][1])

sns.boxplot(data=full, 
            x='TicketFare', 
            ax=axes[1][0])

sns.countplot(data=full[['TicketFare']].astype(int), # Only for having less bars
              x='TicketFare', 
              ax=axes[1][1])

# Adjusting xlabelticks to make them more readble
for ax in [axes[0][1],axes[1][1]]:
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontdict={'fontsize':10})
# Defining age ranges
age_bin = [0,2,15,40,55,80]

# Set label for age ranges
age_bin_labels = ['infant','kid','young','mid-age','old']

# Overide numeriuous age value with discrete bins
full['Age'] = pd.cut(np.array(full['Age']), 
                     bins=age_bin, 
                     labels=age_bin_labels)

# Checking value counts in each new generated age range
full.Age.value_counts().sort_values()
# we need to have 3 quantiles
quantile_list = np.linspace(0,1,4,endpoint=True)

# Finding quiatiles in data
fare_quantiles = full['TicketFare'].quantile(quantile_list)
print(fare_quantiles)

# Visualise the binning
fig, ax = plt.subplots(figsize=(15,10))

full[['TicketFare']].astype(int).hist(bins=50,
                                      color='b',
                                      alpha=.5,
                                      ax=ax)

# Drawing quantile lines in red over the histogram
for q in fare_quantiles:
    qvl = plt.axvline(q,color='r')
    
ax.legend([qvl],['Quantiles'],fontsize=18,loc='upper center')
ax.set_title('Ticket Fare Histogram with Quntiles', fontsize=25)
ax.set_xlabel('Ticket Fare', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)
# Using qunatile binning to bin each each of passenger ticket fare
quantile_label = ['Cheap','Regular','Premium']
full['TicketFare'] = pd.qcut(full['TicketFare'],
                             q=quantile_list,
                             labels=quantile_label)

full.TicketFare.value_counts().sort_values()
full.head()
# Calculating family size for each passenger
full['FamilySize'] = full['Sibling_Spouse'] + \
                     full['Parent_Children'] + \
                     1 # include the passeger itself
# Adding a new column for indicating solo-travellers
full.loc[full.FamilySize > 1, 'IsAlone'] = 0
full.loc[full.FamilySize <= 1, 'IsAlone'] = 1
full['IsAlone'] = full['IsAlone'].astype(int)
full.loc[:891,['FamilySize','Survived']]\
.groupby(['FamilySize'],as_index=False)\
.mean().sort_values(by='Survived', ascending=False)
full.loc[:891,['IsAlone','Survived']].groupby(['IsAlone'],as_index=False)\
.mean().sort_values(by='Survived', ascending=False)
# Dropping unnecessary columns after these columns engineerings
full.drop(['Sibling_Spouse','Parent_Children'], axis=1, inplace=True)
# Check dataset again
full.head()
for nom_feature in ['Sex','Embarked','Title']:
    gle = LabelEncoder()
    labels = gle.fit_transform(full[nom_feature])
    report = {index: label for index,label in enumerate(gle.classes_)}
    full[nom_feature] = labels
    print(nom_feature,':',report,'\n','-'*50)
# Mapping ordinal values in Age column
age_ord_map = {'infant':0, 'kid':1, 'young':2, 'mid-age':3, 'old':4}
full.Age = full.Age.map(age_ord_map)

# Mapping ordinal values in TicketFare column
tf_ord_map = {'Cheap':0, 'Regular':1, 'Premium':2}
full.TicketFare = full.TicketFare.map(tf_ord_map)
# Checking dataset
full.head(10)
# encode all the categorical features using one-hot encoding scheme
list_category_features = ['TicketClass','Sex','Age','TicketFare','Embarked','Title']
dummy_features = pd.get_dummies(full[list_category_features], columns=list_category_features)

# Drop all the features before transforming to dummy variables
full.drop(list_category_features, axis=1,inplace=True)

# Merging remaining columns with dummy variables
full = pd.concat([full, dummy_features], axis=1)

# Checking dataset
full.sample(10)
train_df_new = full.iloc[:891]
y = train_df_new['Survived']
X = train_df_new.drop(['Survived'], axis=1)

test_df_new = full.iloc[891:]
test_df_new = test_df_new.drop(['Survived'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42) 
print(X_train.shape, X_test.shape)
# train and build the model
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# Make the prediction values
y_pred = logistic.predict(X_test)

# Checking model score
print('Logistic Regression model score:',
      np.round(logistic.score(X_test, y_test), 3))
def model_report(y_test, y_pred):
    print('Confusion Matrix:\n',
          metrics.confusion_matrix(y_true=y_test,
                                   y_pred=y_pred,
                                   labels=[0, 1]))
    print('{:-^30}'.format('|'))

    print('{:15}{:.3f}'.format('Accuracy:', 
          metrics.accuracy_score(y_test,
                                 y_pred)))

    print('{:-^30}'.format('|'))

    print('{:15}0:{:.3f}|1:{:.3f}'.format('Precision:', 
          metrics.precision_score(y_test,y_pred,average=None)[0],
          metrics.precision_score(y_test,y_pred,average=None)[1]))

    print('{:-^30}'.format('|'))

    print('{:15}0:{:.3f}|1:{:.3f}'.format('Recall:',
          metrics.recall_score(y_test,y_pred,average=None)[0],
          metrics.recall_score(y_test,y_pred,average=None)[1]))


    print('{:-^30}'.format('|'))

    print('{:15}0:{:.3f}|1:{:.3f}'.format('f1-score:',
          metrics.f1_score(y_test,y_pred,average=None)[0],
          metrics.f1_score(y_test,y_pred,average=None)[1]))
    
    print('{:-^30}'.format('|'))
    
model_report(y_test, y_pred)
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
report = classification_report(y_test, predicted, digits=3)
print(report)
from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10, random_state=0)
model = LogisticRegression()

scoring = 'accuracy'
results = model_selection.cross_val_score(model, 
                                          X, 
                                          y, 
                                          cv=kfold, 
                                          scoring=scoring)

print("Average Accuracy: {:.3f}".format(results.mean()))
# Models names
names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Gaussian Process",
         "Decision Tree", 
         "Random Forest", 
         "Neural Net", 
         "AdaBoost",
         "Naive Bayes", 
         "Logistic Regression"]

# Models instances
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression()]

# A placeholder for all results
results = {}

# Iterate over all models and put the score and confusion matrix into the results dictionary.
for name, clf in zip(names, classifiers):
    results[name] = {}
    results[name]['Score'] = model_selection.cross_val_score(clf,
                                                             X, 
                                                             y, 
                                                             cv=kfold, 
                                                             scoring=scoring).mean()
    
    results[name]['Confusion_matrix'] = metrics.confusion_matrix(y,
                                                                 model_selection.cross_val_predict(clf,
                                                                                                   X,
                                                                                                   y,
                                                                                                   cv=kfold))
# Sort models based on accuracy
for clf_name in sorted(names, key=lambda x: (results[x]['Score'])):
    print('{:19} :{:.3f}'.format(clf_name, results[clf_name]['Score']))
fig, axes = plt.subplots(5,2, figsize=(10,15))
# Adjust padding between plots
plt.tight_layout()

counter=0
for clf_name in sorted(names, key=lambda x: (results[x]['Score'])):
    sns.heatmap(results[clf_name]['Confusion_matrix'],
                ax=axes[counter % 5, math.floor(counter / 5)],
                annot=True,
                fmt='2.0f',
                square=True,
                annot_kws={"size": 20},
                cmap="coolwarm")
    axes[counter % 5, math.floor(counter / 5)].set_title(clf_name)
    counter += 1
# Setting values for hyperparameters
hyper_params={'max_depth':range(5,21,5),
              'min_samples_split':range(2,9,2),
              'min_samples_leaf':range(1,6,1),
              'max_leaf_nodes':range(2,11,1)}

grid = GridSearchCV(RandomForestClassifier(random_state=1),
                   param_grid=hyper_params)

grid.fit(X,y)
print('Best Score: {:.4f}'.format(grid.best_score_))

print('Best Parameters setting:',grid.best_params_)
ranfor_model = RandomForestClassifier(random_state=1,
                                      max_depth=10, 
                                      max_leaf_nodes=10, 
                                      min_samples_leaf=2, 
                                      min_samples_split=2)

ranfor_model.fit(X,y)

y_pred = ranfor_model.predict(test_df_new)
submission = pd.DataFrame({
    "PassengerId": range(892,1310),
    "Survived": y_pred
})
submission.to_csv('titanic.csv', index=False)