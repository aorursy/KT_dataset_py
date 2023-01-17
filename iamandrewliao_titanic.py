# I want to thank the many other Kaggle coders whose notebooks helped a lot, including: 

# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook

# https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import re

import warnings

from statistics import mode

from sklearn import preprocessing

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn import feature_selection, model_selection, metrics

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data["Sex"].replace({"male": 0, "female": 1}, inplace=True)

# another way to do it: train_data['Sex'][train_data['Sex'] == 'male'] = 0

test_data["Sex"].replace({"male": 0, "female": 1}, inplace=True)



#https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()



temp = pd.DataFrame(encoder.fit_transform(train_data[['Sex']]).toarray(), columns=['Male', 'Female'])

train_data = train_data.join(temp)

train_data.drop(columns='Sex', inplace=True)



temp = pd.DataFrame(encoder.fit_transform(test_data[['Sex']]).toarray(), columns=['Male', 'Female'])

test_data = test_data.join(temp)

test_data.drop(columns='Sex', inplace=True)
print("Train Data length = ", len(train_data))

for col in train_data.columns:

    print('Train Data {} missing values: {}'.format(col, train_data[col].isnull().sum()))

print('\n')

print("Test Data length = ", len(test_data))

for col in test_data.columns:

    print('Test Data {} missing values: {}'.format(col, test_data[col].isnull().sum()))
# I am dropping Ticket because I don't think it's relevant

train_data.drop(['Ticket'], 1, inplace=True)

test_data.drop(['Ticket'], 1, inplace=True)



# Cabin might be worth exploring

# we know exactly how the titanic sank and where the staircases were

# those closest to staircases might have had a better chance at survival

train_data['Deck'] = train_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

# sns.barplot(x = 'Deck', y = 'Survived', data=train_data) # to decide what cabins to group together

# train_data['Deck'].unique() # tells us what columns to make when we encode

# print(train_data[train_data['Deck']=='T']) # there's only one guy with a T cabin and he's first class

# I know from research that A, B, and C were 1st class cabins so we'll group him with them

train_data['Deck'] = train_data['Deck'].replace('T', 'A')

#train_data['Deck'] = train_data['Deck'].replace(['D', 'E'], 'DE')

#train_data['Deck'] = train_data['Deck'].replace(['F', 'G'], 'FG')



temp = pd.DataFrame(encoder.fit_transform(train_data[['Deck']]).toarray(), columns=['A','B','C','D','E','F','G','M'])

train_data = train_data.join(temp)

train_data.drop(['Cabin','Deck','M'], 1, inplace=True)



test_data['Deck'] = test_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

# test_data['Deck'].unique() # tells us what columns to make when we encode

#test_data['Deck'] = test_data['Deck'].replace(['A', 'B', 'C'], 'ABC')

#test_data['Deck'] = test_data['Deck'].replace(['D', 'E'], 'DE')

#test_data['Deck'] = test_data['Deck'].replace(['F', 'G'], 'FG')



temp = pd.DataFrame(encoder.fit_transform(test_data[['Deck']]).toarray(), columns=['A','B','C','D','E','F','G','M'])

test_data = test_data.join(temp)

test_data.drop(['Cabin','Deck','M'], 1, inplace=True)
#train_data.drop(['Ticket','Cabin'], 1, inplace=True)

#test_data.drop(['Ticket','Cabin'], 1, inplace=True)
train_data['Embarked'] = train_data['Embarked'].fillna(mode(train_data['Embarked']))
test_data.head()
plt.figure(figsize=(15, 11))



plt.subplot(211)

sns.heatmap(train_data.corr(), annot=True)



plt.subplot(212)

sns.heatmap(test_data.corr(), annot=True)



plt.tight_layout()
# as we saw earlier, Age has some NaNs but not THAT many so we'll fill them

# seems like Pclass is the feature most correlated with Age in both train and test data

# what we will do is replace NaN ages by the median age of passengers within the same Pclass

# I also did some research and 

train_data.loc[train_data.Age.isnull(), 'Age'] = train_data.groupby("Pclass").Age.transform('median')

test_data.loc[test_data.Age.isnull(), 'Age'] = test_data.groupby("Pclass").Age.transform('median')



#test data has one Fare null value

test_data.loc[test_data.Fare.isnull(), 'Fare'] = test_data.groupby("Pclass").Fare.transform('median')

#another way to do it is

#test_data['Fare']  = test_data.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))



#train_data.isnull().sum()
# Outlier detection 

from collections import Counter



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train_data,2,["Age","SibSp","Parch","Fare"])



train_data = train_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_data["family_size"] = train_data["SibSp"]+train_data["Parch"]+1

test_data["family_size"] = test_data["SibSp"]+test_data["Parch"]+1



train_data.drop(['SibSp', 'Parch'], 1, inplace=True)

test_data.drop(['SibSp', 'Parch'], 1, inplace=True)



#I imagine being alone decreases your odds of survival by a lot as opposed to having just one family member

train_data['IsAlone'] = 1 #initialize to 1/yes

train_data['IsAlone'].loc[train_data['family_size'] > 1] = 0

test_data['IsAlone'] = 1 #initialize to 1/yes

test_data['IsAlone'].loc[test_data['family_size'] > 1] = 0
plt.subplot(131)

plt.hist(x = [train_data[train_data['Survived']==1]['Fare'], train_data[train_data['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare')

plt.ylabel('# of Passengers')



plt.subplot(132)

plt.hist(x = [train_data[train_data['Survived']==1]['Age'], train_data[train_data['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age')

plt.ylabel('# of Passengers')



plt.subplot(133)

plt.hist(x = [train_data[train_data['Survived']==1]['family_size'], train_data[train_data['Survived']==0]['family_size']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size')

plt.ylabel('# of Passengers')



plt.tight_layout()
#we will use seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html



#graph individual features by survival

fig, ax = plt.subplots(2, 3, figsize=(16,10))



sns.barplot(x = 'Embarked', y = 'Survived', data=train_data, ax = ax[0,0])



sns.barplot(x = 'Pclass', y = 'Survived', data=train_data, order=[1,2,3], ax = ax[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', data=train_data, order=[1,0], ax = ax[0,2])



sns.violinplot(x = 'Survived', y = 'Fare',  data=train_data, ax=ax[1,0])

sns.violinplot(x = 'Survived', y = 'Age',  data=train_data, ax=ax[1,1])

sns.violinplot(x = 'Survived', y = 'family_size', data=train_data, ax=ax[1,2])
train_data['Title'] = train_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test_data['Title'] = test_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

title_names = (train_data['Title'].value_counts() < 10) #this will create a true false series with title name as index

train_data['Title'] = train_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

title_names = (test_data['Title'].value_counts() < 10) #this will create a true false series with title name as index

test_data['Title'] = test_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

temp = pd.DataFrame(encoder.fit_transform(train_data[['Title']]).toarray(), columns=train_data.Title.unique())

train_data = train_data.join(temp)

#train_data['Miss/Ms/Mlle/Dona/Countess']=train_data['Miss']+train_data['Ms']+train_data['Mlle']+train_data['the Countess']

#train_data['Mrs/Mme/Lady']=train_data['Mrs']+train_data['Mme']+train_data['Lady']

#train_data.drop(['Title','Name','Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms','Major', 'Lady', 'Sir', 'Mlle', 'Col', 'the Countess','Jonkheer'], 1, inplace=True)

#train_data.head()

train_data.drop(['Title','Name','Misc'],axis=1,inplace=True)

#test_data['Title'].unique()

temp = pd.DataFrame(encoder.fit_transform(test_data[['Title']]).toarray(), columns=test_data.Title.unique())

test_data = test_data.join(temp)

#test_data['Miss/Ms/Mlle/Dona/Countess']=test_data['Miss']+test_data['Ms']+test_data['Dona']

#test_data['Mrs/Mme/Lady']=test_data['Mrs']

#test_data.drop(['Title','Name','Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Col', 'Rev', 'Dr', 'Dona'], 1, inplace=True)

#test_data.head()

test_data.drop(['Title','Name','Misc'],axis=1,inplace=True)
# let's examine survival rates by family size; maybe we can turn it into bins

sorted_family_sizes = np.sort(train_data['family_size'].unique()) # because it is of type nparray

for i in sorted_family_sizes:

    survival_rate = len(train_data.loc[(train_data['family_size']==i)&(train_data['Survived']==1)])/len(train_data.loc[train_data['family_size']==i])

    print('Family Size {}: {} instances, {} survival rate'.format(i,len(train_data.loc[train_data['family_size']==i]),round(survival_rate, 2)))



sns.countplot(x='family_size', hue='Survived', data=train_data).set_title('Survival by family size')
'''

family_map = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 11: 3}

train_data['family_size_bins'] = train_data['family_size'].map(family_map)

test_data['family_size_bins'] = test_data['family_size'].map(family_map)



train_data.drop('family_size',1,inplace=True)

test_data.drop('family_size',1,inplace=True)



train_data.head()

'''
train_data["Fare"] = train_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

test_data["Fare"] = test_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
# As we see in the violin plots above, having an age under 16ish or a fare above 90ish increases one's chance of survival

# So, we might want to turn Age and Fare into bins with these cutoffs in mind

# Here, we try to match the quantiles with either the upper or lower cutoffs. This will tell me how many bins to set (1/quantile if lower cutoff and 1/(1-quantile) if upper).

#np.quantile(train_data['Age'], .125) # for instance, we're aiming for 16, the lower cutoff, so the number of bins = 1/.125 = 8

#np.quantile(train_data['Fare'], .93) # here we're aiming for 90, the upper cutoff, so the number of bins = 1/(1-.93) = 14
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()



'''

train_data['FareBin'] = pd.qcut(train_data['Fare'], 10)

test_data['FareBin'] = pd.qcut(test_data['Fare'], 10)



train_data['AgeBin'] = pd.qcut(train_data['Age'].astype(int), 6)

test_data['AgeBin'] = pd.qcut(test_data['Age'].astype(int), 6)



train_data['AgeBin_Code'] = label.fit_transform(train_data[['AgeBin']])

test_data['AgeBin_Code'] = label.fit_transform(test_data[['AgeBin']])



train_data['FareBin_Code'] = label.fit_transform(train_data[['FareBin']])

test_data['FareBin_Code'] = label.fit_transform(test_data[['FareBin']])



train_data.drop(['Age','Fare','FareBin','AgeBin'], 1, inplace=True)

test_data.drop(['Age','Fare','FareBin','AgeBin'], 1, inplace=True)

'''
# just like I did with sex, I make unique columns for the categorical values in Embarked and in Pclass



#temp = pd.DataFrame(encoder.fit_transform(train_data[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])

#train_data = train_data.join(temp)

train_data.drop(['Embarked'], 1, inplace=True)



#temp = pd.DataFrame(encoder.fit_transform(test_data[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])

#test_data = test_data.join(temp)

test_data.drop(['Embarked'], 1, inplace=True)



temp = pd.DataFrame(encoder.fit_transform(train_data[['Pclass']]).toarray(), columns=['Pclass_1', 'Pclass_2', 'Pclass_3'])

train_data = train_data.join(temp)

train_data.drop(['Pclass'], 1, inplace=True)



temp = pd.DataFrame(encoder.fit_transform(test_data[['Pclass']]).toarray(), columns=['Pclass_1', 'Pclass_2', 'Pclass_3'])

test_data = test_data.join(temp)

test_data.drop(['Pclass'], 1, inplace=True)
train_data.head()
test_data.head()
'''

plt.figure(figsize=(25, 21))



plt.subplot(211)

sns.heatmap(train_data.corr(), annot=True)



plt.subplot(212)

sns.heatmap(test_data.corr(), annot=True)



plt.tight_layout()

'''
'''

#training our model using train_data

X_train = train_data.drop(['PassengerId', "Survived"], 1)

Y_train = train_data.Survived



MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 )



#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Test Accuracy Mean']

MLA_compare = pd.DataFrame(columns = MLA_columns)



#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

#    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

    cv_results = model_selection.cross_validate(alg, X_train, Y_train, cv=cv_split)



#    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

#    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean() # only possible if cross_validate parameter return_train_score = True

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   



    alg.fit(X_train, Y_train)

    row_index+=1

    

#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare

'''
X_train = train_data.drop(['PassengerId', "Survived"], 1)

Y_train = train_data.Survived



clf = ensemble.RandomForestClassifier(random_state=1)

clf.fit(X_train, Y_train)



importances_df = pd.DataFrame(clf.feature_importances_, columns=['Feature Importance'], index=X_train.columns)

importances_df.sort_values(by=['Feature Importance'], ascending=False, inplace=True)

print(importances_df)


parameters = { 

    'n_estimators': [100, 400],

    'criterion' : ['gini', 'entropy'],

    'max_depth' : [2, 4, 6]    

}



from sklearn.model_selection import GridSearchCV, cross_val_score



cv = GridSearchCV(estimator = clf, param_grid = parameters, cv=5, n_jobs=-1)

#scores=cross_val_score(cv,X_train,Y_train,scoring='accuracy',cv=5, n_jobs=-1)

#np.mean(scores)

'''

X_train = train_data.drop(['PassengerId', "Survived"], 1)

Y_train = train_data.Survived



clf_svc = svm.SVC(probability=True)

parameters = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 10]}



from sklearn.model_selection import GridSearchCV, cross_val_score



cv_svc = GridSearchCV(estimator = clf_svc, param_grid = parameters, cv=5, n_jobs=-1)

#scores=cross_val_score(cv_svc,X_train,Y_train,scoring='accuracy',cv=5, n_jobs=-1)

#np.mean(scores)

'''
#from sklearn.preprocessing import StandardScaler



#X_train = preprocessing.scale(X_train)

#X_train = StandardScaler().fit_transform(train_data.drop(['PassengerId', "Survived"], 1))

#Y_train = train_data.Survived

#X_test = StandardScaler().fit_transform(test_data.drop('PassengerId', 1))

#X_test = preprocessing.scale(X_test)



#X_train = train_data.drop(['PassengerId', "Survived"], 1)

#Y_train = train_data.Survived

X_test = test_data.drop('PassengerId', 1)



clf = cv.fit(X_train, Y_train)



predictions = clf.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")