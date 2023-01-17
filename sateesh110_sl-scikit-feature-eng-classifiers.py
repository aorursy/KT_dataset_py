import pandas as pd

from pandas import Series,DataFrame

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from IPython.display import Image



from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# 4 ML categories: classification, regression, clustering, or dimensionality reduction



# continuous target variable requires a regression algorithm 

# discrete target variable requires a classification algorithm

##logistic regression is a classification algorithm

#https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

#https://scikit-learn.org/stable/user_guide.html



# Regression ML algos:

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression



# Classification ML algos:

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import metrics



# cross validation and scoring metrics are to rank and compare our algorithms’ performance

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict





import statsmodels.api as sm

from statsmodels.formula.api import ols

from sklearn import metrics

from scipy.stats import norm

from scipy import stats

from sklearn.metrics import mean_squared_error, r2_score
# Train set

df_train = pd.read_csv('../input/train.csv')



# Test set

df_test = pd.read_csv('../input/test.csv')
df_train.columns
df_test.columns
# Survived in Train Dataset

df_train['Survived'].value_counts()



# There is no Survived column in Test Dataset
df_train.head(3)
df_test.head(3)
df_train.describe()
df_test.describe()
df_train.info()
df_test.info()
# Nulls in Columns in Train -> True

df_train.isnull().any()
# Number of Null records in each column

df_train.isnull().sum().sort_values(ascending=False)
# Nulls in Columns in Test -> True

df_test.isnull().any()
# Number of Null records in each column

df_test.isnull().sum().sort_values(ascending=False)
sns.catplot(x="Survived", y="Age", kind="swarm",hue="Sex", data=df_train)
sns.catplot(x="Survived", y="Pclass", kind="bar",hue="Sex", data=df_train)
# Combine Test & Train datasets

train_test_data = [df_train, df_test]
# train set

train_test_data[0].isnull().sum()
# test set

train_test_data[1].isnull().sum()
# Create column - Title

for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
# map to Title column in both datasets

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
df_train['Title'].unique()
genders = {"male": 0, "female": 1}



for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(genders)  
df_train['Sex'].value_counts()
df_train['Sex'].unique()
# info on Embarked

df_train['Embarked'].describe()
# Most commonly used value

embrk_common_value = list(df_train["Embarked"].mode())[0]

embrk_common_value
# two null records

df_train['Embarked'].isnull().sum()
# fillna with most common value

df_train['Embarked']= df_train['Embarked'].fillna(embrk_common_value)
# quick check for nulls

print(df_train['Embarked'].isnull().sum())

print(df_test['Embarked'].isnull().sum())
df_train['Embarked'].unique()
def Embark_cat(x):

    if x =='S' :

        return 0

    elif x =='C':

        return 1

    else:

        return 2
df_train['Embarked']= df_train['Embarked'].apply(Embark_cat)

df_test['Embarked']= df_test['Embarked'].apply(Embark_cat)
# quick check 

df_train['Embarked'].value_counts()
df_train['Embarked'].unique()
df_test['Embarked'].unique()
# Median of Fare

fare_med = df_test['Fare'].median()

fare_med
df_test['Fare'].isnull().sum()
# Fill missing Fare record in df_test dataset with median value

df_test['Fare'] = df_test['Fare'].fillna(fare_med)



# Quick check 

df_test['Fare'].isnull().sum()
#Method: use random value in the normal distribution or Bell curve

def assign_random_age_toNan (dataset):

    mean = dataset["Age"].mean()

    std = dataset["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    rand_age = np.random.randint((mean - std), (mean + std), size = is_null)

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age #assign random number

    dataset["Age"] = age_slice

    dataset["Age"] = dataset["Age"].astype(int)

    return dataset 
df_train = assign_random_age_toNan(df_train)

df_train.isnull().sum()
df_test = assign_random_age_toNan(df_test)

df_test.isnull().sum()
# distinct values

age = list(df_train['Age'].value_counts())[0:5]

age
# categorize 'Age' into 1 to 6 bucket.



def Age_cat(x):

    if x <=11 :

        return 0

    elif x>11 and x<=18:

        return 1

    elif x>18 and x<=22:

        return 2

    elif x>22 and x<=27:

        return 3

    elif x>27 and x<=33:

        return 4

    elif x>33 and x<=40:

        return 5

    elif x>40 and x<=66:

        return 6

    else:

        return 6
df_train['Age'] = df_train['Age'].apply(Age_cat)

df_test['Age'] = df_test['Age'].apply(Age_cat)
df_train['Age'].unique()
df_test['Age'].unique()
df_train.isnull().sum()
df_test.isnull().sum()
df_train['With_someone'] = df_train['SibSp'] | df_train['Parch']

df_test['With_someone'] = df_test['SibSp'] | df_test['Parch']
df_train['With_someone'].unique()
df_test['With_someone'].unique()
df_train['With_someone'].value_counts()
df_train['With_someone'] =df_train['With_someone'].apply(lambda x:1 if x >=1 else 0)

df_test['With_someone'] =df_test['With_someone'].apply(lambda x:1 if x >=1 else 0)
df_test['With_someone'].unique()
df_train['With_someone'].unique()
df_train['Family'] = df_train['SibSp'] + df_train['Parch']

df_test['Family'] = df_test['SibSp'] + df_test['Parch']
df_train['Family'].unique()
df_test['Family'].unique()
df_train = pd.get_dummies(df_train, columns=['Pclass', 'Embarked','Age','Title'], drop_first=True)

df_test = pd.get_dummies(df_test, columns=['Pclass', 'Embarked','Age','Title'], drop_first=True)

df_train.head()
drop_cols = ['Name', 'Ticket', 'Cabin','SibSp','Parch']
df_train=df_train.drop(columns=drop_cols, axis=1)

df_test=df_test.drop(columns=drop_cols, axis=1)
df_train.shape, df_test.shape
sc_X = MinMaxScaler()

df_train[['Fare','Family']] = sc_X.fit_transform(df_train[['Fare','Family']])

df_test[['Fare','Family']] = sc_X.transform(df_test[['Fare','Family']])
df_train.head(5)
# Y axis for the "Survived" where as X axis for other attributes in the data model



X_train = df_train.drop(["Survived","PassengerId"], axis=1)

y_train = df_train["Survived"]



X_test  = df_test.drop("PassengerId", axis=1)

y_test = df_test['PassengerId']
logi_clf = LogisticRegression(random_state=0)

logi_parm = {"penalty": ['l1', 'l2'], "C": [0.1, 0.5, 1, 5, 10, 50]}



svm_clf = SVC(random_state=0)

svm_parm = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 1, 5, 10, 50], 'degree': [3, 5, 7], 

            'gamma': ['auto', 'scale']}



dt_clf = DecisionTreeClassifier(random_state=0)

dt_parm = {'criterion':['gini', 'entropy']}



knn_clf = KNeighborsClassifier()

knn_parm = {'n_neighbors':[5, 10, 15, 20], 'weights':['uniform', 'distance'], 'p': [1,2]}



gnb_clf = GaussianNB()

gnb_parm = {'priors':['None']}



clfs = [logi_clf, svm_clf, dt_clf, knn_clf]

params = [logi_parm, svm_parm, dt_parm, knn_parm] 
clf1 = GradientBoostingClassifier(max_depth=4,n_estimators=300,learning_rate=0.01)

clf1.fit(X_train,y_train)



clf2 = SVC(C=5,degree=3,random_state=0,probability=True)

clf2.fit(X_train,y_train)



clf3 = RandomForestClassifier(max_depth=30,min_samples_split =15 ,n_estimators=500,random_state=0)

clf3.fit(X_train,y_train)
eclf = VotingClassifier(estimators=[('gb',clf1),('svc',clf2),('rf',clf3)],voting='soft',weights=[2,1,1.5])
eclf.fit(X_train,y_train)
kf = KFold(n_splits=10, random_state=5)



cv = cross_val_score(eclf,X_train,y_train,cv=kf)



print(cv)
cv.mean()
pred = eclf.predict(X_test)

print(pred)
cols = ['PassengerId', 'Survived']

submit_df = pd.DataFrame(np.hstack((y_test.values.reshape(-1,1),pred.reshape(-1,1))),columns=cols)
submit_df.to_csv('submission.csv', index=False)
submit_df.head()
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)

print( "Prediction of the Survived", predictions)
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.10,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':9 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(df_train)