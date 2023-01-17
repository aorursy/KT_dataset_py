import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#import train and test CSV files

df = pd.read_csv("../input/train.csv")

df.head()
df.info()
df.describe(include='all')
#Look at percentage complete records

pd.isnull(df).sum()/len(df) * 100
#Look at survival rates by characteristics

#Age

df['Age_Groups'] = pd.cut(df['Age'], bins=[0, 1, 5, 15, 20, 40], labels=False)

Age_Groups = {0:'Baby', 1:'Young Child', 2:'Child', 3:'Young Adult', 4:'Adult'}

df['Age_Groups'] = df['Age_Groups'].map(Age_Groups)



#Cabin

df['Cabin_dummy']=(df["Cabin"].notnull().astype('int'))



#Print results

print("Survival Rates:",)

print('\n', df.groupby(['Age_Groups'])['Survived'].mean()*100)

print('\n', df.groupby(['Sex'])['Survived'].mean()*100)

print('\n',df.groupby(['Pclass'])['Survived'].mean()*100)

print('\n',df.groupby(['SibSp'])['Survived'].mean()*100)

print('\n',df.groupby(['Parch'])['Survived'].mean()*100)

print('\n',df.groupby(['Cabin_dummy'])['Survived'].mean()*100)

print('\n',df.groupby(['Embarked'])['Survived'].mean()*100)
sns.distplot(df['Age'][ (~df['Age'].isnull()) & (df['Survived'] == 0)], label='Died')

sns.distplot(df['Age'][ (~df['Age'].isnull()) & (df['Survived'] == 1)], label='Survived')

plt.legend()
sns.distplot(df['Fare'][ df['Survived'] == 0 ], label='Died')

sns.distplot(df['Fare'][ df['Survived'] == 1 ], label='Survived')

plt.legend()
#Embarked - what is the most common port

df['Embarked'].value_counts()
#fill in missing values Emabarked with most common port

df['Embarked'].fillna("S", inplace=True)
#Get the titles

def getTitle(s):

    sStart = s.find(", ")

    sEnd = s.find(".", sStart)

    return s[sStart+1: sEnd].lstrip()



df['title'] = df['Name'].apply(getTitle)

titles = df['title'].value_counts()

titles
# Group the titles into categories.  Need to have a catch all

title_groups = { 'Mr': 1, "Miss": 2, "Mrs":3, "Ms": 4, "Master": 5,

                "Dr": 6, "Don": 6,

                "Rev": 7, 

                "Col": 8, "Major": 8, "Capt": 8,

                "Sir": 9, "Lady": 9, 

                "the Countess": 9, "Countess": 9, "Lord": 9, "the Lord": 9     

                }



def titleMapping(title):

    if title in title_groups:

        return title_groups[title]

    else:

        return 99

        

df['title_cat'] = df['title'].apply(titleMapping)

df['title_cat'].value_counts()
#Create a dictionary of median age with title_cat key. 

#NB use median as few observations in some groups

medianAges = df.groupby(['title_cat'])['Age'].median().to_dict()

#add medianAge column to dataframe

df['medianAge'] = df['title_cat'].map(medianAges)

#replace missing values in Age column

df['Age'].fillna(df['medianAge'], inplace=True)

#check results

df[['Name','Age','title_cat','medianAge']].head(20)
#Map male and female to categorical variable 

sex_mapping = {'male':1, 'female':0}

df['Sex_dummy'] = df['Sex'].map(sex_mapping)



#Cabin has already been made a dummy when calculating survival rates



#Make dummy variables

Age_Group_dummy = pd.get_dummies(df['Age_Groups'])

Embark_dummy = pd.get_dummies(df['Embarked'])



PassengerClass_dummy = pd.get_dummies(df['Pclass'])

PassengerClass_dummy = PassengerClass_dummy.rename(columns={1:'1st_class',2:'2nd_class',3:'3rd_class'})



title_dummy = pd.get_dummies(df['title_cat'])

title_dummy = title_dummy.rename(columns={1: 'Mr_dummy', 2: "Miss_dummy", 3: "Mrs_dummy",

                                          4: "Ms_dummy", 5: "Master_dummy", 

                                          6: "Academic_dummy", 7: "clergy_dummy", 8: "military_dummy",

                                          9: "Royal_dummy", 99: "Other_dummy"})



#Join back onto main dataset

df = pd.concat([df, Age_Group_dummy, Embark_dummy, PassengerClass_dummy, title_dummy], axis=1)



#Drop duplicate variables

df = df.drop(['PassengerId','Cabin','Ticket','Name','Embarked','Sex',

                    'Age_Groups','Pclass','title', 'title_cat', 'medianAge'], 1)



#Drop non-independant variables - knowledge of other variables class provides information

df = df.drop(['C','1st_class', 'Other_dummy'], 1)



#examine dataset

df.head()
#scale values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



#Age

df['Age_sc'] = sc.fit_transform( df['Age'].reshape(-1,1) )

sns.distplot(df['Age_sc'])
#Fare - use log to normalise distribution of Fare as very skewed, 

#use the +1 as 0 values in the Fare

df['Fare_sc'] = sc.fit_transform( np.log10(df['Fare'] + 1 ).reshape(-1,1) )

sns.distplot(df['Fare_sc'])
#Drop unscalled variables

df = df.drop(['Age','Fare'],1)
#imports for train test split

from sklearn.model_selection import train_test_split



#set target and training sets at 20% split

y = df["Survived"].values

X = df.drop(['Survived'],axis=1).values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Import sklearn ML models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron



#Import kfold cross validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
models = []

models.append(("LR", LogisticRegression()))

models.append(("KNN", KNeighborsClassifier()))

models.append(("DT", DecisionTreeClassifier()))

models.append(("RF", RandomForestClassifier()))

models.append(("GBC", GradientBoostingClassifier()))

models.append(("SVM", SVC()))

models.append(("NB", GaussianNB()))

models.append(("SGD", SGDClassifier()))

models.append(("Per", Perceptron()))
#Before calculating each model - establish baseline

baseline = sum(y)/len(y)

print('Probability of living: {}%'.format( round(baseline*100,2)))

print('Probability of dying: {}%'.format( 100 - round(baseline*100,2)))
kfold = KFold(n_splits=5)



averages = []

sds = []

names = [] 

for name, model in models:

    cv_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")

    names.append(name)

    averages.append(round(cv_result.mean()*100,2))

    sds.append(round(cv_result.std(),3))



results, names, sds = zip(*sorted(zip(averages, names, sds), reverse=True))

for r,n, s in zip(results, names, sds): print("Model {}\t Mean {}\t StDev {}".format(n, r, s))
from sklearn.grid_search import GridSearchCV



param_grid_lr = dict(solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])

grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy') 

grid_lr.fit(X_train, y_train)

grid_lr.grid_scores_
param_grid_svm = dict(kernel = [ 'linear', 'poly', 'rbf', 'sigmoid'],

                      shrinking = [True, False])

grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy') 

grid_svm.fit(X_train, y_train)

grid_svm.grid_scores_
from sklearn import metrics

from sklearn.metrics import confusion_matrix, roc_curve

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):

    cm = confusion_matrix(y_true, y_pred)

    pred_labels = ['Predicted '+ l for l in labels]

    df = pd.DataFrame(cm, index=labels, columns=pred_labels)

    return df
#Confusion matrix for logistic regression

lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred_lr =  lr.predict(X_test)

y_pred_prob_lr = lr.predict_proba(X_test)[:, 1]



pretty_confusion_matrix(y_test, y_pred_lr, ['Died', 'Survived'])
print("Accuracy:\t{:0.3f}".format(accuracy_score(y_test, y_pred_lr)*100))

print("Precision:\t{:0.3f}".format(precision_score(y_test, y_pred_lr)*100))

print("Recall:  \t{:0.3f}".format(recall_score(y_test, y_pred_lr)*100))

print("F1 Score:\t{:0.3f}".format(f1_score(y_test, y_pred_lr)*100))
#Confusion matrix for SVM

svm = SVC(probability=True)

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

y_pred_prob_svm = svm.predict_proba(X_test)[:, 1]



pretty_confusion_matrix(y_test, y_pred_svm, ['Died', 'Survived'])
print("Accuracy:\t{:0.3f}".format(accuracy_score(y_test, y_pred_svm)*100))

print("Precision:\t{:0.3f}".format(precision_score(y_test, y_pred_svm)*100))

print("Recall:  \t{:0.3f}".format(recall_score(y_test, y_pred_svm)*100))

print("F1 Score:\t{:0.3f}".format(f1_score(y_test, y_pred_svm)*100))
fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_test, y_pred_prob_lr)

fpr_svm, tpr_svm, thresholds_svm = metrics.roc_curve(y_test, y_pred_prob_svm)

plt.plot(fpr_lr, tpr_lr, label="LR")

plt.plot(fpr_svm, tpr_svm, label="SVM")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Titanic dataset')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.grid(True)
print("Logistic Regression:\t Area Under Curve\t {}"

              .format(metrics.roc_auc_score(y_test, y_pred_prob_lr)))

print("Support Vector Machine:\t Area Under Curve\t {}"

              .format(metrics.roc_auc_score(y_test, y_pred_prob_svm)))
#read in test data

test = pd.read_csv('../input/train.csv')

ids = test['PassengerId']

test.drop(['PassengerID'], axis=1)



def apply_mappings(df):

    df['Age_Groups'] = pd.cut(df['Age'], bins=[0, 1, 5, 15, 20, 40], labels=False)

    df['Sex'] = df['Sex'].map(sex_mapping)

    df['Cabin']=(df["Cabin"].notnull().astype('int'))

    df['Age_Groups'] = df['Age_Groups'].map(Age_Groups)

    df['title_cat'] = df['Name'].apply(getTitle).apply(titleMapping)



    df['medianAge'] = df['title_cat'].map(medianAges)

    df['Age'].fillna(df['medianAge'], inplace=True)

    df.drop(['medianAge'], axis=1)

    return df



test = apply_mapping(test)
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin



class SeriesImputer(TransformerMixin):

    def __init__(self):

        """

        Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.

        else impute with the mean.  

        """

        

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)

        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)

    

    

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        return X[self.attribute_names]





num_cols = ['Age', 'Fare']

cat_cols = [x for x in test.columns if x not in num_cols]



#numerical pipeline

pipe_num = Pipeline(steps=[ ('selector', DataFrameSelector(num_cols)),

                            ('impute', Imputer(strategy='median', axis=0)),

                            ('scale', StandardScaler())

                            ])

    

#categorical pipeline

pipe_cat  = Pipeline(steps=[('selector', DataFrameSelector(cat_cols)),

                            ('impute_cat', SeriesImputer()),

                            ('encoder', LabelBinarizer())

                            ])



full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", pipe_num),

                                              ("cat_pipeline", pipe_cat)

                                              ])

#finalised prepared data

X = full_pipeline.fit_transform(test)



#make model predictions

predictions = sv.predict(X)
#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)