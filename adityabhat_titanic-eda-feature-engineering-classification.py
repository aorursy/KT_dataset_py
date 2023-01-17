# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import altair as alt

import re



from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.feature_selection import RFE, RFECV, SelectFromModel

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, precision_recall_curve, log_loss

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MinMaxScaler, OneHotEncoder

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import shuffle

from xgboost import XGBClassifier



#from sklearn.preprocessing import (Imputer, StandardScaler, MinMaxScaler) 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
# Lets create copies of dataset to add new features

test_df = test.copy()

train_df = train.copy()
train.info()
test.info()
train.head()
train['Survived'].value_counts(normalize=True)
train['Pclass'].value_counts()
sns.catplot(x='Pclass', y='Survived', kind='bar', data=train)
train_df['LastName'] = train['Name'].str.split(r",", expand=True, n=1).get(0)

test_df['LastName'] = test['Name'].str.split(r",", expand=True, n=1).get(0)

train_df['Title'] = train.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)

test_df['Title'] = test.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)
sns.catplot(x='Title', y='Survived', data=train_df, kind='bar')

plt.xticks(rotation=90);
def title_transform(x):

    if x == 'Mr':

        return x

    elif x in ['Mrs', 'Miss', 'Mme','Ms','Lady', 'Mlle', 'the Countess']:

        return 'Ms'

    elif x == 'Master':

        return x

    else:

        return 'Rare'
train_df['Title'] = train_df.Title.apply(title_transform)

test_df['Title'] = test_df.Title.apply(title_transform)
train_df['Title'].value_counts()
sns.catplot(x='Title', y='Survived', data=train_df, kind='bar')
train['Sex'].value_counts(normalize=True)
sns.catplot(x='Sex', y='Survived', kind='bar', data=train)
sns.catplot(x='Sex', y='Survived', kind='point', data=train)
train['SibSp'].value_counts()
sns.catplot(x='SibSp', y='Survived', kind='bar', data=train)
# First scheme

train_df['sibsp1'] = train_df['SibSp'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))

test_df['sibsp1'] = test_df['SibSp'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))
# Second scheme

train_df['sibsp2'] = train_df['SibSp'].apply(lambda x: 0 if x == 0 else (1 if (x==1 | x==2) else 2))

test_df['sibsp2'] = test_df['SibSp'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))
sns.catplot(x='sibsp2', y='Survived', kind='bar', data=train_df)
train['Parch'].value_counts()
sns.catplot(x='Parch', y='Survived', kind='bar', data=train)
train_df['parch'] = train_df['Parch'].apply(lambda x: 0 if x == 0 else (1 if (x==1 | x==2) else 2))

test_df['parch'] = test_df['Parch'].apply(lambda x: 0 if x == 0 else (1 if (x==1 | x==2) else 2))
sns.catplot(x='parch', y='Survived', kind='bar', data=train_df)
train['Ticket'].tail(20)
train['ticket_prefix'] = train['Ticket'].str.extract(r"(\D+)", expand=True)

train['ticket_prefix'].fillna('X')

train['ticket_prefix'].value_counts()
sns.catplot(x='Survived', y='Fare', kind='box', data=train)
sns.distplot(train.Fare, kde=False, bins=20)
sns.distplot(np.log1p(train.Fare), kde=True, bins=20)
nbins = np.arange(0,700,100)

g = sns.kdeplot(train.loc[train.Survived == 0, 'Fare'], color="red", label="Not Survived", shade = True)

g = sns.kdeplot(train.loc[train.Survived == 1, 'Fare'], color="green", label="Survived", shade = True)

g.set_xlim(0,100);

g.set_xticks(nbins);
nbins = np.arange(0,720,20)

hist_kws = {'bins': nbins}

hist_kws={"rwidth":0.9, 'alpha':0.3, 'bottom': 1}

g = sns.distplot(train.loc[train.Survived == 0, 'Fare'], bins=nbins, kde=False, color="red", label="Not Survived", hist_kws=hist_kws)

g = sns.distplot(train.loc[train.Survived == 1, 'Fare'], bins=nbins, kde=False, color="green", label="Survived", hist_kws=hist_kws)

g.set_xlim(0,100);

g.set_xticks(nbins);

g.set_xticklabels(nbins, rotation=90);
train['Fare'] = np.log1p(train['Fare'])

test['Fare'] = np.log1p(test['Fare'])
# Lets use the log of Fare as its distribution is skewed.

train_df['log_fare'] = np.log1p(train['Fare'])

test_df['log_fare'] = np.log1p(test['Fare'])
train['Cabin'].value_counts()
train['Cabin'].head(10)
# Cabin

# 0: Cabin absent; 1:Cabin present

train_df['has_cabin'] = train_df.Cabin.notna().astype(int)

test_df['has_cabin'] = test_df.Cabin.notna().astype(int)
train_df['has_cabin'].value_counts()
sns.catplot(x='has_cabin', y='Survived', kind='bar', data=train_df)
train['Embarked'].value_counts()
sns.catplot(x='Embarked', y='Survived', kind = 'bar', data=train)
nbins = np.arange(0,105,5)

g = sns.kdeplot(train.loc[train.Survived == 0, 'Age'], color="red", label="Not Survived", shade = True)

g = sns.kdeplot(train.loc[train.Survived == 1, 'Age'], color="green", label="Survived", shade = True)

g.set_xlim(0,100);

g.set_xticks(nbins);
nbins = np.arange(0,105,5)

hist_kws = {'bins': nbins}

hist_kws={"rwidth":0.9, 'alpha':0.3, 'bottom': 1}

g = sns.distplot(train.loc[train.Survived == 0, 'Age'], bins=nbins, kde=False, color="red", label="Not Survived", hist_kws=hist_kws)

g = sns.distplot(train.loc[train.Survived == 1, 'Age'], bins=nbins, kde=False, color="green", label="Survived", hist_kws=hist_kws)

g.set_xlim(0,100);

g.set_xticks(nbins);
sns.catplot(x="Survived", y="Age", kind="box", data=train);
# Bucketize the Age: 0-5; 5-35; 35-50; 50+

def age_bucket(age):

    if 0 <= age < 10:

        return 0

    elif 10 <= age < 35:

        return 1

    elif 35 <= age < 50:

        return 2

    else:

        return 3
train_df['Family_Size'] = train['SibSp']+ train['Parch'] 



test_df['Family_Size'] = test['SibSp']+ test['Parch'] 
train['Family_Size'].value_counts()
sns.catplot(x='Family_Size', y='Survived',kind='bar' ,data=train)
train_df['familysize'] = train.Family_Size.apply(lambda x: 0 if x==0 else (1 if x in [1,2,3] else 2) )

test_df['familysize'] = test.Family_Size.apply(lambda x: 0 if x==0 else (1 if x in [1,2,3] else 2) )
sns.catplot(x='familysize', y='Survived',kind='bar' ,data=train_df)
train_df['is_alone'] = train['Family_Size'].eq(0).astype(int)

test_df['is_alone'] = test['Family_Size'].eq(0).astype(int)
train_df['is_alone'].value_counts()
sns.catplot(x="is_alone", y="Survived", kind="bar", data=train_df);
train_df.info()
features_to_use = ['Survived', 'Pclass', 'Sex', 'Age','SibSp','sibsp2', 'sibsp1','Parch','parch','Fare','log_fare', 'Embarked', 'Title','has_cabin','Family_Size','familysize','is_alone']

categoricals = ['sibsp1','sibsp2','parch','Embarked','Title','familysize']
train_final = train_df[features_to_use]

test_final = test_df[features_to_use[1:]]
train_final[categoricals] = train_final[categoricals].astype('object')

test_final[categoricals] = test_final[categoricals].astype('object')
num_attribs = train_final.drop(columns='Survived').select_dtypes(include=['float64', 'int64']).columns.to_list()
cat_attribs = train_final.select_dtypes(include=['object']).columns.to_list()
train_final.dtypes
print(num_attribs )

print(cat_attribs)
train_final = train_final.sample(frac=1)

y = train_final.pop('Survived')
num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

])



cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('ohe', OneHotEncoder(handle_unknown="ignore")),

])



full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attribs),

    ('cat', cat_pipeline, cat_attribs)

])
test_final.head()
train_final.head()
X = full_pipeline.fit_transform(train_final)

X_test = full_pipeline.transform(test_final)
X_test.shape


# Evaluation metrics

def classification_metrics(y_true,y_pred):

    return {'Accuracy': accuracy_score(y_true, y_pred),

            'Precision': precision_score(y_true, y_pred),

            'Recall': recall_score(y_true, y_pred),

            'F1': f1_score(y_true, y_pred),

            #'AUC': auc(y_true, y_pred),

            'Log Loss': log_loss(y_true, y_pred)

           }



# Cross Validation



def cross_validation(clf, X, y, scoring='accuracy', cv=10):

    scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)

    return {

        'Scores': scores,

        'Mean': scores.mean(),

        'Standard Deviation': scores.std()

    }

    
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.25, random_state=42)
lr_clf = LogisticRegression()

lr_clf.fit(X_train_new, y_train)



print("==========Training Perfromace==============")

y_train_pred = lr_clf.predict(X_train)

metrics = classification_metrics(y_train, y_train_pred)

print(metrics)



print('\n')



print("==========Validation Perfromace==============")

y_val_pred = lr_clf.predict(X_val)

metrics = classification_metrics(y_val, y_val_pred)

print(metrics) 



print('\n')



# Cross Validation

scores = cross_validation(lr_clf, X_train, y_train, scoring='accuracy', cv=5)

print(scores)
# Feature Selection 

model = SelectFromModel(lr_clf, prefit=True)

X_train_new = model.transform(X_train)

X_val_new = model.transform(X_val)

X_test_new = model.transform(X_test)

X_test_new.shape

# Hyperparameter optimization



param_grid =[

    {'penalty': ['none','l2'],

     'C': [0.001,0.01,0.1,1]

     }

]



clf = LogisticRegression()



grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=True)



grid_search.fit(X_train, y_train)
grid_search.best_params_
# Precision Recall Curve



y_scores = cross_val_predict(lr_clf, X_train, y_train, cv=5,method="predict_proba")

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores[:,1])



# Plot Curve

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "r--", label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.legend(loc="center right", fontsize=16) # Not shown in the book

    plt.xlabel("Threshold", fontsize=16)        # Not shown

    plt.grid(True)                              # Not shown

    #plt.axis([-50000, 50000, 0, 1])  

    

plt.figure(figsize=(8, 4))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
y_scores[:,1]
sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train)



print("==========Training Performance==============")

y_train_pred = sgd_clf.predict(X_train)

eval = eval_metrics(y_train_pred, y_train)

print('\n')

print("==========Validation Performance==============")

y_val_pred = sgd_clf.predict(X_val)

eval = eval_metrics(y_val_pred,y_val)

print('\n')

print("==========Cross Validation Results ================================")

scores = cross_val_score(sgd_clf, X_train, y_train,scoring='accuracy', cv=5)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard Deviation: ", scores.std())

# Precision Recall Curve



y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3,method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)



# Plot Curve

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.legend(loc="center right", fontsize=16) # Not shown in the book

    plt.xlabel("Threshold", fontsize=16)        # Not shown

    plt.grid(True)                              # Not shown

    #plt.axis([-50000, 50000, 0, 1])  

    

plt.figure(figsize=(8, 4))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# Precision vs Recall

def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])

    plt.grid(True)



plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions, recalls)

plt.plot([0.4368, 0.4368], [0., 0.9], "r:")

plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")

plt.plot([0.4368], [0.9], "ro")

#save_fig("precision_vs_recall_plot")

plt.show()
dt_clf = DecisionTreeClassifier(max_depth=3, max_features='sqrt', min_samples_leaf=25, min_samples_split=35)

dt_clf.fit(X, y)



print("==========Training Performance==============")

y_train_pred = dt_clf.predict(X_train)

eval = eval_metrics(y_train_pred, y_train)

print('\n')

print("==========Validation Performance==============")

y_val_pred = dt_clf.predict(X_val)

eval = eval_metrics(y_val_pred,y_val)

print('\n')

print("==========Cross Validation Results ================================")

scores = cross_val_score(dt_clf, X_train, y_train,scoring='accuracy', cv=5)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard Deviation: ", scores.std())
param_grid = [

    {'max_depth':[3], 'max_features': ['sqrt'], 'min_samples_split':[30,35,40,45], 

     'min_samples_leaf':[25]}

]



clf = DecisionTreeClassifier()



grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=True)



grid_search.fit(X_train, y_train)
grid_search.best_params_
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=42)

rf_clf.fit(X_train, y_train)



print("==========Training Perfromace==============")

y_train_pred = rf_clf.predict(X_train)

metrics = classification_metrics(y_train, y_train_pred)

print(metrics)



print('\n')



print("==========Validation Perfromace==============")

y_val_pred = rf_clf.predict(X_val)

metrics = classification_metrics(y_val, y_val_pred)

print(metrics) 



print('\n')



# Cross Validation

scores = cross_val_score(rf_clf, X_train, y_train, scoring='accuracy', cv=5)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard Deviation: ", scores.std())

scores = cross_val_score(rf_clf, X_train, y_train,scoring='accuracy', cv=5)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard Deviation: ", scores.std())
gb_clf = GradientBoostingClassifier(random_state=4)

gb_clf.fit(X_train, y_train)



print("==========Training Perfromace==============")

y_train_pred = gb_clf.predict(X_train)

metrics = classification_metrics(y_train, y_train_pred)

print(metrics)



print('\n')



print("==========Validation Perfromace==============")

y_val_pred = gb_clf.predict(X_val)

metrics = classification_metrics(y_val, y_val_pred)

print(metrics) 



print('\n')



# Cross Validation

scores = cross_val_score(gb_clf, X_train, y_train, scoring='accuracy', cv=5)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard Deviation: ", scores.std())

param_grid1  = {'n_estimators':range(10,81,10)}

param_grid2 = {'max_depth':range(2,16,2), 'min_samples_split':range(30,150,30)}



clf = GradientBoostingClassifier(n_estimators = 70,

                                 learning_rate=0.1, 

                                 #min_samples_split=50,

                                 min_samples_leaf=50,

                                 #max_depth=3,

                                 max_features='sqrt',

                                 subsample=0.8,

                                 random_state=10)





grid_search = GridSearchCV(clf, param_grid2, cv=5, scoring='accuracy', return_train_score=True)

grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_
grid_search.cv_results_
len(X_train)
class MyXGBClassifier(XGBClassifier):

    @property

    def coef_(self):

        return None


xgb_clf = MyXGBClassifier(max_depth=2, subsample=0.5, n_estimators=100, learning_rate=0.1)

xgb_clf.fit(X_train_new, y_train)



print("==========Training Perfromace==============")

y_train_pred = xgb_clf.predict(X_train)

metrics = classification_metrics(y_train, y_train_pred)

print(metrics)



print('\n')



print("==========Validation Perfromace==============")

y_val_pred = xgb_clf.predict(X_val)

metrics = classification_metrics(y_val, y_val_pred)

print(metrics) 



print('\n')



# Cross Validation

scores = cross_val_score(xgb_clf, X_train, y_train, scoring='accuracy', cv=10)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard Deviation: ", scores.std())
xgb_clf.feature_importances_
# plot feature importance

from xgboost import plot_importance

plot_importance(xgb_clf)

plt.show()
thresholds = np.sort(xgb_clf.feature_importances_)

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(xgb_clf, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)



    # train model

    selection_model = XGBClassifier()

    selection_model.fit(select_X_train, y_train)



    # eval model

    select_X_val = selection.transform(X_val)

    predictions = selection_model.predict(select_X_val)

    accuracy = accuracy_score(y_val, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],

    accuracy*100.0))
X.shape
X_test.shape
y_test = xgb_clf.predict(X_test_new)

test['Survived'] = y_test
submission = test[['PassengerId', 'Survived']]

submission.to_csv('submission7.csv',mode = 'w', index=False)
import altair as alt