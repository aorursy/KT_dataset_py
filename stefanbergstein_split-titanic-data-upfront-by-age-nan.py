# data processing

import numpy as np

import pandas as pd 





import warnings

warnings.filterwarnings('ignore')



# plotting

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold







# some configuratin flags and variables

verbose = 1 # Use in classifier

quick_run = True # if set to True, use only few variabales during hyperparameter tuning

n_jobs = -1



# Input files

file_train='../input/train.csv'

file_test='../input/test.csv'



# define random seed for reproducibility

seed = 69

np.random.seed(seed)



# read training and test data

train_df = pd.read_csv(file_train,index_col='PassengerId')

test_df = pd.read_csv(file_test,index_col='PassengerId')

# Show the columns

train_df.columns.values
# Show the shape

train_df.shape
# preview the training data

train_df.head()
train_df.describe()
# Show that there is NaN data (Age,Fare Embarked), that needs to be handled.

train_df.isnull().sum()
def prep_data(df):

    # Drop unwanted features

    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    

    # Fill missing data:  Fare with the mean, Embarked with most frequent value

    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())

    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())

    

    # Convert categorical  features into numeric

    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

      

    # Convert Embarked to one-hot

    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')

    df = df.drop('Embarked', axis=1)

    df = df.join(enbarked_one_hot)



    return df

train_df = prep_data(train_df)

train_df_age_no_nan = train_df[train_df['Age'].notnull()]

print("Training data without any NaN in Age:\n", train_df_age_no_nan.isnull().sum())
train_df_age_no_nan.shape
cmap = plt.cm.RdBu

corr = train_df_age_no_nan.corr()

plt.figure(figsize=(12,10))

plt.title('Pearson Features Correlation of training data without NaN in Age', size=15)

sns.heatmap(corr, cmap=cmap,  annot=True, linewidths=1)
g = sns.pairplot(train_df_age_no_nan, hue='Survived', palette = 'seismic',size=1.5,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))

g.set(xticklabels=[])
test_df = prep_data(test_df)

test_df_age_no_nan = test_df[test_df['Age'].notnull()]

print("Test data without any NaN in Age:\n", test_df_age_no_nan.isnull().sum())
test_df_age_no_nan.shape
# X contains all columns except 'Survived'  

X = train_df_age_no_nan.drop(['Survived'], axis=1).values.astype(float)



# Y is just the 'Survived' column

Y = train_df_age_no_nan['Survived'].values
kfold = KFold(n_splits=10, random_state=seed)



if quick_run:

    rf_parameters = {"max_depth": [4,8]

                ,"min_samples_split" :[2,6]

                ,"n_estimators" : [100]

                ,"min_samples_leaf": [1,2]

                ,"max_features": [6,"sqrt"]

                ,"criterion": ['gini']}       

else:

    rf_parameters = {"max_depth": [2,4,6,8,12]

                ,"min_samples_split" :[2,3,5,8]

                ,"n_estimators" : [50, 100,200]

                ,"min_samples_leaf": [2,3,5]

                ,"max_features": [4,6,"sqrt"]

                ,"criterion": ['gini','entropy']}



print('** GridSearchCV RF ...') 

rf_clf = RandomForestClassifier()

rf_grid = GridSearchCV(rf_clf,rf_parameters, n_jobs = n_jobs, verbose = verbose, cv = 10)

rf_grid.fit(X,Y)



print('Best score RF: {}'.format(rf_grid.best_score_))

print('Best parameters RF: {}'.format(rf_grid.best_params_))



rf_clf = RandomForestClassifier(**rf_grid.best_params_)

rf_clf.fit(X,Y)
if quick_run:

    ab_parameters = {'n_estimators':[50,100,200,300],

                  'learning_rate':[0.1,0.5,1.0,2.0]}

else:

    ab_parameters = {'n_estimators':[50,100],

                  'learning_rate':[0.5,1.0]}

    

print('** GridSearchCV AB ...') 

ab_clf = AdaBoostClassifier()

ab_grid = GridSearchCV(ab_clf,ab_parameters, n_jobs = n_jobs, verbose = verbose, cv = 10)

ab_grid.fit(X,Y)



print('Best score AB: {}'.format(ab_grid.best_score_))

print('Best parameters AB: {}'.format(ab_grid.best_params_))



ab_clf = AdaBoostClassifier(**ab_grid.best_params_)

ab_clf.fit(X,Y)
# Create X_test

X_test = test_df_age_no_nan.values.astype(float)





# Predict 'Survived' for Age no NaN

if ( rf_grid.best_score_ > ab_grid.best_score_ ) :

    print('** Predict Survived for data with age using RF {}'.format(rf_grid.best_params_))

    prediction_age_no_nan = rf_clf.predict(X_test)

else:

    print('** Predict Survived for data with age using AB {}'.format(ab_grid.best_params_))

    prediction_age_no_nan = ab_clf.predict(X_test)





subm_no_nan = pd.DataFrame({

    'PassengerId': test_df_age_no_nan.index,

    'Survived': prediction_age_no_nan,

})

    

test_df_age_nan = test_df[test_df['Age'].isnull()]
# Split training data into input X and output Y



train_df_age_nan = train_df[train_df['Age'].isnull()]



# X contains all columns except Age and 'Survived'  

X = train_df.drop(['Age','Survived'], axis=1).values.astype(float)



# Y is just the 'Survived' column

Y = train_df['Survived'].values





kfold = KFold(n_splits=10, random_state=seed)



print('** GridSearchCV (no Age) RF ...') 

rf_clf = RandomForestClassifier()

rf_grid = GridSearchCV(rf_clf,rf_parameters, n_jobs = n_jobs, verbose = verbose, cv = 10)

rf_grid.fit(X,Y)



print('Best score (no Age) RF: {}'.format(rf_grid.best_score_))

print('Best parameters (no Age) RF: {}'.format(rf_grid.best_params_))





rf_clf = RandomForestClassifier(**rf_grid.best_params_)

rf_clf.fit(X,Y)



# Predict test data with Age == NaN

X_test_age_nan = test_df_age_nan.drop(['Age'], axis=1).values.astype(float)



# Predict 'Survived'

prediction_age_nan = rf_grid.predict(X_test_age_nan)   



subm_nan = pd.DataFrame({

    'PassengerId': test_df_age_nan.index,

    'Survived': prediction_age_nan

})
# stack the DataFrames on top of each other

submission = pd.concat([subm_no_nan, subm_nan], axis=0)



submission.sort_values('PassengerId', inplace=True)    

submission.to_csv('submission-splitt-input.csv', index=False)