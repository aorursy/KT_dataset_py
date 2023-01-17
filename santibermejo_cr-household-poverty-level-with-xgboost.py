import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.isnull().sum()[train_df.isnull().sum()>0]
def join_columns(df):
    df.loc[df['tipovivi1']==1, 'tipovivigral'] = 1 #Owns
    df.loc[df['tipovivi2']==1, 'tipovivigral'] = 2 #Paying installments
    df.loc[df['tipovivi3']==1, 'tipovivigral'] = 3 #Rented
    df.loc[df['tipovivi4']==1, 'tipovivigral'] = 4 #Precarious
    df.loc[df['tipovivi5']==1, 'tipovivigral'] = 5 #Other

join_columns(train_df)

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,8))

sns.countplot(train_df['tipovivigral'][train_df['v2a1'].isnull()], ax=ax1)
ax1.set_title('NULL Rent')
ax1.set_xticklabels(['Owns', 'Precarious', 'Other'], rotation=30)
ax1.set_xlabel('Household Situation')

sns.countplot(train_df['tipovivigral'][train_df['v2a1'].notnull()], ax=ax2)
ax2.set_title('Not NULL Rent')
ax2.set_xticklabels(['Paying Installments', 'Rented'], rotation=30)
ax2.set_xlabel('Household Situation')

sns.countplot(train_df['tipovivigral'][train_df['v2a1']==0], ax=ax3)
ax3.set_title('Rent = 0')
ax3.set_xticklabels(['Paying Installments'], rotation=30)
ax3.set_xlabel('Household Situation')

plt.show()
train_df.loc[train_df['v2a1'].isnull(),'v2a1'] = 0

f, (ax1, ax2) = plt.subplots(1,2, figsize=(25,8))

sns.countplot(train_df['tipovivigral'][train_df['v2a1']==0], ax = ax1)
ax1.set_title('Not paying rent')
ax1.set_xticklabels(['Owns','Paying Installments', 'Precarious', 'Other'], rotation=30)
ax1.set_xlabel('Household Situation')

sns.countplot(train_df['tipovivigral'][train_df['v2a1']!=0], ax = ax2)
ax2.set_title('Paying rent')
ax2.set_xticklabels(['Paying Installments', 'Rented'], rotation=30)
ax2.set_xlabel('Household Situation')

plt.show()
sns.countplot(train_df['v18q1'].isnull())
train_df.loc[train_df['v18q1'].isnull(), 'v18q1'] = 0
train_df['v18q1'].value_counts().plot.bar()
train_df[['rez_esc','escolari']][train_df['rez_esc'].notnull()].head(10)
train_df.drop(columns='rez_esc', inplace = True)
train_df.loc[train_df['meaneduc'].isnull(), 'meaneduc'] = abs(train_df['meaneduc'].mean())
train_df.loc[train_df['SQBmeaned'].isnull(), 'SQBmeaned'] = abs(train_df['meaneduc'].mean())**2
train_df.applymap(np.isreal).any()[(train_df.applymap(np.isreal).any())==False]
train_df.groupby('dependency')['Id'].count().sort_values(ascending=False)
train_df.loc[train_df['dependency'] == 'yes', 'dependency'] = train_df.loc[(train_df['dependency']!='yes') & (train_df['dependency']!='no'), 'dependency'].mode()[0]
train_df.loc[train_df['dependency'] == 'no', 'dependency'] = 0
train_df.groupby('edjefe')['Id'].count().sort_values(ascending=False)
def complete_edjefe_edjefa(edjefe_jefa, sex, new_column, value):
    no_records = train_df['idhogar'][train_df[edjefe_jefa]==value].drop_duplicates()
    no_households = train_df[[edjefe_jefa, 'idhogar','escolari','age',sex]].where(lambda x : x['idhogar'].isin(no_records)).dropna()
   
    # I create a new column in which we are going to highlight the yes-no households with jefas 
    no_households.loc[(no_households[sex]==1) & (no_households['age']>=18), new_column] = 1
    no_households.loc[no_households[edjefe_jefa].isnull(), new_column] = 0
    households_by_jefa = no_households.groupby('idhogar')[[new_column,'escolari']].max().reset_index()
   
    # I get rid of the no values that can be replaced by the jefe/jefa's escolari values
    for index, row in households_by_jefa.iterrows():
        if row[new_column] == 1:
            train_df.loc[train_df['idhogar'] == row['idhogar'], edjefe_jefa] = row['escolari']

complete_edjefe_edjefa('edjefe', 'male', 'jefeexists', 'yes')
complete_edjefe_edjefa('edjefa', 'female', 'jefaexists', 'yes')
complete_edjefe_edjefa('edjefe', 'male', 'jefeexists', 'no')
complete_edjefe_edjefa('edjefa', 'female', 'jefaexists', 'no')

#The remaining 'no' values correspond to households where there's no pressence of a father (jefe) or mother (jefa), so we'll turn those values into 0
train_df.loc[train_df['edjefe'] == 'no', 'edjefe'] = 0
train_df.loc[train_df['edjefa'] == 'no', 'edjefa'] = 0 
train_df.drop(columns=['Id','idhogar','tipovivigral'], inplace = True)
train_df.dtypes[(train_df.dtypes != 'int64') & (train_df.dtypes != 'float64')]
train_df['dependency'] = train_df['dependency'].astype(float)
train_df['edjefe'] = train_df['edjefe'].astype(int)
train_df['edjefa'] = train_df['edjefa'].astype(int)
X = train_df.iloc[:,0:len(train_df.columns)-1]
y = train_df.iloc[:,len(train_df.columns)-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

def calculate_accuracy(cm):
    correct=0
    incorrect=0
    for i in range(4):
        for j in range(4):
            if i == j:
                correct = correct + cm[i][j]
            else:
                incorrect = incorrect + cm[i][j]
    return correct/(correct+incorrect)

calculate_accuracy(cm)
from sklearn.model_selection import GridSearchCV
parameters = {'min_child_weight': [0.5, 1.5],
              'gamma': [0.1, 0.3],
              'subsample': [0.7, 0.9],
              'colsample_bytree': [0.9],
              'max_depth': [5, 7]}

grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 5,
                            n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
best_accuracy
best_parameters
classifier = XGBClassifier(colsample_bytree = 0.9, 
                           gamma = 0.1,
                           max_depth = 7,
                           min_child_weight = 0.5,
                           subsample = 0.9)
classifier.fit(X_train, y_train)
feat_imp = pd.DataFrame({'importance':classifier.feature_importances_})    
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.iloc[:10]
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title='Feature Importance')
plt.xlabel('Feature Importance Score')
plt.show() 
test_df.loc[test_df['v2a1'].isnull(),'v2a1'] = 0
test_df.loc[test_df['v18q1'].isnull(), 'v18q1'] = 0
test_df.drop(columns='rez_esc', inplace = True)
test_df.loc[test_df['meaneduc'].isnull(), 'meaneduc'] = abs(test_df['meaneduc'].mean())
test_df.loc[test_df['SQBmeaned'].isnull(), 'SQBmeaned'] = abs(test_df['meaneduc'].mean())**2
test_df.loc[test_df['dependency'] == 'yes', 'dependency'] = test_df.loc[(test_df['dependency']!='yes') & (test_df['dependency']!='no'), 'dependency'].mode()[0]
test_df.loc[test_df['dependency'] == 'no', 'dependency'] = 0
def complete_edjefe_edjefa(edjefe_jefa, sex, new_column, value):
    no_records = test_df['idhogar'][test_df[edjefe_jefa]==value].drop_duplicates()
    no_households = test_df[[edjefe_jefa, 'idhogar','escolari','age',sex]].where(lambda x : x['idhogar'].isin(no_records)).dropna()
    #We create a new column in which we are going to highlight the yes-no households with jefas 
    no_households.loc[(no_households[sex]==1) & (no_households['age']>=18), new_column] = 1
    no_households.loc[no_households[edjefe_jefa].isnull(), new_column] = 0
    households_by_jefa = no_households.groupby('idhogar')[[new_column,'escolari']].max().reset_index()   
    #We get rid of the no values that can be replaced by the jefe/jefa's escolari values
    for index, row in households_by_jefa.iterrows():
        if row[new_column] == 1:
            test_df.loc[test_df['idhogar'] == row['idhogar'], edjefe_jefa] = row['escolari']
complete_edjefe_edjefa('edjefe', 'male', 'jefeexists', 'yes')
complete_edjefe_edjefa('edjefa', 'female', 'jefaexists', 'yes')
complete_edjefe_edjefa('edjefe', 'male', 'jefeexists', 'no')
complete_edjefe_edjefa('edjefa', 'female', 'jefaexists', 'no')
test_df.loc[test_df['edjefe'] == 'no', 'edjefe'] = 0
test_df.loc[test_df['edjefa'] == 'no', 'edjefa'] = 0 
subs = pd.DataFrame()
subs['Id'] = test_df['Id']
test_df.drop(columns=['Id','idhogar'], inplace = True)
test_df['dependency'] = test_df['dependency'].astype(float)
test_df['edjefe'] = test_df['edjefe'].astype(int)
test_df['edjefa'] = test_df['edjefa'].astype(int)
X = test_df.iloc[:,0:len(test_df.columns)]

y_pred = classifier.predict(X)
y_pred = pd.DataFrame(y_pred)

subs['Target'] = y_pred
subs.to_csv('sample_submission.csv', index = False)