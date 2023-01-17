# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train['Survived'].isnull().sum()
survived = train['Survived']
train.drop('Survived', axis = 1, inplace = True)
test['PassengerId'].isnull().sum()
PssId = test['PassengerId']
train_test = pd.concat([train,test], keys = ['train', 'test'], names = ['dataset', 'index'])
train_test.head()
train_test.info()
train_test.drop(['PassengerId','Name', 'Ticket'], axis = 1, inplace = True)
train_test.head()
train_test.isnull().sum()
train_test.drop(['Cabin'], axis = 1, inplace = True)
age_summary = train.groupby(by = ['Sex','Pclass'])['Age'].describe()
age_summary
where_age_is_null = train_test['Age'].isnull()
age_missing_data = train_test[where_age_is_null]
age_missing_data.head()
    
fill_age = []
for index, row in age_missing_data.iterrows():
    fill_age.append(round(age_summary.xs([row['Sex'], row['Pclass']])['mean'],0))

train_test.loc[where_age_is_null, 'Age'] = fill_age#This contains the mean value of age based on gender and class for each missing data in age
train_test[where_age_is_null].head()
train_test.isnull().sum()
train_test['Embarked'].describe()
train_test['Embarked'].value_counts()
train_test['Embarked'] = train_test['Embarked'].fillna('S')
train_test['Fare'] = train_test['Fare'].fillna(train_test['Fare'].mean())
train_test.isnull().sum()
train = train_test.loc['train']
test = train_test.loc['test']
train['Survived'] = survived
train.head()
train.shape
train['Survived'].mean()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Survived', data = train)
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Survived', data = train, hue = 'Sex')
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Sex', data = train)
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
1-train.groupby(by = 'Sex')['Survived'].mean()
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Pclass', data = train)
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
1 - train.groupby(by = 'Pclass')['Survived'].mean()
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Survived', data = train, hue = 'Embarked')
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Embarked', data = train)
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
1 - train.groupby(by = 'Embarked')['Survived'].mean()
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Embarked', data = train, hue = 'Pclass')
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (12, 6))
g = sns.countplot(x = 'Embarked', data = train, hue = 'Sex')
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (12, 6))
g = sns.FacetGrid(train, col = 'Embarked', size = 5, aspect = 0.7)
g = (g.map(sns.countplot, 'Pclass', hue = 'Sex', data = train, palette="Set1")).add_legend()
for ax in g.axes.ravel():
    for p in ax.patches:
        ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    
#sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = train, kind = 'bar')
plt.figure(figsize = (12, 6))
g = sns.FacetGrid(train, row = 'Pclass', col = 'Embarked', size = 5, aspect = 0.7)
g = (g.map(sns.countplot, 'Survived', hue = 'Sex', data = train, palette="Set1")).add_legend()
for ax in g.axes.ravel():
    for p in ax.patches:
        ax.annotate("%0.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
#ECS stands for Embarked, Class, and Sex
ECS = pd.DataFrame(train.groupby(by = ['Embarked','Pclass'])['Sex'].value_counts())
ECS_per = pd.DataFrame(round(1 - train.groupby(by = ['Embarked', 'Pclass', 'Sex'])['Survived'].mean(),3))
ECS_lost = pd.DataFrame(train.groupby(by = ['Embarked','Pclass'])['Sex'].value_counts() - train.groupby(by = ['Embarked', 'Pclass', 'Sex'])['Survived'].sum())
ECS = pd.concat([ECS, ECS_lost, ECS_per], axis = 1)
ECS.columns = ['Total number of passengers', 'Number of passengers lost', 'Percentage of passengers lost']
ECS = ECS.style.set_properties(**{'text-align': 'right'})
ECS
#ECS['Lost Percentage'] = ECS_per
train['Age'].describe()
plt.figure(figsize = (16, 6))
sns.distplot(train['Age'], kde = False, hist_kws = {'edgecolor': 'k'}, bins = 150)
train.groupby(by = 'Sex')['Age'].describe()
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Sex', y = 'Age', data = train)
train.groupby(by = ['Pclass'])['Age'].describe()
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
train.groupby(by = ['Sex','Pclass'])['Age'].describe()
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Sex', y = 'Age', data = train, hue = 'Pclass')
train['Fare'].describe()
plt.figure(figsize = (16, 6))
sns.distplot(train['Fare'], kde = False, hist_kws = {'edgecolor': 'k'}, bins = 150)
plt.figure(figsize = (12, 8))
sns.boxplot(y = 'Fare', data = train)
train.groupby(by = 'Sex')['Fare'].describe()
train.groupby(by = 'Pclass')['Fare'].describe()
train.groupby(by = ['Sex', 'Pclass'])['Fare'].describe()
plt.figure(figsize = (16, 6))
g = sns.countplot(x = 'SibSp', data = train)
for p in g.patches:
        g.annotate("%0.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.figure(figsize = (16, 6))
g = sns.countplot(x = 'Parch', data = train)
for p in g.patches:
        g.annotate("%0.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
train_get_dummy = pd.get_dummies(columns=['Sex', 'Pclass', 'Embarked'], drop_first = True, data = train)
train_get_dummy.head()
plt.figure(figsize = (12, 8))
sns.heatmap(pd.DataFrame(train_get_dummy.corr().unstack()['Survived'], 
                         columns = ['Survived']), annot = True, cmap = 'coolwarm')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
def forward_stepwise_selection(df):
    '''
    This function takes as an input a dataframe with Survived as one of its columns performs forward
    stepwise selection on the features. For a given number of features, R-squared is used to determine
    the best predictor to add. After having the P models (P is the number of predictors), we select the best
    model size using 5-fold cross validation mean score.
    '''
    p = len(df.columns)#the total number of predictors
    X = df.drop('Survived', axis = 1)
    columns = list(X.columns)
    y = df['Survived']
    best_cols_global = []
    for col1 in columns:
        max_score = -1
        for col2 in columns:
            model = LogisticRegression()
            if col2 not in best_cols_global:
                cols = best_cols_global[:]
                #print(cols)
                cols.append(col2)
                #print(cols)
                model.fit(X[cols], y)
                score = model.score(X[cols], y)
                if score > max_score:
                    max_score = score
                    best_col = col2
                    #print(best_col)
        if best_col not in best_cols_global:        
            best_cols_global.append(best_col)
        print(best_cols_global, max_score)
        model = LogisticRegression()
        mean_score = cross_val_score(model, X[best_cols_global], y, cv = 5).mean()
        print('CV mean score is ', mean_score)
forward_stepwise_selection(train_get_dummy)
def summary_results(df):
    '''
    This function takes as input a dataframe that has Survived as one of its columns, and
    does the following:
    
    1- Split the dataframe into a training and testing dataset with a test size of 0.3.
    2- Define a list of models to be used
    3- Train each model
    4- Make predictions on the testing dataset
    
    The outputs of the function is:
    1- The acuuracy of the models using the confusion matrix
    2- the 5-fold cross-validation mean score of the models.
    '''
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis = 1),
                                                       df['Survived'],
                                                       test_size = 0.3,
                                                       random_state = 100)
#     pca = PCA(n_components = 2)
    
#     X_train = pca.fit_transform(X_train)
#     X_test = pca.fit_transform(X_test)

    X_train = scale(X_train)
    X_test = scale(X_test) 
    
    k_max = -10
    mean_score_max = -10
    for kk in range(1, 20, 2):
        KNN_model = KNeighborsClassifier(n_neighbors = kk)
        mean_score = cross_val_score(KNN_model, df.drop('Survived', axis = 1),
                                df['Survived'], cv = 10).mean()
    #print('K = ', kk, 'mean score = ', mean_score)
        if mean_score > mean_score_max:
            mean_score_max = mean_score
            k_max = kk
            
    print('The best K for K-NN using 5-fold CV is ', k_max)
    
    DC = DummyClassifier(strategy = 'most_frequent')
    LR = LogisticRegression()
    LDA = LinearDiscriminantAnalysis()
    QDA = QuadraticDiscriminantAnalysis()
    KNN = KNeighborsClassifier(n_neighbors = k_max)
    DTC = DecisionTreeClassifier()
    BC = BaggingClassifier(n_estimators = 300)
    RFC = RandomForestClassifier(n_estimators = 300)
    SVCL = SVC(probability = True)
    VC = VotingClassifier(estimators=[('lr', LR), 
                                      #('LDA', LDA),
                                      ('QDA', QDA),
                                     # ('DTC', DTC),
                                      ('BC', BC),
                                      ('RFC', RFC),
                                      ('SVC', SVCL),
                                      #('KNN', KNN)
                                     ], 
                          voting = 'soft')
    
    classifiers = [DC, 
                   LR, 
                   LDA, 
                   QDA,
                   DTC,
                   BC,
                   RFC,
                   KNN,
                   SVCL,
                   VC]
    
    list_of_classifiers = ['Null Classifier',
                           'Logistic Regression', 
                           'Linear Discriminant Analaysis', 
                           'Quadratic Discriminant Analysis', 
                           'Decision Tree Classifier',
                           'Bagging Classifier',
                           'Random Forest Classifier',
                           'K-Nearest Neighbors', 
                           'Support Vector Classifier',
                           'Voting Classifier']
    max_len = max(list_of_classifiers, key = len)#this returns the string of the maximum length
    
#     scaler = StandardScaler()
#     X_train_KNN = scaler.fit_transform(X_train)
#     X_test_KNN = scaler.fit_transform(X_test)
    
    
    print('================================================================================')   
    print('{}{}{}{}'.format('Model Name:', ' '*(len(max_len)-len('Model Name:')+3), 'Accuracy', '\t5-fold CV Mean Score'))
    print('================================================================================')   

    i = 0
    for model in classifiers:
        model.fit(X_train, y_train)
        pred = model.predict(X_test) 
        Con_mat = confusion_matrix(y_test, pred)
        acc = (Con_mat[0,0]+Con_mat[1,1])/(Con_mat[0,0]+Con_mat[0,1]+Con_mat[1,0]+Con_mat[1,1])
        mean_score = cross_val_score(model, df.drop('Survived', axis = 1), df['Survived'], cv = 5).mean()
        diff = len(max_len)-len(list_of_classifiers[i])+5
        print('{}{}{:.3f}\t{:.3f}'.format(list_of_classifiers[i], ' '*diff, acc, mean_score))
        #print(list_of_classifiers[i], acc)
        i += 1
        
#     print('================================================================================')   
#     print('\n'*2)
#     print(' '*15+'Summary result of the logistic regression model')
#     print('\n'*2)
#     print('================================================================================') 
#     X_train = sm.add_constant(X_train)
#     model_stats = sm.Logit(y_train, X_train)
#     result = model_stats.fit()
#     print(result.summary2())
summary_results(train_get_dummy[['Survived','Sex_male', 'SibSp', 'Pclass_3', 'Age', 'Embarked_S']])
test.head()
test_get_dummies = pd.get_dummies(test, columns = ['Sex', 'Pclass', 'Embarked'], drop_first = True)
test_get_dummies = test_get_dummies[['Sex_male', 'SibSp', 'Pclass_3', 'Age', 'Embarked_S']]
test_get_dummies.head()
LR = LogisticRegression()
LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis()
KNN = KNeighborsClassifier(n_neighbors = 11)
DTC = DecisionTreeClassifier()
BC = BaggingClassifier(n_estimators = 300)
RFC = RandomForestClassifier(n_estimators = 300)
SVCL = SVC(probability = True)
best_model = VotingClassifier(estimators=[('lr', LR), 
                                      #('LDA', LDA),
                                      ('QDA', QDA),
                                     # ('DTC', DTC),
                                      ('BC', BC),
                                      ('RFC', RFC),
                                      ('SVC', SVCL),
                                      #('KNN', KNN)
                                     ], 
                          voting = 'soft')
best_model.fit(train_get_dummy[['Sex_male', 'SibSp', 'Pclass_3', 'Age', 'Embarked_S']], 
               train_get_dummy['Survived'])
PssID_df = pd.DataFrame(PssId, columns = ['PassengerId'])
PssID_df['Survived'] = best_model.predict(test_get_dummies)
PssID_df.to_csv('final_result.csv', index = False)
PssID_df.head()
