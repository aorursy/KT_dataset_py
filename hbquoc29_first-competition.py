import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc



import string

import warnings

warnings.filterwarnings('ignore')



SEED = 42
def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]



print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)

df_all.head(10)
def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    

for df in dfs:

    print('{}'.format(df.name))

    display_missing(df)
df_all['Age'] = df_all.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all['Embarked'].fillna(df_all['Embarked'].mode()[0], inplace = True)
med_fare = df_all.groupby(['Pclass']).Fare.median()[3]

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

df_all['Fare'] = df_all['Fare'].fillna(med_fare)
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)

df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')



df_all_decks = df_all.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 

                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()



def get_pclass_dist(df):

    

    # Creating a dictionary for every passenger class count in every deck

    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}

    decks = df.columns.levels[0]    

    

    for deck in decks:

        for pclass in range(1, 4):

            try:

                count = df[deck][pclass][0]

                deck_counts[deck][pclass] = count 

            except KeyError:

                deck_counts[deck][pclass] = 0

                

    df_decks = pd.DataFrame(deck_counts)    

    deck_percentages = {}



    # Creating a dictionary for every passenger class percentage in every deck

    for col in df_decks.columns:

        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

        

    return deck_counts, deck_percentages



def display_pclass_dist(percentages):

    

    df_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.85

    

    pclass1 = df_percentages[0]

    pclass2 = df_percentages[1]

    pclass3 = df_percentages[2]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')

    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')

    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')



    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   

    

    plt.show()    



all_deck_count, all_deck_per = get_pclass_dist(df_all_decks)

display_pclass_dist(all_deck_per)
# Passenger in the T deck is changed to A

idx = df_all[df_all['Deck'] == 'T'].index

df_all.loc[idx, 'Deck'] = 'A'
df_all_decks_survived = df_all.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 

                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()



def get_survived_dist(df):

    

    # Creating a dictionary for every survival count in every deck

    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}

    decks = df.columns.levels[0]    



    for deck in decks:

        for survive in range(0, 2):

            surv_counts[deck][survive] = df[deck][survive][0]

            

    df_surv = pd.DataFrame(surv_counts)

    surv_percentages = {}



    for col in df_surv.columns:

        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]

        

    return surv_counts, surv_percentages



def display_surv_dist(percentages):

    

    df_survived_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.85    



    not_survived = df_survived_percentages[0]

    survived = df_survived_percentages[1]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")

    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")

 

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Survival Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

    plt.title('Survival Percentage in Decks', size=18, y=1.05)

    

    plt.show()



all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)

display_surv_dist(all_surv_per)
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')

df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')

df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')



df_all['Deck'].value_counts()
df_all.drop(['Cabin'], inplace=True, axis=1)



df_train, df_test = divide_df(df_all)

dfs = [df_train, df_test]



for df in dfs:

    display_missing(df)
survived = df_train['Survived'].value_counts()[1]

not_survived = df_train['Survived'].value_counts()[0]

survived_per = survived / df_train.shape[0] * 100

not_survived_per = not_survived / df_train.shape[0] * 100



print('{} of {} passengers survived and it is the {:.2f}% of the training set.'.format(survived, df_train.shape[0], survived_per))

print('{} of {} passengers didnt survive and it is the {:.2f}% of the training set.'.format(not_survived, df_train.shape[0], not_survived_per))



plt.figure(figsize=(10, 8))

sns.countplot(df_train['Survived'])



plt.xlabel('Survival', size=15, labelpad=15)

plt.ylabel('Passenger Count', size=15, labelpad=15)

plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])

plt.tick_params(axis='x', labelsize=13)

plt.tick_params(axis='y', labelsize=13)



plt.title('Training Set Survival Distribution', size=15, y=1.05)



plt.show()
df_all = concat_df(df_train, df_test)

df_all.head()
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)



fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Fare', hue='Survived', data=df_all)



plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)



plt.show()
df_all['Age'] = pd.qcut(df_all['Age'], 6)



fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Age', hue='Survived', data=df_all)



plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)



plt.show()
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1



fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)

plt.subplots_adjust(right=1.5)



sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])

sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])



axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)

axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)



family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)



sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, y=df_all['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])

sns.countplot(x='Family_Size_Grouped', hue='Survived', data=df_all, ax=axs[1][1])



axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)

axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)



for i in range(2):

    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

        axs[i][j].set_xlabel('')

        axs[i][j].set_ylabel('')



plt.show()
df_train = df_all.loc[:890]

df_test = df_all.loc[891:]

dfs = [df_train, df_test]
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[0])



axs[0].tick_params(axis='x', labelsize=10)

axs[1].tick_params(axis='x', labelsize=15)



for i in range(2):    

    axs[i].tick_params(axis='y', labelsize=15)



axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)



df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[1])

axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)



plt.show()

df_train = df_all.loc[:890]

df_test = df_all.loc[891:]

dfs = [df_train, df_test]
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']



for df in dfs:

    for feature in non_numeric_features:        

        df[feature] = LabelEncoder().fit_transform(df[feature])
cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']

encoded_features = []



for df in dfs:

    for feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)



df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)

df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)
df_train
df_all = concat_df(df_train, df_test)

drop_cols = ['Deck', 'Embarked', 'Family_Size', 'Family_Size_Grouped', 'Survived',

             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title']



df_all.drop(columns=drop_cols, inplace=True)



df_all.head()
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))

y_train = df_train['Survived'].values

X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
random_state = 4

classifiers = []

classifiers.append(('SVC',SVC(random_state=random_state)))

classifiers.append(('DecisionTree', DecisionTreeClassifier(random_state=random_state)))

classifiers.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),\

                                                  random_state=random_state,learning_rate=0.1)))

classifiers.append(('RandomForest', RandomForestClassifier(random_state=random_state)))

classifiers.append(('GradientBoost', GradientBoostingClassifier(random_state=random_state)))

classifiers.append(('MPL', make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state))))

classifiers.append(('KNN',make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=7))))



# evaluate each model 

results = []

names = []

for name, classifier in classifiers:

    kfold = model_selection.KFold(n_splits= 3, random_state=random_state, shuffle = True)

    cv_results = model_selection.cross_val_score(classifier, X_train, y = y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



def random_forest(X, Y, X_test):

    parameters = {'max_depth' : [2, 4, 5, 10], 

                  'n_estimators' : [200, 500, 1000, 2000], 

                  'min_samples_split' : [3, 4, 5], 



                 }

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    model_RFC = RandomForestClassifier(random_state = 4, n_jobs = -1)

    search_RFC = GridSearchCV(model_RFC, parameters, n_jobs = -1, cv = kfold, scoring = 'accuracy',verbose=1)

    search_RFC.fit(X, Y)

    predicted= search_RFC.predict(X_test)

    

    print("Best parameters are:", search_RFC.best_params_)

    print("Best accuracy achieved:",search_RFC.best_score_)

    

    return search_RFC.best_params_, model_RFC, search_RFC, predicted
# param_RFC_onehot, model_RFC_onehot, search_RFC_onehot, predicted_cv_RFC_onehot = random_forest(X_train, y_train, X_test)
def fit_pred_RF(X, Y, X_test):



    model_RFC = RandomForestClassifier(max_depth =5,  min_samples_split =4, n_estimators = 5000,

                                     random_state = 4, n_jobs = -1)

    model_RFC.fit(X, Y)

    

    predicted= model_RFC.predict(X_test)

    

    return predicted, model_RFC
predicted_RFC, model_RFC= fit_pred_RF(X_train, y_train, X_test)
def grad_boost(X, Y, X_test):



    parameters = {'max_depth' : [2, 4, 10, 15], 

                  'n_estimators' : [10, 50, 100], 

                  'min_samples_split' : [5, 10, 15],

                 }

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    model_GBC = GradientBoostingClassifier(random_state = 4)

    search_GBC = GridSearchCV(model_GBC, parameters, n_jobs = -1, cv = kfold, scoring = 'accuracy',verbose=1)

    search_GBC.fit(X, Y)

    predicted= search_GBC.predict(X_test)

    

    print("Best parameters are:", search_GBC.best_params_)

    print("Best accuracy achieved:",search_GBC.cv_results_['mean_test_score'].mean())

    

    return search_GBC.best_params_, model_GBC, search_GBC, predicted

    
# param_GBC_onehot, model_GBC_onehot, search_GBC_onehot, predicted_cv_GBC_onehot = grad_boost(X_train, y_train, X_test)
def fit_pred_GBC(X, Y, X_test):



    model_GBC = GradientBoostingClassifier(max_depth = 3, min_samples_split = 15, n_estimators = 20,\

                                 random_state = 4, max_features= 'auto')

    model_GBC.fit(X, Y)

    

    predicted= model_GBC.predict(X_test)

    

    return predicted, model_GBC
predicted_GBC, model_GBC = fit_pred_GBC(X_train, y_train, X_test)
def mod_KNN(X, Y, X_test):

    

    model_KNN=make_pipeline(MinMaxScaler(),KNeighborsClassifier())

    #KNN.get_params().keys()

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    parameters=[{'kneighborsclassifier__n_neighbors': [2,3,4,5,6,7,8,9,10]}]

    search_KNN = GridSearchCV(estimator=model_KNN, param_grid=parameters, scoring='accuracy', cv=kfold)

    scores_KNN=cross_val_score(search_KNN, X, Y,scoring='accuracy', cv=kfold, verbose=1)

    search_KNN.fit(X, Y)

    predicted= search_KNN.predict(X_test)

    

    print("Best parameters are:", search_KNN.best_params_)

    print("Best accuracy achieved:",search_KNN.cv_results_['mean_test_score'].mean())

    

    return search_KNN.best_params_, model_KNN, search_KNN, predicted
# param_KNN_onehot, model_KNN_onehot, search_KNN_onehot, predicted_cv_KNN_onehot = mod_KNN(X_train, y_train, X_test)
def fit_pred_KNN(X, Y, X_test):



    model_KNN = make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=12))

    

    model_KNN.fit(X, Y)

    

    predicted= model_KNN.predict(X_test)

    

    return predicted, model_KNN
predicted_KNN, model_KNN = fit_pred_KNN(X_train, y_train, X_test)
def mod_SVC(X, Y, X_test):



    model_SVC=make_pipeline(StandardScaler(),SVC(random_state=1))

    parameters=[{'svc__C': [0.0001,0.001,0.1,1, 10, 100], 

           'svc__gamma':[0.0001,0.001,0.1,1,10,50,100],

           'svc__kernel':['rbf'],

           'svc__degree' : [1,2,3,4]

          }]

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    search_SVC = GridSearchCV(estimator=model_SVC, param_grid = parameters, scoring='accuracy', cv=kfold)

    scores_SVC=cross_val_score(search_SVC, X, Y,scoring='accuracy', cv=kfold, verbose =1)

    search_SVC.fit(X, Y)

    predicted= search_SVC.predict(X_test)

    

    print("Best parameters are:", search_SVC.best_params_)

    print("Best accuracy achieved:",search_SVC.cv_results_['mean_test_score'].mean())

    

    return search_SVC.best_params_, model_SVC, search_SVC, predicted
# param_SVC_onehot, model_SVC_onehot, search_SVC_onehot, predicted_cv_SVC_onehot = mod_SVC(X_train, y_train, X_test)
def fit_pred_SVC(X, Y, X_test):



    model_SVC =SVC(random_state=random_state, C= 1, gamma = 0.001, kernel = 'rbf', degree =1)

    

    model_SVC.fit(X, Y)

    

    predicted= model_SVC.predict(X_test)

    

    return predicted, model_SVC
predicted_SVC, model_SVC = fit_pred_SVC(X_train, y_train, X_test)
predicted = np.where(((predicted_SVC + predicted_KNN+predicted_RFC+predicted_RFC)/4) > 0.5, 1, 0)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = df_test['PassengerId']

submission_df['Survived'] = predicted

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)