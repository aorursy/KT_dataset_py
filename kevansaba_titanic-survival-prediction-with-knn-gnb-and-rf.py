# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

import sklearn.metrics as metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')

template_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

compiled_results = template_submission.drop('Survived', axis=1)

train.head(10)
def aggregate_trainset(argument_array, df):

    df_agg = df.groupby(argument_array).agg(

        count_survived = pd.NamedAgg('Survived', 'sum'),

        count_total = pd.NamedAgg('Survived', 'count')

    ).reset_index()

    df_agg['count_deaths'] = df_agg['count_total']-df_agg['count_survived']

    df_agg['pct_survived'] = df_agg['count_survived']/df_agg['count_total']

    return df_agg



def categorize_age(age):

    if age <= 15:

        return '0 to 15', 1

    if age <= 35:

        return '16 to 35', 2

    if age <= 55:

        return '36 to 55', 3

    if age > 55:

        return '55+', 4

    else:

        return 'N/A', 0

    

def categorize_fare(fare):

    if fare < 15:

        return 'Under $15', 1

    if fare < 35:

        return '$15 to $35', 2

    if fare < 100:

        return '$35 to $100', 3

    if fare > 100:

        return 'Over $100', 4

    else:

        return 'N/A', 0

    

def return_train_test_sets(df, features_list, test_size, random_state):

    df = conduct_feature_engineering(df)

    

    X = df[features_list].copy()

    y = df[['Survived']].copy()



    # Categorical boolean mask

    categorical_feature_mask = X.dtypes==object

    # filter categorical columns using mask and turn it into a list

    categorical_cols = X.columns[categorical_feature_mask].tolist()



    #encode categorical variables

    le = LabelEncoder()

    # apply label encoder on categorical feature columns

    X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))



    # # fill all NaN with -1

    X.fillna(value=-1, inplace=True)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test



def return_final_train_test_sets(df_train, df_test, features_list, test_size, random_state , fillna=True):

    df_test['Survived'] = -1

    full_set = df_train.append(df_test, sort=False).reset_index(drop=True)

    full_set = conduct_feature_engineering(full_set)



    # Categorical boolean mask

    categorical_feature_mask = full_set.dtypes==object

    # filter categorical columns using mask and turn it into a list

    categorical_cols = full_set.columns[categorical_feature_mask].tolist()



    # encode categorical variables

    le = LabelEncoder()

    # apply label encoder on categorical feature columns

    full_set[categorical_cols] = full_set[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))



    if fillna == True:

        # fill all NaN with -1

        full_set.fillna(value=-1, inplace=True)



    # keep only necessary columns + Survived column

    features_list = features_list+['Survived']

    full_set = full_set[features_list]



    #Create final datasets used to train model

    final_X_train = full_set.loc[full_set['Survived']!=-1].drop('Survived', axis=1)

    final_y_train = full_set.loc[full_set['Survived']!=-1, 'Survived'].values

    final_X_test = full_set.loc[full_set['Survived']==-1].drop('Survived', axis=1)

    return final_X_train, final_y_train, final_X_test



def conduct_feature_engineering(df):

    df['age_category'] = df['Age'].apply(lambda x: categorize_age(x)[0])

    df['age_cat_rank'] = df['Age'].apply(lambda x: categorize_age(x)[1])

    df['fare_cat'] = df['Fare'].apply(lambda x: categorize_fare(x)[0])

    df['fare_cat_rank'] = df['Fare'].apply(lambda x: categorize_fare(x)[1])

    df['cabin_zone'] = df['Cabin'].astype(str).apply(lambda x: x[0:1] if x != 'nan' else None)

    return df
fig = px.scatter(train, x='Fare', y='Age', color='Survived')

fig.show()
fig = px.histogram(train, x='Fare', histfunc='count', histnorm='percent', color='Survived')

fig.show()
df = conduct_feature_engineering(train)

aggregation_criteria = ['fare_cat', 'fare_cat_rank']

graph_data = aggregate_trainset(aggregation_criteria, df)

fig = px.bar(graph_data.sort_values('fare_cat_rank'), x='fare_cat', y='pct_survived')

fig.show()
df = conduct_feature_engineering(train)

aggregation_criteria = ['Pclass']

graph_data = aggregate_trainset(aggregation_criteria, df)

fig = px.bar(graph_data, x='Pclass', y='pct_survived')

fig.show()
df = conduct_feature_engineering(train)

aggregation_criteria = ['Embarked']

graph_data = aggregate_trainset(aggregation_criteria, df)

fig = px.bar(graph_data, x='Embarked', y='pct_survived')

fig.show()
df = conduct_feature_engineering(train)

aggregation_criteria = ['Sex']

graph_data = aggregate_trainset(aggregation_criteria, df)

fig = px.bar(graph_data, x='Sex', y='pct_survived')

fig.show()
df = conduct_feature_engineering(train)

aggregation_criteria = ['Parch']

graph_data = aggregate_trainset(aggregation_criteria, df)

fig = px.bar(graph_data, x='Parch', y='pct_survived')

fig.show()
df = conduct_feature_engineering(train)

aggregation_criteria = ['age_category', 'age_cat_rank', 'Sex']

graph_data = aggregate_trainset(aggregation_criteria, df)

fig = px.bar(graph_data.sort_values('age_cat_rank'), x='age_category', y='pct_survived', facet_col='Sex')

fig.show()
features_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'age_category', 'fare_cat', 'cabin_zone']

X_train, X_test, y_train, y_test = return_train_test_sets(df=train, features_list=features_list, test_size=0.2, random_state=4)

final_X_train, final_y_train, final_X_test = return_final_train_test_sets(df_train=train, df_test=test, features_list=features_list, test_size=0.2, random_state=4)
k_range = range(1, 25)

scores = {}

scores_list = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train['Survived'].values)

    y_pred=knn.predict(X_test)

    scores[k] = metrics.accuracy_score(y_test['Survived'].values,y_pred)

    scores_list.append(metrics.accuracy_score(y_test['Survived'].values,y_pred))
graph_data = pd.Series(scores).to_frame('accuracy').reset_index()

graph_data.rename(columns={'index':'k-factor'}, inplace=True)



fig = px.bar(graph_data, x='k-factor', y='accuracy')

fig.show()
#solving test df based on best performing # neighbors

final_knn = KNeighborsClassifier(n_neighbors=10)

final_knn.fit(final_X_train, final_y_train)

y_pred_final = final_knn.predict(final_X_test)
submission = pd.Series(y_pred_final).to_frame('Survived').reset_index()

submission.rename(columns={'index':'PassengerId'}, inplace=True)

submission['PassengerId'] = submission['PassengerId']+892

compiled_results = pd.merge(compiled_results, submission[['PassengerId', 'Survived']], on='PassengerId', how='left')

compiled_results.rename(columns={'Survived':'Survived_KNN'}, inplace=True)

submission.to_csv('submission_KNN.csv', index=False)
features_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'age_category', 'fare_cat', 'cabin_zone']

X_train, X_test, y_train, y_test = return_train_test_sets(df=train, features_list=features_list, test_size=0.2, random_state=4)

final_X_train, final_y_train, final_X_test = return_final_train_test_sets(df_train=train, df_test=test, features_list=features_list, test_size=0.2, random_state=4)
gnb = GaussianNB()

gnb.fit(X_train, y_train['Survived'].values)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#solving test set

final_gnb = GaussianNB()

final_gnb.fit(final_X_train, final_y_train)

y_pred_final = final_gnb.predict(final_X_test)
submission = pd.Series(y_pred_final).to_frame('Survived').reset_index()

submission.rename(columns={'index':'PassengerId'}, inplace=True)

submission['PassengerId'] = submission['PassengerId']+892

compiled_results = pd.merge(compiled_results, submission[['PassengerId', 'Survived']], on='PassengerId', how='left')

compiled_results.rename(columns={'Survived':'Survived_GNB'}, inplace=True)

submission.to_csv('submission_GNB.csv', index=False)
features_list = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'cabin_zone']

X_train, X_test, y_train, y_test = return_train_test_sets(df=train, features_list=features_list, test_size=0.2, random_state=4)

final_X_train, final_y_train, final_X_test = return_final_train_test_sets(df_train=train, df_test=test, features_list=features_list, test_size=0.2, random_state=4)
clf=RandomForestClassifier(n_estimators=1000, max_depth=6)

clf.fit(X_train,y_train['Survived'].values)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#solving on test set

final_clf = RandomForestClassifier(n_estimators=1000)

final_clf.fit(final_X_train, final_y_train)

y_pred_final = final_clf.predict(final_X_test)
submission = pd.Series(y_pred_final).to_frame('Survived').reset_index()

submission.rename(columns={'index':'PassengerId'}, inplace=True)

submission['PassengerId'] = submission['PassengerId']+892

compiled_results = pd.merge(compiled_results, submission[['PassengerId', 'Survived']], on='PassengerId', how='left')

compiled_results.rename(columns={'Survived':'Survived_RF'}, inplace=True)

submission.to_csv('submission_RF.csv', index=False)
features_list = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'cabin_zone']

X_train, X_test, y_train, y_test = return_train_test_sets(df=train, features_list=features_list, test_size=0.2, random_state=4)

final_X_train, final_y_train, final_X_test = return_final_train_test_sets(df_train=train, df_test=test, features_list=features_list, test_size=0.2, random_state=4)
abc = list(range(1,9))
hidden_layers = list(range(1, 9))

neurons_per_layer = list(range(5,51))

neural_net_testing = pd.DataFrame(columns=['nb_hidden_layers', 'nb_neurons', 'accuracy'])

for layer in hidden_layers:

    for neurons in neurons_per_layer:   

        mlpc = MLPClassifier(hidden_layer_sizes=(neurons, layer), max_iter=1500, alpha=1e-4,

                            solver='sgd', verbose=False, tol=1e-4, random_state=1,

                            learning_rate_init=.01)

        mlpc.fit(X_train, y_train['Survived'].values)

        y_pred=mlpc.predict(X_test)

        neural_net_testing = neural_net_testing.append(pd.Series([layer, neurons, metrics.accuracy_score(y_test, y_pred)], index=neural_net_testing.columns ), ignore_index=True)

        

fig = px.scatter(neural_net_testing, x='nb_hidden_layers', y='nb_neurons', color='accuracy')

fig.show()
print('Survival rate based on KNN: {}'.format(sum(compiled_results['Survived_KNN'].values)/compiled_results.shape[0]))

print('Survival rate based on Gaussian Naive Bayes: {}'.format(sum(compiled_results['Survived_GNB'].values)/compiled_results.shape[0]))

print('Survival rate based on Random Forests: {}'.format(sum(compiled_results['Survived_RF'].values)/compiled_results.shape[0]))

print('Average surival rate in train set: {}'.format(sum(train['Survived'].values/train.shape[0])))
compiled_results['AverageSurvival'] = (compiled_results['Survived_KNN']+compiled_results['Survived_GNB']+compiled_results['Survived_RF'])/3

compiled_results['AverageSurvival'].value_counts()