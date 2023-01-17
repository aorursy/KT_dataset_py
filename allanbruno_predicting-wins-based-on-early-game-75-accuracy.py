import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from xgboost import plot_tree as plot_xgboost
filepath = '../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv'

df = pd.read_csv(filepath)

df.head()
rows, columns = df.shape

print(f'Rows: {rows}, columns: {columns}')
nan_column_count = 0

for column in df.isna().sum():

    if column>0:

        print(column)

        nan_column_count+=1

if nan_column_count == 0:

    print('No missing values in your dataset!')
df.dtypes
plt.figure(figsize=(12,8))

plt.title('Distribution of target variable - blueWins')

plt.ylabel('Amount of wins')

sns.countplot(df['blueWins'])

print(df['blueWins'].value_counts())
len(df['gameId'].unique())
#Dropping 'gameId' variable

columns_to_drop = ['gameId']

df = df.drop('gameId', axis=1)
y = df['blueWins']
X = df.drop('blueWins', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)  
def test_classifier(model, X_train, X_test, y_train, y_test):      

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return f'{round(accuracy,4)*100}%'
tree_model = DecisionTreeClassifier(random_state=1)



first_run = test_classifier(tree_model, X_train, X_test, y_train, y_test)

print(f'The accuracy of your model is: {first_run}')

tree_model.fit(X_train, y_train)

predictions = tree_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10,7))

plt.ylabel('actual')

plt.xlabel('predicted')

ax = sns.heatmap(cm, annot=True, fmt='g')

ax.set_ylabel('Actual')

ax.set_xlabel('Predicted')

ax.set_title('Confusion matrix')

 #MELHORAR O PLOT DA CONFUSION MATRIX

    
for x in range(2,21,2):

    tree_model = DecisionTreeClassifier(max_leaf_nodes= x,random_state=1)

    test_model = test_classifier(tree_model, X_train, X_test, y_train, y_test)

    print(f'max leaf nodes: {x}- accuracy: {test_model}')
for depth in range(1,5):    

    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=1)

    test_model = test_classifier(tree_model, X_train, X_test, y_train, y_test)

    print(f'max_depth: {depth} - accuracy: {test_model}')
final_tree_model = DecisionTreeClassifier(max_depth = 3, random_state=1)

final_tree_model.fit(X_train, y_train)

predictions = final_tree_model.predict(X_test)

plt.figure(figsize=(20,10))

_ = plot_tree(final_tree_model, feature_names = X_train.columns, class_names = 'blueWins',

                rounded = True, proportion = False, precision = 2, filled = True)
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10,7))

plt.ylabel('actual')

plt.xlabel('predicted')

ax = sns.heatmap(cm, annot=True, fmt='g')

ax.set_ylabel('Actual')

ax.set_xlabel('Predicted')

ax.set_title('Confusion matrix')
diff_variables = ['WardsPlaced', 'WardsDestroyed','TotalMinionsKilled', 'Kills', 'Deaths', 'Assists', 'EliteMonsters', 'Dragons', 'Heralds', 'TowersDestroyed', 'AvgLevel']

variables_to_drop_red = ['GoldDiff', 'ExperienceDiff']

X_with_fe = pd.DataFrame()

X_with_fe['FirstBlood'] = X['blueFirstBlood']

for var in diff_variables:

    X_with_fe[f'Dif{var}'] = X[f'blue{var}'] - X[f'red{var}']



for var in variables_to_drop_red:

    X_with_fe[var] = X[f'blue{var}']

X_with_fe.head()
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_with_fe, y, random_state=1)
tree_model = DecisionTreeClassifier(max_leaf_nodes=2, random_state=1)

test_model = test_classifier(tree_model, X_train_fe, X_test_fe, y_train_fe, y_test_fe)

print(f'accuracy: {test_model}')
tree_model.fit(X_train_fe, y_train_fe)

predictions = tree_model.predict(X_test_fe)

cm = confusion_matrix(y_test_fe, predictions)

plt.figure(figsize=(10,7))

ax = sns.heatmap(cm, annot=True, fmt='g')

ax.set_ylabel('Actual')

ax.set_xlabel('Predicted')

_ = ax.set_title('Confusion matrix')
dataset_1 = (X_train, X_test, y_train, y_test, 'dataset_1') #Tuple of the unmodified dataset

dataset_2 = (X_train_fe, X_test_fe, y_train_fe, y_test_fe, 'dataset_2') #Tuple of the dataset with feature engineering
def test_classifier_2(model, dataset):

    model.fit(dataset[0], dataset[2])

    predictions = model.predict(dataset[1])

    accuracy = accuracy_score(dataset[3], predictions)

    return f'{round(accuracy,4)*100}%'
for est in range(25, 101, 25):

    for dataset in (dataset_1, dataset_2):

        rf_model = RandomForestClassifier(n_estimators = est, random_state=1)

        print(f'{dataset[4]} - n_estimators:{est} -  Your accuracy is: {test_classifier_2(rf_model, dataset)}')
for depth in range(2,11,2):

    for dataset in (dataset_1, dataset_2):

        rf_model = RandomForestClassifier(max_depth=depth,n_estimators = 100, random_state=1)

        print(f'max_estimators: {est} - depth:{depth} -{dataset[4]} - Your accuracy is: {test_classifier_2(rf_model, dataset)}')
final_rf_model = RandomForestClassifier(max_depth=6,n_estimators = 100, random_state=1)



print(f'Your final accuracy is: {test_classifier_2(final_rf_model, dataset_1)}')
plt.figure(figsize=(8,8))

plot_df = pd.DataFrame({'Models':['Decision tree', 'Random forest'], 'Accuracy':[73.28,74.33]})

ax = sns.barplot(x='Accuracy', y='Models', data=plot_df)
model_dict ={'Support Vector Classification': SVC(random_state=1),'Gaussian Naive Bayes':GaussianNB(), 'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=1), 'XGBoost': XGBClassifier()}

for model in model_dict:

    print(f'model:{model} - accuracy: {test_classifier_2(model_dict[model], dataset_1)}')
def xboost_func(model, dataset):



    model.fit(dataset[0], dataset[2], 

             early_stopping_rounds=50, 

             eval_set=[(dataset[1], dataset[3])], 

             verbose=False)

    predictions = model.predict(dataset[1])

    accuracy = accuracy_score(dataset[3], predictions)

    return f'{round(accuracy,4)*100}%'





XGBoost = XGBClassifier(n_estimators=1000, learning_rate=0.06, max_depth= 3, subsample= 0.9, colsample_bytree= 1, gamma= 1)

for df in (dataset_1, dataset_2):  

    test_model = xboost_func(XGBoost,df)

    print(f'XGBoost accuracy on {df[4]}: {test_model}')
XGBoost.fit(dataset[0], dataset[2], 

             early_stopping_rounds=50, 

             eval_set=[(dataset[1], dataset[3])], 

             verbose=False)

predictions = XGBoost.predict(dataset[1])

fig, ax = plt.subplots()

fig.set_size_inches(16,18)

_ = plot_xgboost(XGBoost, num_trees=4, ax=ax, rankdir='LR')


cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10,7))

ax = sns.heatmap(cm, annot=True, fmt='g')

ax.set_ylabel('Actual')

ax.set_xlabel('Predicted')

_ = ax.set_title('Confusion matrix')
Falses_dataset = pd.DataFrame({'validation':y_test,'predicted':predictions})

Falses_dataset = pd.concat([Falses_dataset,X_test], axis=1)

Falses_dataset.loc[Falses_dataset['validation'] == Falses_dataset['predicted'], "Equal"] = 'Equal'

Falses_dataset.loc[Falses_dataset['validation'] != Falses_dataset['predicted'], "Equal"] = 'Wrong'
Falses_dataset.drop(Falses_dataset[Falses_dataset.Equal != 'Wrong'].index, inplace=True)

Falses_dataset.head()
Falses_dataset.loc[(Falses_dataset.validation == 0) & (Falses_dataset.predicted == 1), 'kind'] = 'FALSE POSITIVE'

Falses_dataset.loc[(Falses_dataset.validation == 1) & (Falses_dataset.predicted == 0), 'kind'] = 'FALSE NEGATIVE'

Falses_dataset.head()

mask_fn = (Falses_dataset['kind'] == 'FALSE NEGATIVE')

mask_fp = (Falses_dataset['kind'] == 'FALSE POSITIVE')
Falses_dataset['redKillsDiff'] = Falses_dataset['redKills'] - Falses_dataset['blueKills']

selected_columns = ['redGoldDiff', 'redExperienceDiff', 'redKillsDiff']

for column in selected_columns:

    col_str = column.replace('red', '')   

    Falses_dataset[column] = Falses_dataset[column].map(abs)

    plt.figure(figsize=(10,8))

    ax = sns.barplot(x = 'kind', y =column, data=Falses_dataset)

    ax.set_ylabel(f' Absolute {col_str}')

    _ = ax.set_title(f'Avg absolute {col_str}')

    ax.set_xlabel('kind of error')

    plt.subplot(ax)
importance_df = pd.DataFrame({'Variables':dataset_2[0].columns,

              'Importance':XGBoost.feature_importances_}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12,8))

plt.xticks(rotation=20)

plt.title('Importance of each variable to XGBoost Classifier model')

_ = sns.barplot(x='Variables', y='Importance',data=importance_df )