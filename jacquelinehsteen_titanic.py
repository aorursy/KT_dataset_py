import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
# load data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# save passengerId for final submission
passengerId = test.PassengerId

# merge train and test (for OHE and more accurate mean values of age & fare)
df = train.append(test, sort=False, ignore_index=True)

#create index so we can separate data later on
train_idx = len(train)
test_idx = len(df)- len(test)

def clean_data(df):

    # drop Ticket & PassengerId columns, encode Sex with labels, fill missing values
    df.drop(columns=['Ticket', 'PassengerId'], inplace=True)
    df['Sex'].replace('female', 0, inplace=True)
    df['Sex'].replace('male', 1, inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Cabin'].fillna('Unknown', inplace=True)
    df['Fare'].fillna(df.Fare.median(), inplace=True)
    # df['Age'].fillna((train['Age'].mean()), inplace=True)

    # feature engineering - people traveling without family will be called "Lonewolf"s :)
    df['Lonewolf'] = np.where((df['SibSp'] == 0) & (df['Parch'] == 0), 1, 0)
    df.drop(columns=['SibSp', 'Parch'], inplace=True)
    
    #feature engineering - extract zone from cabin
    df.Cabin = df.Cabin.map(lambda x: x[0])

    return df

df = clean_data(df)

#extract title from name and use this column as a categorical value instead of free text

def get_title(name):
    """
    Use a regular expression to search for a title.  Titles always consist of
    capital and lowercase letters, and end with a period.
    
    Takes a name as input and returns the title string as output
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    
    # If title exists, extract 1st capture group and return it.
    
    if title_search:
        return title_search.group(1)
    return ""

TitleDictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royal",
                        "Don":        "Royal",
                        "Sir" :       "Royal",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "Countess":   "Royal",
                        "Dona":       "Royal",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royal"

                        }

def titlemap(x):
    """
    Takes a title as input and returns the dict value as output.
    """
    return TitleDictionary[x]

def further_clean(df):
    
    df['Title'] = df['Name'].apply(get_title)
    df['Title'] = df['Title'].apply(titlemap)

    # drop old columns
    df.drop(['Name'], axis=1, inplace=True)

    # group by Sex, Pclass, and Title 
    grouped = df.groupby(['Sex','Pclass', 'Title'])  

    # apply the grouped median value on the Age NaN
    df.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
    
    #print(df)
    
    return df

df = further_clean(df)

# Encode 'Embarkment' harbor & 'Title' & 'Pclass' & 'Cabin'
onehot_encoder = OneHotEncoder(sparse=False)

def encode(df):
    # Embarkment
    OHE = pd.DataFrame(onehot_encoder.fit_transform(df['Embarked'].to_numpy().reshape(-1,1)))
    OHE.index = df.index
    OHE.columns = ['is_C', 'is_Q', 'is_S']

    # Title
    OHT = pd.DataFrame(onehot_encoder.fit_transform(df['Title'].to_numpy().reshape(-1,1)))
    OHT.index = df.index
    OHT.columns = ['is_Mr', 'is_Mrs', 'is_Miss', 'is_Master', 'is_Royal', 'is_Officer']
    
    # Pclass
    OHP = pd.DataFrame(onehot_encoder.fit_transform(df['Pclass'].to_numpy().reshape(-1,1)))
    OHP.index = df.index
    OHP.columns = ['is_1', 'is_2', 'is_3']
    
    # Cabin
    OHC = pd.DataFrame(onehot_encoder.fit_transform(df['Cabin'].to_numpy().reshape(-1,1)))
    OHC.index = df.index
    OHC.columns = ['U', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']
    
    # drop old columns
    df.drop(['Embarked'], axis=1, inplace=True)
    df.drop(['Title'], axis=1, inplace=True)
    df.drop(['Pclass'], axis=1, inplace=True)
    df.drop(['Cabin'], axis=1, inplace=True)

    # add new columns to df
    df = pd.concat([df, OHE, OHT, OHP, OHC], axis=1)

    return df

df = encode(df)
print(df.columns)
corr = train.corr()
plt.figure(figsize=(30,15))
sns.set_style("ticks")
#colormap = sns.color_palette("coolwarm", 7)
sns.heatmap(data=corr, cmap="Spectral", annot=True, vmin = -1, vmax = 1, center = 0, fmt=".3f")
plt.legend()
plt.show()

# separate test from training data
train = df[:train_idx]
test = df[test_idx:]

# create target and feature variables
y = train['Survived'].astype(int)
X = train.drop(columns=['Survived'], axis=1)

# From the heatmap results, the following feature variables seems like a good place to start:
feat = ['Sex', 'Age', 'Fare', 'Lonewolf', 'is_C', 'is_Q', 'is_S',
       'is_Mr', 'is_Mrs', 'is_Miss', 'is_Master', 'is_Royal', 'is_Officer',
       'is_1', 'is_2', 'is_3', 'U', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']
        
X_test = test[feat]
X_trainfull = train[feat]

X_train, X_val, y_train, y_val = train_test_split(X_trainfull, y, random_state=0)
def get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val):
    """In order to optimize no of leaf nodes, calculate mean absolute error for a variety of lead nodes. 
    Takes a possible leaf node number as input together with our training data and returns a mae score.
    """
    
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return mae

for max_leaf_nodes in [3, 10, 20, 30, 100, 200, 300]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
    print(f"Max leaf nodes of: {max_leaf_nodes} resulted in MAE of: {my_mae}")
    
best_tree_model = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)
best_tree_model.fit(X_train, y_train)
y_pred = best_tree_model.predict(X_val)
print("Best tree model resulting in")
print(f"Accuracy score of: ", accuracy_score(y_val, y_pred))
print(f"Confusion matrix of: \n", confusion_matrix(y_val, y_pred))
print(classification_report(y_val,y_pred))
def get_mae(n_estimators, max_depth, X_train, X_val, y_train, y_val):
    """In order to optimize no of estimators, calculate mean absolute error for a variety of estimators. 
    Takes a possible no of estimators as input together with our training data and returns a mae score.
    """
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return mae

# create param grid object 
forrest_params = dict(     
    max_depth = [n for n in range(8, 10)],     
    min_samples_split = [n for n in range(4, 10)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(100, 300, 50)],
)
# gridsearchCV
forrest = RandomForestClassifier()
forrest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5, scoring="precision",
                   refit=True)
forrest_cv.fit(X_trainfull, y)

print("Best score: {}".format(forrest_cv.best_score_))
print("Optimal estimator: {}".format(forrest_cv.best_estimator_))
print("Best params: {}".format(forrest_cv.best_params_))
y_pred = forrest_cv.predict(X_test)
y_pred = y_pred.astype(int)
# dataframe with predictions
kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': y_pred})
# save to csv
kaggle.to_csv('titanic_sub.csv', index=False)
print(kaggle)


best_forest_model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=6, min_samples_leaf=2,  random_state=0)
best_forest_model.fit(X_train, y_train)
y_pred = best_forest_model.predict(X_val)
print("Best tree model resulting in")
print(f"Accuracy score of: ", accuracy_score(y_val, y_pred))
print(f"Confusion matrix of: \n", confusion_matrix(y_val, y_pred))
print(classification_report(y_val,y_pred))
def get_mae(max_depth, gamma, learning_rate, X_train, X_val, y_train, y_val):

    """In order to optimize parameters, calculate mean absolute error for a possible parameter tuning. 
    Takes a possible no of estimators as input together with our training data and returns a mae score.
    """
    model = XGBClassifier(n_estimators=1000, max_depth=max_depth, gamma=gamma, min_child_weight=1, learning_rate=learning_rate, random_state=0)
    model.fit(X_train, y_train,
             early_stopping_rounds=50,
             eval_set=[(X_val, y_val)],
             verbose=False)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return mae

for max_depth in [9, 10, 11]:
    for gamma in [0.1, 0.2, 0.3, 0.4]:
        for learning_rate in [0.05, 0.1]:
            my_mae = get_mae(max_depth, gamma, learning_rate, X_train, X_val, y_train, y_val)
            print(f"max depth of: {max_depth} and gamma of: {gamma} and learning rate of: {learning_rate} resulted in MAE of: {my_mae}")

best_XGB_model = XGBClassifier(n_estimators=1000, max_depth=10, gamma=0.4, min_child_weight=2, learning_rate=0.1, random_state=0)
best_XGB_model.fit(X_train, y_train,
                 early_stopping_rounds=100,
                 eval_set=[(X_val, y_val)],
                 verbose=False)
y_pred = best_XGB_model.predict(X_val)
print("Best tree model resulting in")
print(f"Accuracy score of: ", accuracy_score(y_val, y_pred))
print(f"Confusion matrix of: \n", confusion_matrix(y_val, y_pred))
print(classification_report(y_val,y_pred))
y_pred = best_XGB_model.predict(X_test)
# dataframe with predictions
kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': y_pred})
# save to csv
kaggle.to_csv('titanic_xgb_2.csv', index=False)
print(kaggle)