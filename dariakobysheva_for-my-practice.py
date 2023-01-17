# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
# Explore pattern: woman and man survival rate
# titanic_women = titanic_train_data[titanic_train_data.Sex == 'female']['Survived']
# women_survival_rate = sum(titanic_women) / len(titanic_women)

# titanic_men = titanic_train_data[titanic_train_data.Sex == 'male']['Survived']
# men_survival_rate = sum(titanic_men) / len(titanic_men)

# print("% of men who survived:", women_survival_rate)
# print("% of women who survived:", men_survival_rate)
# Prepare to restructure features
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan

def group_titles(title): 
    if(title in {'Lady.', 'the', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'}):
        return 'Rare.'
    elif(title in {'Mlle.', 'Miss.', 'Ms.'}):
        return 'Miss.'
    elif(title == 'Mme.'):
        return 'Mrs.'
    else:
        return title
# Construct feature title
train_titles = [name.split(',')[1].strip().split(' ')[0] for name in titanic_train_data['Name']]
test_titles = [name.split(',')[1].strip().split(' ')[0] for name in titanic_test_data['Name']]

titanic_train_data['Title'] = train_titles
titanic_test_data['Title'] = test_titles

titanic_train_data['Title'] = titanic_train_data['Title'].apply(group_titles)
titanic_test_data['Title'] = titanic_test_data['Title'].apply(group_titles)

# Turning cabin number into Deck
titanic_train_data.Cabin = titanic_train_data.Cabin.replace(np.nan, 'U')
titanic_test_data.Cabin = titanic_test_data.Cabin.replace(np.nan, 'U')
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'U']
titanic_train_data['Deck']=titanic_train_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
titanic_test_data['Deck']=titanic_test_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

# Creating new family size column
titanic_train_data['FamilySize']=titanic_train_data['SibSp'] + titanic_train_data['Parch'] + 1
titanic_test_data['FamilySize']=titanic_test_data['SibSp'] + titanic_test_data['Parch'] + 1

#pd.set_option('display.max_rows', None)
#titanic_train_data.dtypes
#titanic_train_data[titanic_train_data['Title']=='the']
titanic_train_data.Title.unique()
titanic_train_data[titanic_train_data['Title'] == 'Master.']
# Explore pattern: correlation matrix
drop_list = []
titanic_useful_train_data = titanic_train_data.drop(drop_list,axis = 1)
titanic_useful_test_data = titanic_test_data.drop(drop_list,axis = 1)
#Combine datasets 
combine = [titanic_useful_train_data, titanic_useful_test_data]
# Fill in missing Age
# https://www.kaggle.com/nikhilkmr300/titanic-detailed-eda-and-feature-engineering#Dealing-with-nulls

for dataset in combine:
    corr_with_pclass = dataset['Age'].corr(dataset['Pclass'])
    print(f'Correlation of Age with Pclass = {round(corr_with_pclass, 3)}')

    null_ids_train = dataset[dataset['Age'].isnull()].index.tolist()
    age_per_group = dataset.groupby(by=['Title', 'Pclass']).median()['Age']
    print(age_per_group)
    
    for index in null_ids_train:
        title = dataset.loc[index, 'Title']
        pclass = dataset.loc[index, 'Pclass']
        dataset.loc[index, 'Age'] = age_per_group[(title, pclass)]
        
    print('-' *40)
# Put age into band
concated_age = pd.concat([titanic_useful_train_data['Age'], titanic_useful_test_data['Age']])
pd.qcut(concated_age, q=8).unique()
# Create new feature Age * Class
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 19, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 19) & (dataset['Age'] <= 25), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 40, 'Age'] = 4

    
for dataset in combine:
    dataset['AgeClass'] = (dataset.Age * dataset.Pclass).astype(int)
# Fare for #152 is Nan, he is a 3rd class passenger
# Assign average fare for 3rd class passenger to #152

fare_per_group = titanic_useful_test_data.groupby(by=['Title', 'Pclass']).mean()['Fare']
null_ids_test = titanic_useful_test_data[titanic_useful_test_data['Fare'].isnull()].index.tolist()

for index in null_ids_test:
    title = titanic_useful_test_data.loc[index, 'Title']
    pclass = titanic_useful_test_data.loc[index, 'Pclass']
    titanic_useful_test_data.loc[index, 'Fare'] = fare_per_group[(title, pclass)]

# or imputer = SimpleImputer(strategy='mean')
# Encode Sex
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for dataset in combine:
    dataset['Sex'] = encoder.fit_transform(dataset.Sex)
# Assign most frequent to missing port
frequent_port = titanic_useful_train_data.Embarked.dropna().mode()[0]
titanic_useful_train_data['Embarked'] = titanic_useful_train_data.Embarked.fillna(frequent_port)
titanic_useful_train_data[['Embarked', 'Survived']].groupby(by=['Embarked'],as_index=False).mean()
# Assign numeric value to Embarkation
for dataset in combine:
    dataset['Embarked'] = dataset.Embarked.map({'S': -1, 'C': 0, 'Q': 1} ).astype(int)
# Create IsAlone feature
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset.FamilySize == 1, "IsAlone"] = 1
    
titanic_useful_train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# Farebin
for dataset in combine: 
    dataset['FareBin'] = pd.qcut(dataset.Fare, 6)
    dataset['FareBin'] = encoder.fit_transform(dataset.FareBin)
#Encode Title
for dataset in combine:
     dataset['TitleCode'] = dataset.Title.map({'Mr.': 1, 'Miss.': 2, 'Mrs.': 3, 
                                               'Master.': 4, 'Rare.': 5}).astype(int)
# Encode Cabin
for dataset in combine:
    dataset['CabinCode'] = dataset.Deck.map({'A': 1, 'B': 2, 'C': 3, 
                                              'D': 4, 'E': 5, 'F': 6, 
                                              'T': 7, 'G': 8, 'U': 9}).astype(int)
    
titanic_useful_train_data[['CabinCode', 'Survived']].groupby(['CabinCode'], as_index=False).mean()
# is Mother
for dataset in combine:
    dataset['isMother'] = 0
    dataset.loc[(dataset.Sex == 0) & (dataset.Parch > 0) & (dataset.Age > 0) 
                & (dataset.Title != 'Miss.'), "isMother"] = 1
# Adding Family_Survival Feature as suggested by:
# https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83

data_df = titanic_useful_train_data.append(titanic_useful_test_data)

data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])

for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

# Family_Survival in TRAIN_DF and TEST_DF:
titanic_useful_train_data['Family_Survival'] = data_df['Family_Survival'][:891]
titanic_useful_test_data['Family_Survival'] = data_df['Family_Survival'][891:]
corr = titanic_useful_train_data.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps
# Random Forest Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

seed = 5

features = ["Sex", "TitleCode", "AgeClass", "Pclass", "Fare", "CabinCode", "FamilySize", "Family_Survival"]
X = pd.get_dummies(titanic_useful_train_data[features])
y = titanic_useful_train_data["Survived"]
X_competition = pd.get_dummies(titanic_useful_test_data[features])

titanicRandomForestModel = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)

scores = cross_val_score(titanicRandomForestModel, X, y, cv=7)
print("Scores: ", scores)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
# Fit and print important feature
titanicRandomForestModel.fit(X, y)
predictions = titanicRandomForestModel.predict(X_competition)

feature_imp = pd.Series(titanicRandomForestModel.feature_importances_,index=X.columns).sort_values(ascending=False)
print('Feature Importance')
print('-' *40)
print(feature_imp)
# Save output
from datetime import datetime
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_random_forest_' + timestamp + '.csv', index=False)
print("Your submission was successfully saved!")
# Visualize Forest
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(titanicRandomForestModel.estimators_[1], 
                out_file='tree.dot', 
                feature_names = X_competition.columns,
                class_names = ['Survived', 'Dead'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
# Using XGBoost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

params = {"objective":"reg:logistic",'colsample_bytree': 0.2,'learning_rate': 0.1,
                'max_depth': 5, 'lambda': 1, 'random_state' : seed}

data_dmatrix = xgb.DMatrix(data=X,label=y)
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=7,
                    num_boost_round=50,early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=seed)
model_xgb = xgb.XGBClassifier(**params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# make predictions for test data
predictions = model_xgb.predict(X_competition)
print(predictions)
# Save output
from datetime import datetime
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_xgboost' + timestamp + '.csv', index=False)
print("Your submission was successfully saved!")
## Use TensorFlow Boosted Trees
import tensorflow as tf
tf.random.set_seed(seed)
print(tf.__version__)
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from matplotlib import pyplot as plt
#Create Feature Columns
#features = ["Sex", "TitleCode", "AgeClass", "Pclass", "Fare", "CabinCode", "FamilySize", "Family_Survival"]

CATEGORICAL_COLUMNS = ['Sex', 'TitleCode', 'Pclass', 'CabinCode']
NUMERIC_COLUMNS = ['AgeClass', 'Fare', 'FamilySize', 'Family_Survival']

def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = X_train[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                           dtype=tf.float32))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_test, y_test, shuffle=False, n_epochs=1)
# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.

params = {
    'n_batches_per_layer': 1,
    'n_trees': 200, # 100
    'max_depth': 5, # 6
    'center_bias': True, # False
    'l2_regularization': 0.01
}

est = tf.estimator.BoostedTreesClassifier(feature_columns,**params)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))
# make predictions for test data
# https://www.kaggle.com/ysanojpn/titanic-boosted-tree-tensorflow2
def input_fn(features,batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

tf_predictions = est.predict(input_fn=lambda: input_fn(X_competition,1))

predictions = pd.Series([pred['class_ids'][0] for pred in tf_predictions])
print(predictions)


# Save output
from datetime import datetime
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_tfBoostedTrees' + timestamp + '.csv', index=False)
print("Your submission was successfully saved!")
print(y)