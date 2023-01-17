import pandas as pd
#import autosklearn.classification
import featuretools as ft
from featuretools.primitives import *
from featuretools.variable_types import Numeric
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
answers = pd.read_csv('../input/gender_submission.csv')
print(train_df.columns.values)
combine = train_df.append(test_df)

passenger_id=test_df['PassengerId']
#combine.drop(['PassengerId'], axis=1, inplace=True)
combine = combine.drop(['Ticket', 'Cabin'], axis=1)

combine.Fare.fillna(combine.Fare.mean(), inplace=True)

combine['Sex'] = combine.Sex.apply(lambda x: 0 if x == "female" else 1)

for name_string in combine['Name']:
    combine['Title']=combine['Name'].str.extract('([A-Za-z]+)\.',expand=True)
    
#replacing the rare title with more common one.
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
combine.replace({'Title': mapping}, inplace=True)

combine = combine.drop(['Name'], axis=1)

titles=['Mr','Miss','Mrs','Master','Rev','Dr']
for title in titles:
    age_to_impute = combine.groupby('Title')['Age'].median()[titles.index(title)]
    combine.loc[(combine['Age'].isnull()) & (combine['Title'] == title), 'Age'] = age_to_impute
combine.isnull().sum()

freq_port = train_df.Embarked.dropna().mode()[0]
combine['Embarked'] = combine['Embarked'].fillna(freq_port)
    
combine['Embarked'] = combine['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
combine['Title'] = combine['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rev': 4, 'Dr': 5} ).astype(int)
combine.fillna(0, inplace=True)
combine.info()
es = ft.EntitySet(id = 'titanic_data')

es = es.entity_from_dataframe(entity_id = 'combine', dataframe = combine.drop(['Survived'], axis=1), 
                              variable_types = 
                              {
                                  'Embarked': ft.variable_types.Categorical,
                                  'Sex': ft.variable_types.Boolean,
                                  'Title': ft.variable_types.Categorical
                              },
                              index = 'PassengerId')

es
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Embarked', index='Embarked')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Sex', index='Sex')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Title', index='Title')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Pclass', index='Pclass')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Parch', index='Parch')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='SibSp', index='SibSp')
es
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(primitives[primitives['type'] == 'aggregation'].shape[0])
primitives[primitives['type'] == 'transform'].head(primitives[primitives['type'] == 'transform'].shape[0])
features, feature_names = ft.dfs(entityset = es, 
                                 target_entity = 'combine', 
                                 max_depth = 2)
feature_names
len(feature_names)
features[features['Age'] == 22][["Title.SUM(combine.Age)","Age","Title"]].head()
# Threshold for removing correlated variables
threshold = 0.95

# Absolute value correlation matrix
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head(50)
# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d features to remove.' % (len(collinear_features)))
features_filtered = features.drop(columns = collinear_features)

print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])
features_positive = features_filtered.loc[:, features_filtered.ge(0).all()]

train_X = features_positive[:train_df.shape[0]]
train_y = train_df['Survived']

test_X = features_positive[train_df.shape[0]:]
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_X, train_y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(train_X)
X_selected_df = pd.DataFrame(X_new, columns=[train_X.columns[i] for i in range(len(train_X.columns)) if model.get_support()[i]])
X_selected_df.shape
X_selected_df.columns
random_forest = RandomForestClassifier(n_estimators=2000,oob_score=True)
random_forest.fit(X_selected_df, train_y)
X_selected_df.shape
Y_pred = random_forest.predict(test_X[X_selected_df.columns])
print(Y_pred)
my_submission = pd.DataFrame({'PassengerId': passenger_id, 'Survived': Y_pred})
my_submission.to_csv('submission.csv', index=False)