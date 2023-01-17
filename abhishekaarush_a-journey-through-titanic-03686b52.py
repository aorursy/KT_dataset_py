import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

# Fare column is just one value empty so we will fill it with median value
big_X['Fare'] = big_X['Fare'].fillna(big_X['Fare'].median())

# Creating probabilty distribution for ages from the existing column values
ages_probabilities = big_X['Age'].value_counts().to_frame()
ages_probabilities['index1'] = ages_probabilities.index
ages_probabilities = ages_probabilities.rename(columns={'Age': 'Count', 'index1': 'Age'})
ages_probabilities = ages_probabilities.reindex_axis(['Age','Count'], axis=1)
ages_probabilities = ages_probabilities.reset_index()
ages_probabilities = ages_probabilities.drop(["index"],axis=1)
ages_probabilities['Probability'] = ages_probabilities['Count'] / big_X['Age'].value_counts().sum()

input_ages_list = ages_probabilities['Age'].values.tolist()
props_ages_list = ages_probabilities['Probability'].values.tolist()
newAges = np.random.choice(input_ages_list, big_X['Age'].isnull().sum(), props_ages_list)

# fill Ages null values with this distribution
AgeNulls = big_X[pd.isnull(big_X['Age'])]
for i, ni in enumerate(AgeNulls.index[:len(newAges)]):
    big_X['Age'].loc[ni] = newAges[i]

big_X_imputed = big_X

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
big_X_imputed['Sex'] = le.fit_transform(big_X_imputed['Sex'])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)