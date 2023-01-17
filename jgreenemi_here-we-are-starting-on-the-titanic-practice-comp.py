import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Set
# This is scratchspace where Python code will go.

pd.DataFrame(columns=['test', 'test2']).describe()

# Scratchspace ends here.
# -----------------------------------------------
# Input data is in "../input". Here we have the Titanic data files.
print(os.listdir("../input"))
PATH = '../input'
file_train = f'{PATH}/train.csv'
file_test = f'{PATH}/test.csv'
file_gender_submission = f'{PATH}/gender_submission.csv'
data_train_all = pd.read_csv(file_train, sep=',')
data_test = pd.read_csv(file_test, sep=',')
data_test.head()
# Verify the contents of the files are what we expect.
print(f'{data_train_all.shape} for training\n{data_test.shape} for testing')
features_categorical = [
    'Survived',
    'PassengerId',
    'Pclass',  # Socioeconomic status, generally.
    'Sex'
]

features_continuous = [
    'Age',
    'SibSp', # Sibling or Spouse
    'Parch' # Parents or Children
]
# When we go to create the data_test object below we want to get all features except 'Survived' since that won't be present outside the training data.
print(features_categorical[1:])

# Drop the columns we don't plan to use.
data_train_all = data_train_all[features_continuous + features_categorical]
data_test = data_test[features_continuous + features_categorical[1:]]
data_test.head()
# Let's have a look through the describe() results for the numerical columns.
data_test.describe()
#data_train_all.get(features_categorical).head()
#data_train_all.get(features_continuous).head()
for featurename in (features_categorical + features_continuous):
    if featurename not in ['PassengerId']:
        print(f'{featurename} Unique Values:')
        print(data_train_all[featurename].unique())
    #if featurename in ['Age']:
    #    for age in data_train[featurename]:
    #        if 'nan' in str(age).lower():
    #            print(age)

mean_age = data_train_all['Age'].mean()  # 29.69911764705882
data_train_all['Age'] = data_train_all['Age'].fillna(mean_age)

if 'nan' in str(data_train_all['Age']).lower():
    print('Missed at least one!')    
    
mean_age_test = data_test['Age'].mean()  # 29.69911764705882.  Wasn't expecting that.
data_test['Age'] = data_test['Age'].fillna(mean_age_test)

if 'nan' in str(data_test['Age']).lower():
    print('Missed at least one!')    
print(f'{mean_age} vs {mean_age_test}')
for featurename in (features_categorical + features_continuous):
    if featurename not in ['PassengerId']:
        print(f'{featurename} Unique Values:')
        print(data_train_all[featurename].unique())
for featurename in (features_categorical[1:] + features_continuous):
    if featurename not in ['PassengerId']:
        print(f'{featurename} Unique Values:')
        print(data_test[featurename].unique())
data_train_all['Sex'] = data_train_all['Sex'].replace(to_replace='male', value=0)
data_train_all['Sex'] = data_train_all['Sex'].replace(to_replace='female', value=1)
data_test['Sex'] = data_test['Sex'].replace(to_replace='male', value=0)
data_test['Sex'] = data_test['Sex'].replace(to_replace='female', value=1)
data_test['Sex'].unique()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Split the data into training and evaluation sets. This must be done after the preprocessing to avoid duplicating work across two DataFrames.
data_train, data_eval = np.split(
    data_train_all, 
    [
        int(0.8*len(data_train_all))
    ]
)

# Could have done this with sklearn with the shufflesplit. Can try on later notebooks.

print(f'{len(data_train)} / {len(data_eval)}\n{len(data_train)/len(data_train_all)} / {len(data_eval)/len(data_train_all)}')
X = data_train.drop(columns=['Survived'])
y = data_train['Survived']
X_eval = data_eval.drop(columns=['Survived'])
y_eval = data_eval['Survived']

y_eval.head()
# Standardize our features.
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_eval_std = scaler.fit_transform(X_eval)
# Create the Logistic Regression.
clf = LogisticRegression(random_state=0)
# And now we train!
#model = clf.fit(X_std, y)

# Gonna try feeding in unstandardized features.
model = clf.fit(X, y)
# Once the model has been trained, we want to pass it a prediction to test its output. Let's build one.
prediction = pd.DataFrame({
    'Age': [12, 30],
    'SibSp': [2, 1],
    'Parch': [0, 1],
    'Pclass': [1, 3],
    'Sex': [0, 1],
    'Survived': [0, 1]
})

model.predict(prediction)
#model.predict_proba(prediction)
evaluation_predictions = model.predict(X_eval)
total_eval_predictions = len(evaluation_predictions)
incorrect_counter = 0
for i in range(0, total_eval_predictions):
    if evaluation_predictions[i] != y_eval.iloc[i]:
        incorrect_counter += 1
print(f'The resulting accuracy: {(total_eval_predictions - incorrect_counter) / total_eval_predictions * 100}% correct!')
X_test = data_test
X_test_std = scaler.fit_transform(X_test)

test_prediction_results = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': X_test['PassengerId'],
    'Survived': test_prediction_results
})
submission.to_csv('titanic-test-results.csv', sep=',', index=False)
pd.read_csv('titanic-test-results.csv')