# Basic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# File

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.shape
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.shape
df_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df_sub.shape
df_train.head()
df_train.info()
df_train.isnull().sum()
women = df_train.loc[df_train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = df_train.loc[df_train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
#df = pd.concat([df_train, df_test])#.reset_index(drop=True)

#df.shape
df_train = df_train.fillna(-999)

df_test = df_test.fillna(-999)
'''

split = len(df_train)

train = df[:split]

test = df[split:]

'''
# Get train and validation sub-datasets

from sklearn.model_selection import train_test_split



X = df_train.drop(["Survived"], axis=1)

y = df_train["Survived"]



#Do train data splitting

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, random_state=42)
'''

from sklearn.ensemble import RandomForestClassifier



y = df_train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(df_train[features])

X_test = pd.get_dummies(df_test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

'''
# Libs

from catboost import CatBoostClassifier, Pool, cv

from sklearn.metrics import accuracy_score



# Select categorical indices

cat_features_indices = np.where(X.dtypes != float)[0]



# Define the model

model = CatBoostClassifier(

    eval_metric='Accuracy',

    loss_function='Logloss',

    #iterations=150,

    use_best_model=True,

    random_seed=42,

    logging_level='Silent'

)



#now just to make the model to fit the data

model.fit(X_train,y_train,cat_features=cat_features_indices,eval_set=(X_test,y_test), plot=True)
#TODO: Early stopping
cv_params = model.get_params()

cv_params.update({

    'loss_function': 'Logloss'

})

cv_data = cv(

    Pool(X, y, cat_features=cat_features_indices),

    cv_params,

    plot=True

)
print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(

    np.max(cv_data['test-Accuracy-mean']),

    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],

    np.argmax(cv_data['test-Accuracy-mean'])

))
print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))
# Create pool

train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)

validate_pool = Pool(X_test, y_test, cat_features=cat_features_indices)
# Feature importance

feature_importances = model.get_feature_importance(train_pool)

feature_names = X_train.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    print('{}: {}'.format(name, score))
eval_metrics = model.eval_metrics(validate_pool, ['AUC'], plot=True)
# Re-train model with full data

#model.fit(X,y,cat_features=cat_features_indices)
# Make predictions

predictions = model.predict(df_test)

predictions_probs = model.predict_proba(df_test)

print(predictions[:10])

print(predictions_probs[:10])
# Save results

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Submission was successfully saved!")