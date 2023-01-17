import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets, linear_model,tree, ensemble, neural_network
from sklearn.feature_extraction.text import CountVectorizer
import_df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
resources = pd.read_csv('../input/resources.csv')
resources['total'] = resources['quantity'] * resources['price']
resources = resources.groupby('id').sum()
resources['id'] = resources.index

test = test.merge(resources, how = 'left', on = 'id')
import_df = import_df.merge(resources, how = 'left', on = 'id')
import_df['submitted_hour'] = pd.to_datetime(import_df['project_submitted_datetime']).dt.hour
import_df['submitted_month'] = pd.to_datetime(import_df['project_submitted_datetime']).dt.month
import_df['submitted_weekday'] = pd.to_datetime(import_df['project_submitted_datetime']).dt.weekday

X = pd.DataFrame()
Y = import_df['project_is_approved']
X['number_of_previously_posted_projects'] = import_df['teacher_number_of_previously_posted_projects']
X['len_essay_1'] = import_df['project_essay_1'].str.len()
X['len_essay_2'] = import_df['project_essay_2'].str.len()
X[pd.get_dummies(import_df['school_state']).columns] = pd.get_dummies(import_df['school_state'])
X[pd.get_dummies(import_df['project_subject_categories']).columns] = pd.get_dummies(import_df['project_subject_categories'])
X[['hour' + str(s) for s in pd.get_dummies(import_df['submitted_hour']).columns.values]] = pd.get_dummies(import_df['submitted_hour'])
X[['month' + str(s) for s in pd.get_dummies(import_df['submitted_month']).columns.values]] = pd.get_dummies(import_df['submitted_month'])
X[['weekday' + str(s) for s in pd.get_dummies(import_df['submitted_weekday']).columns.values]] = pd.get_dummies(import_df['submitted_weekday'])
X['total'] = import_df['total']
bow_transformer = CountVectorizer(max_df = 0.4, min_df = 0.1)
bow_transformer.fit(import_df['project_essay_1']+import_df['project_essay_2'])
bow4 = bow_transformer.transform(import_df['project_essay_1']+import_df['project_essay_2'])
for i in range(len(bow_transformer.get_feature_names())):
    X[bow_transformer.get_feature_names()[i]] = bow4.toarray().transpose()[i]
regr1 = ensemble.ExtraTreesRegressor(n_estimators = 50, max_depth=12, random_state=0, n_jobs=-1)
regr1.fit(X,Y)
features = pd.DataFrame()
features[0] = X.columns
features[1] = regr1.feature_importances_
print(features.sort_values(by = 1, ascending = False))
columns = features.sort_values(by = 1, ascending = False)[0].values[0:30]
regr = ensemble.ExtraTreesRegressor(n_estimators = 1000, max_depth=12, random_state=0, bootstrap=True, oob_score = True, n_jobs=-1)
regr.fit(X[columns],Y)
print(regr.oob_score_)
features2 = pd.DataFrame()
features2[0] = X[columns].columns
features2[1] = regr.feature_importances_
print(features2.sort_values(by = 1, ascending = False))
test['submitted_hour'] = pd.to_datetime(test['project_submitted_datetime']).dt.hour
test['submitted_month'] = pd.to_datetime(test['project_submitted_datetime']).dt.month
test['submitted_weekday'] = pd.to_datetime(test['project_submitted_datetime']).dt.weekday

X_test = pd.DataFrame()
X_test['number_of_previously_posted_projects'] = test['teacher_number_of_previously_posted_projects']
X_test['len_essay_1'] = test['project_essay_1'].str.len()
X_test['len_essay_2'] = test['project_essay_2'].str.len()
X_test[pd.get_dummies(import_df['school_state']).columns] = pd.get_dummies(test['school_state'], columns = pd.get_dummies(import_df['school_state']).columns)
X_test[pd.get_dummies(import_df['project_subject_categories']).columns] = pd.get_dummies(test['project_subject_categories'], columns = pd.get_dummies(import_df['project_subject_categories']).columns)
X_test[['hour' + str(s) for s in pd.get_dummies(import_df['submitted_hour']).columns.values]] = pd.get_dummies(test['submitted_hour'])
X_test[['month' + str(s) for s in pd.get_dummies(import_df['submitted_month']).columns.values]] = pd.get_dummies(test['submitted_month'])
X_test[['weekday' + str(s) for s in pd.get_dummies(import_df['submitted_weekday']).columns.values]] = pd.get_dummies(test['submitted_weekday'])
X_test['total'] = test['total']
#X_test['count_I'] = (test['project_essay_1']+test['project_essay_2']).str.count('I ')

bow5 = bow_transformer.transform(test['project_essay_1']+test['project_essay_2'])
for i in range(len(bow_transformer.get_feature_names())):
    X_test[bow_transformer.get_feature_names()[i]] = bow5.toarray().transpose()[i]
    
Y_test = regr.predict(X_test[columns])
data_to_submit = pd.DataFrame({
    'id':test['id'],
    'project_is_approved':Y_test
})
data_to_submit.to_csv('csv_to_submit.csv', index = False)
