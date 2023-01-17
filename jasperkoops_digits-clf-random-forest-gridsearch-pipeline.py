import pandas as pd

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler
df = pd.read_csv('../input/train.csv')

X = df.drop('label', axis=1)

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
df.hist('label')
pipe = Pipeline(steps=[

    ('pre', None),

    ('feature_selection', None),

    ('pca', None),

    ('clf', RandomForestClassifier(n_estimators=100))

])



params = {

    'pre': [None, StandardScaler(), MinMaxScaler()],

    'feature_selection': [None],

    'pca': [PCA(n_components=0.95), None],

}



clf = GridSearchCV(pipe, param_grid=params, cv=10)

clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)

print(clf.best_params_)
test_df = pd.read_csv('../input/test.csv')

predictions = clf.predict(test_df)

submission = {

    'Label': predictions

}



submission_df = pd.DataFrame(submission)

submission_df.index += 1

submission_df.reset_index(level=0, inplace=True)

submission_df.columns = ['ImageId', 'Label']

submission_df.to_csv('submission.csv', index=False)