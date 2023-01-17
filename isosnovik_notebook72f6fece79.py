import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



# train

x_train = train.drop(['id', 'species'], axis=1)

y_train = train['species']

le = LabelEncoder()

y_train = le.fit_transform(y_train)

# test

test_ids = test.pop('id')

x_test = test



scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)
params = {'C':[1000], 'tol': [0.0008, 0.0007]}

log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='False', cv=3)

clf.fit(x_train, y_train)



print("best params: " + str(clf.best_params_))

for params, mean_score, scores in clf.grid_scores_:

    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

    print(scores)
clf = LogisticRegression(solver='lbfgs',multi_class='multinomial', C=1000, tol=0.0008)

clf.fit(x_train, y_train)



x_test = scaler.transform(x_test)

y_test = clf.predict_proba(x_test)



submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)

submission.to_csv('submission_log_reg.csv')
clf = xgb.XGBClassifier(silent=False, n_estimators=400, learning_rate=0.03)

fit_params = {

    'eval_metric': 'mlogloss'

}

cv = -cross_val_score(clf, x_train, y_train, scoring='log_loss').mean()

print(cv)