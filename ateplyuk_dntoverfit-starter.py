import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/train.csv").drop('id', axis=1)

train_df.head()
test_df = pd.read_csv('../input/test.csv').drop('id', axis = 1)

test_df.head()
plt.bar(range(2), (train_df.shape[0], test_df.shape[0])) 

plt.xticks(range(2), ('Train', 'Test'))

plt.ylabel('Count') 

plt.show()
y = train_df['target']

X = train_df.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
best_score = 0

for penalty in ['l1', 'l2']:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:       

        logreg = LogisticRegression(class_weight='balanced',  penalty=penalty, C=C, solver='liblinear')

        logreg.fit(X_train, y_train)

        score = logreg.score(X_test, y_test)

        if score > best_score:

            best_score = score

            best_parameters = {'C': C, 'penalty': penalty}            
logreg = LogisticRegression(**best_parameters)

logreg.fit(X_train, y_train)

test_score = logreg.score(X_test, y_test)
print("Best score: {:.2f}".format(best_score))

print("Best parameters: {}".format(best_parameters))

print("Best score on test data: {:.2f}".format(test_score))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = logreg.predict_proba(test_df)[:,1]

sub.to_csv('submission.csv', index=False)