import pandas as pd

data = pd.read_csv('../input/Training.csv')
data.head()
from sklearn.model_selection import train_test_split

# drop string features (like ids, names, urls...)
data.drop(['id', 'album', 'analysis_url', 'name', 'track_href', 'type', 'uri'], axis=1, inplace=True)

# the songs, "X" axis
X = data.drop(['class'], axis=1)

# and their label/class, "y" axis
y = data['class']

# split the data into train sets (X_train, y_train) and test sets (X_test, y_test).
# the testing sets are usually smaller than the training sets.
# this time I'll use 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

dtc = DecisionTreeClassifier()
nbc = GaussianNB()
rfc = RandomForestClassifier()

dtc.fit(X_train, y_train)
nbc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
tree_score = dtc.score(X_test, y_test)
naive_score = nbc.score(X_test, y_test)
random_forest_score = rfc.score(X_test, y_test)

print('Decision Tree score:', tree_score)
print('Naive Bayes score:', naive_score)
print('Random Forest score:', random_forest_score)
test_playlist = pd.read_csv('../input/Test.csv')
test_playlist.drop(['id', 'album', 'analysis_url', 'name', 'track_href', 'type', 'uri'], axis=1, inplace=True)

test_playlist_X = test_playlist.drop(['class'], axis=1)
test_playlist_y = test_playlist['class']

test_tree_score = dtc.score(test_playlist_X, test_playlist_y)
test_naive_score = nbc.score(test_playlist_X, test_playlist_y)
test_random_forest_score = rfc.score(test_playlist_X, test_playlist_y)

print('Decision Tree score:', test_tree_score)
print('Naive Bayes score:', test_naive_score)
print('Random Forest score:', test_random_forest_score)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

actual = test_playlist['class']
decision_tree_prob = [p[1] for p in dtc.predict_proba(test_playlist_X)]
naive_bayes_prob = [p[1] for p in nbc.predict_proba(test_playlist_X)]
random_forest_prob = [p[1] for p in rfc.predict_proba(test_playlist_X)]

dt_false_pos, dt_true_pos, _ = roc_curve(actual, decision_tree_prob)
dt_auc = auc(dt_false_pos, dt_true_pos)

nb_false_pos, nb_true_pos, _ = roc_curve(actual, naive_bayes_prob)
nb_auc = auc(nb_false_pos, nb_true_pos)

rf_false_pos, rf_true_pos, _ = roc_curve(actual, random_forest_prob)
rf_auc = auc(rf_false_pos, rf_true_pos)

# plot
plt.plot(dt_false_pos, dt_true_pos, 'r', label='Decision Tree = %0.2f'% dt_auc)
plt.plot(nb_false_pos, nb_true_pos, 'g', label='Naive Bayes = %0.2f'% nb_auc)
plt.plot(rf_false_pos, rf_true_pos, 'b', label='Random Forest = %0.2f'% rf_auc)

plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
