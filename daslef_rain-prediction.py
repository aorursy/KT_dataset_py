raw = pd.read_csv('../input/weatherAUS.csv')
raw['RISK_MM'].describe()
raw['WindDir9am'].value_counts()

raw['WindDir9am'] = raw['WindDir9am'].fillna('N')
raw['WindDir3pm'].value_counts()

raw['WindDir3pm'] = raw['WindDir3pm'].fillna('SE')
raw['WindGustDir'].value_counts()

raw['WindGustDir'] = raw['WindGustDir'].fillna('W')

raw.head(2)
raw.shape
raw.describe().loc['mean']
raw.describe().loc['mean'].values
means = raw.describe().loc['mean']



for i in range(len(means)):

    raw.loc[:, means.index[i]] = raw.loc[:, means.index[i]].fillna(means.values[i])
raw.info()
raw = pd.get_dummies(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'], data=raw)
raw['RainToday'] = raw['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
raw['RainTomorrow'] = raw['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

raw.drop(['Date','Location','RISK_MM'],axis=1, inplace=True)



y = raw['RainTomorrow']

X = raw.drop(['RainTomorrow'], axis=1)
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, sep='; ')
from sklearn.linear_model import LogisticRegression
logres = LogisticRegression(n_jobs=-1)

logres.fit(X=X_train ,y=y_train)

logres.predict_proba(X_test)
y_pred = logres.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred=y_pred, y_true=y_test)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, max_features=10, min_samples_leaf=3)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
accuracy_score(y_pred=y_pred_tree, y_true=y_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_score(y_pred_rf, y_test)
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier()

params = {'max_depth': range(1, 12), 'max_features': range(1, 15), 'min_samples_leaf': range(1, 5)}

gs = GridSearchCV(estimator=tree, param_grid=params, scoring='accuracy', verbose=10)
gs.fit(X_train, y_train)
gs.best_score_, gs.best_params_
rf = RandomForestClassifier(n_estimators=50)

params = {'max_depth': (8,None)}

gs_rf = GridSearchCV(estimator=tree, param_grid=params, scoring='accuracy', verbose=10)
gs_rf.fit(X_train, y_train)
gs_rf.best_score_, gs_rf.best_params_
logres = LogisticRegression(n_jobs=-1)

logres.fit(X=X_train ,y=y_train)

y_proba_logres = logres.predict_proba(X_test)
tree = DecisionTreeClassifier(max_depth=8, max_features=14).fit(X_train ,y_train)
y_proba_tree = tree.predict_proba(X_test)
rf = RandomForestClassifier(n_estimators=100, max_depth=8).fit(X_train ,y_train)

y_proba_rf = rf.predict_proba(X_test)
assert len(y_proba_logres) == len(y_proba_tree) == len(y_proba_rf)
average_proba = []

for i in range(len(y_proba_logres)):

    proba0 = (y_proba_logres[i][0] + y_proba_tree[i][0] + y_proba_rf[i][0])/3

    proba1 = (y_proba_logres[i][1] + y_proba_tree[i][1] + y_proba_rf[i][1])/3

    average_proba.append([proba0, proba1])

print(average_proba)



y_average = [0 if i[0] > i[1] else 1 for i in average_proba]



print(y_average)
accuracy_score(y_average, y_test)