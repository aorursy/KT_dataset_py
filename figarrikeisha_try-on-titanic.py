import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('../input/train.csv')
df.head(5)
df.isnull().sum(axis = 0)
df = df.drop(df.columns[[8, 10]], axis = 1)
df = df.fillna(df.mean())
df['Survived'].value_counts()
df.pivot_table(['Survived'], ['Sex', 'Pclass']).sort_values(by = ['Survived'], ascending = False)
g = sns.PairGrid(df, y_vars="Survived",
                 x_vars=["Pclass", "Sex"],
                 size=5, aspect=.5)

g.map(sns.pointplot, color=sns.xkcd_rgb["green"])
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
df[['SibSp', 'Parch']].hist(figsize=(16, 10), xlabelsize=8, ylabelsize=8);
Alone = [0 for k in range(len(df))]
for p in range(len(df)):
    if df['SibSp'][p] == 0 and df['Parch'][p] == 0:
        Alone[p] = 1
df = df.assign(IsAlone =Alone)
df['IsAlone'].value_counts()
plt.rcParams['figure.figsize'] = (10, 8)
sns.countplot(x='Survived', hue='IsAlone', data=df);
expensive_ = df.sort_values(by='Fare', ascending = False)
expensive_.head(10)
df = df.drop(df[["PassengerId", "Name"]], axis=1)
df = df.dropna(axis = 0, how = 'any')
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Sex'] = lb.fit_transform(df['Sex'])
embarked_ = pd.get_dummies(df['Embarked'],prefix = 'Embarked' )
df = df.assign(C=embarked_['Embarked_C'], Q=embarked_['Embarked_Q'], S=embarked_['Embarked_S'])
df = df.drop(['Embarked'], axis=1)
df.head(5)
df['Age'] = np.round(df['Age'])
corr_ = df.corr()
corr_[abs(corr_) < 0.5] = 0
plt.figure(figsize=(16,10))
sns.heatmap(corr_, annot=True)
plt.show()
df_corr = df.corr()['Survived'][1:]
goldlist = df_corr[abs(df_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Survived:\n{}".format(len(goldlist), goldlist))
X = df.drop('Survived', axis=1)
y = df['Survived']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)
X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_std, y)
result = pd.DataFrame(model.coef_, columns = X.columns)
result = result.T
result.columns = ['coefficient']
np.abs(result).sort_values(by='coefficient', ascending=False)
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
log_reg = smf.ols(formula = 'y ~ Pclass+Sex+Age+SibSp+Parch+Fare+IsAlone+C+Q+S', data=X)
benchmark = log_reg.fit()
print('r2 score : \t', r2_score(y, benchmark.predict(X)))
print('mse : \t', mean_squared_error(y, benchmark.predict(X)))
log_reg = smf.ols(formula = 'y ~ Pclass+Age+SibSp+Parch+Fare+IsAlone+C+Q+S', data=X)
benchmark = log_reg.fit()
print('r2 score : \t', r2_score(y, benchmark.predict(X)))
print('mse : \t', mean_squared_error(y, benchmark.predict(X)))
log_reg = smf.ols(formula = 'y ~ Pclass+Sex+Age+SibSp+Parch+Fare+IsAlone+Q+S', data=X)
benchmark = log_reg.fit()
print('r2 score : \t', r2_score(y, benchmark.predict(X)))
print('mse : \t', mean_squared_error(y, benchmark.predict(X)))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
clf_rbf = svm.SVC(kernel='rbf', degree = 10, C = 1)
clf_rbf.fit(X_std, y)
res_clf_rbf = cross_val_score(clf_rbf, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_rbf)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_rbf)))
from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier()
clf_gb.fit(X_std, y)
res_clf_gb = cross_val_score(clf_gb, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_gb)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_gb)))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(min_samples_leaf = 10, n_estimators = 10)
rf.fit(X_std, y)
res_rf = cross_val_score(rf, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_rf)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_rf)))
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_std, y)
res_clf_lr = cross_val_score(clf_lr, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_lr)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_lr)))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_std, y)
res_knn = cross_val_score(knn, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_knn)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_knn)))
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier()
clf_ada.fit(X_std, y)
res_clf_ada = cross_val_score(clf_ada, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_ada)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_ada)))
import xgboost as xgb
clf_xgb = xgb.XGBClassifier(random_state = 42)
clf_xgb.fit(X_std, y)
res_clf_xgb = cross_val_score(clf_xgb, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_xgb)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_xgb)))
classifier = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Gradient Boosting', 'Random Forest', 
            'Logistic Regression', 'KNN', 
              'Adaboost', 'XGBoost'],
    'Score': [np.mean(res_clf_rbf), np.mean(res_clf_gb), np.mean(res_rf), 
              np.mean(res_clf_lr), np.mean(res_knn), 
              np.mean(res_clf_ada), np.mean(res_clf_xgb)]})
classifier.sort_values(by='Score', ascending=False)
from sklearn.model_selection import GridSearchCV
params = {'max_depth':(5, 10, 25, 50), 
          'n_estimators':(50, 200, 500, 1000)} 
clf_xgb_grid = GridSearchCV(clf_xgb, params, n_jobs=-1,
                            cv=3, verbose=1, scoring='accuracy')
clf_xgb_grid.fit(X_std, y)
clf_xgb_grid.best_estimator_.get_params
clf_xgb_ = xgb.XGBClassifier(random_state = 42, learning_rate = 0.1, max_depth = 5, n_estimators=50, n_jobs=1)
clf_xgb_.fit(X_std, y)
res_clf_xgb_ = cross_val_score(clf_xgb_, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_xgb_)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_xgb_)))
columns_to_show = ['Pclass', 'Sex', 'Age', 'SibSp']
X_ = df[columns_to_show]
y_ = df['Survived']
X_std_ = sc.fit_transform(X_)
clf_xgb_.fit(X_std_, y_)
res_clf_xgb_t = cross_val_score(clf_xgb_, X_std_, y_, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_xgb_t)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_xgb_t)))
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier_ = Sequential()
classifier_.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 10))
classifier_.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier_.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid'))
classifier_.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_.fit(X_std, y, batch_size = 20, epochs = 200)
metr_ = classifier_.evaluate(X_std, y_)
print("Loss metric: \t {0:.4f}".format(metr_[0]))
print("Accuracy metric: \t{0:.4f}".format(metr_[1]))
classifier_update = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Gradient Boosting', 'Random Forest', 
            'Logistic Regression', 'KNN', 
              'Adaboost', 'XGBoost', 'ANN'],
    'Score': [np.mean(res_clf_rbf), np.mean(res_clf_gb), np.mean(res_rf), 
              np.mean(res_clf_lr), np.mean(res_knn), 
              np.mean(res_clf_ada), np.mean(res_clf_xgb_), metr_[1]]})
classifier_update['Score'] = np.round(classifier_update['Score'], decimals = 4)
classifier_update.sort_values(by='Score', ascending=False)
plt.figure(figsize=(16, 8))

x = classifier_update['Model']
y = classifier_update['Score']
sns.barplot(x, y, palette="Set3")
plt.ylabel("Score")
plt.xlabel("Models")
plt.ylim(0.75, 0.86);
df2 = pd.read_csv('../input/test.csv')
df2 = df2.drop(df2.columns[[2,7,9]], axis = 1)
df2 = df2.fillna(df2.mean())
df2 = df2.dropna(axis = 0, how = 'any')
alone = [0 for k in range(len(df2))]
for p in range(len(df2)):
    if df2['SibSp'][p] == 0 and df2['Parch'][p] == 0:
        alone[p] = 1
df2 = df2.assign(IsAlone=alone)
df2 = df2.dropna(axis = 0, how = 'any')
from sklearn.preprocessing import LabelEncoder
lb_t = LabelEncoder()
df2['Sex'] = lb_t.fit_transform(df2['Sex'])

embarked_ = pd.get_dummies(df2['Embarked'],prefix = 'Embarked' )
df2 = df2.assign(C=embarked_['Embarked_C'], Q=embarked_['Embarked_Q'], S=embarked_['Embarked_S'])
df2 = df2.drop(['Embarked'], axis=1)

df2['Age'] = np.round(df2['Age'])
df2.head(3)
X_test = df2.drop('PassengerId', axis = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)
X_test_std = sc.transform(X_test)
y_pred = classifier_.predict(X_test_std)
y_pred = (y_pred > 0.5)
y_pred = pd.Series(list(y_pred))

submission = pd.DataFrame({"PassengerId": df2["PassengerId"], "Survived": y_pred})
submission['Survived'] = submission['Survived'].astype(int)
submission.head()
#submission.to_csv('..\submission.csv', index = False)