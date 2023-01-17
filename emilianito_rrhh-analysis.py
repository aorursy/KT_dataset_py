import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/recursos_humanos.csv')
df.columns.values
df.head()
df.info()
df['promotion_last_5years'] = df['promotion_last_5years'].astype('object')
df['Work_accident'] = df['Work_accident'].astype('object')
df['left'] = df['left'].astype('object')
df['salary_num'] = df['salary'].apply(lambda s: 0 if s == 'low' else 1 if s == 'medium' else 2) # Salary numerical variable
df.info()
df.describe()
df.corr()
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
res = df[['number_project', 'average_montly_hours']].groupby(['number_project'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['number_project'], res['average_montly_hours'])
res = df[['number_project', 'last_evaluation']].groupby(['number_project'], as_index=False).mean().sort_values(by='last_evaluation', ascending=False)
plt.plot(res['number_project'], res['last_evaluation'])
res = df[['time_spend_company', 'satisfaction_level']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['satisfaction_level'])
res = df[['time_spend_company', 'number_project']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['number_project'])
res = df[['number_project', 'time_spend_company']].groupby(['number_project'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
res
res = df[['number_project', 'time_spend_company']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project', ascending=False)
plt.plot(res['number_project'], res['time_spend_company'])
res = df[['average_montly_hours', 'last_evaluation']].groupby(['average_montly_hours'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['average_montly_hours'], res['last_evaluation'])
res = df[['average_montly_hours', 'satisfaction_level']].groupby(['average_montly_hours'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['average_montly_hours'], res['satisfaction_level'])
res = df[['time_spend_company', 'satisfaction_level']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['satisfaction_level'])
res = df[['number_project', 'satisfaction_level']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project', ascending=False)
plt.plot(res['number_project'], res['satisfaction_level'])
df.describe(include=['O'])
df['left'] = df['left'].astype('int64')
ret = df[['salary', 'left']].groupby(['salary'], as_index=False).mean().sort_values(by='left', ascending=False)
ret.plot.bar(x="salary", y="left", legend=False )
ret=df[['salary', 'satisfaction_level']].groupby(['salary'], as_index=False).mean().sort_values(by='satisfaction_level', ascending=False)
ret.plot.bar(x="salary", y="satisfaction_level", legend=False )
df[['sales', 'left']].groupby(['sales'], as_index=False).mean().sort_values(by='left', ascending=False)
ret=df[['sales', 'satisfaction_level']].groupby(['sales'], as_index=False).mean().sort_values(by='satisfaction_level', ascending=False)
ret.plot.bar(x="sales", y="satisfaction_level", legend=False )
df[['Work_accident', 'left']].groupby(['Work_accident'], as_index=False).mean().sort_values(by='left', ascending=False)
df[['promotion_last_5years', 'left']].groupby(['promotion_last_5years'], as_index=False).mean().sort_values(by='left', ascending=False)
df[['number_project', 'left']].groupby(['number_project'], as_index=False).mean().sort_values(by='left', ascending=False)
res = df[['time_spend_company', 'left']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['left'])
res = df[['average_montly_hours', 'left']].groupby(['average_montly_hours'], as_index=False).mean().sort_values(by='average_montly_hours', ascending=False)
plt.plot(res['average_montly_hours'], res['left'])
res = df[['number_project', 'left']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project', ascending=False)
plt.plot(res['number_project'], res['left'])
res = df[['time_spend_company', 'salary_num']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company', ascending=False)
plt.plot(res['time_spend_company'], res['salary_num'])
g = sns.FacetGrid(df, col='left')
g.map(plt.hist, 'time_spend_company')

grid = sns.FacetGrid(df, col='left', row='time_spend_company', size=2.2, aspect=1.6)
grid.map(plt.hist, 'average_montly_hours', alpha=.5)
grid.add_legend();
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
X = df.drop('left', axis=1)
labels = df['left']
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, random_state=1, test_size = 0.3)
predictors = ["satisfaction_level","number_project","average_montly_hours","time_spend_company"]

model = DecisionTreeClassifier()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)
pd.DataFrame(
    confusion_matrix(labels_test, labels_predict),
    columns=['Predicted Not Left', 'Predicted Left'],
    index=['True No Left', 'True Left']
)
model = RandomForestClassifier()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)
pd.DataFrame(
    confusion_matrix(labels_test, labels_predict),
    columns=['Predicted Not Left', 'Predicted Left'],
    index=['True No Left', 'True Left']
)
pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
model = KNeighborsClassifier()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)
pd.DataFrame(
    confusion_matrix(labels_test, labels_predict),
    columns=['Predicted Not Left', 'Predicted Left'],
    index=['True No Left', 'True Left']
)
model = LogisticRegression()
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)
model = SVC(kernel='linear')
model.fit(X_train[predictors], labels_train)
labels_predict = model.predict(X_test[predictors])
accuracy_score(labels_test, labels_predict)