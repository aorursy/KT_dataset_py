import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, plot_confusion_matrix, f1_score, recall_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from scipy import stats
from scipy.stats import norm, skew


data = pd.read_csv('../input/heart-failure/heart_failure_clinical_records_dataset.csv')
data.head()
data.DEATH_EVENT.value_counts()
train_na = (data.isnull().sum() / len(data)) * 100
train_na = train_na.drop(train_na[train_na==0].index).sort_values(ascending=False)
data_missing = pd.DataFrame({'Missing Ratio' :train_na})
len(data_missing)
data.describe()
def plot_histogram(dataframe, column, color, bins, title, width=700, height=500):
    figure = px.histogram(
        dataframe, 
        column, 
        color=color,
        nbins=bins, 
        title=title, 
        width=width,
        height=height
    )
    figure.show()

bins = 50
def pie_visual(dataframe, column):
    ds = dataframe[column].value_counts().reset_index()
    ds.columns = [column, 'count']

    fig = px.pie(
        ds, 
        values='count', 
        names=column, 
        title= column+' bar chart', 
        width=700, 
        height=500
    )

    fig.show()
plot_histogram(data, 'age', 'sex', bins, 'Patients age distribution')
plot_histogram(data, 'age', 'DEATH_EVENT', bins, 'Patients age distribution')
plot_histogram(data, 'creatinine_phosphokinase', 'DEATH_EVENT', 2 * bins, 'Creatinine phosphokinase distribution')
pie_visual(data, 'anaemia')
pie_visual(data, 'diabetes')
plot_histogram(data, 'ejection_fraction', 'DEATH_EVENT', bins, 'ejection fraction distribution')
pie_visual(data, 'high_blood_pressure')
plot_histogram(data, 'platelets', 'DEATH_EVENT', bins, 'ejection fraction distribution')
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), vmin=-1, cmap='coolwarm', annot=True);
adacls = AdaBoostClassifier(random_state=42)
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=True)
adacls.fit(X, y)
prediction = adacls.predict(X_test)

print('Adaboost Classifier ', accuracy_score(y_test, prediction))
# lets check confusion matrix and pay attention at f1 score 
cm = confusion_matrix(y_test, prediction)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
print('AdaBoostClassifier f1-score', f1_score(y_test, prediction))
print('AdaBoostClassifier precision', precision_score(y_test, prediction))
print('AdaBoostClassifier recall', recall_score(y_test, prediction))
dict_feature_importance = {i:j for i, j in zip(X.columns, adacls.feature_importances_)}
most_important_values = dict(sorted(dict_feature_importance.items(),
                               key=lambda x: x[1], reverse=True)[:10])
print(most_important_values)
plt.figure(figsize=(15, 5))
plt.xticks(rotation=45)
plt.bar(most_important_values.keys(), most_important_values.values())
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.20, shuffle=True)

for col in X.columns:
    if abs(X[col].corr(y)) < 0.15:
        X = X.drop([col], axis=1)
        X_test = X_test.drop([col], axis=1)
adacls.fit(X, y)
prediction = adacls.predict(X_test)

print('AdaBoostClassifier accuraccy', accuracy_score(y_test, prediction))
print('AdaBoostClassifier f1-score', f1_score(y_test, prediction))
print('AdaBoostClassifier precision', precision_score(y_test, prediction))
print('AdaBoostClassifier recall', recall_score(y_test, prediction))
fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="time", 
    points='all',
    height=500,
    width=700,
    title='Age & DEATH_EVENT box plot'
)

fig.show()
data.drop(data[(data['time'] > 198) & (data['DEATH_EVENT'] == 1)].index,inplace=True)
data.drop(data[(data['platelets'] > 700000)].index,inplace=True)
X = data.drop(['DEATH_EVENT', 'anaemia', 'diabetes'], axis=1)
y = data['DEATH_EVENT']
X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.20, shuffle=True)

for col in X.columns:
    if abs(X[col].corr(y)) < 0.15:
        X = X.drop([col], axis=1)
        X_test = X_test.drop([col], axis=1)

adacls.fit(X, y)
prediction = adacls.predict(X_test)

print('AdaBoostClassifier accuraccy', accuracy_score(y_test, prediction))
print('AdaBoostClassifier f1-score', f1_score(y_test, prediction))
print('AdaBoostClassifier precision', precision_score(y_test, prediction))
print('AdaBoostClassifier recall', recall_score(y_test, prediction))
cm = confusion_matrix(y_test, prediction)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
data_skew = data.drop('DEATH_EVENT', axis =1 )
numeric_f = data_skew.dtypes[data_skew.dtypes != "object"].index

# Check the skew of all numerical features

skewed_f = data_skew[numeric_f].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew in train data' :skewed_f})
skewness.head()
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features in train data to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    data_skew[feat] = boxcox1p(data_skew[feat], lam)
X = data_skew
y = data['DEATH_EVENT']
X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.20, shuffle=True)

for col in X.columns:
    if abs(X[col].corr(y)) < 0.15:
        X = X.drop([col], axis=1)
        X_test = X_test.drop([col], axis=1)

adacls.fit(X, y)
prediction = adacls.predict(X_test)

print('AdaBoostClassifier accuraccy', accuracy_score(y_test, prediction))
print('AdaBoostClassifier f1-score', f1_score(y_test, prediction))
print('AdaBoostClassifier precision', precision_score(y_test, prediction))
print('AdaBoostClassifier recall', recall_score(y_test, prediction))
model = XGBClassifier(random_state=0)
model.fit(X, y)
preds = model.predict(X_test)

print('XGBClassifier f1-score', f1_score(y_test, preds))
print('XGBClassifier precision', precision_score(y_test, preds))
print('XGBClassifier recall', recall_score(y_test, preds))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
final_rf = RandomForestClassifier(random_state=11)
gscv = GridSearchCV(estimator=final_rf,param_grid={
    "n_estimators":[100,500,1000,5000],
    "criterion":["gini","entropy"]
},cv=5,n_jobs=-1,scoring="f1_weighted")

gscv.fit(X_train,y_train)
FINAL_MODEL = gscv.best_estimator_
from sklearn.metrics import classification_report
f1_score = FINAL_MODEL.score(X_test, y_test)
f1_score
predict = FINAL_MODEL.predict(X_test)
print(classification_report(predict, y_test))
gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)
gradientboost_clf.fit(X_train,y_train)
gradientboost_pred = gradientboost_clf.predict(X_test)

print('gradientboost f1-score', f1_score(y_test, gradientboost_pred))
print('gradientboost precision', precision_score(y_test, gradientboost_pred))
print('gradientboost recall', recall_score(y_test, gradientboost_pred))
cm = confusion_matrix(y_test, gradientboost_pred)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')