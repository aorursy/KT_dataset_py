import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
file = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
file.head()
file.shape
bins= [45,50,55,60,65,70,75,80,85,90,95,120]
labels = ['45-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96-100']
file['AgeGroup'] = pd.cut(file['age'], bins=bins, labels=labels, right=False)

plt.rcParams["figure.figsize"] = 10,8
df1 = file.groupby(["AgeGroup", "sex"]).agg({"DEATH_EVENT": "count"}).unstack()
df1.plot(kind = "bar", stacked = True)
plt.show()

sns.boxplot(file["AgeGroup"], file["serum_creatinine"], hue= file["DEATH_EVENT"])
sns.boxplot(file["AgeGroup"], file["serum_sodium"], hue= file["DEATH_EVENT"])
sns.boxplot(file["AgeGroup"], file["creatinine_phosphokinase"], hue = file["DEATH_EVENT"])
plt.rcParams["figure.figsize"] = 12, 8
df4 = file.groupby(["AgeGroup", "diabetes"]).agg({"DEATH_EVENT": "count"}).unstack()
df4.plot(kind = "bar")
plt.show()
sns.boxplot(file["AgeGroup"], file["ejection_fraction"], hue = file["DEATH_EVENT"])
plt.rcParams["figure.figsize"] = 8,6
sns.countplot(file["high_blood_pressure"],  hue = file["DEATH_EVENT"])
plt.rcParams["figure.figsize"] = 10, 6
sns.stripplot(file["anaemia"], file["platelets"], hue=file["DEATH_EVENT"])
plt.rcParams["figure.figsize"] = 8,6
sns.countplot(file["smoking"],  hue = file["DEATH_EVENT"])
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
#In such problems we should always choose our metrics which suit the problem. 
def metric(model, preds, y_valid):
    
    f1 = f1_score(y_valid, preds)
    precision = precision_score(y_valid, preds)
    recall = recall_score(y_valid, preds)
    if hasattr(model, 'oob_score_'):
        return f1, precision, recall, model.oob_score_
    else:
        return f1, precision, recall

    
def feat_imp(model, cols):
    return pd.DataFrame({"Col_names": cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)

def plot_i(fi, x, y):
    return fi.plot(x, y, "barh", figsize = (12,8))
X = file.drop(["DEATH_EVENT", "AgeGroup"], axis = 1)
y = file["DEATH_EVENT"]
x_train, x_valid, y_train, y_valid = train_test_split(X, y)
%%time
Rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, max_features= 0.6, oob_score=True)
model_Rf = Rf.fit(x_train, y_train)
preds = model_Rf.predict(x_valid)
print(metric(model_Rf, preds, y_valid))
feat10 = feat_imp(model_Rf, x_train.columns)
feat10
plot_i(feat10, "Col_names", "Importance")
to_keep = feat10[feat10["Importance"] > 0.03]
len(to_keep)
X = X[to_keep.Col_names]
x_train, x_valid, y_train, y_valid = train_test_split(X, y)
%%time
Rf = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=3, max_features= 0.5, oob_score=True)
model_Rf = Rf.fit(x_train, y_train)
preds = model_Rf.predict(x_valid)
print(metric(model_Rf, preds, y_valid))
feat_2 = feat_imp(model_Rf, x_train.columns)
feat_2
to_keep = feat_2[feat_2["Importance"] > 0.1]
to_keep
X = X[to_keep.Col_names]
X.head(2)
x_train, x_valid, y_train, y_valid = train_test_split(X, y)

solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalty = ['l1', 'l2', 'elasticnet']
C = [0.2, 0.4, 0.6, 0.8, 1]
Lr = LogisticRegression()
param_grid = dict(solver = solver, penalty = penalty, C= C)
grid = GridSearchCV(Lr, param_grid=param_grid, n_jobs=-1, cv = 3)         

%%time
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
Lr = LogisticRegression()
model_Lr = Lr.fit(x_train, y_train)
preds = model_Lr.predict(x_valid)
print(metric(model_Lr, preds, y_valid))
