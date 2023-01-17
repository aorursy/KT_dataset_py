#importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os
import warnings 
warnings.filterwarnings('ignore')
#read the data
data=pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.info()
#measuring datatype count
dtype_data =data.dtypes.reset_index()
dtype_data.columns = ["Count", "Column Type"]
dtype_data.groupby("Column Type").aggregate('count').reset_index()
data.isnull().values.any()
#still there is missing value in totalcharges, which considered it as a string,
#now we are replacing it with zero n change to float type 
data['TotalCharges'] = data['TotalCharges'].str.replace(" ","0").astype(float)
sns.swarmplot(x="gender", y="TotalCharges",hue="Churn", data=data)
plt.show()
ax1 = sns.barplot(x="PhoneService", y="TotalCharges", hue="Churn", data=data)
ax2 = sns.barplot(x="MultipleLines", y="TotalCharges", hue="Churn", data=data)
ax3 = sns.barplot(x="InternetService", y="TotalCharges", hue="Churn", data=data)
ax4 = sns.barplot(x="OnlineSecurity", y="TotalCharges", hue="Churn", data=data)

ax5 = sns.barplot(x="OnlineBackup", y="TotalCharges", hue="Churn", data=data)
ax6 = sns.barplot(x="DeviceProtection", y="TotalCharges", hue="Churn", data=data)

ax7 = sns.barplot(x="TechSupport", y="TotalCharges", hue="Churn", data=data)

ax8 = sns.barplot(x="StreamingTV", y="TotalCharges", hue="Churn", data=data)
ax9 = sns.barplot(x="StreamingMovies", y="TotalCharges", hue="Churn", data=data)
#label encoding of the variables
for f in data.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[f].values)) 
        data[f] = lbl.transform(list(data[f].values))

data.head()
#using randomforest to find the feature importance
train_y = data['Churn'].values
train_X = data.drop(['customerID', 'Churn'], axis=1)

from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feat_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="b", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()

#importing required model packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree

#split the data as 80/20
X_train, X_test, y_train, y_test = train_test_split(train_X,train_y, test_size=0.2, random_state=42)

# Construct some pipelines
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('clf', svm.SVC(random_state=42))])

pipe_dt = Pipeline([('scl', StandardScaler()),
                    ('clf', tree.DecisionTreeClassifier(random_state=42))])
# List of pipelines for ease of iteration
pipelines = [pipe_lr, pipe_svm, pipe_dt]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Support Vector Machine', 2: 'Decision Tree'}


# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)


# Compare accuracies
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X_test, y_test)))


# Identify the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(pipelines):
    if val.score(X_test, y_test) > best_acc:
        best_acc = val.score(X_test, y_test)
        best_pipe = val
        best_clf = idx
print('Classifier with best accuracy: %s' % pipe_dict[best_clf])