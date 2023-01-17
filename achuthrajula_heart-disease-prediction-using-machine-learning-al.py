import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import tree

from sklearn.cluster import KMeans

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import seaborn as sns

import shap

import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

X = data.iloc[:,0:13]

y = data.iloc[:,13]

data.head()
plt.figure(figsize=(12,10))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap='coolwarm')

plt.show()
#Correlation with output variable

cor_target = abs(cor["target"])#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.4]

relevant_features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
classifiers = [LogisticRegression, MLPClassifier, KNeighborsClassifier, tree.DecisionTreeClassifier, RandomForestClassifier, SVC]

for model in classifiers:

    classifier = model()

    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test) 

    print("{} model accuracy: {}".format(model, np.amax(cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10), axis = 0)))
model = RandomForestClassifier()

model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)
shap.initjs()

shap_values = explainer.shap_values(X_train.iloc[:50])

shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[:50])
f, axes = plt.subplots(1, 2, figsize=(20, 10))

sns.lineplot(X_train['chest_pain_type'],y_train,ax=axes[0])

sns.lineplot(X_test['chest_pain_type'],y_pred,ax=axes[0])

sns.lineplot(X_train['exercise_induced_angina'],y_train,ax=axes[1])

sns.lineplot(X_test['exercise_induced_angina'],y_pred,ax=axes[1])