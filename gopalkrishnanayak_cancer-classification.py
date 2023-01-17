import numpy as np

import pandas as pd

import matplotlib as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

%matplotlib inline
import pandas as pd

df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.info()
cols = list(df)

cols.append(cols.pop(cols.index('diagnosis')))

df = df.loc[:, cols]
df.head()
df.drop('Unnamed: 32', inplace=True, axis=1)
b_count,m_count = df['diagnosis'].value_counts()

print("Percentage Benign : ",(b_count/len(df))*100)

print("Percentage Maliignent : ",((m_count/len(df))*100))
cols = ['radius_mean',

        'texture_mean',

        'perimeter_mean',

        'area_mean',

        'smoothness_mean',

        'compactness_mean',

        'concavity_mean',

       'concave points_mean',

       'symmetry_mean',

       'fractal_dimension_mean']

corr_mat = df[cols].corr()

plt.pyplot.matshow(corr_mat, cmap="Blues")
# first, drop all "worst" columns

cols = ['radius_worst', 

        'texture_worst', 

        'perimeter_worst', 

        'area_worst', 

        'smoothness_worst', 

        'compactness_worst', 

        'concavity_worst',

        'concave points_worst', 

        'symmetry_worst', 

        'fractal_dimension_worst']

df = df.drop(cols, axis=1)



# then, drop all columns related to the "perimeter" and "area" attributes

cols = ['perimeter_mean',

        'perimeter_se', 

        'area_mean', 

        'area_se']

df = df.drop(cols, axis=1)



# lastly, drop all columns related to the "concavity" and "concave points" attributes

cols = ['concavity_mean',

        'concavity_se', 

        'concave points_mean', 

        'concave points_se']

df = df.drop(cols, axis=1)



# verify remaining columns

df.columns
corr_mat = df.corr()

plt.pyplot.matshow(corr_mat, cmap="Blues")
lable_encoder = LabelEncoder()

X = df.drop('diagnosis', axis=1)

y = lable_encoder.fit_transform(df['diagnosis'])



X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

lable_encoder.classes_
sgd_clf = SGDClassifier()

cross_val_pred = cross_val_predict(sgd_clf, X_Train, y_train, cv=3)

confusion_matrix(y_train, cross_val_pred)

precision_score(y_train, cross_val_pred), recall_score(y_train, cross_val_pred), f1_score(y_train, cross_val_pred)
random_forest = RandomForestClassifier()

#random_forest.fit(X_Train, y_train)

cross_val_pred = cross_val_predict(random_forest, X_Train, y_train, cv=4)

print(confusion_matrix(y_train, cross_val_pred))

print(precision_score(y_train, cross_val_pred), recall_score(y_train, cross_val_pred), f1_score(y_train, cross_val_pred))
random_forest_proba = RandomForestClassifier()

random_forest_proba.fit(X_Train, y_train)

cross_val_predProba = cross_val_predict(random_forest_proba, X_Train, y_train, method="predict_proba")

cross_val_predProba = cross_val_predProba[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, cross_val_predProba)

plt.pyplot.plot(fpr_forest, tpr_forest)

plt.pyplot.plot([0,1],[0,1],'k--')

plt.pyplot.show()

roc_auc_score(y_train, cross_val_predProba)
random_forest_proba = RandomForestClassifier()

random_forest_proba.fit(X_Train, y_train)

cross_val_predProba = cross_val_predict(random_forest_proba, X_Train, y_train, method="predict_proba")

cross_val_predProba = cross_val_predProba[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, cross_val_predProba)

plt.pyplot.plot(fpr_forest, tpr_forest)

plt.pyplot.plot([0,1],[0,1],'k--')

plt.pyplot.show()

roc_auc_score(y_train, cross_val_predProba)
random_Forest_test = random_forest_proba.predict(X_Test)

print(confusion_matrix(y_test, random_Forest_test))

print(precision_score(y_test, random_Forest_test), recall_score(y_test, random_Forest_test), f1_score(y_test, random_Forest_test))
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_Train, y_train)

model_predict = model.predict(X_Train)

print(confusion_matrix(y_train, model_predict))

print(precision_score(y_train, model_predict), recall_score(y_train, model_predict), f1_score(y_train, model_predict))
model_test = model.predict(X_Test)

print(confusion_matrix(y_test, model_test))

print(precision_score(y_test, model_test), recall_score(y_test, model_test), f1_score(y_test, model_test))