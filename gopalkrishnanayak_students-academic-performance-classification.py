# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/xAPI-Edu-Data/xAPI-Edu-Data.csv")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,LabelBinarizer

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

import seaborn as sb

%matplotlib inline
df.head()
df.info()
df[['gender','NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays','Class']].iloc[1:20]
df_encode = df.copy(deep = True)



label_encode = LabelEncoder()



gender_encode = label_encode.fit_transform(df_encode['gender'])

nationality_encode = label_encode.fit_transform(df_encode['NationalITy'])

PlaceofBirth_encode = label_encode.fit_transform(df_encode['PlaceofBirth'])

StageID_encode = label_encode.fit_transform(df_encode['StageID'])

GradeID_encode = label_encode.fit_transform(df_encode['GradeID'])

SectionID_encode = label_encode.fit_transform(df_encode['SectionID'])

Topic_encode = label_encode.fit_transform(df_encode['Topic'])

Semester_encode = label_encode.fit_transform(df_encode['Semester'])

Relation_encode = label_encode.fit_transform(df_encode['Relation'])

ParentAnsweringSurvey_encode = label_encode.fit_transform(df_encode['ParentAnsweringSurvey'])

ParentschoolSatisfaction_encode = label_encode.fit_transform(df_encode['ParentschoolSatisfaction'])

StudentAbsenceDays_encode = label_encode.fit_transform(df_encode['StudentAbsenceDays'])

Class_encode = label_encode.fit_transform(df_encode['Class'])



df_encode['gender_encode'] = gender_encode

df_encode['nationality_encode'] = nationality_encode

df_encode['PlaceofBirth_encode'] = PlaceofBirth_encode

df_encode['StageID_encode'] = StageID_encode

df_encode['GradeID_encode'] = GradeID_encode

df_encode['SectionID_encode'] = SectionID_encode

df_encode['Topic_encode'] = Topic_encode

df_encode['Semester_encode'] = Semester_encode

df_encode['Relation_encode'] = Relation_encode

df_encode['ParentAnsweringSurvey_encode'] = ParentAnsweringSurvey_encode

df_encode['ParentschoolSatisfaction_encode'] = ParentschoolSatisfaction_encode

df_encode['StudentAbsenceDays_encode'] = StudentAbsenceDays_encode

df_encode['Class_encode'] = Class_encode
df_encode.head()
cols = ['gender',

        'NationalITy',

        'PlaceofBirth',

        'StageID',

        'GradeID',

        'SectionID',

        'Topic',

        'Semester',

        'Relation',

        'ParentAnsweringSurvey',

        'ParentschoolSatisfaction',

        'StudentAbsenceDays',

        'Class']

df_encode.drop(cols, inplace=True, axis=1)
corr_mat = df_encode.corr()

corr_mat['Class_encode'].sort_values(ascending=False)
X = df_encode.drop('Class_encode',axis=1)

y= df_encode['Class_encode'].ravel()

X_Train, X_Test, y_train, y_test = train_test_split(X, y, random_state =42, test_size=0.2)
sgd_clf = SGDClassifier()

sgd_clf.fit(X_Train, y_train)

sgd_pred = sgd_clf.predict(X_Train)

precision_score(y_train, sgd_pred, average=None), recall_score(y_train, sgd_pred, average=None)
sgd_clf_crs = SGDClassifier()

cross_val_pred = cross_val_predict(sgd_clf_crs, X_Train, y_train, cv=5)
print("Precision Score : ",precision_score(y_train, cross_val_pred, average=None))

print("Recall Score : ",recall_score(y_train, cross_val_pred, average=None))

print("f1 score : ",f1_score(y_train, cross_val_pred, average=None))
forest = RandomForestClassifier(n_estimators =10)

cros_val_forest = cross_val_predict(forest, X_Train, y_train, cv=3)

print("Precision Score : ",precision_score(y_train, cros_val_forest, average=None))

print("Recall Score : ",recall_score(y_train, cros_val_forest, average=None))

print("f1 score : ",f1_score(y_train, cros_val_forest, average=None))
forest.fit(X_Train, y_train)

forest_pred = forest.predict(X_Train)

print("Precision Score : ",precision_score(y_train, forest_pred, average=None))

print("Recall Score : ",recall_score(y_train, forest_pred, average=None))

print("f1 score : ",f1_score(y_train, forest_pred, average=None))
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = LabelBinarizer()

    lb.fit(y_test)

    y_test = lb.transform(y_test)

    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)
multiclass_roc_auc_score(y_train, forest_pred)
forest_test_pred = forest.predict(X_Test)

multiclass_roc_auc_score(y_test, forest_test_pred)

conf_matrix = confusion_matrix(y_test, forest_test_pred)

#df_label = df.pivot("Class")

x_label = ["High-Level","Low-Level","Middle-Level"]

y_label = ["High-Level","Low-Level","Middle-Level"]

sb.heatmap(conf_matrix, annot=True, linewidths = 0.5, cbar=False, xticklabels =  x_label, yticklabels = y_label)