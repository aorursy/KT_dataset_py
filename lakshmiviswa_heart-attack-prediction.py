# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
heart = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
heart.head()
age_sort=heart['age'].sort_values()
age_sort.value_counts(ascending=True).plot(kind='bar',figsize=(10,10))

Gender=heart['sex'].value_counts()
print(Gender)
Gender.plot(kind='bar', title='Count of Male and Female patients')
#gender_male=heart.loc[heart['sex']==1]
#gender_female=heart.loc[heart['sex']==0]

#heart[(heart.sex == 1) & (heart.target == 1)].count()
grouping=heart.groupby(['sex','target'])['target'].agg(['count'])
grouping
grouping.plot(kind='bar')
#The analysis is based on the target class 0 = less chance of heart attack and the target class 1 = higher chance of heart attack..
high_chance = heart.loc[heart['target']==1]
less_chance = heart.loc[heart['target'] == 0]
print("Higher chance of Heart attack numbered " + str(len(high_chance)))
print("Less chance of Heart attack numbered " + str(len(less_chance)))
heart['target'].value_counts().plot(kind='bar')

highrisk = heart.loc[heart['target'] == 1]
lowrisk = heart.loc[heart['target'] == 0]

high,low = heart.target.value_counts()     #Finding the count of classes

normal_target_0_over = lowrisk.sample(high, replace=True)
Combining_together_over = pd.concat([highrisk, normal_target_0_over], axis = 0)

print(Combining_together_over.target.value_counts())
Combining_together_over.target.value_counts().plot(kind='bar', title = 'Data Over Sampling')
#SEPERATING INDEPENDENT VARIABLES AND TARGET VARIABLE
#REFERENCE = FROM WWW.KAGGLE.COM/RENJITHMADHAVAN/CREDIT-CARD-FRAUD-DETECTION-USING-PYTHON
independent_var = Combining_together_over.iloc[:, 0:13].columns
target_var = Combining_together_over.iloc[:0, 13:].columns
print(independent_var)
print(target_var)
from sklearn.model_selection import train_test_split
independent_var = Combining_together_over.iloc[:, 0:13].columns
target_var = Combining_together_over.iloc[:0, 13:].columns
print(independent_var)
print(target_var)

data_independent = Combining_together_over[independent_var]
data_target = Combining_together_over[target_var]
print(data_target)
print(data_independent)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
predict_NB= cross_val_predict(gnb, data_independent, data_target, cv=10)
confmat_NB = confusion_matrix(data_target, predict_NB)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Class_Names = ['LowRisk', 'HighRisk']
dataframe = pd.DataFrame(confmat_NB, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - Naive Bayes - Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
print(classification_report(data_target,predict_NB))
print(accuracy_score(data_target,predict_NB))
print(roc_auc_score(data_target,predict_NB))
print(f1_score(data_target,predict_NB))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
clf_KNN = cross_validate(knn, data_independent,data_target, cv=10)
predict_KNN= cross_val_predict(knn, data_independent, data_target, cv=10)
confmat_KNN = confusion_matrix(data_target, predict_KNN)
Class_Names = ['Low Risk', 'High Risk']
dataframe = pd.DataFrame(confmat_KNN, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - K-Nearest Neighbours - Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
print(classification_report(data_target,predict_KNN))
print(accuracy_score(data_target,predict_KNN))
print(roc_auc_score(data_target,predict_KNN))
print(f1_score(data_target,predict_KNN))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
Ada_gbc = GradientBoostingClassifier()
Adaboostgbc_classifier = AdaBoostClassifier(base_estimator = Ada_gbc, n_estimators=100, random_state=0)
predict_Adagbc= cross_val_predict(Adaboostgbc_classifier, data_independent, data_target, cv=10)
confmat_Adagbc = confusion_matrix(data_target, predict_Adagbc)

print(classification_report(data_target,predict_Adagbc))
print(accuracy_score(data_target,predict_Adagbc))
print(roc_auc_score(data_target,predict_Adagbc))
print(f1_score(data_target,predict_Adagbc))

Class_Names = ['Low Risk', 'High Risk']
dataframe = pd.DataFrame(confmat_Adagbc, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - Adaboost Gradient Boost Classifier - Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
from xgboost import XGBClassifier
model = XGBClassifier()
predict_xgb= cross_val_predict(model, data_independent, data_target, cv=10)
confmat_xgb = confusion_matrix(data_target, predict_xgb)

print(classification_report(data_target,predict_xgb))
print(accuracy_score(data_target,predict_xgb))
print(roc_auc_score(data_target,predict_xgb))
print(f1_score(data_target,predict_xgb))
Class_Names = ['Low Risk', 'High Risk']
dataframe = pd.DataFrame(confmat_xgb, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - XGboost - Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
from sklearn.tree import DecisionTreeClassifier

Ada_dtc = DecisionTreeClassifier(random_state=0)
AdaboostDTC_classifier = AdaBoostClassifier(base_estimator = Ada_dtc, n_estimators=100, random_state=0)
Adaclf_dtc = cross_validate(AdaboostDTC_classifier, data_independent, data_target, cv=10)
predict_Adadtc= cross_val_predict(AdaboostDTC_classifier, data_independent, data_target, cv=10)
confmat_Adadtc = confusion_matrix(data_target, predict_Adadtc)

Class_Names = ['Low Risk', 'High Risk']
dataframe = pd.DataFrame(confmat_Adadtc, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - Adaboost Decision Tree - Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
print(classification_report(data_target,predict_Adadtc))
print(accuracy_score(data_target,predict_Adadtc))
print(roc_auc_score(data_target,predict_Adadtc))
print(f1_score(data_target,predict_Adadtc))

from sklearn.svm import SVC
svc =  SVC(kernel='linear', probability =True)
svc_cv = cross_validate(svc, data_independent, data_target, cv=10)
predict_svc= cross_val_predict(svc, data_independent, data_target, cv=10)
confmat_svc = confusion_matrix(data_target, predict_svc)
Class_Names = ['Low Risk', 'High Risk']
dataframe = pd.DataFrame(confmat_svc, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - SVC- Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
print(classification_report(data_target,predict_svc))
print(accuracy_score(data_target,predict_svc))
print(roc_auc_score(data_target,predict_svc))
print(f1_score(data_target,predict_svc))

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
AdaboostSVM_classifier = AdaBoostClassifier(SVC(probability=True, kernel='linear'), n_estimators = 100, random_state=0)
Adaclf_svm = cross_validate(AdaboostSVM_classifier, data_independent, data_target, cv=10)
predict_Adasvm= cross_val_predict(AdaboostSVM_classifier, data_independent, data_target, cv=10)
confmat_Adasvm = confusion_matrix(data_target, predict_Adasvm)
Class_Names = ['Low Risk', 'High Risk']
dataframe = pd.DataFrame(confmat_Adasvm, index=Class_Names, columns=Class_Names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="rocket_r", fmt = 'g')
plt.title("Confusion Matrix - Adaboost Support Vector - Cross Validation"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
print(classification_report(data_target,predict_Adasvm))
print(accuracy_score(data_target,predict_Adasvm))
print(roc_auc_score(data_target,predict_Adasvm))
print(f1_score(data_target,predict_Adasvm))

Eval_result = pd.DataFrame({'Model': ['Naive Bayes','K-NearestNeighbour','ADAGradient Boost',
                    'XGBClassifier','ADADecision Tree','Support Vector Machine','ADASVC'], 'Recall-HighRisk': [recall_score(data_target,predict_NB)*100,
                    recall_score(data_target, predict_KNN)*100,recall_score(data_target, predict_Adagbc)*100,recall_score(data_target,predict_xgb)*100,recall_score(data_target,predict_Adadtc)*100,recall_score(data_target, predict_svc)*100,recall_score(data_target,predict_Adasvm)*100]})
Eval_result
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=Eval_result['Model'], y=Eval_result['Recall-HighRisk'],line=dict(color='black', width=2, dash='dashdot')))