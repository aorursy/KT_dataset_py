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
#Load the data



dt = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")



# Take a look at the data



dt.head()
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'resting_electrocard', 'max_heart_rate_achieved', 'exercice_induced_angina', 'oldpeak', 'slope', 'nb_major_vessels', 'thalassemia', 'target']

dt.dtypes
dt.describe()
dt.info()
dt.target.value_counts()
len(dt[dt.target == 0])/len(dt.target)
dt.groupby('target').mean()
import seaborn as sns

import matplotlib.pyplot as plt

table = pd.crosstab(dt.sex,dt.target)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Correlation between sex and heart disease')

plt.xlabel("0: female, 1: male")

plt.ylabel('Heart disease')

plt.savefig('heart_disease_by_sex')
table = pd.crosstab(dt["exercice_induced_angina"],dt.target)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Correlation between exercice_induced_angina and heart disease')

plt.xlabel("0: no, 1: yes")

plt.ylabel('Heart disease')

plt.savefig('heart_disease_by_exercice_induced_angina')

table = pd.crosstab(dt["fasting_blood_sugar"],dt.target)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Correlation between fasting_blood_sugar and heart disease')

plt.xlabel("0: no, 1: yes")

plt.ylabel('Heart disease')

plt.savefig('heart_disease_by_fbs')
table = pd.crosstab(dt["chest_pain_type"],dt.target)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Correlation between chest_pain_type and heart disease')

plt.xlabel("0: no, 1: yes")

plt.ylabel('Heart disease')

plt.savefig('heart_disease_by_chest_pain_type')
table = pd.crosstab(dt["slope"],dt.target)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Correlation between slope and heart disease')

plt.xlabel("0: no, 1: yes")

plt.ylabel('Heart disease')

plt.savefig('heart_disease_by_slope')
dt['chest_pain_type'][dt['chest_pain_type'] == 0] = 'typical_angina'

dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'atypical_angina'

dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'non_anginal_pain'

dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'asymptomatic'



dt['resting_electrocard'][dt['resting_electrocard'] == 0] = 'normal'

dt['resting_electrocard'][dt['resting_electrocard'] == 1] = 'medium'

dt['resting_electrocard'][dt['resting_electrocard'] == 2] = 'high'



dt['slope'][dt['slope'] == 1] = 'upsloping'

dt['slope'][dt['slope'] == 2] = 'flat'

dt['slope'][dt['slope'] == 3] = 'downsloping'



dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'

dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'

dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'



dt = pd.get_dummies(dt, drop_first=True)
pd.crosstab(dt.age,dt.target).plot(kind="bar",figsize=(15,5))

plt.title('Heart Disease Frequency by Age')

plt.xlabel('Age')

plt.ylabel('Heart Disease Frequency')

plt.savefig('heartDiseaseByAge.png')

plt.show()
#let's define a variable "Senior", where 1 is >=55 and 0 is <55, and drop the age variable

dt["senior"] = pd.cut(dt.age,bins=[0,54,99],labels=["0", "1"])



pd.crosstab(dt.senior, dt.target).plot(kind="bar",figsize=(10,3))



plt.title('Heart Disease by Age Category')

plt.xlabel(['Adult', 'Senior'])

plt.ylabel('Heart Disease Frequency')

plt.savefig('heartDiseaseByAgeCategory.png')

plt.show()

dt = dt.drop("age", 1)
plt.figure(figsize=(15,15))

sns.heatmap(dt.corr(),vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler



y = dt["target"]

x_brut = dt.drop("target", 1)



# Normalize data

scaler = MinMaxScaler()

scaler.fit(x_brut)

x = scaler.transform(x_brut)

# Split between train and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state=10) 



def compute_score(clf, x, y):

    xval = cross_val_score(clf, x, y, cv=5)

    return print(np.mean(xval))



lr = LogisticRegression(max_iter = 4000)

compute_score(lr, x_train, y_train)

lr.fit(x_train, y_train)

print(lr.coef_, dt.columns)

from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test = train_test_split(x_brut, y, test_size = .2, random_state=10) 



model = RandomForestClassifier(max_depth=5)

compute_score(model, x_train, y_train)

model.fit(x_train, y_train)
from sklearn.tree import export_graphviz



estimator = model.estimators_[1]

feature_names = [i for i in x_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no heart disease'

y_train_str[y_train_str == '1'] = 'heart disease'

y_train_str = y_train_str.values



#code from https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c



export_graphviz(estimator, out_file='heart_disease_tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'heart_disease_tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.metrics import confusion_matrix, roc_curve, auc



y_predict = model.predict(x_test)

y_predict_proba = model.predict_proba(x_test)[:, 1]

y_pred_bin = model.predict(x_test)

confusion_matrix = confusion_matrix(y_test, y_pred_bin)

print(confusion_matrix)



total=sum(sum(confusion_matrix))



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity : ', specificity)



fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)



print('AUC', auc(fpr, tpr))
import shap

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x_test)



shap.summary_plot(shap_values[1], x_test, plot_type="bar")