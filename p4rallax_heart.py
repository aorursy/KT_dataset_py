# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn.metrics import roc_curve ,auc

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

import shap

from pdpbox import pdp, info_plots

np.random.seed(123)

pd.options.mode.chained_assignment = None 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')

data.head()

#Change column names to better ones

data.columns = ['age' , 'sex' , 'chest_pain_type', 'resting_bp' , 'cholestrol' , 'fasting_sugar', 'rest_ecg' , 'max_heartrate', 'angina' , 'depression', 'st_slope' , 'num_major_vessels' , 'thalassemia' , 'target']
data['sex'][data['sex'] == 0] = 'female'

data['sex'][data['sex'] == 1] = 'male'



data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'

data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'

data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'

data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'



data['fasting_sugar'][data['fasting_sugar'] == 0] = 'lower than 120mg/ml'

data['fasting_sugar'][data['fasting_sugar'] == 1] = 'greater than 120mg/ml'



data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'

data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'

data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'



data['angina'][data['angina'] == 0] = 'no'

data['angina'][data['angina'] == 1] = 'yes'



data['st_slope'][data['st_slope'] == 1] = 'upsloping'

data['st_slope'][data['st_slope'] == 2] = 'flat'

data['st_slope'][data['st_slope'] == 3] = 'downsloping'



data['thalassemia'][data['thalassemia'] == 1] = 'normal'

data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'

data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'



data.dtypes
data['sex'] = data['sex'].astype('object')

data['chest_pain_type'] = data['chest_pain_type'].astype('object')

data['fasting_sugar'] = data['fasting_sugar'].astype('object')

data['rest_ecg'] = data['rest_ecg'].astype('object')

data['angina'] = data['angina'].astype('object')

data['st_slope'] = data['st_slope'].astype('object')

data['thalassemia'] = data['thalassemia'].astype('object')

data.dtypes

data = pd.get_dummies(data , drop_first = True)

data.head()

x_train , x_test , y_train , y_test = train_test_split (data.drop('target', 1) , data['target'] ,test_size = .2 , random_state = 10)

model = RandomForestClassifier(max_depth =5)

model.fit(x_train , y_train)
y_predict = model.predict(x_test)

#y_predict

y_pred_quant = model.predict_proba(x_test)[:,1]

#y_pred_quant

y_pred_bin  = model.predict(x_test)
cm = confusion_matrix(y_test , y_pred_bin)

cm
sensi = cm[0,0]/(cm[0,0]+cm[1,0])

speci = cm[1,1]/(cm[1,1]+cm[0,1])

print("Sensitivity = " ,sensi)

print("Specificity =" , speci)

fpr, tpr,  thresholds = roc_curve(y_test , y_pred_quant)



fig , ax = plt.subplots()

ax.plot(fpr,tpr)

ax.plot([0,1] , [0,1] , transform= ax.transAxes , ls="--" , c=".3")

plt.xlim([0.0 , 1.0])

plt.ylim([0.0 , 1.0])

plt.xlabel('False Positivity Rate')

plt.ylabel('True Positivity Rate')

plt.grid(True)

plt.title('ROC Curve for Classifier')
auc(fpr,tpr)
perm = PermutationImportance(model , random_state =1).fit(x_test ,y_test)

eli5.show_weights(perm ,feature_names = x_test.columns.tolist())
base_features = data.columns.values.tolist()

base_features.remove('target')



fname = 'num_major_vessels'

pdp_dist = pdp.pdp_isolate(model=model ,dataset = x_test , model_features = base_features , feature = fname)

pdp.pdp_plot(pdp_dist , fname)

plt.show()

fname = 'age'

pdp_age = pdp.pdp_isolate(model = model ,dataset = x_test,model_features = base_features , feature = fname)

pdp.pdp_plot(pdp_age,fname)

plt.show()
fname = 'depression'

pdp_dep = pdp.pdp_isolate(model = model , dataset = x_train , model_features = base_features , feature =  fname)

pdp.pdp_plot(pdp_dep , fname)

plt.show()
def riskfactors( model , patient):

    explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(patient)

    shap.initjs

    return shap.force_plot(explainer.expected_value[1] , values[1] , patient)
shap.initjs()
pred_patient = x_test.iloc[1,:].astype(float)

riskfactors(model , pred_patient)