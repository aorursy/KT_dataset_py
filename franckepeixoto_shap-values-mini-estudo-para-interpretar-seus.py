import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')



data.Age.fillna(value=data.Age.median() ,inplace=True)

data.Embarked.fillna(value='S', inplace=True)



dropColumns =["PassengerId","Name","Ticket","Cabin","Embarked"]

for col in dropColumns:

  data.drop(columns=[col], inplace=True)



y = data.Survived
feature_names = ["Pclass","Sex","Age","SibSp","Parch","Fare"]

dummies =pd.get_dummies(data[feature_names])

dummies.head(2)
x= dummies

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
model = RandomForestClassifier(random_state=0).fit(train_x, train_y)
row = 50

data_prediction = val_x.iloc[row] 

data_prediction_array = data_prediction.values.reshape(1, -1)

model.predict_proba(data_prediction_array)
import shap 

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(data_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_prediction)
k_explainer = shap.KernelExplainer(model.predict_proba, train_x)

k_shap_values = k_explainer.shap_values(data_prediction)

shap.initjs()

shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_prediction)
import eli5 #!pip install eli5 

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(val_x, val_y)

eli5.show_weights(perm, feature_names = val_x.columns.tolist())
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots  #!pip install pdpbox

feature_names = dummies.columns

pdp_isolate_ = pdp.pdp_isolate(model=model, dataset=val_x, model_features=feature_names, feature='Age')

pdp.pdp_plot(pdp_isolate_, 'Age')

plt.show()
shap_values = explainer.shap_values(val_x)

shap.summary_plot(shap_values[1], val_x)
shap_values = explainer.shap_values(x)

shap.dependence_plot('Age', shap_values[1], x, interaction_index="Pclass")