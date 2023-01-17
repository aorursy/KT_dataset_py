import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

import eli5 

from eli5.sklearn import PermutationImportance
df_train = pd.read_csv('../input/heart.csv')

df_train.head()
df_train.columns
df_train.shape

df_train.describe()
df_train.nunique()
df_train.dtypes
df_train.isnull().sum()
plt.rcParams['figure.figsize']=(20,20)

hm=sns.heatmap(df_train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']].corr(), annot = True)
df_train = pd.get_dummies(df_train, drop_first=True)
df_train.head()
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('target', 1), df_train['target'], test_size = 0.2, random_state=1)

model = RandomForestClassifier(max_depth=5)

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_predict)

confusion_matrix
sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity : ', specificity)
permutation = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(permutation, feature_names = X_test.columns.tolist())
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



feature_names = [i for i in X_train.columns]



pdp_tha = pdp.pdp_isolate(model=model, dataset=X_test, model_features=feature_names, feature='thal')



pdp.pdp_plot(pdp_tha, 'thal')







plt.show()
feature_names = [i for i in X_train.columns]



pdp_tha = pdp.pdp_isolate(model=model, dataset=X_test, model_features=feature_names, feature='restecg')



pdp.pdp_plot(pdp_tha, 'restecg')







plt.show()
feature_names = [i for i in X_train.columns]



pdp_tha = pdp.pdp_isolate(model=model, dataset=X_test, model_features=feature_names, feature='oldpeak')



pdp.pdp_plot(pdp_tha, 'oldpeak')







plt.show()
import shap 

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test, plot_type="bar")
sample_data_for_prediction = X_test.iloc[0].astype(float)  



def patient_risk_factors(model, patient_data):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(patient_data)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)





patient_risk_factors(model,sample_data_for_prediction)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train.iloc[:60])

shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[:60])
explainer = shap.TreeExplainer(model)



shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test)


shap.dependence_plot('slope', shap_values[1], X_test, interaction_index='thal')





shap.dependence_plot('slope', shap_values[1], X_test, interaction_index="ca")
features_to_plot = ['slope', 'ca']

inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')



plt.show()
features_to_plot = ['slope', 'thal']

inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')



plt.show()