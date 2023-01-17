from IPython.display import YouTubeVideo

YouTubeVideo('BLw62AhW_Kc', width=700, height=400)
!pip install lofo-importance
from lofo import LOFOImportance, Dataset, plot_importance

import shap 

import warnings  

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,f1_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import os

import seaborn as sns

print(os.listdir("../input"))

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.model_selection import KFold

# Any results you write to the current directory are saved as output.
#Reading dataset 

dt=pd.read_csv('../input/heart.csv')
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']




dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'

dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'

dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'

dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'







dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'

dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'

dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'







dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'

dt['st_slope'][dt['st_slope'] == 2] = 'flat'

dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'



dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'

dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'

dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'
# Chacking datatypes of all features 

dt.dtypes
dt.head()
dt.describe()
dt.info()
## null count analysis

import missingno as msno

p=msno.bar(dt)
f,ax=plt.subplots(1,2,figsize=(18,8))

dt['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('target',data=dt,ax=ax[1])

ax[1].set_title('target')

plt.show()
dataset2=dt.drop(['target'],axis=1)

p = dataset2.hist(figsize = (12,8))
sns.boxplot(data=dt,x="target", y="cholesterol");
sns.boxplot(data=dt,x="target", y="max_heart_rate_achieved");
sns.boxplot(data=dt,x="target", y="resting_blood_pressure");
cols_drp=['age','sex','fasting_blood_sugar','exercise_induced_angina','st_depression','num_major_vessels','target']

dt_o=dt.drop(cols_drp,axis=1)



Q1 = dt_o.quantile(0.25)

Q3 = dt_o.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
dt_clean = dt.loc[~dt['cholesterol'].isin([63.5])]

dt_clean= dt_clean.loc[~dt_clean['resting_blood_pressure'].isin([20.0])]

dt_clean= dt_clean.loc[~dt_clean['max_heart_rate_achieved'].isin([32.5])]



dt_clean.shape

sns.boxplot(data=dt_clean,x="target", y="resting_blood_pressure");
sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(dt['age']);
sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(dt['cholesterol']);
sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(dt['max_heart_rate_achieved']);
sns.swarmplot(data=dt,x="target", y="age");
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(dataset2.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
dt1=pd.get_dummies(dt,drop_first=True)
dt1.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(dt1.drop(["target"],axis = 1),),

        columns=['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression',

       'num_major_vessels', 'sex_male', 'chest_pain_type_atypical angina','chest_pain_type_non-anginal pain','chest_pain_type_typical angina','fasting_blood_sugar_lower than 120mg/ml','rest_ecg_left ventricular hypertrophy','rest_ecg_normal','exercise_induced_angina_yes','st_slope_flat','st_slope_upsloping','thalassemia_fixed defect','thalassemia_normal','thalassemia_reversable defect'])
y=dt['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,stratify=y, random_state=5)
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state = 0)

logit.fit(X_train, y_train)



# Predicting Test Set

y_pred = logit.predict(X_test)
from sklearn.model_selection import cross_val_score

roc=roc_auc_score(y_test, y_pred)

accuracies = cross_val_score(estimator = logit, X = X_test, y = y_test, cv = 10)

acc = accuracies.mean()

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Base - Logistic Regression', acc,prec,rec, f1,roc]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])



results
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score

random_forest = RandomForestClassifier(n_estimators=500,criterion='entropy',max_depth=5).fit(X_train, y_train)

y_pred_random = random_forest.predict(X_test)
roc=roc_auc_score(y_test, y_pred)

accuracies = cross_val_score(estimator = random_forest, X = X_test, y = y_test, cv = 10)

acc = accuracies.mean()

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest', acc,prec,rec, f1,roc]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results,sort=True)

results
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(random_forest, random_state=123).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X.columns.tolist(),top=24)
from eli5 import explain_prediction





eli5.show_prediction(random_forest, X_test.iloc[50], 

                     feature_names=X_test.columns.tolist(), show_feature_values=True)
features = [c for c in X_test.columns]
from pdpbox import pdp, get_dataset, info_plots



pdp_thal = pdp.pdp_isolate(model=random_forest, dataset=X_test, model_features=features, feature='thalassemia_reversable defect')



# plot it

pdp.pdp_plot(pdp_thal, 'thalassemia_reversable defect')



plt.show()
pdp_resting_bp = pdp.pdp_isolate(model=random_forest, dataset=X_test, model_features=features, feature='resting_blood_pressure')



# plot it

pdp.pdp_plot(pdp_resting_bp, 'resting_blood_pressure')



plt.show()
def plot_pdp(model, df, feature, cluster_flag=False, nb_clusters=None, lines_flag=False):

    

    # Create the data that we will plot

    pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns.tolist(), feature=feature)



    # plot it

    pdp.pdp_plot(pdp_goals, feature, cluster=cluster_flag, n_cluster_centers=nb_clusters, plot_lines=lines_flag)

    plt.show()
plot_pdp(random_forest, X_train, 'thalassemia_reversable defect', cluster_flag=True, nb_clusters=24, lines_flag=True)
plot_pdp(random_forest, X_train, 'resting_blood_pressure', cluster_flag=True, nb_clusters=24, lines_flag=True)
plot_pdp(random_forest, X_train, 'age', cluster_flag=True, nb_clusters=24, lines_flag=True)
plot_pdp(random_forest, X_train, 'st_slope_flat', cluster_flag=True, nb_clusters=24, lines_flag=True)



inter1  =  pdp.pdp_interact(model=random_forest, dataset=X_test, model_features=features, features=['thalassemia_reversable defect', 'resting_blood_pressure'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['thalassemia_reversable defect', 'resting_blood_pressure'], plot_type='contour')

plt.show()



inter1  =  pdp.pdp_interact(model=random_forest, dataset=X_test, model_features=features, features=['age', 'resting_blood_pressure'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['age', 'resting_blood_pressure'], plot_type='contour')

plt.show()
inter1  =  pdp.pdp_interact(model=random_forest, dataset=X_test, model_features=features, features=['age', 'st_slope_flat'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['age', 'st_slope_flat'], plot_type='contour')

plt.show()
fig, axes, summary_df = info_plots.actual_plot_interact(

    model=random_forest, X=X_test, features=['age', 'thalassemia_reversable defect'], feature_names=['age', 'thalassemia_reversable defect']

)
fig, axes, summary_df = info_plots.actual_plot_interact(

    model=random_forest, X=X_test, features=['age', 'resting_blood_pressure'], feature_names=['age', 'resting_blood_pressure']

)
row_to_show = 17

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





random_forest.predict_proba(data_for_prediction_array)



import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(random_forest)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
row_to_show = 9

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





random_forest.predict_proba(data_for_prediction_array)



import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(random_forest)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
explainer = shap.TreeExplainer(random_forest)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)
explainer = shap.TreeExplainer(random_forest)



# calculate shap values. This is what we will plot.

shap_values = explainer.shap_values(X_test)



# make plot.

shap.dependence_plot('chest_pain_type_non-anginal pain', shap_values[0], X_test, interaction_index="age")
shap.dependence_plot('chest_pain_type_non-anginal pain', shap_values[0], X_test, interaction_index="resting_blood_pressure")
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1],plot_cmap="DrDb")
# extract a sample of the data

sample_df = dt1.sample(frac=0.5, random_state=0)
cv = KFold(n_splits=4, shuffle=False, random_state=0)
# define the binary target and the features

dataset = Dataset(df=sample_df, target="target", features=[col for col in dt1.columns if col != 'target'])
# define the validation scheme and scorer. The default model is LightGBM

lofo_imp = LOFOImportance(dataset, cv=cv, scoring="roc_auc")

# get the mean and standard deviation of the importances in pandas format

importance_df = lofo_imp.get_importance()
# plot the means and standard deviations of the importances

plot_importance(importance_df, figsize=(12, 20))
!pip install alibi
from alibi.explainers import AnchorTabular
predict_fn = lambda x: random_forest.predict_proba(x)
explainer = AnchorTabular(predict_fn, features)
explainer.fit(X_train.values, disc_perc=[25, 50, 75])
class_names=['Healthy','Disease']



idx = 3

explanation = explainer.explain(X_test.values[idx], threshold=0.95)

print('Anchor: %s' % (' AND '.join(explanation['names'])))

print('Precision: %.2f' % explanation['precision'])

print('Coverage: %.2f' % explanation['coverage'])
idx = 19

explanation = explainer.explain(X_test.values[idx], threshold=0.95)

print('Anchor: %s' % (' AND '.join(explanation['names'])))

print('Precision: %.2f' % explanation['precision'])

print('Coverage: %.2f' % explanation['coverage'])
import lime

import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=features, class_names=class_names, discretize_continuous=True)
i = 12



print('Actual Label:', y_test[i])

print('Predicted Label:', y_pred[i])



exp = explainer.explain_instance(X_test.iloc[i].values, random_forest.predict_proba).show_in_notebook()
!pip install git+https://github.com/bondyra/pyBreakDown.git
from pyBreakDown.explainer import Explainer

from pyBreakDown.explanation import Explanation
#make explainer object

exp = Explainer(clf=random_forest, data=X_train, colnames=features)
#make explanation object that contains all information

explanation = exp.explain(observation=X.iloc[302,:],direction="up")
#get information in text form

explanation.text()
#customized text form

explanation.text(fwidth=40, contwidth=40, cumulwidth = 40, digits=4)
explanation.visualize()
#customize height, width and dpi of plot

explanation.visualize(figsize=(8,5),dpi=100)
explanation = exp.explain(observation=X.iloc[302,:],direction="up",useIntercept=True)  # baseline==intercept

explanation.visualize(figsize=(8,5),dpi=100)