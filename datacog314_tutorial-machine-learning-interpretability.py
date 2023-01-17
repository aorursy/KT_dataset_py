### Elementary my Dear Watson...

import pandas as pd

import numpy as np



### Graphic libraries

import matplotlib.pyplot as plt

import seaborn as sns 



### Some Scikit-learn utils

from sklearn.model_selection import train_test_split



### Metrics

from sklearn import metrics

from sklearn.metrics import accuracy_score, roc_curve, auc



### Models

from xgboost import XGBClassifier, plot_importance



########################################################

### For an easier workflow, Interpretability libraries

### will be installed/loaded on the fly of the tutorial

########################################################



### Some cosmetics add-ons

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

%matplotlib inline

# loading the csv dataset in a dataframe

df_raw = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df_raw.shape
# Let's visually check the first lines of our wine collection

df_raw.head()
# types of data

df_raw.info()
# Synthetic descriptive statistics

df_raw.describe()
# make for security a copy of the original dataframe before further processing 

wines = df_raw.copy()
# Extracting our target variable 

# and creating a usefull feature list of dependant variables

target = 'quality'

features_list = list(wines.columns)

features_list.remove(target)
wines[features_list].hist(bins=40, edgecolor='b', linewidth=1.0,

                          xlabelsize=8, ylabelsize=8, grid=False, 

                          figsize=(16,6), color='red')    

plt.tight_layout(rect=(0, 0, 1.2, 1.2))   

plt.suptitle('Red Wine Univariate Plots', x=0.65, y=1.25, fontsize=14);  
wines[target].hist(bins=40, edgecolor='b', linewidth=1.0,

              xlabelsize=8, ylabelsize=8, grid=False, figsize=(6,2), color='red')    

plt.tight_layout(rect=(0, 0, 1.2, 1.2))   

plt.suptitle('Red Wine Quality Plot', x=0.65, y=1.25, fontsize=14);  
# for visualizing correlations

f, ax = plt.subplots(figsize=(10, 6))

corr = wines.corr()

hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="Reds",fmt='.2f',

            linewidths=.05)

f.subplots_adjust(top=0.93)

t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
sns.set()

sns.relplot(data=wines, x='alcohol', y=target, kind='line', height=5, aspect=2, color='red');    
# create our separate target vector 

y = wines.pop('quality')



# mapping the target to a binary class at quality = 5

y = y.apply(lambda x: 0 if x <= 5 else 1)



# quickly check that we have a balanced target partition

y.sum() / len(y)
# building train/test datasets on a 70/30 ratio

X_train, X_test, y_train, y_test = train_test_split(wines, y, test_size=0.3, random_state=33)

X_train.shape, X_test.shape
%%time



# ML in two lines ;)

xgb = XGBClassifier(objective='binary:logistic', random_state=33, n_jobs=-1)

xgb.fit(X_train, y_train)
# make predictions for test data

xgb_predictions = xgb.predict(X_test)
# We design a simple classification evaluative function

def evaluation_scores(test, prediction, target_names=None):

    print('Accuracy:', np.round(metrics.accuracy_score(test, prediction), 4)) 

    print('-'*60)

    print('classification report:\n\n', metrics.classification_report(y_true=test, y_pred=prediction, target_names=target_names)) 

    

    classes = [0, 1]

    total_classes = len(classes)

    level_labels = [total_classes*[0], list(range(total_classes))]



    cm = metrics.confusion_matrix(y_true=test, y_pred=prediction, labels=classes)

    cm_frame = pd.DataFrame(data=cm, columns=pd.MultiIndex(levels=[['Predicted:'], classes], labels=level_labels), 

                            index=pd.MultiIndex(levels=[['Actual:'], classes], labels=level_labels))

    

    print('-'*60)

    print('Confusion matrix:\n')

    print(cm_frame) 
# Evaluate predictions

evaluation_scores(y_test, xgb_predictions, target_names=['Low Quality', 'Hight Quality'])
# calculate the FPR and TPR for all thresholds of the classification

probs = xgb.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'red', label = 'ROC AUC score = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'b--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# ploting XGBoost default feature importances

fig = plt.figure(figsize = (18, 10))

title = fig.suptitle("Native Feature Importances from XGBoost", fontsize=14)



ax1 = fig.add_subplot(2, 2, 1)

plot_importance(xgb, importance_type='weight', ax=ax1, color='red')

ax1.set_title("Feature Importance with Feature Weight");



ax2 = fig.add_subplot(2, 2, 2)

plot_importance(xgb, importance_type='cover', ax=ax2, color='red')

ax2.set_title("Feature Importance with Sample Coverage");



ax3 = fig.add_subplot(2, 2, 3)

plot_importance(xgb, importance_type='gain', ax=ax3, color='red')

ax3.set_title("Feature Importance with Split Mean Gain");
# pip install eli5

import eli5

from eli5.sklearn import PermutationImportance
eli5.show_weights(xgb.get_booster())
wine_nb = 0

print('Reference:', y_test.iloc[wine_nb])

print('Predicted:', xgb_predictions[wine_nb])

eli5.show_prediction(xgb.get_booster(), X_test.iloc[wine_nb], 

                     feature_names=list(wines.columns), show_feature_values=True)
wine_nb = 4

print('Reference:', y_test.iloc[wine_nb])

print('Predicted:', xgb_predictions[wine_nb])

eli5.show_prediction(xgb.get_booster(), X_test.iloc[wine_nb], 

                     feature_names=list(wines.columns), show_feature_values=True)
%%time



# we need to retrain a new model with arrays

# as eli5 has a bug with Dataframes and XGBoost

# cf. https://github.com/TeamHG-Memex/eli5/pull/261

xgb_array = XGBClassifier(objective='binary:logistic', random_state=33, n_jobs=-1)

xgb_array.fit(X_train, y_train)
feat_permut = PermutationImportance(xgb_array, random_state=33).fit(X_train, y_train)

eli5.show_weights(feat_permut, feature_names = features_list)
# pip install pdpbox

from pdpbox import pdp, get_dataset, info_plots
def plot_pdp(model, df, feature, cluster_flag=False, nb_clusters=None, lines_flag=False):

    

    # Create the data that we will plot

    pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns.tolist(), feature=feature)



    # plot it

    pdp.pdp_plot(pdp_goals, feature, cluster=cluster_flag, n_cluster_centers=nb_clusters, plot_lines=lines_flag)

    plt.show()
# plot the PD univariate plot

plot_pdp(xgb, X_train, 'alcohol')
# for ICE plot we must specify the numbers of similarity clusters we want

# here 24

plot_pdp(xgb, X_train, 'alcohol', cluster_flag=True, nb_clusters=24, lines_flag=True)
features_to_plot = ['pH', 'fixed acidity']

inter1  =  pdp.pdp_interact(model=xgb, dataset=X_train, model_features=features_list, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='grid')



# we use plot_type='grid' as the default and better option 'contour' has a bug which is being corrected

# cf. https://github.com/SauceCat/PDPbox/issues/40



plt.show()
# installation of skater can be tricky, try :

# pip install -U skater

# conda install --yes -c conda-forge skater



# check out skater installation instructions at 

# https://oracle.github.io/Skater/install.html



# for testing the installation

# ! python -c "from skater.tests.all_tests import run_tests; run_tests()"
from skater.core.explanations import Interpretation

from skater.model import InMemoryModel
interpreter = Interpretation(training_data=X_test, training_labels=y_test, feature_names=features_list)

im_model = InMemoryModel(xgb.predict_proba, examples=X_train, target_names=['Low Quality', 'Hight Quality'])
plots = interpreter.feature_importance.plot_feature_importance(im_model, ascending=True, progressbar=False)
r = interpreter.partial_dependence.plot_partial_dependence(['pH'], im_model, grid_resolution=50, 

                                                           grid_range=(0,1), n_samples=1000, 

                                                           with_variance=True, figsize = (6, 4), n_jobs=-1)

yl = r[0][1].set_ylim(0, 1) 
%%time

# beware : this process is computationally slow/heavy



plots_list = interpreter.partial_dependence.plot_partial_dependence([('pH', 'fixed acidity')], 

                                                                    im_model, grid_range=(0,1), 

                                                                    n_samples=1000,

                                                                    figsize=(16, 6),

                                                                    grid_resolution=100,

                                                                    progressbar=True,

                                                                    n_jobs=-1)
predictions = xgb_array.predict_proba(X_test.values)
from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer



exp = LimeTabularExplainer(X_test.values, feature_names=features_list, discretize_continuous=True, class_names=['Low Quality', 'High Quality'])
wine_nb = 0

print('Reference:', y_test.iloc[wine_nb])

print('Predicted:', predictions[wine_nb])

exp.explain_instance(X_test.iloc[wine_nb].values, xgb_array.predict_proba).show_in_notebook()
wine_nb = 4

print('Reference:', y_test.iloc[wine_nb])

print('Predicted:', predictions[wine_nb])

exp.explain_instance(X_test.iloc[wine_nb].values, xgb_array.predict_proba).show_in_notebook()
surrogate_explainer = interpreter.tree_surrogate(oracle=im_model, seed=33)
f1 = surrogate_explainer.fit(X_train, y_train, use_oracle=True, prune='pre', scorer_type='f1')

print('F1 score for the surrogate tree: ', f1)
# A reminder for referencing the originals feature names 

# since these names are not kept in the surrogate tree

pd.DataFrame([('X'+str(idx), feature) for (idx, feature) in enumerate(wines.columns)]).T
from skater.util.dataops import show_in_notebook



# 'Low Quality' (score <= 5) class in pink and 'High Quality' class (score > 5) in red

surrogate_explainer.plot_global_decisions(colors=['pink', 'red'], file_name='test_tree_sur.png', fig_size=(8,8))



show_in_notebook('test_tree_sur.png', width=1200, height=800);
# using our evaluation_scores function 

surrogate_predictions = surrogate_explainer.predict(X_test)

evaluation_scores(y_test, surrogate_predictions, target_names=['low quality', 'hight quality'])
# calculate the ROC AUC score for the tree surrogated model

roc_auc = metrics.roc_auc_score(y_test, surrogate_predictions)

print('ROC AUC score: ', round(roc_auc, 2))
# ! pip install shap



import shap



# load JS visualization code to notebook

shap.initjs()
# explain the model's predictions using SHAP values

# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

explainer = shap.TreeExplainer(xgb)

shap_values = explainer.shap_values(X_test)
X_shap = pd.DataFrame(shap_values)

X_shap.tail()
print('Expected Value: ', explainer.expected_value)
shap.summary_plot(shap_values, X_test, plot_type="bar", color='red')
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[4,:], X_test.iloc[4,:])
shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_test.iloc[:1000,:])
shap.summary_plot(shap_values, X_test)
shap.dependence_plot(ind='pH', interaction_index='fixed acidity',

                     shap_values=shap_values, 

                     features=X_test,  

                     display_features=X_test)


# ! pip install https://github.com/adebayoj/fairml/archive/master.zip



from fairml import audit_model

from fairml import plot_dependencies
%%time



xgb_fair = XGBClassifier(objective='binary:logistic', random_state=33, n_jobs=-1)



xgb_fair.fit(X_train.values, y_train)
%%time



# call audit model

feat_importances, _ = audit_model(xgb_fair.predict, X_train, distance_metric='accuracy', direct_input_pertubation_strategy='constant-zero',

                                 number_of_runs=50, include_interactions=True)



# print feature importance

print(feat_importances)
# generate feature dependence plot

fig = plot_dependencies(

    feat_importances.median(),

    reverse_values=False,

    title="FairML feature dependence XGB model",

    fig_size=(8,3)

    )



# Print it in a file

file_name = "fairml_wine_quality.png"

plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)