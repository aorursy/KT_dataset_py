# Basic operations

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None 

from scipy.stats import norm

# Data visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

#Modeling: Random Forest Classifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

#Model explanation

import eli5 

from eli5.sklearn import PermutationImportance

import shap 

from pdpbox import pdp, info_plots #for partial plots



%matplotlib inline



# for providing path

import os

#print(os.listdir('../input/'))
# reading the data

data = pd.read_csv('../input/heart-disease-uci/heart.csv')



##/kaggle/input/heart-disease-uci/heart.csv





# getting the shape

data.shape
# Checking dataset head

data.head()
# Describing dataset

data.describe()
#Pairplot

sns.pairplot(data)
# Heatmap

plt.rcParams['figure.figsize'] = (15, 10)

plt.style.use('ggplot')

sns.heatmap(data.corr(), annot = True)

plt.title('Heatmap for the Dataset', fontsize = 20)

plt.show()
# Checking the distribution of age amonng the patients

plt.rcParams['figure.figsize'] = (12, 9)

sns.distplot(data['age'], fit=norm, kde=False)

plt.title('Distribution of Age', fontsize = 15)

plt.show()
# Checking Target

plt.rcParams['figure.figsize'] = (12, 9)

sns.countplot(data['target'])

plt.xlabel(" Target")

plt.ylabel("Count")

plt.show()
# Checking gender distribuition

plt.rcParams['figure.figsize'] = (12, 9)

size = data['sex'].value_counts()

colors = ['lightblue', 'lightgreen']

labels = "Male", "Female"

explode = [0, 0.01]

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')

plt.title('Distribution of Gender', fontsize = 15)

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.legend()

plt.show()
# Checking chest type

plt.rcParams['figure.figsize'] = (12, 9)

sns.countplot(data['cp'])

plt.xlabel(" Chest type")

plt.ylabel("Count")

plt.show()
# tresbps vs target

plt.rcParams['figure.figsize'] = (12, 9)

sns.boxplot(data['target'], data['trestbps'], palette = 'viridis')

plt.title('Relation of tresbps with target', fontsize = 20)

plt.show()
# cholestrol vs target

plt.rcParams['figure.figsize'] = (12, 9)

sns.violinplot(data['target'], data['chol'], palette = 'colorblind')

plt.title('Relation of Cholestrol with Target', fontsize = 20, fontweight = 30)

plt.show()
# Resting electrocardiographic measurement vs target

plt.rcParams['figure.figsize'] = (12, 9)

dat = pd.crosstab(data['target'], data['restecg']) 

dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 

                                                 stacked = False, 

                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))

plt.title('Relation of ECG measurement with Target', fontsize = 20, fontweight = 30)

plt.show()
# Relation between age and target

plt.rcParams['figure.figsize'] = (12, 9)

sns.swarmplot(data['target'], data['age'], palette = 'winter', size = 10)

plt.title('Relation of Age and target', fontsize = 20, fontweight = 30)

plt.show()
# Relation between sex and target

sns.boxenplot(data['target'], data['sex'], palette = 'Set3')

plt.title('Relation of Sex and target', fontsize = 20, fontweight = 30)

plt.show()
# Relation between thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

sns.boxenplot(data['target'], data['thal'], palette = 'magma')

plt.title('Relation between Target and Blood disorder-Thalessemia', fontsize = 20, fontweight = 30)

plt.show()
# target vs chol and hue = thalach

plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('fivethirtyeight')

plt.scatter(x = data['target'], y = data['chol'], s = data['thalach']*100, color = 'yellow')

plt.title('Relation of target with cholestrol and thalessemia', fontsize = 20, fontweight = 30)

plt.show()
# multi-variate analysis

sns.boxplot(x = data['target'], y = data['trestbps'], hue = data['sex'], palette = 'rainbow')

plt.title('Checking relation of tresbps with genders to target', fontsize = 20, fontweight = 30)

plt.show()
# Changing columns names

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

data.columns
# Preparing labels to encode

data['sex'][data['sex'] == 0] = 'female'

data['sex'][data['sex'] == 1] = 'male'



data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'

data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'

data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'

data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'



data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'

data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'

data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'



data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'

data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'



data['st_slope'][data['st_slope'] == 1] = 'upsloping'

data['st_slope'][data['st_slope'] == 2] = 'flat'

data['st_slope'][data['st_slope'] == 3] = 'downsloping'



data['thalassemia'][data['thalassemia'] == 1] = 'normal'

data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'

data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'
# Checking type of data in our dataframe

data.dtypes
# Converting columns in objects

data['sex'] = data['sex'].astype('object')

data['chest_pain_type'] = data['chest_pain_type'].astype('object')

data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')

data['rest_ecg'] = data['rest_ecg'].astype('object')

data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')

data['st_slope'] = data['st_slope'].astype('object')

data['thalassemia'] = data['thalassemia'].astype('object')
# Taking the labels out from the data 

y = data['target']

data = data.drop('target', axis = 1)

print("Shape of y:", y.shape)
# OneHot encoding of the data

# drop_first = True, means dropping the first categories from each of the attribues 

# for ex gender having gender_male and gender-female would be male having values 1 and 0

data = pd.get_dummies(data, drop_first=True)
# Checking the dataset after encoding

data.head()
# Splitting the dependent and independent variables from the data

x = data
# Checking the shapes of x and y

print("Shape of x:", x.shape)

print("Shape of y:", y.shape)
y.value_counts()
# Splitting the sets into training and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# getting the shapes

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
#Creating a function to find the best value for n_estimators, considering 3 folders cross-validation 

def get_score(n_estimators):

    """Return the precision over 5 CV folds"""

    random_forest_pipeline = Pipeline(steps=[('model', RandomForestClassifier(n_estimators, max_depth=5))])

    scores = cross_val_score(random_forest_pipeline, x, y,

                                  cv=5,

                                  scoring='precision')

    return scores.mean()
results_random_forest = {}

for i in range(1,9):

    results_random_forest[50*i] = get_score(50*i)
plt.plot(list(results_random_forest.keys()), list(results_random_forest.values()))

plt.xlabel('n_estimators')

plt.ylabel('Precision')

plt.show()
#Finding the best n_estimators

n_estimators_random_forest = min(results_random_forest, key=results_random_forest.get)

n_estimators_random_forest
#Fitting Model

model = RandomForestClassifier(n_estimators = n_estimators_random_forest, random_state=0, max_depth=5, n_jobs=8)

model.fit(x_train, y_train)
# Getting predictions

y_predict = model.predict(x_test)

y_pred_quant = model.predict_proba(x_test)[:, 1]  #predict_proba(self, X) ---Probability estimates.

y_pred = model.predict(x_test)
# Generating Cofusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
# Generating Classification Report

cr = classification_report(y_test, y_pred)

print(cr)
# Computing Accuracy for training and test data

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))
#Computing Matthews Correlation Coefficient

print('Matthews Correlation Coefficient:', matthews_corrcoef(y_test, y_pred))
# Computing Specificity and Sensitivity scores

total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])

print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])

print('Specificity : ', specificity)
#ROC Curve



#Computing fpr (false positives rate), tpr (true positives rate) and thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



#Ploting ROC Curve

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for diabetes classifier', fontweight = 30)

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
# Checking AUC score

auc = auc(fpr, tpr)

print("AUC Score :", auc)
#Tree for Model Explanation

from sklearn.tree import export_graphviz

estimator = model.estimators_[1]

feature_names = [i for i in x_train.columns]

y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=50'])

from IPython.display import Image

Image(filename = 'tree.png')
# Checking the importance of each attributes

perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
# Partial dependence plot for thalassemia_fixed defect

base_features = data.columns.values.tolist()

feat_name = 'thalassemia_fixed defect'

pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# Partial dependence plot for num_major_vessels

base_features = data.columns.values.tolist()

feat_name = 'num_major_vessels'

pdp_dist = pdp.pdp_isolate(model=model, dataset=x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# Partial dependence graph for st_depression

base_features = data.columns.values.tolist()

feat_name = 'st_depression'

pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()


# Partial dependence plot for thalassemia_reversable defect

base_features = data.columns.values.tolist()

feat_name = 'thalassemia_reversable defect'

pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# Partial dependence graph for chest_pain_type_atypical angina

base_features = data.columns.values.tolist()

feat_name = 'chest_pain_type_atypical angina'

pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# Checking shap values

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values[1], x_test, plot_type="bar")
shap.summary_plot(shap_values[1], x_test)
# Checking the patient's conditions

def patient_analysis(model, patient):

  explainer = shap.TreeExplainer(model)

  shap_values = explainer.shap_values(patient)

  shap.initjs()

  return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)
# Real time prediction for patients

patients = x_test.iloc[1,:].astype(float)

patient_analysis(model, patients)
patients = x_test.iloc[:, 2].astype(float)

patient_analysis(model, patients)
patients = x_test.iloc[:,3].astype(float)

patient_analysis(model, patients)
# dependence plot

shap.dependence_plot('num_major_vessels', shap_values[1], x_test, interaction_index = "st_depression")
shap_values = explainer.shap_values(x_train.iloc[:50])

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], x_test.iloc[:50])
