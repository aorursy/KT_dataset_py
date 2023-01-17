# basic python libraries
import pandas as pd
import numpy as np
import matplotlib
import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from matplotlib import cm
from collections import OrderedDict
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (15, 5)
from scipy.stats import norm, shapiro
from colorama import Fore, Back, Style
from mlxtend.plotting import plot_confusion_matrix
from plotly.offline import plot, iplot, init_notebook_mode
from statsmodels.formula.api import ols
from scipy import stats


# sklearn libraries
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import GridSearchCV

# feature selection library
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel

# model building libraries
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier,StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier

# warning library
import warnings
warnings.filterwarnings("ignore")

# setting basic options
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', None)
%matplotlib inline
# importing dataset
data = pd.read_csv(r'../input/heart-disease/heart.csv', error_bad_lines = False)
print("Shape of the data is {}.".format(data.shape))
data.head().style.background_gradient(cmap='viridis')
# getting the death event distribution
labels = data.target.value_counts(normalize = True)*100 
fig = px.pie(labels, values= 'target', names = ['Disease', 'No Disease'], title='Target Distribution across whole dataset')
fig.show()
# unique values of columns
for col in data.columns.tolist():
    print(col,": ",display(data[col].describe().to_frame().style.background_gradient(cmap='RdPu')), "\n")
# age distribution across whole dataset

hist_data =[data["age"].values]
group_labels = ['age'] 

fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text='Age Distribution plot')

fig.show()
# lets see the distribution of age with respect to the target label
fig = px.box(data, x='target', y='age', points="all")
fig.update_layout(title_text="Target wise Age Spread - Unhealthy = 1 Healthy =0")
fig.show()
# lets see those people who are having age of 40-60 and get the gender distribution
temp_df = data[(data['age']>=40) & (data['age']<=60)]
male = temp_df[temp_df["sex"]==1]
female = temp_df[temp_df["sex"]==0]

male_survi = male[temp_df["target"]==0]
male_not = male[temp_df["target"]==1]
female_survi = female[temp_df["target"]==0]
female_not = female[temp_df["target"]==1]

labels = ['Male - Healthy','Male - Not Healthy', "Female -  Healthy", "Female - Not Healthy"]
values = [len(male[temp_df["target"]==0]),len(male[temp_df["target"]==1]),
         len(female[temp_df["target"]==0]),len(female[temp_df["target"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Age (40-60)yrs - Analysis on Healthy - Sex")
fig.show()

# Analysis on Healthy Ratio between Male and Female
labels = ['Male - Healthy Ratio',"Female -  Healthy Ratio"]
values = [(len(male[temp_df["target"]==0])/(len(male[temp_df["target"]==0]) + len(male[temp_df["target"]==1]))),
          (len(female[temp_df["target"]==0])/(len(female[temp_df["target"]==0]) + len(female[temp_df["target"]==1])))]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Age (40-60)yrs - Analysis on Healthy Ratio between Male and Female")
fig.show()
fig = px.violin(data, y="age", x="exang", box=True, color="target", points="all", hover_data=data.columns)
fig.update_layout(title_text="Analysis in Age and Excercise induced Angina on Healthy Status")
fig.show()
# count of different values present in Chest pain column
f,ax = plt.subplots(1,1,figsize=(15,5))
sns.set(font_scale=1)
sns.countplot('cp',data=data, palette='Set2')
# Analysis on Chest Pain where the patients are healthy.
labels = ['typical', 'asymptotic', 'nonanginal', 'nontypical']
values = [len(data[(data['cp']==0) & (data['target'] == 1)]), 
          len(data[(data['cp']==1) & (data['target'] == 1)]),
          len(data[(data['cp']==2) & (data['target'] == 1)]),
          len(data[(data['cp']==3) & (data['target'] == 1)])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Healthy percentage of Chest Pain types")
fig.show()
temp_df = data.groupby(['cp', 'ca'])['target'].count().unstack().reset_index().fillna(0)
cp=['Chest Pain - Typical', 'Chest Pain - Asymptotic', 'Chest Pain - Non Anginal', 'Chest Pain - Non Typical']

fig = go.Figure(data=[
    go.Bar(name='ca - 0', x=cp, y=temp_df[0]),
    go.Bar(name='ca - 1', x=cp, y=temp_df[1]),
    go.Bar(name='ca - 2', x=cp, y=temp_df[2]),
    go.Bar(name='ca - 3', x=cp, y=temp_df[3]),
    go.Bar(name='ca - 4', x=cp, y=temp_df[4]),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()
# Resting BP distribution across whole dataset

hist_data =[data["trestbps"].values]
group_labels = ['Resting Blood Pressure'] 

fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text='Resting Blood Pressure Distribution plot')

fig.show()
# survival distribution on the basis of resting BP
s = data[data['target']==0]['trestbps']
ns = data[data['target']==1]['trestbps']
hist_data = [s,ns]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(
    title_text="Distribution in Resting BP on Survival Status")
fig.show()
# getting the distribution of different continuous columns
continuous_variables = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak'
]
sns.pairplot(data, hue="target", palette="husl", vars = continuous_variables, kind = 'scatter', markers=["o", "s"],
             corner = True, diag_kind = 'kde')
# distribution of serum cholestoral with respect to the target variable
s, ns = data[data['target']==0]['chol'], data[data['target']==1]['chol']
hist_data = [s,ns]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(
    title_text="Distribution in Serum Cholestoral on Survival Status")
fig.show()

# distribution of maximum Heart rate recieved with respect to the target variable
s, ns = data[data['target']==0]['thalach'], data[data['target']==1]['thalach']
hist_data = [s,ns]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(
    title_text="Distribution in Maximum Heartrate recieved on Survival Status")
fig.show()

# distribution of oldpeak with respect to the target variable
s, ns = data[data['target']==0]['oldpeak'], data[data['target']==1]['oldpeak']
hist_data = [s,ns]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
fig.update_layout(
    title_text="Distribution in Oldpeak on Survival Status")
fig.show()
# categorical columns
# Analysis on Fasting blood sugar > 120 mg/dL, where the patients are healthy or unhealthy.
labels = ['FBS >= 120mg/dL and Healthy', 'FBS >= 120mg/dL and Not Healthy', 'FBS < 120mg/dL and Healthy', 'FBS < 120mg/dL and Not Healthy']
values = [len(data[(data['fbs']==1) & (data['target'] == 1)]), 
          len(data[(data['fbs']==1) & (data['target'] == 0)]),
          len(data[(data['fbs']==0) & (data['target'] == 1)]),
          len(data[(data['fbs']==0) & (data['target'] == 0)])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Healthy percentage of Fasting blood sugar > 120 mg/dL")
fig.show()

# Analysis on Resting ECG, where the patients are healthy or unhealthy.
labels = ['rest ecg = 0 and Healthy', 'rest ecg = 0 and Not Healthy',
          'rest ecg = 1 and Healthy', 'rest ecg = 1 and Not Healthy',
          'rest ecg = 2 and Healthy', 'rest ecg = 2 and Not Healthy'
         ]
values = [len(data[(data['restecg']==0) & (data['target'] == 1)]), 
          len(data[(data['restecg']==0) & (data['target'] == 0)]),
          len(data[(data['restecg']==1) & (data['target'] == 1)]),
          len(data[(data['restecg']==1) & (data['target'] == 0)]),
          len(data[(data['restecg']==2) & (data['target'] == 1)]),
          len(data[(data['restecg']==2) & (data['target'] == 0)])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Healthy percentage of Resting Electrographic results")
fig.show()

# Analysis on Excercise induced angina, where the patients are healthy or unhealthy.
labels = ['Exang is present and Healthy', 'Exang is present and Not Healthy',
          'Exang is absent and Healthy', 'Exang is absent and Not Healthy'
         ]
values = [len(data[(data['exang']==1) & (data['target'] == 1)]), 
          len(data[(data['exang']==1) & (data['target'] == 0)]),
          len(data[(data['exang']==0) & (data['target'] == 1)]),
          len(data[(data['exang']==0) & (data['target'] == 0)])
         ]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Healthy percentage of Excercise induced Angina")
fig.show()

# Analysis on Slope, where the patients are healthy or unhealthy.
labels = ['slope = 0 and Healthy', 'slope = 0 and Not Healthy',
          'slope = 1 and Healthy', 'slope = 1 and Not Healthy',
          'slope = 2 and Healthy', 'slope = 2 and Not Healthy'
         ]
values = [len(data[(data['slope']==0) & (data['target'] == 1)]), 
          len(data[(data['slope']==0) & (data['target'] == 0)]),
          len(data[(data['slope']==1) & (data['target'] == 1)]),
          len(data[(data['slope']==1) & (data['target'] == 0)]),
          len(data[(data['slope']==2) & (data['target'] == 1)]),
          len(data[(data['slope']==2) & (data['target'] == 0)])
         ]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Healthy percentage of Slope")
fig.show()
# analysis of different continuous columns over chest pain
fig = px.violin(data, y="trestbps", x="cp", box=True, color="target", points="all", hover_data=data.columns)
fig.update_layout(title_text="Analysis in Resting BP and Chest Pain on Healthy Status")
fig.show()

fig = px.violin(data, y="chol", x="cp", box=True, color="target", points="all", hover_data=data.columns)
fig.update_layout(title_text="Analysis in Serum Cholestoral and Chest Pain on Healthy Status")
fig.show()

fig = px.violin(data, y="thalach", x="cp", box=True, color="target", points="all", hover_data=data.columns)
fig.update_layout(title_text="Analysis in Max heart rate and Chest Pain on Healthy Status")
fig.show()

fig = px.violin(data, y="oldpeak", x="cp", box=True, color="target", points="all", hover_data=data.columns)
fig.update_layout(title_text="Analysis in Oldpeak and Chest Pain on Healthy Status")
fig.show()
features = [
     'age',
     'sex',
     'cp',
     'trestbps',
     'chol',
     'fbs',
     'restecg',
     'thalach',
     'exang',
     'oldpeak',
     'slope',
     'ca',
     'thal'
]

target = ['target']
# splitting the dataset into train and test set
xtrain, xtest, ytrain, ytest = train_test_split(data[features],
                                                    data[target],
                                                    stratify=data[target], test_size=0.20, random_state=42)

print("Shape of train set: ",(xtrain.shape, ytrain.shape))
print("Shape of test set: ",(xtest.shape, ytest.shape))
mask = np.zeros_like(xtrain.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(xtrain.corr(), mask=mask, square=True, annot = True)
# Sequential Feature Selector Object and Configuring the Parameters -> Backward Elimination
sfs_bw = SequentialFeatureSelector(LGBMClassifier(random_state = 0, n_jobs = -1, n_estimators = 20, max_depth = 4),
          k_features = 5,
          forward = False, 
          floating = False,
          scoring = 'accuracy',
          cv = 3,
          n_jobs = -1)

# Fit the object to the Training Data.
print(sfs_bw.fit(xtrain, np.ravel(ytrain)))
# Print the Selected Features.
selected_features_BW = xtrain.columns[list(sfs_bw.k_feature_idx_)]
print(selected_features_BW)
top_features = ['sex', 'cp', 'exang', 'oldpeak', 'ca']
model = LGBMClassifier(random_state = 0, n_jobs = -1, n_estimators = 20, max_depth = 3)
model.fit(xtrain[top_features], ytrain)
pred = model.predict(xtest[top_features])
pred_proba = model.predict_proba(xtest[top_features])
print("\n\nClassification_score: \n", classification_report(ytest, pred))
print("\n\nThe ROC AUC Score is: ",roc_auc_score(ytest, pred_proba[:,1]))
cm = confusion_matrix(ytest, pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Light GBM Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()