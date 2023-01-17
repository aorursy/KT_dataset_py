!pip install jovian opendatasets --upgrade --quiet
# Change this
dataset_url = "https://www.kaggle.com/andrewmvd/heart-failure-clinical-data"
import opendatasets as od
od.download(dataset_url)
# Change this
data_dir = './heart-failure-clinical-data'
import os
os.listdir(data_dir)
project_name = "health-failure-prediction" # change this (use lowercase letters and hyphens only)
!pip install jovian --upgrade -q
import jovian
jovian.commit(project=project_name)
#Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
#from plotly.offline import plot,iplot,init_notebook_mode
import plotly.graph_objects as go
import plotly.express as px
import warnings
import lightgbm as lgb
#from IPython.display import HTML 

#init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
heart_failure_raw_df = pd.read_csv("heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
heart_failure_raw_df
heart_failure_raw_df.columns
selected_columns = heart_failure_raw_df.columns
len(selected_columns)
heart_failure_df = heart_failure_raw_df[selected_columns].copy()
heart_failure_df.shape
heart_failure_df.info()
heart_failure_df.isnull().sum()
heart_failure_df.describe()
import jovian
jovian.commit(project=project_name)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
target = heart_failure_df["DEATH_EVENT"]
sns.countplot(heart_failure_df.DEATH_EVENT)
# Getting the count and percentage column by using target column

counts = heart_failure_df.DEATH_EVENT.value_counts()
percentage = heart_failure_df.DEATH_EVENT.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
pd.DataFrame({'Counts':counts,'Percentage': percentage})
from sklearn.model_selection import train_test_split

# Removing the target column from real dataset
y = heart_failure_df.loc[:,"DEATH_EVENT"]
y_dropped = heart_failure_df.drop("DEATH_EVENT",axis = 1)
X = y_dropped.loc[:,:]

# Data Splitting to determine Feature Importance 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.30, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import lightgbm as lgb

# Indicating all specific params
fit_params={"early_stopping_rounds":10, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'feature_name': 'auto', # that's actually the default
            'categorical_feature': 'auto' # that's actually the default
           }

# n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 1000 define only the absolute maximum
clf = lgb.LGBMClassifier(num_leaves= 15, max_depth=-1, 
                         random_state=314, 
                         silent=True, 
                         metric='None', 
                         n_jobs=4, 
                         n_estimators=1000,
                         colsample_bytree=0.9,
                         subsample=0.9,
                         learning_rate=0.1)

# force larger number of max trees and smaller learning rate
clf.fit(X_train, y_train, **fit_params)

feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))
class LGBMClassifier_GainFE(lgb.LGBMClassifier):
    @property 
    # We used @property decorator to give "special" functionality such as deleters to certain methods
    def feature_importances_(self):
        if self._n_features is None:
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self.booster_.feature_importance(importance_type='gain')

clf2 = LGBMClassifier_GainFE(num_leaves= 15, max_depth=-1, 
                         random_state=314, 
                         silent=True, 
                         metric='None', 
                         n_jobs=4, 
                         n_estimators=1000,
                         colsample_bytree=0.9,
                         subsample=0.9,
                         learning_rate=0.1) 

clf2.fit(X_train, y_train, **fit_params)

feat_imp = pd.Series(clf2.feature_importances_, index=X.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))
%matplotlib inline
fig = px.histogram(heart_failure_df, x="time", color="DEATH_EVENT", marginal="violin", hover_data=heart_failure_df.columns, 
                   title ="Distribution of TIME Vs DEATH_EVENT", 
                   labels={"time": "TIME"})
fig.show()
fig = px.histogram(heart_failure_df, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=heart_failure_df.columns, 
                   title ="Distribution of SERUM CREATININE Vs DEATH_EVENT", 
                   labels={"serum_creatinine": "SERUM CREATININE"})
fig.show()
fig = px.histogram(heart_failure_df, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=heart_failure_df.columns, 
                   title ="Distribution of EJECTION FRACTION Vs DEATH_EVENT", 
                   labels={"ejection_fraction": "EJECTION FRACTION"})
fig.show()
fig = px.histogram(heart_failure_df, x="age", color="DEATH_EVENT", marginal="violin", hover_data=heart_failure_df.columns, 
                   title ="Distribution of AGE Vs DEATH_EVENT", 
                   labels={"age": "AGE"})
fig.show()
fig = px.histogram(heart_failure_df, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=heart_failure_df.columns, 
                   title ="Distribution of CREATININE PHOSPHOKINASE Vs DEATH_EVENT", 
                   labels={"creatinine_phosphokinase": "CREATININE PHOSPHOKINASE"})
fig.show()
!pip install --upgrade plotly
sun = heart_failure_df.groupby(["sex","diabetes","smoking","anaemia","high_blood_pressure","DEATH_EVENT"])["age"].count().reset_index()
sun.columns= ["sex","diabetes","smoking","anaemia","high_blood_pressure","DEATH_EVENT", "count"]
sun.loc[sun["sex"]== 0 , "sex"] = "female"
sun.loc[sun["sex"]== 1, "sex"] = "male"

sun.loc[sun["diabetes"]== 0 , "diabetes"] = "no diabetes"
sun.loc[sun["diabetes"]== 1, "diabetes"] = "diabetes"

sun.loc[sun['DEATH_EVENT'] == 0,'DEATH_EVENT'] = "LIVE"
sun.loc[sun['DEATH_EVENT'] == 1, 'DEATH_EVENT'] = 'DEATH'

fig = px.sunburst(sun, 
                  path=["sex","diabetes","DEATH_EVENT"],
                  values="count",
                  title="Gender & Diabetes Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
sun.loc[sun["smoking"]== 0 , "smoking"] = "non smoking"
sun.loc[sun["smoking"]== 1, "smoking"] = "smoking"

fig = px.sunburst(sun, 
                  path=["sex","smoking","DEATH_EVENT"],
                  values="count",
                  title="Gender & Smoking Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
sun.loc[sun["anaemia"]== 0 , "anaemia"] = "no anaemia"
sun.loc[sun["anaemia"]== 1, "anaemia"] = "anaemia"

fig = px.sunburst(sun, 
                  path=["sex","anaemia","DEATH_EVENT"],
                  values="count",
                  title="Gender & Anaemia  Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
sun.loc[sun["high_blood_pressure"]== 0 , "high_blood_pressure"] = "no high_blood_pressure"
sun.loc[sun["high_blood_pressure"]== 1, "high_blood_pressure"] = "high_blood_pressure"

fig = px.sunburst(sun, 
                  path=["sex","high_blood_pressure","DEATH_EVENT"],
                  values="count",
                  title="Gender & High Blood Pressure Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
import jovian
jovian.commit(project=project_name)
fig = px.violin(heart_failure_df, x="age",y="sex", color="DEATH_EVENT",
                box=True,points="all",hover_data=heart_failure_df.columns,
                title="Analysis of Gender and Age on Survival People",
                orientation="h")

fig.show()
fig = px.violin(heart_failure_df, x="age",y="smoking", color="DEATH_EVENT",
                box=True,points="all",hover_data=heart_failure_df.columns,
                title="Analysis of Age and Smoking on Survival People",
                orientation="h")
fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["smoking"]==1],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["smoking"]==1],
                        legendgroup=1, scalegroup=1,name="Death",
                        side="negative",line_color="blue"))

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["smoking"]==0],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["smoking"]==0],
                        legendgroup=0, scalegroup=0,name="Live",
                        side="positive",line_color="pink"))



fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0,violinmode="overlay",
                  title= "Analysis of Gender & Smoking on Survival Person",
                  xaxis_title="Gender",
                  yaxis_title="Smoking",
                  legend_title="Death Event")
fig.show()
fig = px.violin(heart_failure_df, x="age",y="diabetes", color="DEATH_EVENT",
                box=True,points="all",hover_data=heart_failure_df.columns,
                title="Analysis of Age and Diabetes on Survival People",
                orientation="h")
fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["diabetes"]==1],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["diabetes"]==1],
                        legendgroup=1, scalegroup=1,name="Death",
                        side="negative",line_color="blue"))

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["diabetes"]==0],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["diabetes"]==0],
                        legendgroup=0, scalegroup=0,name="Live",
                        side="positive",line_color="pink"))



fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0,violinmode="overlay",
                  title= "Analysis of Gender & Diabetes on Survival Person",
                  xaxis_title="Gender",
                  yaxis_title="Diabetes",
                  legend_title="Death Event")
fig.show()
fig = px.violin(heart_failure_df, x="age",y="anaemia", color="DEATH_EVENT",
                box=True,points="all",hover_data=heart_failure_df.columns,
                title="Analysis of Age and Anaemia on Survival People",
                orientation="h")
fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["anaemia"]==1],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["anaemia"]==1],
                        legendgroup=1, scalegroup=1,name="Death",
                        side="negative",line_color="blue"))

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["anaemia"]==0],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["anaemia"]==0],
                        legendgroup=0, scalegroup=0,name="Live",
                        side="positive",line_color="pink"))



fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0,violinmode="overlay",
                  title= "Analysis of Gender & Anaemia on Survival Person",
                  xaxis_title="Gender",
                  yaxis_title="Anaemia",
                  legend_title="Death Event")
fig.show()
fig = px.violin(heart_failure_df, x="age",y="high_blood_pressure", color="DEATH_EVENT",
                box=True,points="all",hover_data=heart_failure_df.columns,
                title="Analysis of Age and High Blood Pressure on Survival People",
                orientation="h")
fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["high_blood_pressure"]==1],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["high_blood_pressure"]==1],
                        legendgroup=1, scalegroup=1,name="Death",
                        side="negative",line_color="blue"))

fig.add_trace(go.Violin(x=heart_failure_df["sex"][heart_failure_df["high_blood_pressure"]==0],
                        y=heart_failure_df["DEATH_EVENT"][heart_failure_df["high_blood_pressure"]==0],
                        legendgroup=0, scalegroup=0,name="Live",
                        side="positive",line_color="pink"))



fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0,violinmode="overlay",
                  title= "Analysis of Gender & High Blood Pressur on Survival Person",
                  xaxis_title="Gender",
                  yaxis_title="High Blood Pressure",
                  legend_title="Death Event")
fig.show()
sns.pairplot(heart_failure_df,hue="DEATH_EVENT",vars =["time","serum_creatinine","ejection_fraction","age","creatinine_phosphokinase","platelets","serum_sodium"])
import plotly.graph_objects as go

anaemia = heart_failure_df[heart_failure_df["anaemia"]==1]
no_anaemia = heart_failure_df[heart_failure_df["anaemia"]==0]

fig = go.Figure(go.Pie(
    title="Why does anaemia affect death events so little?",
    values=[len(anaemia[heart_failure_df["DEATH_EVENT"]==1]),
           len(anaemia[heart_failure_df["DEATH_EVENT"]==0]),
           len(no_anaemia[heart_failure_df["DEATH_EVENT"]==1]),
           len(no_anaemia[heart_failure_df["DEATH_EVENT"]==0])],
    labels=["Anaemia + No Survived",
          "Anaemia + Survived",
          "No Anaemia + No Survived",
          "No Anaemia + Survived"],
    hovertemplate = "%{label}: <br>Rate: %{percent} </br> %{text}"
))

fig.show()


import plotly.graph_objects as go

high_blood_pressure = heart_failure_df[heart_failure_df["high_blood_pressure"]==1]
no_high_blood_pressure = heart_failure_df[heart_failure_df["high_blood_pressure"]==0]

fig = go.Figure(go.Pie(
    title="Why does high blood pressure affect death events so little?",
    values=[len(high_blood_pressure[heart_failure_df["DEATH_EVENT"]==1]),
           len(high_blood_pressure[heart_failure_df["DEATH_EVENT"]==0]),
           len(no_high_blood_pressure[heart_failure_df["DEATH_EVENT"]==1]),
           len(no_high_blood_pressure[heart_failure_df["DEATH_EVENT"]==0])],
    labels=["High Blood Pressure + No Survived",
          "High Blood Pressure + Survived",
          "No High Blood Pressure + No Survived",
          "No High Blood Pressure + Survived"],
    hovertemplate = "%{label}: <br>Rate: %{percent} </br> %{text}"
))

fig.show()
import plotly.graph_objects as go

diabetes = heart_failure_df[heart_failure_df["diabetes"]==1]
no_diabetes = heart_failure_df[heart_failure_df["diabetes"]==0]

fig = go.Figure(go.Pie(
    title="Why does diabetes affect death events so little?",
    values=[len(diabetes[heart_failure_df["DEATH_EVENT"]==1]),
           len(diabetes[heart_failure_df["DEATH_EVENT"]==0]),
           len(no_diabetes[heart_failure_df["DEATH_EVENT"]==1]),
           len(no_diabetes[heart_failure_df["DEATH_EVENT"]==0])],
    labels=["Diabetes + No Survived",
          "Diabetes + Survived",
          "No Diabetes + No Survived",
          "No Diabetes + Survived"],
    hovertemplate = "%{label}: <br>Rate: %{percent} </br> %{text}"
))

fig.show()
import plotly.graph_objects as go

male = heart_failure_df[heart_failure_df["sex"]==1]
female = heart_failure_df[heart_failure_df["sex"]==0]

fig = go.Figure(go.Pie(
    title="Why does gender affect death events so little?",
    values=[len(male[heart_failure_df["DEATH_EVENT"]==1]),
           len(male[heart_failure_df["DEATH_EVENT"]==0]),
           len(female[heart_failure_df["DEATH_EVENT"]==1]),
           len(female[heart_failure_df["DEATH_EVENT"]==0])],
    labels=["Male + No Survived",
          "Male + Survived",
          "Female + No Survived",
          "Female + Survived"],
    hovertemplate = "%{label}: <br>Rate: %{percent} </br> %{text}"
))

fig.show()
import plotly.graph_objects as go

smoking = heart_failure_df[heart_failure_df["smoking"]==1]
no_smoking = heart_failure_df[heart_failure_df["smoking"]==0]

fig = go.Figure(go.Pie(
    title="Why does smoking affect death events so little?", 
    values=[len(smoking[heart_failure_df["DEATH_EVENT"]==1]),
           len(smoking[heart_failure_df["DEATH_EVENT"]==0]),
           len(no_smoking[heart_failure_df["DEATH_EVENT"]==1]),
           len(no_smoking[heart_failure_df["DEATH_EVENT"]==0])],
    labels=["Smoking + No Survived",
          "Smoking + Survived",
          "No Smoking + No Survived",
          "No Smoking + Survived"],
    hovertemplate = "%{label}: <br>Rate: %{percent} </br> %{text}"
))

fig.show()
import jovian
jovian.commit(project=project_name)
import jovian
jovian.commit(project=project_name)
import jovian
jovian.commit(project=project_name)