import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Loading The Libraries



#For visualizations



import matplotlib.pyplot as plt

import seaborn as sns

!pip install --upgrade pip -q

!pip install dexplot -q

!pip install dabl -q

!pip install sweetviz -q

!pip install autoviz -q

!pip install pycaret -q

!pip install seaborn -q

# for visualizations

plt.style.use('fivethirtyeight')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler



from pandas_profiling import ProfileReport

import dexplot as dxp

import dabl
dt = pd.read_csv("/kaggle/input/heart-disease-dataset-by-openml/heart-statlog_csv.csv")
display(dt.head())
dt['class'][dt['class'] == 'present']=1

dt['class'][dt['class'] == 'absent']=0
display(dt.head())
dt[['class']] = dt[['class']].apply(pd.to_numeric) 
dt.info()
dt.describe()
df_plot = dt.copy()



dj = {0 : 'normal', 1 :'fixed defect', 2 : 'reversable defect'}

dk = {0:'Less chance of Heart Attack',1:'High Chance of Heart Attack'}





df_plot['thal'].replace(dj, inplace=True)

df_plot['class'].replace(dk, inplace=True)

print(df_plot)
profile = ProfileReport(df_plot, title='Pandas Profiling Report')
profile.to_notebook_iframe()
clean_data = dabl.clean(dt, verbose=1)

clean_data.describe()
types = dabl.detect_types(clean_data)

types
dabl.plot(clean_data,'class')
dabl_classifer = dabl.SimpleClassifier(random_state=0)
X = clean_data.drop('class', axis=1)

y = clean_data.maximum_heart_rate_achieved

sc = StandardScaler()

X = sc.fit_transform(X)
dabl_classifer.fit(X, y)
from autoviz.AutoViz_Class import AutoViz_Class



AV = AutoViz_Class()
sep = ','

target = 'class'

filename = '../input/heart-disease-dataset-by-openml/heart-statlog_csv.csv'
dft = AV.AutoViz(filename, sep=sep, depVar=target, dfte=df_plot, header=0, verbose=2,lowess=False,chart_format='svg',max_rows_analyzed=1500,max_cols_analyzed=30)
from pycaret.classification import *

classifier = setup(dt, target = 'class', session_id=42, experiment_name='heart',normalize=True,silent=True)

best = compare_models()

# return best model based on AUC

best = compare_models(sort = 'AUC') 



# return top 3 models based on 'Accuracy'

top3 = compare_models(n_select = 2)
lr = create_model('lr', fold = 10)
rf = create_model('rf', fold = 5)
et = create_model('et',fold = 5)
models(type='ensemble').index.tolist()
tuned_lr = tune_model(lr)
plot_model(lr)
plot_model(rf)
plot_model(et)
plot_model(lr, plot = 'confusion_matrix')
plot_model(rf, plot = 'confusion_matrix')
plot_model(et, plot = 'confusion_matrix')
plot_model(lr, plot = 'feature')
plot_model(rf, plot = 'feature')
plot_model(et, plot = 'feature')
plot_model(lr, plot = 'class_report')
plot_model(rf, plot = 'class_report')
plot_model(et, plot = 'class_report')