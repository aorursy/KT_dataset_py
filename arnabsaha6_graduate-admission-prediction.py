# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
admission_df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")
admission_df.head()
admission_df.drop(columns = "Serial No.", inplace = True)

admission_df.head()
admission_df.isnull().sum()
admission_df.dtypes
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style("darkgrid")
plt1 = sns.distplot(admission_df['GRE Score'])

plt.title("Distribution of GRE score")

plt.show()
plt1 = sns.distplot(admission_df['TOEFL Score'])

plt.title("Distribution of TOEFL score")

plt.show()
plt1 = sns.distplot(admission_df['University Rating'])

plt.title("Distribution of University Rating")

plt.show()
plt1 = sns.distplot(admission_df['SOP'])

plt.title("Distribution of SOP")

plt.show()
plt1 = sns.distplot(admission_df['LOR '])

plt.title("Distribution of LOR")

plt.show()
plt1 = sns.distplot(admission_df['CGPA'])

plt.title("Distribution of CGPA")

plt.show()
import plotly

import plotly.graph_objs as go



avg_gre_score = admission_df['GRE Score'].mean()

data = [go.Histogram(

        x = admission_df['GRE Score']

)]



# Vertical dashed line to indicate the average app rating

layout = {'shapes': [{

              'type' :'line',

              'x0': avg_gre_score,

              'y0': 0,

              'x1': avg_gre_score,

              'y1': 100,

              'line': { 'dash': 'dashdot'}

          }]

          }



print(avg_gre_score)

plotly.offline.iplot({'data': data, 'layout': layout})
avg_toefl_score = admission_df['TOEFL Score'].mean()

data = [go.Histogram(

        x = admission_df['TOEFL Score']

)]



# Vertical dashed line to indicate the average app rating

layout = {'shapes': [{

              'type' :'line',

              'x0': avg_toefl_score,

              'y0': 0,

              'x1': avg_toefl_score,

              'y1': 100,

              'line': { 'dash': 'dashdot'}

          }]

          }

print(avg_toefl_score)

plotly.offline.iplot({'data': data, 'layout': layout})
avg_sop_score = admission_df['SOP'].mean()

data = [go.Histogram(

        x = admission_df['SOP']

)]



# Vertical dashed line to indicate the average app rating

layout = {'shapes': [{

              'type' :'line',

              'x0': avg_sop_score,

              'y0': 0,

              'x1': avg_sop_score,

              'y1': 100,

              'line': { 'dash': 'dashdot'}

          }]

          }

print(avg_sop_score)

plotly.offline.iplot({'data': data, 'layout': layout})
avg_lor_score = admission_df['LOR '].mean()

data = [go.Histogram(

        x = admission_df['LOR ']

)]



# Vertical dashed line to indicate the average app rating

layout = {'shapes': [{

              'type' :'line',

              'x0': avg_lor_score,

              'y0': 0,

              'x1': avg_lor_score,

              'y1': 100,

              'line': { 'dash': 'dashdot'}

          }]

          }

print(avg_lor_score)

plotly.offline.iplot({'data': data, 'layout': layout})
avg_cgpa_score = admission_df['CGPA'].mean()

data = [go.Histogram(

        x = admission_df['CGPA']

)]



# Vertical dashed line to indicate the average app rating

layout = {'shapes': [{

              'type' :'line',

              'x0': avg_cgpa_score,

              'y0': 0,

              'x1': avg_cgpa_score,

              'y1': 100,

              'line': { 'dash': 'dashdot'}

          }]

          }

print(avg_cgpa_score)

plotly.offline.iplot({'data': data, 'layout': layout})
plt2 = sns.jointplot(x = admission_df['GRE Score'], y = admission_df['TOEFL Score'], kind = 'hex')

plt3 = sns.jointplot(x = admission_df['GRE Score'], y = admission_df['SOP'], kind = 'hex')

plt4 = sns.jointplot(x = admission_df['GRE Score'], y = admission_df['CGPA'], kind = 'hex')

plt5 = sns.jointplot(x = admission_df['GRE Score'], y = admission_df['University Rating'], kind = 'hex')

plt5 = sns.jointplot(x = admission_df['GRE Score'], y = admission_df['LOR '], kind = 'hex')
fig = sns.lmplot(x="TOEFL Score", y="CGPA", data=admission_df, hue="Research")

plt.title("TOEFL Score vs CGPA")

plt.show()
fig = sns.lmplot(x="TOEFL Score", y="SOP", data=admission_df, hue="Research")

plt.title("TOEFL Score vs SOP")

plt.show()
fig = sns.lmplot(x="TOEFL Score", y="LOR ", data=admission_df, hue="Research")

plt.title("TOEFL Score vs LOR")

plt.show()
fig = sns.lmplot(x="CGPA", y="SOP", data=admission_df, hue="Research")

plt.title("CGPA vs SOP")

plt.show()
fig = sns.lmplot(x="CGPA", y="LOR ", data=admission_df, hue="Research")

plt.title("CGPA vs LOR")

plt.show()

corr_1 = admission_df.corr()

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(corr_1, linewidths=.5, annot=True, fmt=".2f")

plt.show()
from sklearn.model_selection import train_test_split



X = admission_df.drop(['Chance of Admit '], axis=1)

y = admission_df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30)
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error



model1 = DecisionTreeRegressor()

model1.fit(X_train, y_train)

predictions1 = model1.predict(X_test)

print("Decision Tree: ",np.sqrt(mean_squared_error(y_test, predictions1)))
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor



models = {"Linear Regression": LinearRegression(), 'Random Forest':RandomForestRegressor(), 

          'KNN':KNeighborsRegressor(),'SVM':SVR(), 'GradientBoost':GradientBoostingRegressor()}



for model_name, model in models.items():

    predictor_model = model

    predictor_model.fit(X_train, y_train)

    predictions = predictor_model.predict(X_test)

    print(str(model_name) + ": "+ str(np.sqrt(mean_squared_error(y_test, predictions))))