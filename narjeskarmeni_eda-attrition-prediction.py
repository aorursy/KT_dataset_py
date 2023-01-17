# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.cluster import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
from sklearn.feature_selection import *

from sklearn.decomposition import *

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

## -------------------- ##

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
df_attrition = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df_attrition.head()
# Looking for NaN

display(df_attrition.isnull().any())
len(set(df_attrition['TotalWorkingYears']))
variable_x = 'MonthlyIncome'

df_yes = df_attrition.loc[df_attrition["Attrition"] == 'Yes'][variable_x].values.tolist()

df_no = df_attrition.loc[df_attrition["Attrition"] == 'No'][variable_x].values.tolist()

df_age = df_attrition[variable_x].values.tolist()



#First plot

trace0 = go.Histogram(

    x=df_no,

    histnorm='probability',

    name="No Attrition"

)

#Second plot

trace1 = go.Histogram(

    x=df_yes,

    histnorm='probability',

    name="Attrition"

)

#Third plot

trace2 = go.Histogram(

    x=df_age,

    histnorm='probability',

    name="Overall Age"

)



#Creating the grid

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('No','Yes', 'General Distribuition'))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)



fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
features_list = list(df_attrition)

print(features_list)
df_attrition.dtypes
features_categorical = [name for name in features_list if df_attrition[name].dtype == np.dtype('O')]

features_numerical = [name for name in features_list if df_attrition[name].dtype == np.dtype('int')]

features_categorical
df_attr_replaced = df_attrition.copy()

for fe in features_categorical:

    list_modes = list(set(df_attrition[fe]))

    df_attr_replaced[fe].replace(list_modes , list(range(len(list_modes))),inplace = True)
df_attr_replaced.head()
df_attr_replaced_num = df_attr_replaced[features_numerical]
#Using Pearson Correlation of numerical data

plt.figure(figsize=(20,20))

cor = df_attr_replaced_num.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Adding constant column of ones, mandatory for sm.OLS model

X_1 = sm.add_constant(df_attr_replaced)

#Fitting sm.OLS model

target_value = df_attr_replaced['Attrition'].copy()

del df_attr_replaced['Attrition']

model = sm.OLS(target_value,X_1).fit()

model.pvalues
#Backward Elimination

cols = list(df_attr_replaced_num.columns)

pmax = 1

while (len(cols)>0):

    p= []

    X_1 = df_attr_replaced_num[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(target_value,X_1).fit()

    p = pd.Series(model.pvalues.values,index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)
len(selected_features_BE)
X = df_attr_replaced_num

y = target_value

#no of features

nof_list=np.arange(1,26)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    model = LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))