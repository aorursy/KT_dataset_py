# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.ticker as mtick # For specifying the axes tick format 
import matplotlib.pyplot as plt
from PIL import  Image
%matplotlib inline
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
telco = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# telco.head(5)
telco.describe()
# dropping missing values from the data
telco.shape[0] - telco.dropna().shape[0]
telco.isnull().values.ravel().sum()
# replacing 'No internet service' and 'No phone service' to 'No'
telco.replace(['No internet service','No phone service'], 'No')
# replacing blank values with 0
telco.replace([' '],0)
# replacing yes and no with numeric data
telco['SeniorCitizen'].replace({1:'Yes',0:'No'})
# replacing yes and no of churn data
# telco['Churn'].replace({'Yes':1,'No':0}).astype(int)
#Convertin the predictor variable in a binary numeric variable
telco['Churn'].replace(to_replace='Yes', value=1, inplace=True)
telco['Churn'].replace(to_replace='No',  value=0, inplace=True)
# removing cust_id from the data
df = telco.iloc[:,1:]
list(telco)
churn = sns.catplot(y="Churn", kind="count", data=telco, height=2.6, aspect=2.5, orient='a')
count = telco['Churn'].value_counts().to_dict()
count
# categorizing tenure data
def tenure_lab(telco) :
    
    if telco["tenure"] <= 12 :
        return "Tenure_0-12"
    elif (telco["tenure"] > 12) & (telco["tenure"] <= 24 ):
        return "Tenure_12-24"
    elif (telco["tenure"] > 24) & (telco["tenure"] <= 48) :
        return "Tenure_24-48"
    elif (telco["tenure"] > 48) & (telco["tenure"] <= 60) :
        return "Tenure_48-60"
    elif telco["tenure"] > 60 :
        return "Tenure_gt_60"
telco["tenure_cat"] = telco.apply(lambda telco:tenure_lab(telco),
                                      axis = 1)
df.dtypes
def barplot_percentages(feature, orient='v', axis_name = "percentage of customers"):
    ratios = pd.DataFrame()
    g = df.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
barplot_percentages("SeniorCitizen")
def barplot_percentages(feature, orient='v', axis_name = "percentage of customers"):
    ratios = pd.DataFrame()
    g = df.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
barplot_percentages("Dependents")
def barplot_percentages(feature, orient='v', axis_name = "percentage of customers"):
    ratios = pd.DataFrame()
    g = df.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
barplot_percentages("gender")
def barplot_percentages(feature, orient='v', axis_name = "percentage of customers"):
    ratios = pd.DataFrame()
    g = df.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
barplot_percentages("Partner")
# creating dummy variables
df_dummies = pd.get_dummies(df)
df_dummies.head()
# Converting churn from string to numeric
df['Churn'].astype(str).astype(int)
# pd.to_numeric(df['Churn'])
# Converting totalCharges column from string to numeric
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df.info()
sns.pairplot(df,vars = ['tenure','MonthlyCharges','TotalCharges'], hue="Churn")
#Separating churn and non churn customers
churn     = telco[telco["Churn"] == 1]
not_churn = telco[telco["Churn"] == 0]

tg_ch  =  churn["tenure_cat"].value_counts().reset_index()
tg_ch.columns  = ["tenure_cat","count"]
tg_nch =  not_churn["tenure_cat"].value_counts().reset_index()
tg_nch.columns = ["tenure_cat","count"]

#bar - churn
bill1 = go.Bar(x = tg_ch["tenure_cat"]  , y = tg_ch["count"],
                name = "Churn Customers",
                marker = dict(line = dict(width = .5,color = "black")),
                opacity = .9)

#bar - not churn
bill2 = go.Bar(x = tg_nch["tenure_cat"] , y = tg_nch["count"],
                name = "Non Churn Customers",
                marker = dict(line = dict(width = .5,color = "black")),
                opacity = .9)

layout = go.Layout(dict(title = "Customer churn in tenure categories",
                        plot_bgcolor  = "rgb(260,260,260)",
                        paper_bgcolor = "rgb(180,180,180)",
                        xaxis = dict(gridcolor = 'rgb(210, 210, 210)',
                                     title = "tenure group",
 zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(210, 210, 210)',
                                     title = "count",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                       )
                  )
data = [bill1,bill2]
fig  = go.Figure(data=data,layout=layout)
py.iplot(fig)

solo = sns.FacetGrid(df, row='SeniorCitizen', col="gender", hue="Churn", height=3.5)
solo.map(plt.scatter, "tenure", "MonthlyCharges", alpha=0.5)
plt.gray()
solo.add_legend();
plt.figure(figsize=(12, 10))
telco.drop(['customerID'],
        axis=1, inplace=True)
corr = telco.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                 linewidths=.2, cmap="PuBuGn")