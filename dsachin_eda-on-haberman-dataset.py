#Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='Set1')
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

%matplotlib inline
col_head = ['Age', 'Op_year','Axil_nodes','Survived_more_than_5years']
df_haberman = pd.read_csv('../input/haberman.csv', header= None)
# Set Column headers
df_haberman.columns = col_head
df_haberman.head(5)
df_haberman.info()
df_haberman.Survived_more_than_5years = df_haberman.Survived_more_than_5years.map({1:'Yes',2:'No'}) 
df_haberman.info()
df_haberman.describe()
np.percentile(df_haberman['Axil_nodes'],[25,50,75,100])
np.percentile(df_haberman['Op_year'],[25,50,75,100])
df_haberman.groupby('Survived_more_than_5years').mean()
df_haberman.groupby('Survived_more_than_5years').median()
df_haberman.Survived_more_than_5years.value_counts().plot('bar', title = 'Histogram for Class Variable')
plt.plot()
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16,4)
fig.suptitle('Histograms for Features')
ax[0].hist(df_haberman.Age)
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Count')
ax[1].hist(df_haberman.Op_year)
ax[1].set_xlabel('Year of Operation')
ax[1].set_ylabel('Count')
ax[2].hist(df_haberman.Axil_nodes)
ax[2].set_xlabel('No. of Nodes')
ax[2].set_ylabel('Count')
'''
->Distribution functions helps to visualize spread of datapoints.
->Here, histogram shows count of variable in a particular bin, higher the hight of bar, more are the values in that bin.
->PDF shows the probablity of variable for a particular value, for ex, in the 1st plot below, the probality of person with
  age 50 and who has not survied more than 5 years nearly .027
-> As the total probablity for outcomes of any event has to be 1, the area under the curve of PDF with KDE is 1. 
'''
for idx, col in enumerate(list(df_haberman.columns[:-1])):
    fig = sns.FacetGrid(df_haberman, hue='Survived_more_than_5years', size=5)
    fig.map(sns.distplot, col).add_legend()
'''
-->PDF shows for a particular value, what is the percentage of readings of that variable in the whole dataset. For ex-
for the case of not survived more than 5 years, nearly 10% of patients have got age 50
-->CDF shows the percentile or percentage of datapoints less than or eqaul to a give value for a variable. For ex-
for the case of not survived more than 5 years, nearly 40% of patients have got age 50 or below

'''

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16,4)
fig.suptitle('PDF & CDF charts - Not Survived more than five years', fontsize=12)
for idx, col in enumerate(list(df_haberman.columns[:-1])):
    counts, bin_edges = np.histogram(df_haberman[df_haberman['Survived_more_than_5years']== 'No'][col],bins= 10, density=True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    ax[idx].plot(bin_edges[1:], pdf)
    ax[idx].plot(bin_edges[1:], cdf)
    ax[idx].set_xlabel(col)
    ax[idx].legend(['PDF','CDF'])
    
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16,4)
fig.suptitle('PDF & CDF charts - Survived more than five years', fontsize=12)
for idx, col in enumerate(list(df_haberman.columns[:-1])):
    counts, bin_edges = np.histogram(df_haberman[df_haberman['Survived_more_than_5years']== 'Yes'][col],bins= 10, density=True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    ax[idx].plot(bin_edges[1:], pdf)
    ax[idx].plot(bin_edges[1:], cdf)
    ax[idx].set_xlabel(col)
    ax[idx].legend(['PDF','CDF'])
'''
Box plots helps in representation of - 
Q1 = 25% percentile 
Q2 = 50% percentile which is median value
Q3 = 75% percentil
IQR = Q3 - Q1
lesser of (Q1-1.5*IQR or min value)
lesser of (Q3+1.5*IQR or max values).
'''
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Box Plots for Features')
for idx, cols in enumerate(list(df_haberman.columns[:-1])):
    sns.boxplot('Survived_more_than_5years', cols, data= df_haberman, ax=ax[idx])
'''
Vilon plot is combination of PDF and Box plot
'''
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Vilon Plots for Features')
for idx, cols in enumerate(list(df_haberman.columns[:-1])):
    sns.violinplot('Survived_more_than_5years', cols, data= df_haberman, ax=ax[idx])
'''
Pair Plot shows the scatter plot between pair of all combinations between columns of a dataframe
'''
sns.pairplot(df_haberman,hue = 'Survived_more_than_5years',vars=['Age', 'Op_year','Axil_nodes'], size= 3 ).fig.suptitle('Pairplot of Features')
plt.figure(figsize=(16,8))

for idx1, col1 in enumerate(list(df_haberman.columns[:-1])):
    for idx2, col2 in enumerate(list(df_haberman.columns[idx1:-1])):
        if col1 != col2:
            sns.jointplot(col1, col2, df_haberman[df_haberman['Survived_more_than_5years']== 'Yes'], kind = 'kde')
        else:
            pass   
                         
                            
def colx(x):
    if x == 'Yes':
        return 'green'
    elif x == 'No':
        return 'orchid'
df_haberman['color'] = df_haberman.Survived_more_than_5years.apply(colx)
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
trace1 = go.Scatter3d(
    x=df_haberman['Age'],
    y=df_haberman['Op_year'],
    z=df_haberman['Axil_nodes'],
    mode='markers',
    marker=dict(
        symbol='circle',
        color=df_haberman['color'],
        colorscale='Viridis',
        opacity=0.8,
    ))
data = [trace1]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='3d-scatter-colorscale')
