# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/insurance/insurance.csv")
data1 = data.copy()
display(data1.head(10))
display(data1.tail(10))
data1.info()
data1.describe().T
display(data1.sex.unique())
display(data1.children.unique())
display(data1.smoker.unique())
display(data1.region.unique())
display(data1.sex.value_counts())
display(data1.children.value_counts())
display(data1.smoker.value_counts())
display(data1.region.value_counts())
data1.isna().sum()
plt.subplots(1,1)
sns.countplot(data1.sex)
plt.title("gender",color = 'blue',fontsize=15)
plt.show()

plt.subplots(1,1)
sns.countplot(data1.children)
plt.title("number of children",color = 'blue',fontsize=15)
plt.show()

plt.subplots(1,1)
sns.countplot(data1.smoker)
plt.title("smoker",color = 'blue',fontsize=15)
plt.show()

plt.subplots(1,1)
sns.countplot(data1.region)
plt.title("region",color = 'blue',fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.sex, y=data1.charges);
plt.xticks(rotation= 45)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10, 5))
sns.barplot(x=data1.children, y=data1.charges);
plt.xticks(rotation= 0)
plt.xlabel('Number of Children', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Number of Children', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.smoker, y=data1.charges);
plt.xticks(rotation= 45)
plt.xlabel('Smoker', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.region, y=data1.charges);
plt.xticks(rotation= 45)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Region', color = 'blue', fontsize=15)
plt.show()
age_cat = pd.cut(data1.age, [17,35,51,64])
age_cat
data1['age_cat'] = age_cat
sns.countplot(data1.age_cat)
plt.title("age_cat",color = 'blue',fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.age_cat, y=data1.charges);
plt.xticks(rotation= 0)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Age Categories', color = 'blue', fontsize=15)
plt.show()
bmi_cat = pd.cut(data1.bmi, [15,18.4,24.9,40,55])
data1['bmi_cat'] = bmi_cat
data1.bmi_cat.value_counts()
sns.countplot(data1.bmi_cat)
plt.title("bmi_cat",color = 'blue',fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.bmi_cat, y=data1.charges);
plt.xticks(rotation= 0)
plt.xlabel('BMI', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by BMI Categories', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10, 5))
sns.barplot(x=data1.children, y=data1.charges, hue=data1.sex);
plt.xticks(rotation= 0)
plt.xlabel('Number of Children', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Number of Children and Gender', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.region, y=data1.charges, hue=data1.sex);
plt.xticks(rotation= 45)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Region and Gender', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.region, y=data1.charges, hue=data1.children);
plt.xticks(rotation= 45)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Region and Number of Children', color = 'blue', fontsize=15)
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x=data1.smoker, y=data1.charges, hue=data1.children);
plt.xticks(rotation= 45)
plt.xlabel('Smoker', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Smoking and Number of Children', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="age_cat", y="charges", data=data1)
plt.title('Medical Costs by Age Categories', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="bmi_cat", y="charges", data=data1)
plt.title('Medical Costs by BMI', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="sex", y="charges", data=data1)
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="children", y="charges", data=data1)
plt.title('Medical Costs by Number of Children', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="charges", data=data1)
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="region", y="charges", data=data1)
plt.title('Medical Costs by Region', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="age_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Age Categories and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="bmi_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by BMI and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="sex", y="charges",hue="smoker", data=data1)
plt.title('Medical Costs by Gender and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="children", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Number of Children and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="region", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Region and Smoking', color = 'blue', fontsize=15)
plt.show()
sns.swarmplot(x="age_cat", y="bmi", data=data1)
plt.title('BMI by Age Categories', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="sex", y="bmi", data=data1)
plt.title('BMI by Gender', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="children", y="bmi", data=data1)
plt.title('BMI by Number of Children', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="bmi", data=data1)
plt.title('BMI by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="region", y="bmi", data=data1)
plt.title('BMI by Region', color = 'blue', fontsize=15)
plt.show()
sns.boxplot(x="age_cat", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Age Categories', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="bmi_cat", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by BMI', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="sex", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="children", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Number of Children', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="smoker", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="region", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Region', color = 'blue', fontsize=15)
plt.show()
sns.boxplot(x="age_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Age Categories and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="bmi_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by BMI Categories and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="sex", y="charges",hue="smoker", data=data1)
plt.title('Medical Costs by Gender and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="children", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Number of Children and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="region", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Region and Smoking', color = 'blue', fontsize=15)
plt.show()
data3 = data1.copy()
data3['smoker'].replace('yes',1,inplace=True)
data3['smoker'].replace('no',0,inplace=True)
data3['sex'].replace('male',1,inplace=True)
data3['sex'].replace('female',0,inplace=True)
data3['region'].replace('southwest',0,inplace=True)
data3['region'].replace('southeast',1,inplace=True)
data3['region'].replace('northwest',2,inplace=True)
data3['region'].replace('northeast',3,inplace=True)
data3.corr()
f,ax = plt.subplots(figsize=(10, 5))
sns.heatmap(data3.corr(), annot=True, linewidths=0.5, linecolor="red", fmt= '.3f',ax=ax)
plt.show()
sns.distplot(data1.charges, bins = 20, kde = True);
sns.distplot(data1.bmi, bins = 20, kde = True);
sns.pairplot(data1[['bmi','charges','age']],kind='reg')
plt.show()
sns.lmplot(x='bmi', y='charges', data=data1)
plt.show()
sns.lmplot(x='bmi', y='charges', hue='smoker', data=data1)
plt.show()
sns.lineplot(x='bmi',y='charges', data = data1)
plt.show()
sns.lineplot(x='bmi',y='charges', hue='smoker', data = data1)
plt.show()
sns.lmplot(x='age', y='charges', hue='smoker', data=data1)
plt.show()
sns.lmplot(x='bmi', y='charges', hue='smoker', col='age_cat', data=data1)
plt.show()
sns.lineplot(x='age',y='charges',data = data1)
plt.show()
data3.groupby('smoker')[['charges','bmi']].corr()
g = sns.jointplot(data1.bmi, data1.charges, kind="kde", size=7)
plt.show()
(sns
 .FacetGrid(data1,
              hue = "smoker",
              height = 5,
              xlim = (0, 70000))
 .map(sns.kdeplot, "charges", shade= True)
 .add_legend()
);
(sns
 .FacetGrid(data1,
              hue = "age_cat",
              height = 5,
              xlim = (0, 70000))
 .map(sns.kdeplot, "charges", shade= True)
 .add_legend()
);
sns.catplot(x = "bmi_cat", y = "charges", hue = "smoker", kind = "point", data = data1);
sns.catplot(x = "age_cat", y = "charges", hue = "smoker", kind = "point", data = data1);
data_bmi = data1.bmi/data1.bmi.max()
data_charges = data1.charges/data1.charges.max()
data_com = pd.concat([data_bmi,data_charges,data1.age,data.region],axis=1)
data_com.sort_values("age", ascending=True, inplace=True)
new_index = np.arange(len(data_com))
data_com = data_com.set_index(new_index)
data_com
sns.scatterplot(x = "bmi", y = "charges", hue= "smoker", size = "age", data = data1);
f,ax1 = plt.subplots(figsize =(15,10))
sns.pointplot(x='age',y='charges',data=data_com,color='lime',alpha=0.8)
sns.pointplot(x='age',y='bmi',data=data_com,color='red',alpha=0.8)
plt.text(40,0.15,'bmi ratio',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.10,'charges ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Age',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Charges vs. Bmi',fontsize = 20,color='blue')
plt.grid()
charges_age = data_com.groupby('age')['charges'].mean()
bmi_age = data_com.groupby('age')['bmi'].mean()
ages = data_com.groupby('age')['bmi'].mean().index

trace1 = go.Scatter(
                    x = ages,
                    y = charges_age,
                    mode = "lines",
                    name = "charges",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= data_com.index)
# Creating trace2
trace2 = go.Scatter(
                    x = ages,
                    y = bmi_age,
                    mode = "lines+markers",
                    name = "bmi",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= data_com.index)
data = [trace1, trace2]
layout = dict(title = 'Medical Costs vs BMI by Age',
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
trace1 = go.Histogram(
    x=data1.charges,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='medical costs distribution',
                   xaxis=dict(title='charges'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace0 = go.Box(
    y=data1.charges,
    name = 'medical costs',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

data = [trace0]
iplot(data)
data4 = data1.loc[:,["age","charges", "bmi"]]
data4["index"] = np.arange(1,len(data4)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data4, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
bmi_size  = [each for each in data1.bmi]
smoker_color = [float(each) for each in data3.smoker]
data = [
    {
        'y': data1.charges,
        'x': data1.age,
        'mode': 'markers',
        'marker': {
            'color': smoker_color,
            'size': bmi_size,
            'showscale': True
        },
        "text" :  data1.index    
    }
]
iplot(data)
age_size  = [each for each in data1.age]
smoker_color = [float(each) for each in data3.smoker]
data = [
    {
        'y': data1.charges,
        'x': data1.bmi,
        'mode': 'markers',
        'marker': {
            'color': smoker_color,
            'size': age_size,
            'showscale': True
        },
        "text" :  data1.index    
    }
]
iplot(data)
trace1 = go.Scatter3d(
    x=data1.bmi,
    y=data1.age,
    z=data1.charges,
    mode='markers',
    marker=dict(
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="3D ScatterPlot",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
