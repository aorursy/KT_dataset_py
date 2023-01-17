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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots
!pip install creme
from creme import *
df = pd.read_csv("/kaggle/input/insurance/insurance.csv")

df.head()
sns.heatmap(df.isnull())
sns.pairplot(df)
df.describe()
df.info()
sns.heatmap(df.corr(),annot=True)
df.head(2)
fig = px.scatter(df, x='age', y='bmi',color='sex')

fig.update_layout(title="Age v/s BMI",xaxis_title="Age",yaxis_title="BMI",title_x=0.5)

fig.show()
children=df['children'].value_counts().to_frame().reset_index().rename(columns={'index':'children','children':'Count'})

children
fig = go.Figure(data=[go.Scatter(

    x=children['children'], y=children['Count'],

    mode='markers',

    marker=dict(

        color=children['Count'],

        size = 20,

        showscale=True

    ))])



fig.update_layout(title='Number of children is each category',xaxis_title="children",yaxis_title="Number of Children",title_x=0.5)

fig.show()
fig = go.Figure(go.Bar(

    x=children['children'],y=children['Count'],marker={'color': children['Count'], 

    'colorscale': 'Viridis'},  

    text=children['Count'],

    textposition = "outside",

))

fig.update_layout(title_text='Count of Children',xaxis_title="Children",yaxis_title="Number of Children",title_x=0.5)

fig.show()
df.head(2)
region=df['region'].value_counts().to_frame().reset_index().rename(columns={'index':'region','region':'count'})

region
colors=['orange','pink','yellow','lightblue']

fig = go.Figure([go.Pie(labels=region['region'],values=region['count'])])

fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=2)))

fig.update_layout(title="Region Types",title_x=0.5)

fig.show()
df.head(1)
fig = px.sunburst(df, path=['sex','smoker','region'], values='charges', color='charges',color_continuous_scale='viridis')

fig.update_layout(title="Sunburst Distribution by Sex, smoker and region",title_x=0.5)

fig.show()
male=df[df['sex']=="male"]['charges']

female=df[df['sex']=="female"]['charges']



fig = go.Figure()

fig.add_trace(go.Box(y=male,

                     marker_color="blue",

                     name="Male charges",

                    boxpoints='suspectedoutliers', # Display suspected outliers

                     boxmean = 'sd',

                     marker=dict(

                     color='black',

                     outliercolor='red')))

fig.add_trace(go.Box(y=female,

                     marker_color="magenta",

                     name="Female charges",

                    boxpoints='suspectedoutliers', # Display suspected outliers

                     boxmean = 'sd',

                     marker=dict(

                     color='black',

                     outliercolor='black')))

fig.update_layout(title="Distribution of Charges by Sex with their outliers",title_x=0.5)

fig.show()
fig = go.Figure(data=[go.Histogram(x=male,  

                                  marker_color="royalblue")])

fig.update_layout(title="Distribution of Charges for Male",xaxis_title="Charges",yaxis_title="Counts")

fig.show()
min(male)
histdata = [male,female]

grouplabels = ['Charge distribution for Male',"Charge distribution for Female"]

colors = ['red','royalblue','orange']

fig = ff.create_distplot(histdata,grouplabels,colors = colors,show_hist=False)

fig.show()
df.head()
fig = go.Figure(go.Histogram2dContour(x=df['age'],

        y=df['bmi']))

fig.update_layout(title='Density of Age v/s BMI',xaxis_title="AGE",yaxis_title="BMI",title_x=0.5)

fig.show()
fig = go.Figure(go.Histogram2d(x=df['age'],

        y=df['bmi']))

fig.update_layout(title='Density of Age v/s BMI',xaxis_title="AGE",yaxis_title="BMI",title_x=0.5)

fig.show()
fig = px.density_heatmap(df, x="sex", y="charges", facet_row="smoker", facet_col="region")

fig.update_layout(title='Density heatmap of Sex vs charges with smoker and region')

fig.show()
male=df[df['sex']=="male"]



x = df['age']

y = df['charges']



fig = go.Figure()

fig.add_trace(go.Histogram2dContour(

        x = x,

        y = y,

        colorscale = 'gray',

        reversescale = True,

        xaxis = 'x',

        yaxis = 'y'

    ))

fig.add_trace(go.Scatter(

        x = x,

        y = y,

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = "red", #'rgba(0,0,0,0.3)',

            size = 3

        )

    ))

fig.add_trace(go.Histogram(

        y = y,

        xaxis = 'x2',

        marker = dict(

            color = "blue", #'rgba(0,0,0,1)'

        )

    ))

fig.add_trace(go.Histogram(

        x = x,

        yaxis = 'y2',

        marker = dict(

            color = "blue",# 'rgba(0,0,0,1)'

        )

    ))



fig.update_layout(

    autosize = False,

    xaxis = dict(

        zeroline = False,

       domain = [0,0.85],

        showgrid = False

    ),

    yaxis = dict(

        zeroline = False,

       domain = [0,0.85],

        showgrid = False

    ),

    xaxis2 = dict(

        zeroline = False,

       domain = [0.85,1],

        showgrid = False

    ),

    yaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    height = 600,

    width = 600,

    bargap = 0,

    hovermode = 'closest',

    showlegend = False,

    title_text="Density Contour of charges and age for Males",title_x=0.5

)



fig.show()
temp_list=[]

children=list(df['children'].value_counts()[:10].to_frame().reset_index()['index'])

children
for i in children:

    temp_df=df[df['children']==i]['charges']

    temp_list.append(temp_df)

final_arr=np.array(temp_list)

colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 10, colortype='rgb')



fig = go.Figure()

for dataline,color,n in zip(final_arr,colors,children):

    fig.add_trace(go.Violin(x=dataline,line_color=color,name=n))

fig.update_traces(orientation='h',side='positive',width=2,points=False)

fig.update_layout(title="Distribution of Children with charges",xaxis_showgrid=False, xaxis_zeroline=True,height=800)

fig.show()
fig = px.parallel_categories(df, dimensions=['sex','children','smoker'],

                color="age", color_continuous_scale=px.colors.sequential.Inferno)

fig.update_layout(title="Parallel Categories Diagram ")

fig.show()
fig = px.parallel_categories(df, dimensions=['sex','region','smoker'],

                color="age", color_continuous_scale=px.colors.sequential.Inferno)

fig.update_layout(title="Parallel Categories Diagram ")

fig.show()
fig = go.Figure(go.Heatmap(z=df['charges'],x=df['sex'],

                   y=df['children'],hoverongaps = False))

fig.update_layout(title='Heatmap of Charges Vs Sex & Children',xaxis_title="Gender",yaxis_title="Children")

fig.show()
df['children'].unique()
df['region'].unique()
def required(region):

    data=df[(df['region']==region)&((df['children']==0)|(df['children']==1)|(df['children']==2)|(df['children']==3))][['children','charges']].groupby('children').sum().reset_index()

#     df["month"] = pd.to_datetime(df.month, format='%B').dt.month 

    data = data.sort_values(by="children",ascending=False)

    children = ['0','1','2','3']

    data['children'] = children

    return data
southwest = required("southwest")

southeast = required("southeast")

northwest = required("northwest")

northeast = required("northeast")



northeast
fig = go.Figure()



fig.add_trace(go.Funnel(

    name = 'southwest',

    y = southwest['children'].tolist(),

    x = southwest['charges'].tolist(),

    textinfo = "value"))



fig.add_trace(go.Funnel(

    name = 'southeast',

    orientation = "h",

    y = southeast['children'].tolist(),

    x = southeast['charges'].tolist(),

    textposition = "inside",

    textinfo = "value"))



fig.add_trace(go.Funnel(

    name = 'northwest',

    orientation = "h",

    y = northwest['children'].tolist(),

    x = northwest['charges'].tolist(),

    textinfo = "value"))



fig.add_trace(go.Funnel(

    name = 'northeast',

    orientation = "h",

    y = northeast['children'].tolist(),

    x = northeast['charges'].tolist(),

    textinfo = "value"))



fig.update_layout(title = "Funnel chart for children and charges based on region",title_x=0.5)



fig.show()
df.head()
fig = px.scatter_3d(data_frame=df,x='age',y='bmi',z='charges',color='sex')

fig.update_layout(title = "3D scatter plot for age, bmi and charges based on sex",title_x=0.5)

fig.show()
fig = go.Figure(data=[go.Histogram(x=df['charges'],  

                                  marker_color="royalblue")])

fig.update_layout(title="Distribution of Charges",xaxis_title="Charges",yaxis_title="Counts")

fig.show()
IQR=df['charges'].quantile(0.75)-df['charges'].quantile(0.25)

lower_bridge=df['charges'].quantile(0.25)-(IQR*1.5)

upper_bridge=df['charges'].quantile(0.75)+(IQR*1.5)

print(lower_bridge), print(upper_bridge)
#### Extreme outliers

lower_bridge=df['charges'].quantile(0.25)-(IQR*3)

upper_bridge=df['charges'].quantile(0.75)+(IQR*3)

print(lower_bridge), print(upper_bridge)
data = df.copy()

data.loc[data['charges']>=52338.78861,'charges']=52338.78861

fig = go.Figure(data=[go.Histogram(x=data['charges'],  

                                  marker_color="royalblue")])

fig.update_layout(title="Distribution of Charges after fixing outliers",xaxis_title="Charges",yaxis_title="Counts")

fig.show()
data.head(3)
categorical = data.columns[data.dtypes==object]

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

for i in categorical:

    data[i] = enc.fit_transform(data[i])
data.head()
X = data.drop('charges',axis=1)

y = data['charges']
from creme import compose

from creme import linear_model

from creme import preprocessing

from creme import stream

from creme import metrics



scaler = preprocessing.StandardScaler()

linreg= linear_model.LinearRegression()

y_true = []

y_pred = []



for xi, yi in stream.iter_pandas(X,y, shuffle=True, seed=42):

    xi_scaled = scaler.fit_one(xi).transform_one(xi)

    yi_pred = linreg.predict_one(xi_scaled)

    linreg.fit_one(xi_scaled, yi)

    y_true.append(yi)

    y_pred.append(yi_pred)
from sklearn import metrics
metrics.r2_score(y_true,y_pred)