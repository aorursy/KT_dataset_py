import numpy as np 

import pandas as pd 



import seaborn as sns 

sns.set(style = "whitegrid")

import matplotlib.pyplot as plt 



import warnings

warnings.filterwarnings("ignore")



import warnings

warnings.filterwarnings("ignore")

import plotly.figure_factory as ff 

import  plotly.offline as py

import plotly.graph_objs as go 



from plotly.offline import download_plotlyjs,init_notebook_mode, iplot, plot

from plotly import tools 

py.init_notebook_mode(connected = True)



import cufflinks as cf 

cf.go_offline()
df=pd.read_csv('../input/BlackFriday.csv')
df.head(20)
Data_types = df.dtypes.value_counts()

print(Data_types)



plt.figure(figsize = (14,4))

sns.barplot(x = Data_types.index, y = Data_types.values)

plt.title("Data Type Distribution")
df.isnull().sum()
Numerical_data=df.select_dtypes(include=("float64", "int64"))

Categr_data=df.select_dtypes(include=("object"))





fig, (ax1, ax2) =plt.subplots(nrows=2, ncols=1, figsize = (15,10))



sns.heatmap(Categr_data.isnull(), cbar = False, annot = False, cmap ="cividis", yticklabels=False, ax=ax1)

plt.title("Missing Values in Categorical Columns")

sns.heatmap(Numerical_data.isnull(), cbar = False, annot = False, cmap ="cividis", yticklabels=False)

plt.title("Missing Values in Numberical Columns")

plt.tight_layout()
df['Product_Category_2'].mean()
df['Product_Category_3'].mean()
df["Product_Category_2"].fillna("9.842144034798471", inplace = True) 

df["Product_Category_3"].fillna("12.669840149015693", inplace = True) 
# check null value agine 

df.isnull().sum()
df['Gender'].value_counts()
Female = df[(df['Gender'] == 'F')]

Male = df[(df['Gender'] != 'F')]



trace = go.Pie(labels = ['Male', 'Females'], values = df['Gender'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['blueviolet','cadetblue'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of Gender as Pie')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)

trace0 = go.Bar(

    y=df['Gender'].value_counts(),

  

)



data = [trace0]



fig = go.Figure(data=data)

layout = dict(title =  'Distribution of Gender as Bar')

           

fig = dict(data = [trace0], layout=layout)

py.iplot(fig)





Looging_Age = df['Age'].values.tolist()

Age_MAle = df['Age'].loc[df['Gender'] == 'M'].values.tolist()

Age_Female = df['Age'].loc[df['Gender'] != 'M'].values.tolist()



trace0 = go.Histogram(

    x=Age_MAle,

    histnorm='probability',

    name="GMale",

    marker = dict(

        color = 'rgba(100, 149, 237, 0.6)',

    )

)

trace1 = go.Histogram(

    x=Age_Female,

    histnorm='probability',

    name=" Female",

    marker = dict(

        color = 'rgba(255, 182, 193, 0.6)',

    )

)

trace2 = go.Histogram(

    x=Looging_Age,

    histnorm='probability',

    name="Overall Gender",

     marker = dict(

        color = 'rgba(169, 169, 169, 0.6)',

    )

)

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Males','Female', 'All Genders'))



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)



fig['layout'].update(showlegend=True, title='Distribution of Age Male and Female ', bargap=0.05)

iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
Age = df['Age'].value_counts()

trace = go.Pie(labels = ['26-35', '36-45','18-25','46-50','51-55','55+','0-17'], values = df['Age'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['darkgoldenrod','darkturquoise'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of Age')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
df.groupby(['Product_Category_1','Gender'])['Gender'].count().unstack(1).plot.bar()

df.groupby(['Product_Category_2','Gender'])['Gender'].count().unstack(1).plot.bar()
df.groupby(['Product_Category_3','Gender'])['Gender'].count().unstack(1).plot.bar()
df.groupby(['Gender','Age'])['Gender'].count().unstack(1).plot.bar()
df.info()
df.groupby(['City_Category','Gender'])['Gender'].count().unstack(1).plot.bar()
Gender_M = df.loc[df["Gender"]=='M']

City_Category_Male = df["City_Category"].unique().tolist()



A_m = Gender_M["Age"].loc[Gender_M["City_Category"] == "A"].values

B_m = Gender_M["Age"].loc[Gender_M["City_Category"] == "B"].values

C_m = Gender_M["Age"].loc[Gender_M["City_Category"] == "C"].values



Ages = [A_m, B_m, C_m]







colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',

          'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 

          'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',

         'rgba(229, 126, 56, 0.5)', 'rgba(229, 56, 56, 0.5)',

         'rgba(174, 229, 56, 0.5)', 'rgba(229, 56, 56, 0.5)']



traces = []



for xd, yd, cls in zip(City_Category_Male, Ages, colors):

        traces.append(go.Box(

            y=yd,

            name=xd,

            boxpoints='all',

            jitter=0.5,

            whiskerwidth=0.2,

            fillcolor=cls,

            marker=dict(

                size=2,

            ),

            line=dict(width=1),

        ))



layout = go.Layout(

    title='Distribution of Ages by City_Category_Male',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(224,255,246)',

    plot_bgcolor='rgb(251,251,251)',

    showlegend=False

)



fig = go.Figure(data=traces, layout=layout)

iplot(fig)







Gender_Female = df.loc[df["Gender"]=='F']

City_Category_Male = df["City_Category"].unique().tolist()



A_Female = Gender_Female["Age"].loc[Gender_Female["City_Category"] == "A"].values

B_Female = Gender_Female["Age"].loc[Gender_Female["City_Category"] == "B"].values

C_Female = Gender_Female["Age"].loc[Gender_Female["City_Category"] == "C"].values



Ages = [A_Female, B_Female, C_Female]







colors = ['rgba(200, 180, 214, 0.5)', 'rgba(255, 160, 34, 0.5)',

          'rgba(180, 160, 110, 0.5)', 'rgba(255, 65, 84, 0.5)', 

          'rgba(300, 90, 255, 0.5)', 'rgba(190, 96, 0, 0.5)',

         'rgba(150, 150, 255, 0.5)', 'rgba(229, 56, 56, 0.5)',

         'rgba(230, 210, 156, 0.5)', 'rgba(229, 56, 56, 0.5)']



traces = []



for xd, yd, cls in zip(City_Category_Male, Ages, colors):

        traces.append(go.Box(

            y=yd,

            name=xd,

            boxpoints='all',

            jitter=0.5,

            whiskerwidth=0.2,

            fillcolor=cls,

            marker=dict(

                size=2,

            ),

            line=dict(width=1),

        ))



layout = go.Layout(

    title='Distribution of Ages by City_Category_Female',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(210,230,180)',

    plot_bgcolor='rgb(180,200,180)',

    showlegend=False

)



fig = go.Figure(data=traces, layout=layout)

iplot(fig)











df['Occupation'].unique()
df['Occupation'].value_counts()
Occupation_4 = df[(df['Occupation'] == 4)]

Occupation_0 = df[(df['Occupation'] == 0)]

Occupation_7 = df[(df['Occupation'] == 7)]

Occupation_1 = df[(df['Occupation'] == 1)]

Occupation_17 = df[(df['Occupation'] == 17)]

Occupation_20 = df[(df['Occupation'] == 20)]

Occupation_12 = df[(df['Occupation'] == 12)]

Occupation_14 = df[(df['Occupation'] == 14)]

Occupation_2 = df[(df['Occupation'] == 2)]

Occupation_16 = df[(df['Occupation'] == 16)]

Occupation_6 = df[(df['Occupation'] == 6)]

Occupation_3 = df[(df['Occupation'] == 3)]

Occupation_10 = df[(df['Occupation'] == 10)]

Occupation_5 = df[(df['Occupation'] == 5)]

Occupation_15 = df[(df['Occupation'] == 15)]

Occupation_11 = df[(df['Occupation'] == 11)]

Occupation_19 = df[(df['Occupation'] == 19)]

Occupation_13 = df[(df['Occupation'] == 13)]

Occupation_18 = df[(df['Occupation'] == 18)]

Occupation_9 = df[(df['Occupation'] == 9)]

Occupation_8 = df[(df['Occupation'] == 8)]





trace8 = go.Pie(labels = [4, 0, 7,  1, 17,  20,  12, 14, 2,  16,  6,  3, 10,  5, 15,  11, 19,

        13, 18, 9,  8], values = df['Occupation'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['Turquoise','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of Occupation')

           

fig = dict(data = [trace8], layout=layout)

py.iplot(fig)
df_Occupation = df.Occupation.value_counts()

trace_df_Occupation = go.Bar(

    x=df_Occupation.index,

    y=df['Product_Category_1'].value_counts(),



    name='Product_Category_1',

    marker=dict(

        color='rgb(200, 70, 150)'

    )

)



trace_df_Occupation2 = go.Bar(

    x=df_Occupation.index,

    y=df['Product_Category_2'].value_counts(),



    name='Product_Category_2',

    marker=dict(

        color='rgb(400, 180, 30)'

    )

)









trace_df_Occupation3 = go.Bar(

    x=df_Occupation.index,

    y=df['Product_Category_3'].value_counts(),



    name='Product_Category_3',

    marker=dict(

        color='rgb(100, 200, 300)'

    )

)







data = [trace_df_Occupation,trace_df_Occupation2,trace_df_Occupation3]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='group',

    

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='angled-text-bar')
sns.pairplot(df.select_dtypes(include=[np.number]), dropna=True)
#correlation

correlation = df.corr()

#tick labels

matrix_cols = correlation.columns.tolist()

#convert to array

corr_array  = np.array(correlation)
# Correalation plot in order to identify the relationship between the Numberical Features:

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(), linewidths=.1, annot=True, cmap='magma')

df.corr()["Purchase"].sort_values(ascending = False).head(5)