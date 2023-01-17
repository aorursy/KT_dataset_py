# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



MAIN_PATH = '../input/'

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(MAIN_PATH +'bank.csv')

original_df = df.copy()

# Have a grasp of how our data looks.

df.head()

df.describe()
# No missing values.

df.info()
f, ax = plt.subplots(1,2, figsize=(16,8))



colors = ["#FA5858", "#64FE2E"]

labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"



plt.suptitle('Information on Term Suscriptions', fontsize=20)



df["deposit"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 

                                             labels=labels, fontsize=12, startangle=25)





# ax[0].set_title('State of Loan', fontsize=16)

ax[0].set_ylabel('% of Condition of Loans', fontsize=14)



# sns.countplot('loan_condition', data=df, ax=ax[1], palette=colors)

# ax[1].set_title('Condition of Loans', fontsize=20)

# ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')

palette = ["#64FE2E", "#FA5858"]



sns.barplot(x="education", y="balance", hue="deposit", data=df, palette=palette, estimator=lambda x: len(x) / len(df) * 100)

ax[1].set(ylabel="(%)")

ax[1].set_xticklabels(df["education"].unique(), rotation=0, rotation_mode="anchor")

plt.show()
# Let's see how the numeric data is distributed.

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



df.hist(bins=20, figsize=(14,10), color='#E14906')

plt.show()
df['deposit'].value_counts()
# plt.style.use('dark_background')

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(212)



g = sns.boxplot(x="default", y="balance", hue="deposit",

                    data=df, palette="muted", ax=ax1)



g.set_title("Amount of Balance by Term Suscriptions")



# ax.set_xticklabels(df["default"].unique(), rotation=45, rotation_mode="anchor")



g1 = sns.boxplot(x="job", y="balance", hue="deposit",

                 data=df, palette="RdBu", ax=ax2)



g1.set_xticklabels(df["job"].unique(), rotation=90, rotation_mode="anchor")

g1.set_title("Type of Work by Term Suscriptions")



g2 = sns.violinplot(data=df, x="education", y="balance", hue="deposit", palette="RdBu_r")



g2.set_title("Distribution of Balance by Education")





plt.show()
df.columns
import squarify

df = df.drop(df.loc[df["balance"] == 0].index)





x = 0

y = 0

width = 100

height = 100



job_names = df['job'].value_counts().index

values = df['job'].value_counts().tolist()



normed = squarify.normalize_sizes(values, width, height)

rects = squarify.squarify(normed, x, y, width, height)



colors = ['rgb(200, 255, 144)','rgb(135, 206, 235)',

          'rgb(235, 164, 135)','rgb(220, 208, 255)',

          'rgb(253, 253, 150)','rgb(255, 127, 80)', 

         'rgb(218, 156, 133)', 'rgb(245, 92, 76)',

         'rgb(252,64,68)', 'rgb(154,123,91)']



shapes = []

annotations = []

counter = 0



for r in rects:

    shapes.append(

        dict(

            type = 'rect',

            x0 = r['x'],

            y0 = r['y'],

            x1 = r['x'] + r['dx'],

            y1 = r['y'] + r['dy'],

            line = dict(width=2),

            fillcolor = colors[counter]

        )

    )

    annotations.append(

        dict(

            x = r['x']+(r['dx']/2),

            y = r['y']+(r['dy']/2),

            text = values[counter],

            showarrow = False

        )

    )

    counter = counter + 1

    if counter >= len(colors):

        counter = 0

    

# For hover text

trace0 = go.Scatter(

    x = [ r['x']+(r['dx']/2) for r in rects],

    y = [ r['y']+(r['dy']/2) for r in rects],

    text = [ str(v) for v in job_names],

    mode='text',

)



layout = dict(

    title='Number of Occupations <br> <i>(From our Sample Population)</i>',

    height=700, 

    width=700,

    xaxis=dict(showgrid=False,zeroline=False),

    yaxis=dict(showgrid=False,zeroline=False),

    shapes=shapes,

    annotations=annotations,

    hovermode='closest'

)



# With hovertext

figure = dict(data=[trace0], layout=layout)



iplot(figure, filename='squarify-treemap')
# Now let's see which occupation tended to have more balance in their accounts



suscribed_df = df.loc[df["deposit"] == "yes"]



occupations = df["job"].unique().tolist()



# Get the balances by jobs

management = suscribed_df["age"].loc[suscribed_df["job"] == "management"].values

technician = suscribed_df["age"].loc[suscribed_df["job"] == "technician"].values

services = suscribed_df["age"].loc[suscribed_df["job"] == "services"].values

retired = suscribed_df["age"].loc[suscribed_df["job"] == "retired"].values

blue_collar = suscribed_df["age"].loc[suscribed_df["job"] == "blue-collar"].values

unemployed = suscribed_df["age"].loc[suscribed_df["job"] == "unemployed"].values

entrepreneur = suscribed_df["age"].loc[suscribed_df["job"] == "entrepreneur"].values

housemaid = suscribed_df["age"].loc[suscribed_df["job"] == "housemaid"].values

self_employed = suscribed_df["age"].loc[suscribed_df["job"] == "self-employed"].values

student = suscribed_df["age"].loc[suscribed_df["job"] == "student"].values





ages = [management, technician, services, retired, blue_collar, unemployed, 

         entrepreneur, housemaid, self_employed, student]



colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',

          'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 

          'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',

         'rgba(229, 126, 56, 0.5)', 'rgba(229, 56, 56, 0.5)',

         'rgba(174, 229, 56, 0.5)', 'rgba(229, 56, 56, 0.5)']



traces = []



for xd, yd, cls in zip(occupations, ages, colors):

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

    title='Distribution of Ages by Occupation',

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
# Balance Distribution



# Create a Balance Category

df["balance_status"] = np.nan

lst = [df]



for col in lst:

    col.loc[col["balance"] < 0, "balance_status"] = "negative"

    col.loc[(col["balance"] >= 0) & (col["balance"] <= 30000), "balance_status"] = "low"

    col.loc[(col["balance"] > 30000) & (col["balance"] <= 40000), "balance_status"] = "middle"

    col.loc[col["balance"] > 40000, "balance_status"] = "high"

    

# balance by balance_status

negative = df["balance"].loc[df["balance_status"] == "negative"].values.tolist()

low = df["balance"].loc[df["balance_status"] == "low"].values.tolist()

middle = df["balance"].loc[df["balance_status"] == "middle"].values.tolist()

high = df["balance"].loc[df["balance_status"] == "high"].values.tolist()





# Get the average by occupation in each balance category

job_balance = df.groupby(['job', 'balance_status'])['balance'].mean()





trace1 = go.Barpolar(

    r=[-199.0, -392.0, -209.0, -247.0, -233.0, -270.0, -271.0, 0, -276.0, -134.5],

    text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",

         "services", "student", "technician", "unemployed"],

    name='Negative Balance',

    marker=dict(

        color='rgb(246, 46, 46)'

    )

)

trace2 = go.Barpolar(

    r=[319.5, 283.0, 212.0, 313.0, 409.0, 274.5, 308.5, 253.0, 316.0, 330.0],

    text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",

         "services", "student", "technician", "unemployed"],

    name='Low Balance',

    marker=dict(

        color='rgb(246, 97, 46)'

    )

)

trace3 = go.Barpolar(

    r=[2128.5, 2686.0, 2290.0, 2366.0, 2579.0, 2293.5, 2005.5, 2488.0, 2362.0, 1976.0],

    text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",

         "services", "student", "technician", "unemployed"],

    name='Middle Balance',

    marker=dict(

        color='rgb(246, 179, 46)'

    )

)

trace4 = go.Barpolar(

    r=[14247.5, 20138.5, 12278.5, 12956.0, 20723.0, 12159.0, 12223.0, 13107.0, 12063.0, 15107.5],

    text=["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",

         "services", "student", "technician", "unemployed"],

    name='High Balance',

    marker=dict(

        color='rgb(46, 246, 78)'

    )

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    title='Mean Balance in Account<br> <i> by Job Occupation</i>',

    font=dict(

        size=12

    ),

    legend=dict(

        font=dict(

            size=16

        )

    ),

    radialaxis=dict(

        ticksuffix='%'

    ),

    orientation=270

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='polar-area-chart')
df['marital'].value_counts()
df['marital'].unique()
df['marital'].value_counts().tolist()
vals = df['marital'].value_counts().tolist()

labels = ['married', 'divorced', 'single']



data = [go.Bar(

            x=labels,

            y=vals,

    marker=dict(

    color="#FE9A2E")

    )]



layout = go.Layout(

    title="Count by Marital Status",

)



fig = go.Figure(data=data, layout=layout)







iplot(fig, filename='basic-bar')
# Distribution of Balances by Marital status

single = df['balance'].loc[df['marital'] == 'single'].values

married = df['balance'].loc[df['marital'] == 'married'].values

divorced = df['balance'].loc[df['marital'] == 'divorced'].values





single_dist = go.Histogram(

    x=single,

    histnorm='density', 

    name='single',

    marker=dict(

        color='#6E6E6E'

    )

)





married_dist = go.Histogram(

    x=married,

    histnorm='density', 

    name='married',

    marker=dict(

        color='#2E9AFE'

    )

)



divorced_dist = go.Histogram(

    x=divorced,

    histnorm='density', 

    name='divorced',

    marker=dict(

        color='#FA5858'

    )

)





fig = tools.make_subplots(rows=3, print_grid=False)



fig.append_trace(single_dist, 1, 1)

fig.append_trace(married_dist, 2, 1)

fig.append_trace(divorced_dist, 3, 1)





fig['layout'].update(showlegend=False, title="Price Distributions by Marital Status",

                    height=1000, width=800)



iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
# Notice how divorced have a considerably low amount of balance.

fig = ff.create_facet_grid(

    df,

    x='duration',

    y='balance',

    color_name='marital',

    show_boxes=False,

    marker={'size': 10, 'opacity': 1.0},

    colormap={'single': 'rgb(165, 242, 242)', 'married': 'rgb(253, 174, 216)', 'divorced': 'rgba(201, 109, 59, 0.82)'}

)



iplot(fig, filename='facet - custom colormap')
# Hmmm We have missed some important clients with some high balances. 

# This shouldn't be happening.

fig = ff.create_facet_grid(

    df,

    y='balance',

    facet_row='marital',

    facet_col='deposit',

    trace_type='box',

)



iplot(fig, filename='facet - box traces')
df = df.drop(df.loc[df["education"] == "unknown"].index)

df['education'].unique()
df['marital/education'] = np.nan

lst = [df]



for col in lst:

    col.loc[(col['marital'] == 'single') & (df['education'] == 'primary'), 'marital/education'] = 'single/primary'

    col.loc[(col['marital'] == 'married') & (df['education'] == 'primary'), 'marital/education'] = 'married/primary'

    col.loc[(col['marital'] == 'divorced') & (df['education'] == 'primary'), 'marital/education'] = 'divorced/primary'

    col.loc[(col['marital'] == 'single') & (df['education'] == 'secondary'), 'marital/education'] = 'single/secondary'

    col.loc[(col['marital'] == 'married') & (df['education'] == 'secondary'), 'marital/education'] = 'married/secondary'

    col.loc[(col['marital'] == 'divorced') & (df['education'] == 'secondary'), 'marital/education'] = 'divorced/secondary'

    col.loc[(col['marital'] == 'single') & (df['education'] == 'tertiary'), 'marital/education'] = 'single/tertiary'

    col.loc[(col['marital'] == 'married') & (df['education'] == 'tertiary'), 'marital/education'] = 'married/tertiary'

    col.loc[(col['marital'] == 'divorced') & (df['education'] == 'tertiary'), 'marital/education'] = 'divorced/tertiary'

    

    

df.head()
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

g = sns.FacetGrid(df, row="marital/education", hue="marital/education", aspect=12, palette=pal)



g.map(sns.kdeplot, "balance", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)

g.map(sns.kdeplot, "balance", clip_on=False, color="w", lw=1, bw=0)

g.map(plt.axhline, y=0, lw=2, clip_on=False)
education_groups = df.groupby(['marital/education'], as_index=False)['balance'].median()



fig = plt.figure(figsize=(12,8))









sns.barplot(x="balance", y="marital/education", data=education_groups,

            label="Total", palette="RdBu")



plt.title('Median Balance by Educational/Marital Group', fontsize=16)
# Let's see the group who had loans from the marital/education group



loan_balance = df.groupby(['marital/education', 'loan'], as_index=False)['balance'].median()





no_loan = loan_balance['balance'].loc[loan_balance['loan'] == 'no'].values

has_loan = loan_balance['balance'].loc[loan_balance['loan'] == 'yes'].values





labels = loan_balance['marital/education'].unique().tolist()





trace0 = go.Scatter(

    x=no_loan,

    y=labels,

    mode='markers',

    name='No Loan',

    marker=dict(

        color='rgb(175,238,238)',

        line=dict(

            color='rgb(0,139,139)',

            width=1,

        ),

        symbol='circle',

        size=16,

    )

)

trace1 = go.Scatter(

    x=has_loan,

    y=labels,

    mode='markers',

    name='Has a Previous Loan',

    marker=dict(

        color='rgb(250,128,114)',

        line=dict(

            color='rgb(178,34,34)',

            width=1,

        ),

        symbol='circle',

        size=16,

    )

)



data = [trace0, trace1]

layout = go.Layout(

    title="The Impact of Loans to Married/Educational Clusters",

    xaxis=dict(

        showgrid=False,

        showline=True,

        linecolor='rgb(102, 102, 102)',

        titlefont=dict(

            color='rgb(204, 204, 204)'

        ),

        tickfont=dict(

            color='rgb(102, 102, 102)',

        ),

        showticklabels=False,

        dtick=10,

        ticks='outside',

        tickcolor='rgb(102, 102, 102)',

    ),

    margin=dict(

        l=140,

        r=40,

        b=50,

        t=80

    ),

    legend=dict(

        font=dict(

            size=10,

        ),

        yanchor='middle',

        xanchor='right',

    ),

    width=1000,

    height=800,

    paper_bgcolor='rgb(255,250,250)',

    plot_bgcolor='rgb(255,255,255)',

    hovermode='closest',

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='lowest-oecd-votes-cast')
import seaborn as sns

sns.set(style="ticks")



sns.pairplot(df, hue="marital/education", palette="Set1")

plt.show()
fig = plt.figure(figsize=(12,8))



sns.violinplot(x="balance", y="job", hue="deposit", palette="RdBu_r",

            data=df);



plt.title("Job Distribution of Balances by Deposit Status", fontsize=16)



plt.show()