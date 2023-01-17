import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression

import warnings

import matplotlib.pyplot as plt

%matplotlib inline
warnings.filterwarnings('ignore')
housing_datasets = pd.read_csv('../input/boston-housing/housing.data', delim_whitespace=True, header = None)
housing_datasets.head()
columns_names = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_datasets.columns = columns_names
sns.pairplot(housing_datasets, size=1.5)

plt.tight_layout()

plt.show()
column_analysis = ['ZN', 'INDUS','NOX', 'RM']

sns.pairplot(housing_datasets[column_analysis],size = (2.5))

plt.tight_layout()

plt.show()
housing_datasets.corr()
plt.figure(figsize = (10, 8))

sns.heatmap(housing_datasets.corr(), annot=True)

plt.tight_layout()

plt.show()
model = LinearRegression()
independent_attribute = housing_datasets['RM'].values.reshape(-1, 1)

dependent_attribute = housing_datasets['MEDV'].values
model.fit(independent_attribute, dependent_attribute)
model.intercept_
model.coef_
plt.figure(figsize=(10, 8))

sns.regplot(independent_attribute, dependent_attribute)

plt.xlabel('Average Number of rooms per dwelling (Independent Attribute)')

plt.ylabel('Median value of Owner-occupid homes in $1000\'s (Dependent Attribute)')

plt.tight_layout()

plt.show()
sns.jointplot(x = 'MEDV', y = 'RM', data = housing_datasets, kind = 'reg', size = 10)

plt.show()
plt.scatter('MEDV', 'RM',

             s=100,

             c='blue',

             alpha=0.5, data=housing_datasets)

plt.xlabel("MEDV", size=16)

plt.ylabel("RM", size=16)

plt.title("Bubble Plot with Colors: Matplotlib", size=18)
sns.set_context("talk", font_scale=1.1)

plt.figure(figsize=(10,6))

sns.scatterplot(x='MEDV', 

                y='RM',

                size=80,

                sizes=(20,500),

                alpha=0.5,

                hue="RAD",

                c='blue',

                data=housing_datasets)

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.1, 1),borderaxespad=0)

# Put the legend out of the figure

#plt.legend(bbox_to_anchor=(1.01, 0.54),  borderaxespad=0.)

plt.xlabel("MEDV")

plt.ylabel("RM")

plt.title("Bubble plot with Colors in Seaborn")

plt.tight_layout()

plt.show()
import plotly.graph_objects as go



import pandas as pd



# load dataset

df = housing_datasets



# create figure

fig = go.Figure()



# Add surface trace

fig.add_trace(go.Surface(z=df.values.tolist(), colorscale="Viridis"))



# Update plot sizing

fig.update_layout(

    width=800,

    height=900,

    autosize=False,

    margin=dict(t=0, b=0, l=0, r=0),

    template="plotly_white",

)



# Update 3D scene options

fig.update_scenes(

    aspectratio=dict(x=1, y=1, z=0.7),

    aspectmode="manual"

)



# Add dropdown

fig.update_layout(

    updatemenus=[

        dict(

            type = "buttons",

            direction = "left",

            buttons=list([

                dict(

                    args=["type", "surface"],

                    label="3D Surface",

                    method="restyle"

                ),

                dict(

                    args=["type", "heatmap"],

                    label="Heatmap",

                    method="restyle"

                )

            ]),

            pad={"r": 10, "t": 10},

            showactive=True,

            x=0.11,

            xanchor="left",

            y=1.1,

            yanchor="top"

        ),

    ]

)



# Add annotation

fig.update_layout(

    annotations=[

        dict(text="Trace type:", showarrow=False,

                             x=0, y=1.08, yref="paper", align="left")

    ]

)



fig.show()
import plotly.graph_objects as go



# Generate dataset

import numpy as np

np.random.seed(1)



x0 = housing_datasets['MEDV']

y0 = housing_datasets['RM']

x1 = housing_datasets['CHAS']

y1 = housing_datasets['RM']

x2 = housing_datasets['CRIM']

y2 = housing_datasets['RM']



# Create figure

fig = go.Figure()



# Add traces

fig.add_trace(

    go.Scatter(

        x=x0,

        y=y0,

        mode="markers",

        marker=dict(color="DarkOrange")

    )

)



fig.add_trace(

    go.Scatter(

        x=x1,

        y=y1,

        mode="markers",

        marker=dict(color="Crimson")

    )

)



fig.add_trace(

    go.Scatter(

        x=x2,

        y=y2,

        mode="markers",

        marker=dict(color="RebeccaPurple")

    )

)



# Add buttons that add shapes

cluster0 = [dict(type="circle",

                            xref="x", yref="y",

                            x0=min(x0), y0=min(y0),

                            x1=max(x0), y1=max(y0),

                            line=dict(color="DarkOrange"))]

cluster1 = [dict(type="circle",

                            xref="x", yref="y",

                            x0=min(x1), y0=min(y1),

                            x1=max(x1), y1=max(y1),

                            line=dict(color="Crimson"))]

cluster2 = [dict(type="circle",

                            xref="x", yref="y",

                            x0=min(x2), y0=min(y2),

                            x1=max(x2), y1=max(y2),

                            line=dict(color="RebeccaPurple"))]



fig.update_layout(

    updatemenus=[

        dict(

            type="buttons",

            buttons=[

                dict(label="None",

                     method="relayout",

                     args=["shapes", []]),

                dict(label="Cluster 0",

                     method="relayout",

                     args=["shapes", cluster0]),

                dict(label="Cluster 1",

                     method="relayout",

                     args=["shapes", cluster1]),

                dict(label="Cluster 2",

                     method="relayout",

                     args=["shapes", cluster2]),

                dict(label="All",

                     method="relayout",

                     args=["shapes", cluster0 + cluster1 + cluster2])

            ],

        )

    ]

)



# Update remaining layout properties

fig.update_layout(

    title_text="Highlight Clusters",

    showlegend=False,

)



fig.show()