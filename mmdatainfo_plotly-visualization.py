# Import Plotly and set to offline mode (not credentials needed)

import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)
import numpy as np

import pandas as pd

import scipy as sp

# To scale data before plotting

from sklearn.preprocessing import StandardScaler

# Load scikit-learn data set

from sklearn.datasets import load_boston
boston = load_boston();

df = pd.DataFrame(boston.data,columns=boston.feature_names)

df = df.assign(target=boston.target);

df.head(3)
# First create layout (will be used for figure later)

layout = go.Layout(

    # set figure size manually

    width=400,

    height=250,

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(

        l=5,r=5, # left & right

        b=5, # bottom

        t=40, # top for the title

    ),

    title="Plotly example", # figure title

    # Inside the layout, user can define axis properties

    xaxis=go.layout.XAxis(

        showgrid=False,

        zeroline=False,

        showticklabels=False

    ),

    yaxis=go.layout.YAxis(

        showgrid=False,

        zeroline=False,

        showticklabels=False

    )

)



# Now create the "data" to be plotted Can be multiple scatter plots

data = [

    go.Scatter(x=[1,1,2,3,3],y=[-1,1,0,1,-1],mode="lines")

]



# Create figure with data using the defined layout

fig = go.Figure(data=data, layout=layout)



# Show the figure inside Jupyter

py.offline.iplot(fig)
p = np.polyfit(df.target,df.RM,deg=1)

# First create layout (will be used for figure later)

layout = go.Layout(

    # set figure size manually

    width=600,

    height=350,

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(

        l=30,r=5, # left & right

        b=30, # bottom

        t=40, # top for the title

    ),

    title="Plotly interactive scatter plot", # figure title

    font=dict(size=11),

    # Set x/y labels. Legend names are defined in "data" (legend in layout sets position)

    xaxis=go.layout.XAxis(

        title="number of rooms"

    ),

    yaxis=go.layout.YAxis(

        title="price"

    )

)



# Now create the "data" to be plotted. Insert two data sets = 

data = [

    go.Scatter(x=df.target,y=df.RM, 

               mode="markers",

               marker=dict(color="black"),

               name='input'

            ),

    go.Scatter(x=df.target,y=np.polyval(p,df.target),

              mode="lines",

              line=dict(color="red",width=2),

              name="fit"

            )

]



# Create figure with data using the defined layout

fig = go.Figure(data=data, layout=layout)



# Show the figure inside Jupyter

py.offline.iplot(fig)
# Generate time vector (same length as x1) from first to forth month of 2010 with `D`aily sampling

t = np.arange("2010-01-01","2010-04-01", dtype="datetime64[D]")

# Generate `y` value (use `t.astype('float')` to convert to float days)

y = np.sin(2*np.pi*1/30*t.astype("float")+10);

# add some noise

y1 = y + np.random.randn(y.size)/10;
# Set layout = 2 Axes at different (y coordinate) position

layout = go.Layout(

    xaxis2=dict(

        domain=[0, 1]

    ),

    yaxis2=dict(

        domain=[0.52, 1],

    ),

    # Set the main x axis including date-time tick format

    xaxis=dict(

        domain=[0, 1],

        tickformat="%Y-%m-%d"

    ),

    yaxis=dict(

        domain=[0.0, 0.48]

    ),

    # Set legend posistion and title

    legend=dict(x=0.8, y=0,orientation ="v"),

    title="Plotly time series subplot with shared time axis",

    font=dict(size=11),

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(l=50,r=50,b=50,t=50),

    # Set figure size

    width=600,

    height=300,

)



# Assign data to axis

data = [

    go.Scatter(x=t,y=y,

               mode="lines",

               marker=dict(color="black"),

               name="target/label",

               xaxis="x",

               yaxis="y2"

    ),

    go.Scatter(x=t,y=y1,

               mode="markers",

               marker=dict(color="blue"),

               name="feature",

               xaxis="x",

               yaxis="y"

    ),

]



# Show the result

py.offline.iplot(go.Figure(data=data, layout=layout))
layout = go.Layout(

    # Figure size ration will be ignored 

    width=600,

    height=400,

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(l=30,r=30,t=30,b=30),

    title="Plotly 3D scatter plot", # figure title

    font=dict(size=11),

    # Set x/y/z labels. The main difference to standard (2D) plot is that

    # the objects are in "Scene" (graph_objs) 

    scene=go.Scene(

        xaxis=go.layout.scene.XAxis(title="x = RM"),

        yaxis=go.layout.scene.YAxis(title='y = MEDV'),

        zaxis=go.layout.scene.ZAxis(title='z axis title'),

        # set default "camera" view

        camera=dict(eye=dict(x=1.4, y=1.5, z=0.3))

    ),

)



# Use Scater3d function

data = [

    go.Scatter3d(x=df.RM,

                 y=df.LSTAT,

                 z=df.DIS,

                 mode="markers",

                 marker=dict(size=6,

                             color=df.target,

                             colorscale="Viridis"

                            )

                )

]



# Show figure

py.offline.iplot(go.Figure(data=data,layout=layout))
layout = go.Layout(

    # Figure size ration will be ignored 

    width=600,

    height=350,

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(l=50,r=30,t=30,b=30),

    # figure title and font size

    title="Plotly select histogram", 

    font=dict(size=11),

    # Force to plot in the same axis

    barmode="overlay",

    yaxis=go.layout.YAxis(

        title="probability"

    ),

    xaxis=go.layout.XAxis(

        title="unscaled value"

    ),

    # Legend name is set (as usual) inside "data"

)

# Loop over all columns setting transparency and normalized histogram for each

# Set default visibility to selected columns and constant number of bins

data = [go.Histogram(x=df[i],

                     opacity=0.75,

                     nbinsx = np.int(np.sqrt(df.shape[0])),

                     histnorm="probability",

                     visible = True if i in ["target","LSTAT","AGE"] else "legendonly",

                     name = i) for i in df.columns];



# Show the plot

py.offline.iplot(go.Figure(data=data, layout=layout))
bplot = StandardScaler().fit_transform(df.values);
layout = go.Layout(

    # Figure size ration will be ignored 

    width=600,

    height=350,

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(l=50,r=30,t=30,b=30),

    # figure title and font size

    title="Plotly boxplot", 

    font=dict(size=11),

    yaxis=go.layout.YAxis(

        title="scaled values"

    )

)

# Loop over all columns setting transparency and normalized histogram for each

# Set default visibility to selected columns and constant number of bins

data = [go.Box(y=bplot[:,i],

               name = df.columns[i]) for i in range(0,bplot.shape[1])];



# Show the plot

py.offline.iplot(go.Figure(data=data, layout=layout))
# Use pandas methods to compute correlation matrix for all columns, 

# round the result to 3 decimals and show the dataframe with background color

df.corr().round(3).style.background_gradient(cmap="viridis")
layout = go.Layout(

    width=600,

    height=600,

    # Reduce default (rather big) margins between Figure edges and axes

    margin=go.layout.Margin(l=50,r=40,t=40,b=50),

    # figure title and font size

    title="Plotly correlation heatmap", 

    font=dict(size=11)

)

# flip the numpy correlation matrix upside down to get correct labeling. 

data = [

    go.Heatmap(x=df.columns,

               y=df.columns,

               z=np.flipud(np.corrcoef(df.values.T)),

               colorscale = "Viridis")

];

# Show the plot

py.offline.iplot(go.Figure(data=data, layout=layout))