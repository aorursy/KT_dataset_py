# Importing numpy, pandas and Series + DataFrame:

import numpy as np

import pandas as pd

from pandas import Series, DataFrame



# Imports for plotly:

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



# Imports for plotting:

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# Importing breast cancer dataset:

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
# Check first 5 rows of dataset:

df.head()
# Function to describe variables

def desc(df):

    d = pd.DataFrame(df.dtypes,columns=['Data_Types'])

    d = d.reset_index()

    d['Columns'] = d['index']

    d = d[['Columns','Data_Types']]

    d['Missing'] = df.isnull().sum().values    

    d['Uniques'] = df.nunique().values

    return d



# Apply function on df:

desc(df)
# Preview of diagnosis :

print(df.diagnosis.unique())
df['target'] = df.diagnosis.map({'B':0, 'M':1})

df = pd.DataFrame(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1))
trg_df = pd.DataFrame(df.groupby(['target'])['target'].count())



# Target distribution:





data=go.Bar(x = trg_df.index

           , y = trg_df.target

           ,  marker=dict( color=['#3198b7', '#fd6190'])

           , text=trg_df.target

           , textposition='auto' 

           )







layout = go.Layout(title = 'Target distribution'

                   , xaxis = dict(title = 'Target')

                   , yaxis = dict(title = 'Volume')

                  )



fig = go.Figure(data,layout)

fig.show()
# Plotting the features of our dataset (this gives us density graphs and scatter plots): 



cols = ['radius', 'texture', 'perimeter', 'area','smoothness', 'compactness', 'concavity','concave points', 'symmetry', 'fractal', 'target']



df_mean = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean', 'compactness_mean',

              'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','target']]

df_mean.columns = cols





sns.set(style="ticks")

sns.pairplot(df_mean[cols]

             , hue='target'

             , palette=['#3198b7', '#fd6190']

             , diag_kind = 'kde'

             #, height = 2

             , corner = True)

plt.show()
df['target'] = df.target.astype(str)



fig = px.scatter(df

                 , x='area_mean'

                 , y='smoothness_mean'

                 , color='target'

                 , size='perimeter_mean'

                 , color_discrete_sequence=['#fd6190', '#3198b7']

                 , title='Scatter for Area vs Smoothness (sized by Perimeter)'

                )

fig.show()
fig = px.scatter(df

                 , x='concave points_mean'

                 , y=  'fractal_dimension_mean'         

                 , color='target'

                 , size='concavity_mean'

                 , color_discrete_sequence=[ '#fd6190', '#3198b7']

                 , title='Scatter for Concave Points vs Fractal Dimension (sized by Concavity)'

                )

fig.show()
fig = px.scatter(df

                 , x='concave points_mean'

                 , y=  'concavity_mean'     

                 , color='target'

                 , size='area_mean'

                 , color_discrete_sequence=[ '#fd6190', '#3198b7']

                 , title='Scatter for Concave Points vs Concavity (sized by Area)'

                )

fig.show()
# Boxplot with dropdown menu for Mean Features:



fig = go.Figure()



# Add Traces



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.radius_mean[df.target=='0'], name='Radius (0)', fillcolor='#3198b7'))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.radius_mean[df.target=='1'], name='Radius (1)', fillcolor='#fd6190'))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.texture_mean[df.target=='0'], name='Texture (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.texture_mean[df.target=='1'], name='Texture (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.perimeter_mean[df.target=='0'], name='Perimeter (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.perimeter_mean[df.target=='1'], name='Perimeter (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.area_mean[df.target=='0'], name='Area (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.area_mean[df.target=='1'], name='Area (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.smoothness_mean[df.target=='0'], name='Smoothness (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.smoothness_mean[df.target=='1'], name='Smoothness (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.compactness_mean[df.target=='0'], name='Compactness (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.compactness_mean[df.target=='1'], name='Compactness (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.concavity_mean[df.target=='0'], name='Concavity (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.concavity_mean[df.target=='1'], name='Concavity (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.fractal_dimension_mean[df.target=='0'], name='Fractal Dimention (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.fractal_dimension_mean[df.target=='1'], name='Fractal Dimention (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.symmetry_mean[df.target=='0'], name='Symmetry (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box( x=df.target[df.target=='1'], y=df.symmetry_mean[df.target=='1'], name='Symmetry (1)', fillcolor='#fd6190', visible=False))  





# Add Buttons



fig.update_layout(

    updatemenus=[

        dict(

            active=0,

            buttons=list([ 

                

                dict(label='Radius',

                     method='update',

                     args=[{'visible': [True, True, False, False, False, False,False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Radius)"}]),

                

                dict(label='Texture',

                     method='update',

                     args=[{'visible': [False, False, True, True, False, False,False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Texture)"}]),

                

                dict(label='Perimeter',

                     method='update',

                     args=[{'visible': [False, False, False, False, True, True, False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Perimeter)"}]),

                

                dict(label='Area',

                     method='update',

                     args=[{'visible': [False, False, False, False,False, False, True, True, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Area)"}]),

                

                dict(label='Smoothness',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False,True, True, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Smoothness)"}]),

                

                dict(label='Compactness',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, True, True, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Compactness)"}]),

                

                dict(label="Concavity",

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False ,False, False, False, False, True, True, False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Concavity)"}]),

                

                dict(label='Fractal Dimention',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, False, False,False, False, True, True, False, False]},

                           {"title": "Boxplot for Malignant & Benign (MEAN Fractal Dimention)"}]),

                

                dict(label='Symmetry',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True]},

                           {'title': "Boxplot for Malignant & Benign (MEAN Symmetry)"}]),                



            ]),

        )

    ])



# Set title

fig.update_layout(title_text="Boxplot for Malignant & Benign (MEAN Features)")



fig.show()
# Boxplot with dropdown menu for SE Features:



fig = go.Figure()



# Add Traces



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.radius_se[df.target=='0'], name='Radius (0)', fillcolor='#3198b7'))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.radius_se[df.target=='1'], name='Radius (1)', fillcolor='#fd6190'))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.texture_se[df.target=='0'], name='Texture (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.texture_se[df.target=='1'], name='Texture (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.perimeter_se[df.target=='0'], name='Perimeter (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.perimeter_se[df.target=='1'], name='Perimeter (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.area_se[df.target=='0'], name='Area (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.area_se[df.target=='1'], name='Area (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.smoothness_se[df.target=='0'], name='Smoothness (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.smoothness_se[df.target=='1'], name='Smoothness (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.compactness_se[df.target=='0'], name='Compactness (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.compactness_se[df.target=='1'], name='Compactness (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.concavity_se[df.target=='0'], name='Concavity (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.concavity_se[df.target=='1'], name='Concavity (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.fractal_dimension_se[df.target=='0'], name='Fractal Dimention (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.fractal_dimension_se[df.target=='1'], name='Fractal Dimention (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.symmetry_se[df.target=='0'], name='Symmetry (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.symmetry_se[df.target=='1'], name='Symmetry (1)', fillcolor='#fd6190', visible=False))  





# Add Buttons



fig.update_layout(

    updatemenus=[

        dict(

            active=0,

            buttons=list([ 

                

                dict(label='Radius',

                     method='update',

                     args=[{'visible': [True, True, False, False, False, False,False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Radius)"}]),

                

                dict(label='Texture',

                     method='update',

                     args=[{'visible': [False, False, True, True, False, False,False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Texture)"}]),

                

                dict(label='Perimeter',

                     method='update',

                     args=[{'visible': [False, False, False, False, True, True, False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Perimeter)"}]),

                

                dict(label='Area',

                     method='update',

                     args=[{'visible': [False, False, False, False,False, False, True, True, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Area)"}]),

                

                dict(label='Smoothness',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False,True, True, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Smoothness)"}]),

                

                dict(label='Compactness',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, True, True, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Compactness)"}]),

                

                dict(label="Concavity",

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False ,False, False, False, False, True, True, False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (SE Concavity)"}]),

                

                dict(label='Fractal Dimention',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, False, False,False, False, True, True, False, False]},

                           {"title": "Boxplot for Malignant & Benign (SE Fractal Dimention)"}]),

                

                dict(label='Symmetry',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True]},

                           {'title': "Boxplot for Malignant & Benign (SE Symmetry)"}]),

                



            ]),

        )

    ])



# Set title

fig.update_layout(title_text="Boxplot for Malignant & Benign (SE Features)")



fig.show()
# Boxplot with dropdown menu for WORST Features:



fig = go.Figure()



# Add Traces



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.radius_worst[df.target=='0'], name='Radius (0)', fillcolor='#3198b7'))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.radius_worst[df.target=='1'], name='Radius (1)', fillcolor='#fd6190'))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.texture_worst[df.target=='0'], name='Texture (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.texture_worst[df.target=='1'], name='Texture (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.perimeter_worst[df.target=='0'], name='Perimeter (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.perimeter_worst[df.target=='1'], name='Perimeter (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.area_worst[df.target=='0'], name='Area (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.area_worst[df.target=='1'], name='Area (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.smoothness_worst[df.target=='0'], name='Smoothness (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.smoothness_worst[df.target=='1'], name='Smoothness (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.compactness_worst[df.target=='0'], name='Compactness (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.compactness_worst[df.target=='1'], name='Compactness (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.concavity_worst[df.target=='0'], name='Concavity (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.concavity_worst[df.target=='1'], name='Concavity (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.fractal_dimension_worst[df.target=='0'], name='Fractal Dimention (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.fractal_dimension_worst[df.target=='1'], name='Fractal Dimention (1)', fillcolor='#fd6190', visible=False))  



fig.add_trace(

    go.Box(x=df.target[df.target=='0'], y=df.symmetry_worst[df.target=='0'], name='Symmetry (0)', fillcolor='#3198b7', visible=False))

fig.add_trace(

    go.Box(x=df.target[df.target=='1'], y=df.symmetry_worst[df.target=='1'], name='Symmetry (1)', fillcolor='#fd6190', visible=False))  





# Add Buttons



fig.update_layout(

    updatemenus=[

        dict(

            active=0,

            buttons=list([ 

                

                dict(label='Radius',

                     method='update',

                     args=[{'visible': [True, True, False, False, False, False,False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Radius)"}]),

                

                dict(label='Texture',

                     method='update',

                     args=[{'visible': [False, False, True, True, False, False,False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Texture)"}]),

                

                dict(label='Perimeter',

                     method='update',

                     args=[{'visible': [False, False, False, False, True, True, False, False, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Perimeter)"}]),

                

                dict(label='Area',

                     method='update',

                     args=[{'visible': [False, False, False, False,False, False, True, True, False, False, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Area)"}]),

                

                dict(label='Smoothness',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False,True, True, False, False, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Smoothness)"}]),

                

                dict(label='Compactness',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, True, True, False, False,False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Compactness)"}]),

                

                dict(label="Concavity",

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False ,False, False, False, False, True, True, False, False, False, False]},

                           {'title': "Boxplot for Malignant & Benign (WORST Concavity)"}]),

                

                dict(label='Fractal Dimention',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, False, False,False, False, True, True, False, False]},

                           {"title": "Boxplot for Malignant & Benign (WORST Fractal Dimention)"}]),

                

                dict(label='Symmetry',

                     method='update',

                     args=[{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True]},

                           {'title': "Boxplot for Malignant & Benign (WORST Symmetry)"}]),

                



            ]),

        )

    ])



# Set title

fig.update_layout(title_text="Boxplot for Malignant & Benign (WORST Features)")



fig.show()
# Correlation matrix for Wisconsin dataset features:

df['target'] = df.target.astype(int)



corr = df.corr()



fig = go.Figure(data=go.Heatmap(

                   z=corr

                 , x=df.columns

                 , y=df.columns

                 , hoverongaps = False

                 , colorscale= 'Sunsetdark'

))



fig.update_layout(title='Correlation for Features of Wisconsin Dataset')





fig.show()
# Define input values, or X by dropping the target values:

X = df.drop(['target'], axis = 1)



# Define output values - this is the target:

y = df['target']
# For splitting data we will be using train_test_split from sklearn:

from sklearn.model_selection import train_test_split
# Splitting the data into test and train, we are testing on 0.2 = 20% of dataset:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Imports for training data:

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix



model = SVC()
# Training the model:

model.fit(X_train,y_train)
# Precictions for X_test:

y_predict = model.predict(X_test)
print(classification_report(y_test,y_predict))
p = confusion_matrix(y_test,y_predict)



q = ['0', '1']

r = ['0', '1']



# change each element of z to type string for annotations

z_text = [[str(r) for r in q] for q in p]



# set up figure 

fig = ff.create_annotated_heatmap(p, x=q, y=r, annotation_text=z_text, colorscale='Portland')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',

                  #xaxis = dict(title='x'),

                  #yaxis = dict(title='x')

                 )



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))



# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.35,

                        y=0.5,

                        showarrow=False,

                        text="Real value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=200))



# add colorbar

fig['data'][0]['showscale'] = True

fig.show()

# Perform normalisation:

minx = X.min()

rangex = (X - minx).max()

X_scaled = (X - minx)/rangex
# Splitting the data into test and train, we are testing on 0.2 = 20% of dataset:

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
fig = px.scatter(X_train

                 , x='area_mean'

                 , y=  'smoothness_mean'     

                 , color=y_train.astype(str)

                 , color_discrete_sequence=[  '#3198b7', '#fd6190']

                 , title='Training data (Not Normalised)'

                )

fig.show()
fig = px.scatter(X_train_scaled

                 , x='area_mean'

                 , y='smoothness_mean'     

                 , color=y_train.astype(str)

                 , color_discrete_sequence=[  '#3198b7', '#fd6190']

                 , title='Training data (Normalised)'

                )

fig.show()
# Training the model:

model.fit(X_train_scaled,y_train)
y_predict = model.predict(X_test_scaled)
print(classification_report(y_test,y_predict))
p = confusion_matrix(y_test,y_predict)



q = ['0', '1']

r = ['0', '1']



# change each element of z to type string for annotations

z_text = [[str(r) for r in q] for q in p]



# set up figure 

fig = ff.create_annotated_heatmap(p, x=q, y=r, annotation_text=z_text, colorscale='Portland')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',

                  #xaxis = dict(title='x'),

                  #yaxis = dict(title='x')

                 )



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))



# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.35,

                        y=0.5,

                        showarrow=False,

                        text="Real value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=200))



# add colorbar

fig['data'][0]['showscale'] = True

fig.show()