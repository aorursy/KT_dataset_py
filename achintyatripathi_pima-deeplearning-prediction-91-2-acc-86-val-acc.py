# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data viz. and EDA
import matplotlib.pyplot as plt 
%matplotlib inline  
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)

## For scaling data 
from mlxtend.preprocessing import minmax_scaling 

# Tensorflow 
import tensorflow as tf


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

# checking missing values if any
display(data.info(),data.head())
## lets see how many are affected by diabeties 
D = data[data['Outcome'] == 1]
H = data[data['Outcome'] == 0]

## here I am using graph_obs as I am not able to costimize px. 

def target_count():
    trace = go.Bar( x = data['Outcome'].value_counts().values.tolist(), 
                    y = ['healthy','diabetic' ], 
                    orientation = 'h', 
                    text=data['Outcome'].value_counts().values.tolist(), 
                    textfont=dict(size=15),
                    textposition = 'auto',
                    opacity = 0.5,marker=dict(
                    color=['lightskyblue', ' indigo'],
                    line=dict(color='#000000',width=1.5)))

    layout = dict(title =  'Count of affectes females')

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)

# --------------- donut chart to show there percentage -------------------- # 

def target_per():
    trace = go.Pie(labels=['healthy','diabetic' ],values=data['Outcome'].value_counts(),
                   textfont=dict(size=15),
                   opacity = 0.5,marker=dict(
                   colors=['lightskyblue','indigo'],line=dict(color='#000000', width=1.5)),
                   hole=0.6
                  )
    layout = dict(title='Donut chart to see the %age of affected.')
    fig = dict(data=[trace],layout=layout)
    py.iplot(fig)
target_count()
target_per()
## As seen earlier there is no null value. However on close inspection we find that null values are filled with '0'

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)    
## Checking the new null values found.
data.isnull().sum()
# Define missing plot to detect all missing values in dataset
def missing_plot(dataset, key) :
    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])
    percentage_null = pd.DataFrame((dataset.isnull().sum())/len(dataset[key])*100, columns = ['Count'])
    percentage_null = percentage_null.round(2)

    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'],  textposition = 'auto',marker=dict(color = '#7EC0EE',
            line=dict(color='#000000',width=1.5)))

    layout = dict(title =  "Missing Values (count & %)")

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)
    
missing_plot(data,'Outcome')
## to find the median for filling null values

def find_median(var):
    temp = data[data[var].notnull()]
    temp = data[[var,'Outcome']].groupby('Outcome')[[var]].median().reset_index()
    return temp
def density_plot(var,size_bin):
    tmp1 = D[var]
    tmp2 = H[var]
    
    hist_data = [tmp1,tmp2]
    labels = ['Diabeties','Healthy']
    color = ['skyblue','indigo']
    fig = ff.create_distplot(hist_data,labels,colors = color,show_hist=True,bin_size=size_bin,curve_type='kde')
    
    fig['layout'].update(title = var)

    py.iplot(fig, filename = 'Density plot')
    
density_plot('Insulin',0)
find_median('Insulin')
## Now we will be filling these values instead of null values

data.loc[(data['Outcome'] == 0) & (data['Insulin'].isnull()), 'Insulin'] = 102.5
data.loc[(data['Outcome'] == 1) & (data['Insulin'].isnull()), 'Insulin'] = 169.5
# SkinThickness density plot 

density_plot('SkinThickness',0)
find_median('SkinThickness')
## Now we will be filling these values instead of null values

data.loc[(data['Outcome'] == 0) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27.0
data.loc[(data['Outcome'] == 1) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32.0
density_plot('BloodPressure',0)
find_median('BloodPressure')
data.loc[(data['Outcome'] == 0) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 27.0
data.loc[(data['Outcome'] == 1) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 32.0
density_plot('BMI',0)
find_median('BMI')
data.loc[(data['Outcome'] == 0) & (data['BMI'].isnull()), 'BMI'] = 30.1
data.loc[(data['Outcome'] == 1) & (data['BMI'].isnull()), 'BMI'] = 34.3
density_plot('Glucose',0)
find_median('Glucose')
data.loc[(data['Outcome'] == 0) & (data['Glucose'].isnull()) , 'Glucose'] = 107.0
data.loc[(data['Outcome'] == 1) & (data['Glucose'].isnull()) , 'Glucose'] = 140.0
## lets check if any null value is still left

display(data.isnull().sum())
def correlation_plot():
    #correlation
    correlation = data.corr()
    #tick labels
    matrix_cols = correlation.columns.tolist()
    #convert to array
    corr_array  = np.array(correlation)
    trace = go.Heatmap(z = corr_array,
                       x = matrix_cols,
                       y = matrix_cols,
                       colorscale='Viridis',
                       colorbar   = dict() 
                      )
    layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                            #autosize = False,
                            #height  = 1400,
                            #width   = 1600,
                            margin  = dict(r = 0 ,l = 100,
                                           t = 0,b = 100,
                                         ),
                            yaxis   = dict(tickfont = dict(size = 9)),
                            xaxis   = dict(tickfont = dict(size = 9)),
                           )
                      )
    fig = go.Figure(data = [trace],layout = layout)
    py.iplot(fig)
correlation_plot()
def plot_feat1_feat2(feat1, feat2) :  
    D = data[(data['Outcome'] != 0)]
    H = data[(data['Outcome'] == 0)]
    trace0 = go.Scatter(
        x = D[feat1],
        y = D[feat2],
        name = 'diabetic',
        mode = 'markers', 
        opacity=0.8,
        marker = dict(color = 'lightskyblue',
            line = dict(
                width = 1)))

    trace1 = go.Scatter(
        x = H[feat1],
        y = H[feat2],
        name = 'healthy',
        opacity=0.8,
        mode = 'markers',
        marker = dict(color = 'indigo',
            line = dict(
                width = 1)))

    layout = dict(title = feat1 +" "+"vs"+" "+ feat2,
                  yaxis = dict(title = feat2,zeroline = False),
                  xaxis = dict(title = feat1, zeroline = False)
                 )

    plots = [trace0, trace1]

    fig = dict(data = plots, layout=layout)
    py.iplot(fig)

plot_feat1_feat2('Pregnancies', 'Age')
plot_feat1_feat2('Glucose', 'Insulin')

plot_feat1_feat2('SkinThickness', 'BMI')
scaled_data = minmax_scaling(data,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
def build_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=[len(scaled_data.keys())]),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = build_model()
model.summary()
EPOCHS = 1000

history = model.fit(scaled_data, data['Outcome'],epochs=EPOCHS, validation_split=0.2, verbose=2)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
acc = (hist['accuracy'].tail().sum())*100/5 
val_acc = (hist['val_accuracy'].tail().sum())*100/5 

print("Training Accuracy = {}% and Validation Accuracy= {}%".format(acc,val_acc))