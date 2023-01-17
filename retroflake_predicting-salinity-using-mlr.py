#Importing all the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings('ignore')
#Let's read the data from the file

df = pd.read_csv('../input/bottle.csv')

df = pd.DataFrame(df)

df.shape
#As we can see, the data has myriad rows

#So to reduce the number of rows for our model, we'll use 10% of them

df = df.sample(frac=0.1, random_state = 123)
df.head()
#Now that those rows are reduced we see that there are a lot of entries with NaN values



#print(df[df.columns[0]].dtype)

#df.dtypes



#Let's try removing them

df.dropna()
#What!!!!? Every row from our randomly selected 10% data has atleast 1 NaN value



#df.fillna(value=0, inplace=True)



#We need a better approach than substituting every entry with 0

#Let's first remove the columns with "object" 

temp_df=df

for i in temp_df.columns:

    if(df[i].dtype=='object'):

        del df[i]

    

df.shape
#As we can see we are now left with 70 columns 



#print(len(df.columns))



#Now, let's replace them with the mean value of their respective columns

for i in df.columns:

    df[i].fillna((df[i].mean()),inplace=True)

    

#print(df[df.columns[0]])
df.head()
#Boom, all NaN values replaced with mean values



#Plotting a Heatmap now, yes a heatmap for 70 features 

import seaborn as sns

cor = df.corr()

fig = plt.figure(figsize = (24, 18))



#Plotting the heatmap

sns.heatmap(cor, cmap="YlGnBu")

plt.show()
#Since we are predicting Salinity of water, let's select the columns which show correlation 

df.columns
#Selecting the target and feature variables

target='Salnty'

features =['Depthm', 'T_degC','O2ml_L', 'STheta','O2Sat', 'Oxy_µmol/Kg','PO4uM', 'SiO3uM','NH3uM','R_Depth', 'R_TEMP', 'R_POTEMP',

       'R_SALINITY', 'R_SIGMA', 'R_SVA', 'R_DYNHT', 'R_O2', 'R_O2Sat','R_SIO3', 'R_PO4', 'R_NO3','R_PRES']

x=df[features]

y=df[target]

print(x.shape,y.shape)
#Splitting them into test and training set with a 3:1 ratio

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(x_train.shape,y_train.shape)
#Hello Linear Regression!

clf = LinearRegression()

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print(accuracy*100)
#Over 99.99% accuracy, I wonder if I made any mistake or it's just great :)



# Make predictions using the new model

y_pred = clf.predict(x_test)

for i in range(10):

    print('Actual value: {:.3f} Predicted Value: {:.3f}'.format(y_test.values[i],y_pred[i]))

#As expected, the actual and the predicted values are almost same



#Let's make an interactive visualization of predicted and actual values

trace=go.Scatter(x=y_test.values, y=y_pred,marker=dict(color='red'))
layout = go.Layout(

    title=go.layout.Title(

        text='Predicted Vs Actual Salinity ',

        xref='paper',

        x=0

    ),

    xaxis=go.layout.XAxis(

        title=go.layout.xaxis.Title(

            text='Actual Values',

            font=dict(

                family='Courier New, monospace',

                size=18,

                color='#7f7f7f'

            )

        )

    ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text='Predicted Values',

            font=dict(

                family='Courier New, monospace',

                size=18,

                color='#7f7f7f'

            )

        )

    )

)



fig = go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='results')
#Feel free to hover your cursor over the trace line :)