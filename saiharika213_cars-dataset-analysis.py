import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt 

import plotly.express as px

import plotly

import plotly.graph_objects as go

plotly.offline.init_notebook_mode()
df1 = pd.read_csv('../input/car-dataset/car-mpg.csv')

df1.head()#printing the first 5 rows to view the data
#printing the shape of the data

print('Shape of dataframe:',df1.shape)

print('Number of rows: ',df1.shape[0])

print('Number of Columns: ',df1.shape[1])
#printing information of columns type and how many rows are there and the memory size utilized

df1.info()
#As we noted ['Hp'] attribute values are missing
df1.replace(to_replace='?',value=np.NAN, inplace=True)
df1[df1.hp.isna()==True]
df1.dropna(inplace=True)
df1[df1.hp.isna()==True]
# we see the different attributes of car data set
df1.columns
# replacing column names
new_names =  {'mpg': 'Mile_per_gallon',

               'cyl': 'Cylinder',

               'disp': 'Display',

                'hp': 'Horse_Power',

               'wt': 'Weight',

               'acc': 'Acceleration',

               'yr': 'Year',

               'origin': 'Car_origin',

               'car_name': 'Car_Name'}
df1.rename(columns=new_names, inplace=True)

df1.head()
df1.isna().sum()
df1["Car_Name"].is_unique
#describe the summary statistics of the data



df1.describe()
#no of cars based on the car origin

dfcar_origin = df1.Car_origin.value_counts()

dfcar_origin
#no of cars based on year 

dfcar_yr_count = df1.Year.value_counts()

dfcar_yr_count
unique_Horse_power = df1.Horse_Power.unique()

print('No .of Horse_Power ',unique_Horse_power.size)

unique_Acceleration = df1.Acceleration.unique()

print('No .of Acceleration ',unique_Acceleration.size)

unique_Weight = df1.Weight.unique()

print('No .of Weight ',unique_Weight.size)

unique_Display = df1.Display.unique()

print('No .of Display ',unique_Display.size)

unique_Cylinder = df1.Cylinder.unique()

print('No .of Cylinder ',unique_Cylinder.size)

unique_Car_Name = df1.Car_Name.unique()

print('No .of Car_Name ',unique_Car_Name.size)

unique_Car_origin = df1.Car_origin.unique()

print('No .of Car_origin ',unique_Car_origin.size)

unique_Year = df1.Year.unique()

print('No .of Year ',unique_Year.size)

unique_Mile_per_gallon = df1.Mile_per_gallon.unique()

print('No .of Mile_per_gallon ',unique_Mile_per_gallon.size)
#looking at the above no of unique values for each column the best 

#grouping column can be done on year, car origin and cylinder
#grouping on year and car_origin

dfgyear = df1.groupby(by=['Year'])

dfgyear.count() 
#grouping on year and car_origin

dfgcylinder = df1.groupby(by=['Cylinder'])

dfgcylinder.count() 
dfgCar_origin = df1.groupby(by=['Car_origin'])

dfgCar_origin.count() 
#grouping on horse_power

dfg = df1.groupby(by='Horse_Power')

dfg.count()
#grouping on horse power results more no of groouped data where as Horse power is not a categorical data

dfghmax = df1.groupby(by='Horse_Power').max()

dfghmax
dfghmin = df1.groupby(by='Horse_Power').min()

dfghmin
dfghmean = df1.groupby(by='Horse_Power').mean()

dfghmean
#grouping on Mile_per_gallon results more no of grouped data where as Mile_per_gallon is not a categorical data

dfgmmax = df1.groupby(by='Mile_per_gallon').max()

dfgmmax
#Ploting the Count of Cars based on origin


fig = go.Figure(go.Pie(

    name = '',

    values = dfcar_origin,

    labels = dfcar_origin.index, 

    text =  dfcar_origin.index,texttemplate = "%{label}:<br> Count:%{value:s} <br>(%{percent})",

    hovertemplate =  "Car Origin:%{label} <br>Count: %{value:s} </br> Percent:%{percent}" ,

    title = 'Count of Cars based on car origin'

))

fig.update_layout(

    autosize=False,

    width=500,

    height=500

)



fig.show() 
#Ploting the Count of Cars based on Years
dfcar_yr_count.index.name='Year'

fig = go.Figure(go.Pie(

    name = '',

    values = dfcar_yr_count,

    labels = dfcar_yr_count.index, 

    text =  dfcar_yr_count.index,texttemplate = "%{label}:<br> Count:%{value:s} <br>(%{percent})",

    hovertemplate =  "Year:%{label} <br>Count: %{value:s} </br> Percent:%{percent}" ,

    title = 'Count Of Cars Year Wise'

))

fig.update_layout(

    autosize=False,

    width=600,

    height=600

)



fig.show() 
#Plotting the box plot for Horse power to see   min max and outliers 

fig = px.box(data_frame= df1, y="Horse_Power",title='Box Plot for Horse Power')

fig.show()
#Plotting the box plot for Miles pergallon to see   min max and outliers 

fig = px.box(data_frame= df1, y="Mile_per_gallon",title='Box Plot for Mile Per Gallon')

fig.show()

fig = px.scatter(data_frame= df1, y= "Horse_Power", x="Cylinder",title='Plot b/w Horse Power and Cylinder')

fig.show()
df_car_8 = df1[(df1['Horse_Power']=='230')& (df1['Cylinder']==8) ]

df_car_6 = df1[(df1['Horse_Power']=='165') & (df1['Cylinder']==6)]

df_car_5 = df1[(df1['Horse_Power']=='103') & (df1['Cylinder']==5)]

df_car_4 = df1[(df1['Horse_Power']=='115') & (df1['Cylinder']==4)]

df_car_3 = df1[(df1['Horse_Power']=='110') & (df1['Cylinder']==3)]

df_sprt_car_cyl = pd.concat([df_car_8,df_car_6,df_car_5,df_car_4,df_car_3])

df_sprt_car_cyl
fig = px.bar(df_sprt_car_cyl, x='Cylinder', y='Horse_Power',

             labels ={'Car_Name' : 'Car Name', 'Car_origin': 'Origin Of Car','Horse_Power':'Horse Power','Mile_per_gallon':'Mile Per Gallon'},

             text = 'Horse_Power',hover_data=['Car_Name','Car_origin','Mile_per_gallon','Year'],

             height=600,orientation='v',

             title='Top Sports Car in Each Cylinder')



fig.update_traces(texttemplate='%{text:s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
#As per the Probelm statment to figure out the best sports car form 

#the attributes consider with max Mile Per Gallon for each cylinder type
fig = px.scatter(data_frame= df1, y= "Mile_per_gallon", x="Cylinder",title='Plot b/w Mile Per Gallon and Cylinder')

fig.show()
df_car_8 = df1[(df1['Mile_per_gallon']==26.6)& (df1['Cylinder']==8) ]

df_car_6 = df1[(df1['Mile_per_gallon']==38) & (df1['Cylinder']==6)]

df_car_5 = df1[(df1['Mile_per_gallon']==36.4) & (df1['Cylinder']==5)]

df_car_4 = df1[(df1['Mile_per_gallon']==46.6) & (df1['Cylinder']==4)]

df_car_3 = df1[(df1['Mile_per_gallon']==23.7) & (df1['Cylinder']==3)]

df_economic_car_cyl = pd.concat([df_car_8,df_car_6,df_car_5,df_car_4,df_car_3])

df_economic_car_cyl
fig = px.bar(df_economic_car_cyl, x='Cylinder', y='Mile_per_gallon',

             labels ={'Car_Name' : 'Car Name', 'Car_origin': 'Origin Of Car','Horse_Power':'Horse Power','Mile_per_gallon':'Mile Per Gallon'},

             text = 'Mile_per_gallon',hover_data=['Car_Name','Car_origin','Horse_Power','Year'],

             height=600,

             title='Top Economical Cars In Each Cylinder Category')



fig.update_traces(texttemplate='%{text:s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.scatter(data_frame= df1, y= "Mile_per_gallon", x="Car_origin",title='Best Milage Car in Origin Wise',

                hover_data ={'Cylinder','Car_Name','Horse_Power','Year'})

fig.show()
fig = px.scatter(data_frame= df1, y= "Horse_Power", x="Car_origin",title='Best Horse Power providing Car in Origin Wise',

                hover_data ={'Cylinder','Car_Name','Mile_per_gallon','Year'})

fig.show()