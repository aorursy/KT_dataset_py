# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import all the necessary libraries here

import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import plotly.offline as pyoff

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go

%matplotlib inline

import math

def generate_layout_bar(col_name):

    layout_bar = go.Layout(

        autosize=False, # auto size the graph? use False if you are specifying the height and width

        width=1600, # height of the figure in pixels

        height=600, # height of the figure in pixels

        title = "Distribution of {} column".format(col_name), # title of the figure

        # more granular control on the title font 

        titlefont=dict( 

            family='Courier New, monospace', # font family

            size=14, # size of the font

            color='black' # color of the font

        ),

        # granular control on the axes objects 

        xaxis=dict( 

        tickfont=dict(

            family='Courier New, monospace', # font family

            size=14, # size of ticks displayed on the x axis

            color='black'  # color of the font

            )

        ),

        yaxis=dict(

#         range=[0,100],

            title='Percentage',

            titlefont=dict(

                size=14,

                color='black'

            ),

        tickfont=dict(

            family='Courier New, monospace', # font family

            size=14, # size of ticks displayed on the y axis

            color='black', # color of the font

            tickangle=45

            )

        ),

        font = dict(

            family='Courier New, monospace', # font family

            color = "white",# color of the font

            size = 12 # size of the font displayed on the bar

                )  

        )

    return layout_bar
init_notebook_mode(connected=True)
data=pd.read_excel("../input/Train.xlsx")
datacopy_master=data
data.head()
Records=data.shape[0]

Attributes=data.shape[1]

print("Number of Records in dataset = ",Records)

print("Number of Attributes in dataset = ",Attributes)

data['Suspicious'].describe()
value,count=np.unique(data['Suspicious'],return_counts=True)

percent=(count/Records)*100

print(np.asarray([value,count,percent]).T)
count
data1 = [

    go.Bar(

        x=value, # assign x as the dataframe column 'x'

        y=count,

        text=count,

        textposition='auto'

    )

]



layout = go.Layout(

    autosize=True,

    title='Distribution of the Suspicious column',

    xaxis=dict(title='Suspicious'),

    yaxis=dict(title='Count')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
data.head()
#sum of the total sales value

summation_of_total_Sales=data['TotalSalesValue'].sum()

summation_of_total_Sales
#sum of the total quantity

summation_of_total_Quantity=data['Quantity'].sum()

summation_of_total_Quantity
temp=data.groupby(['SalesPersonID']).sum()
top_ten_quantity_by_SalesPersonID=temp.sort_values(by='Quantity',ascending=False).head(10)

last_ten_quantity_by_SalesPersonID=temp.sort_values(by='Quantity',ascending=False).tail(10)
((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100)

((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100)

data1 = [

    go.Bar(

        x=top_ten_quantity_by_SalesPersonID.index, # assign x as the dataframe column 'x'

        y=((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),

        text=((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='Top performers based on Quantity',

        xaxis=dict(title='SalesPersonID'),

        yaxis=dict(title='Total Quantity')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
data1 = [

    go.Bar(

        x=last_ten_quantity_by_SalesPersonID.index, # assign x as the dataframe column 'x'

        y=((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),

        text=((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='Under performers based on Quantity',

        xaxis=dict(title='SalesPersonID'),

        yaxis=dict(title='Total Quantity')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
#top ten guys total contribution by quantity

((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100).sum()

#top ten guys contributed 13.948894548010024 percent to the total quantity sold.
#bottom ten guys total contribution by quantity

((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100).sum()

#Bottom ten guys contributed 0.00014734402037043685 percent to the total quantity sold.
temp=data.groupby(['SalesPersonID']).sum()
temp.head()
top_ten_TotalSales_by_SalesPersonID=temp.sort_values(by='TotalSalesValue',ascending=False).head(10)

last_ten_TotalSales_by_SalesPersonID=temp.sort_values(by='TotalSalesValue',ascending=False).tail(10)
last_ten_TotalSales_by_SalesPersonID
data1 = [

    go.Bar(

        x=top_ten_TotalSales_by_SalesPersonID.index, # assign x as the dataframe column 'x'

        y=((top_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),

        text=((top_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='Top performers based on TotalSales Percentage',

        xaxis=dict(title='SalesPersonID'),

        yaxis=dict(title='Total Sales')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
data1 = [

    go.Bar(

        x=last_ten_quantity_by_SalesPersonID.index, # assign x as the dataframe column 'x'

        y=((last_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),

        text=((last_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='least performers based on TotalSales Percentage',

        xaxis=dict(title='SalesPersonID'),

        yaxis=dict(title='Total Sales by percentage')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
top_ten_quantity_by_SalesPersonID.head()
data.head()
temp=data.groupby(['ProductID']).sum()
top_ten_quantity_by_ProductID=temp.sort_values(by='Quantity',ascending=False).head(10)

last_ten_quantity_by_ProductID=temp.sort_values(by='Quantity',ascending=False).tail(10)
top_ten_quantity_by_ProductID.head()
((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100)

((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100)

data1 = [

    go.Bar(

        x=top_ten_quantity_by_ProductID.index, # assign x as the dataframe column 'x'

        y=((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),

        text=((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='Top performers based on productID Quantity',

        xaxis=dict(title='ProductID'),

        yaxis=dict(title='Quantity percentage')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
sum(((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100))
data1 = [

    go.Bar(

        x=last_ten_quantity_by_ProductID.index, # assign x as the dataframe column 'x'

        y=((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),

        text=((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='last ten performers based on ProductID Quantity',

        xaxis=dict(title='ProductID'),

        yaxis=dict(title='Quantity percentage')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
sum(((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100))
top_ten_TotalSales_by_ProductID=temp.sort_values(by='TotalSalesValue',ascending=False).head(10)

last_ten_TotalSales_by_ProductID=temp.sort_values(by='TotalSalesValue',ascending=False).tail(10)
((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100)

((last_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100)

data1 = [

    go.Bar(

        x=top_ten_TotalSales_by_ProductID.index, # assign x as the dataframe column 'x'

        y=((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100),

        text=((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100),

        textposition='auto'

    )

]



layout = go.Layout(

        autosize=True,

        title='Top products perform based on TotalSales',

        xaxis=dict(title='Product ID'),

        yaxis=dict(title='Total Sales')

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
sum(((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100))
dataframe_Yes=data[data['Suspicious']=='Yes']
dataframe_No=data[data['Suspicious']=='No']
dataframe_Indeterminate=data[data['Suspicious']=='indeterminate']
print(dataframe_Yes.shape)

print(dataframe_No.shape)

print(dataframe_Indeterminate.shape)
data['PricePerUnit']=data.TotalSalesValue/data.Quantity

temp=data.groupby(['SalesPersonID','ProductID']).mean()['Quantity']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')

#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_guy_prdID

data=data.rename(index=str, columns={"Quantity_y": "Average_qty_guy_prdID","Quantity_x":"Quantity"})
temp=data.groupby(['SalesPersonID','ProductID']).mean()['TotalSalesValue']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')

#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and Average_TotalSalesValue_guy_prdID

data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSalesValue_guy_prdID","TotalSalesValue_x":"TotalSalesValue"})
temp=data.groupby(['SalesPersonID','ProductID']).mean()['PricePerUnit']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')
#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and Average_PricePerUnit_guy_prdID

data=data.rename(index=str, columns={"PricePerUnit_y": "Average_PricePerUnit_guy_prdID","PricePerUnit_x":"PricePerUnit"})
temp=data.groupby(['SalesPersonID']).mean()['Quantity']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_guy

data=data.rename(index=str, columns={"Quantity_y": "Average_qty_guy","Quantity_x":"Quantity"})
temp=data.groupby(['SalesPersonID']).mean()['TotalSalesValue']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')

#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and Average_TotalSalesValue_guy

data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSalesValue_guy","TotalSalesValue_x":"TotalSalesValue"})
temp=data.groupby(['SalesPersonID']).mean()['PricePerUnit']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and Average_PricePerUnit_guy

data=data.rename(index=str, columns={"PricePerUnit_y": "Average_PricePerUnit_guy","PricePerUnit_x":"PricePerUnit"})
temp=data.groupby(['ProductID']).mean()['Quantity']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['ProductID'],how='left')

#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_prdID

data=data.rename(index=str, columns={"Quantity_y": "Average_qty_prdID","Quantity_x":"Quantity"})
temp=data.groupby(['ProductID']).mean()['TotalSalesValue']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['ProductID'],how='left')

#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and Average_TotalSalesValue_guy_prdID

data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSalesValue_prdID","TotalSalesValue_x":"TotalSalesValue"})
temp=data.groupby(['ProductID']).mean()['PricePerUnit']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['ProductID'],how='left')
#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and Average_PricePerUnit_guy_prdID

data=data.rename(index=str, columns={"PricePerUnit_y": "Average_PricePerUnit_prdID","PricePerUnit_x":"PricePerUnit"})
temp=data.groupby(['SalesPersonID','ProductID']).std()['Quantity']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')

#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_guy_prdID

data=data.rename(index=str, columns={"Quantity_y": "std_qty_guy_prdID","Quantity_x":"Quantity"})
temp=data.groupby(['SalesPersonID','ProductID']).std()['TotalSalesValue']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')

#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and std_totalsalesvalue_guy_prdID

data=data.rename(index=str, columns={"TotalSalesValue_y": "std_TotalSalesValue_guy_prdID","TotalSalesValue_x":"TotalSalesValue"})
temp=data.groupby(['SalesPersonID','ProductID']).std()['PricePerUnit']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')

#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and std_PricePerUnit_guy_prdID

data=data.rename(index=str, columns={"PricePerUnit_y": "std_PricePerUnit_guy_prdID","PricePerUnit_x":"PricePerUnit"})
temp=data.groupby(['SalesPersonID']).std()['Quantity']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
#renaming the Quantity_x and Quantity_y columns to Quantity and std_qty_guy

data=data.rename(index=str, columns={"Quantity_y": "std_qty_guy","Quantity_x":"Quantity"})
temp=data.groupby(['SalesPersonID']).std()['TotalSalesValue']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and std_TotalSalesValue_guy

data=data.rename(index=str, columns={"TotalSalesValue_y": "std_TotalSalesValue_guy","TotalSalesValue_x":"TotalSalesValue"})
temp=data.groupby(['SalesPersonID']).std()['PricePerUnit']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and std_PricePerUnit_guy

data=data.rename(index=str, columns={"PricePerUnit_y": "std_PricePerUnit_guy","PricePerUnit_x":"PricePerUnit"})
temp=data.groupby(['ProductID']).std()['Quantity']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['ProductID'],how='left')
#renaming the Quantity_x and Quantity_y columns to Quantity and std_Quantity_prdID

data=data.rename(index=str, columns={"Quantity_y": "std_Quantity_prdID","Quantity_x":"Quantity"})
temp=data.groupby(['ProductID']).std()['TotalSalesValue']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['ProductID'],how='left')
#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and std_TotalSalesValue_prdID

data=data.rename(index=str, columns={"TotalSalesValue_y": "std_TotalSalesValue_prdID","TotalSalesValue_x":"TotalSalesValue"})
temp=data.groupby(['ProductID']).std()['PricePerUnit']

temp=pd.DataFrame(temp)

temp=temp.reset_index()
data=pd.merge(data,temp,on=['ProductID'],how='left')
#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and std_PricePerUnit_prdID

data=data.rename(index=str, columns={"PricePerUnit_y": "std_PricePerUnit_prdID","PricePerUnit_x":"PricePerUnit"})
data.head()
len(data.columns)
feature_engineered_dataSet=data.iloc[ : ,0:7]
feature_engineered_dataSet['diff_Average_qty_guy_prdID']=data['Quantity']-data['Average_qty_guy_prdID']

feature_engineered_dataSet['diff_Average_TotalSalesValue_guy_prdID']=data['TotalSalesValue']-data['Average_TotalSalesValue_guy_prdID']

feature_engineered_dataSet['diff_Average_PricePerUnit_guy_prdID']=data['PricePerUnit']-data['Average_PricePerUnit_guy_prdID']
feature_engineered_dataSet['diff_Average_qty_guy']=data['Quantity']-data['Average_qty_guy']

feature_engineered_dataSet['diff_Average_TotalSalesValue_guy']=data['TotalSalesValue']-data['Average_TotalSalesValue_guy']

feature_engineered_dataSet['diff_Average_PricePerUnit_guy']=data['PricePerUnit']-data['Average_PricePerUnit_guy']

feature_engineered_dataSet['diff_Average_qty_prdID']=data['Quantity']-data['Average_qty_prdID']

feature_engineered_dataSet['diff_Average_TotalSalesValue_prdID']=data['TotalSalesValue']-data['Average_TotalSalesValue_prdID']

feature_engineered_dataSet['diff_Average_PricePerUnit_prdID']=data['PricePerUnit']-data['Average_PricePerUnit_prdID']

data.head()
feature_engineered_dataSet.head()
feature_engineered_dataSet.columns
data.to_csv('basic_feature_engineering.csv',index=False)
feature_engineered_dataSet.to_csv('advanced_feature_engineering_using_mean_iteration_1.csv',index=False)
data.head()
feature_engineered_dataSet.head()