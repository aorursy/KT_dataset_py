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

        width=800, # height of the figure in pixels

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

            color='black' # color of the font

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
labels = 'No', 'Yes', 'Indeterminate'

explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Yes')



fig1, ax1 = plt.subplots()

ax1.pie(count, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

count[0]
data1 = [

    go.Bar(

        x=value, # assign x as the dataframe column 'x'

        y=count,

        text=count,

        textposition='auto'

    )

]



layout = go.Layout(

    barmode='stack',

    title='Stacked Bar with Pandas'

)



fig = go.Figure(data=data1, layout=layout)



# IPython notebook

iplot(fig)
plt.bar(np.unique(data['SalesPersonID']),data.groupby(['SalesPersonID']).mean()['Quantity'])



plt.ylim([0,120000])

plt.show()
datacopy=data
datacopy.head()
datacopy['new_Column']=0
temp=data.groupby(['SalesPersonID','ProductID']).mean()['Quantity']
temp = (pd.DataFrame(temp))
temp = temp.reset_index()
temp.columns
result=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')
result=result.drop(['new_Column'],axis=1)
data=data.drop(['new_Column'],axis=1)
result.head()
datacopy.head()
datacopy['new_Column']=datacopy.TotalSalesValue/datacopy.Quantity
datacopy.head()
data['price_Per_Unit']=data.TotalSalesValue/data.Quantity
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')
data=data.rename(index=str, columns={"Quantity_y": "Average_qty_/guy_/prdID"})

data.head()
temp=data.groupby(['SalesPersonID','ProductID']).mean()['TotalSalesValue']
temp = (pd.DataFrame(temp))
temp = temp.reset_index()
data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')
data.head()
data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSales_/guy_/prdID"})

data.head()
temp=data.groupby(['SalesPersonID']).mean()['TotalSalesValue_x']
temp=pd.DataFrame(temp)
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
data=data.rename(index=str, columns={"TotalSalesValue_x_y": "Average_TotalSales_/guy"})

temp=data.groupby(['SalesPersonID']).mean()['Quantity_x']

temp=pd.DataFrame(temp)
data=pd.merge(data,temp,on=['SalesPersonID'],how='left')
data=data.rename(index=str, columns={"Quantity_x_y": "Average_Quantity_/guy"})

temp=data.groupby(['ProductID']).mean()['Quantity_x_x']

temp=pd.DataFrame(temp)
data=pd.merge(data,temp,on=['ProductID'],how='left')
data=data.rename(index=str, columns={"Quantity_x_x_y": "Average_Quantity_/Product"})

temp=data.groupby(['ProductID']).mean()['TotalSalesValue_x_x']

temp=pd.DataFrame(temp)
data=pd.merge(data,temp,on=['ProductID'],how='left')
data=data.rename(index=str, columns={"TotalSalesValue_x_x_y": "Average_TotalSales_/ProductID"})

data.describe()
data.columns
data=data.rename(index=str, columns={'Average_qty_/guy_/prdID':'Average_qty_guy_prdID',

                                     'Average_TotalSales_/guy_/prdID':'Average_TotalSales_guy_prdID',

                                     'Average_TotalSales_/guy':'Average_TotalSales_guy',

                                     'Average_Quantity_/guy':'Average_Quantity_guy',

                                     'Average_Quantity_/Product':'Average_Quantity_Product',

                                     'Average_TotalSales_/ProductID':'Average_TotalSales_ProductID',

                                     'Quantity_x_x_x':'Quantity',

                                     'TotalSalesValue_x_x_x':'TotalSalesValue'})

data.head()
data.to_csv('feature_engineering.csv',index=False)
data.groupby('ProductID').mean()
ax = sns.scatterplot(x="SalesPersonID", y="price_Per_Unit", hue="Suspicious",data=data)
data_for_Yes=datacopy_master[datacopy_master.Suspicious=='Yes']
data_for_No=datacopy_master[datacopy_master.Suspicious=='No']
data_for_indeterminate=datacopy_master[datacopy_master.Suspicious=='indeterminate']
data_for_Yes.head()
temp=data_for_Yes.groupby(['SalesPersonID']).sum()

temp=pd.DataFrame(temp)

dataframe_with_Yes = temp.reset_index()

temp=data_for_No.groupby(['SalesPersonID']).sum()

temp=pd.DataFrame(temp)

dataframe_with_No = temp.reset_index()

temp=data_for_indeterminate.groupby(['SalesPersonID']).sum()

temp=pd.DataFrame(temp)

dataframe_with_indeterminate = temp.reset_index()

temp1=dataframe_with_Yes.sort_values('Quantity',ascending=False).head(10)

temp2=dataframe_with_No.sort_values('Quantity',ascending=False).head(10)

temp3=dataframe_with_indeterminate.sort_values('Quantity',ascending=False).head(10)

temp4=pd.merge(temp1,temp2,on=['SalesPersonID'],how='left')
temp4
data.groupby('Suspicious').count()