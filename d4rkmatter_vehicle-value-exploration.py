# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/852k-used-car-listings/tc20171021.csv',error_bad_lines=False)

data.head
def make_data(data,**kwargs):

    make = kwargs['make']

    

    #cleaning routine

    #round prices to nearest 1000

    data.Price = data.Price.round(-3)

    #round mileage to nearest 10000

    data.Mileage = data.Mileage.round(-4)



    #build df

    make_df = data[data.Make == make] 

    #if model is provided

    if len(kwargs.items())==2:

        model = kwargs['model']

        model_df = make_df[make_df['Model'].str.contains(model)]

        model_df = model_df.drop(['Make','City','Vin','Id'],axis=1)

        return(model_df)

    #if only make is provided

    else:

        make_df = make_df.drop(['Make','City','Vin','Id'],axis=1)

        return(make_df)
def trendline(df,column):

    y_arr = []

    #get set of years

    x_arr = df[column].unique()

    x_arr = np.sort(x_arr)

    for x in x_arr[::-1]:

        #calculate average per year

        subset = df[df[column] == x]

        avg = np.mean(subset.Price)

        y_arr.append(avg)

    

    return(x_arr[::-1], y_arr)



def get_slope(x_arr,y_arr,column):

    if column == 'Year':

        x_ends = [min(x_arr),max(x_arr)]

        y_ends = [min(y_arr), max(y_arr)]

        m = (y_ends[1]-y_ends[0])/(x_ends[1]-x_ends[0])

    elif column =='Mileage':

        x_ends = [x_arr[0],x_arr[-1]]

        y_ends = [min(y_arr), max(y_arr)]

        m = (y_ends[1]-y_ends[0])/(x_ends[1]-x_ends[0])

    else:

        print('Column not supported')

        return(None)

    return(x_ends, y_ends, m)



def drop_outliers(df,column):

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR =  Q3-Q1

    left = Q1-1.5*IQR

    right = Q3 + 1.5*IQR

    return(df[(df[column] >= left) & (df[column] <= right)])
def slope_plotter(df, column):

    def derivative(x,y):

        slope_arr = [[],[]]

        for i in range(len(x)-1):

            slope_arr[1].append(y[i]-y[i+1])

            slope_arr[0].append(x[i+1])

        return(slope_arr)

    fig, ax = plt.subplots(2,figsize=(15,10))

    #remove outliers

    df = drop_outliers(df,column)

    #function calls

    x_trend, y_trend = trendline(df, column)

    slope_arr = derivative(x_trend,y_trend)

    slope_x = [slope_arr[0][0],slope_arr[0][-1]]

    slope_y = [slope_arr[1][0],slope_arr[1][-1]]

    

    ax[0].plot(slope_arr[0],slope_arr[1])

    ax[0].plot(slope_x,slope_y, color='red',linestyle='--')

    

    ax[0].set_title('First Derivatives')

    ax[0].set_ylabel('Slope')

    ax[0].set_xlabel('Year+1')



    slope_arr1 = derivative(slope_arr[0],slope_arr[1])

    slope_x = [slope_arr1[0][0],slope_arr1[0][-1]]

    slope_y = [slope_arr1[1][0],slope_arr1[1][-1]]

    ax[1].plot(slope_arr1[0],slope_arr1[1])

    

    ax[1].plot(slope_x,slope_y, color='red',linestyle='--')

    ax[1].set_title('Second Derivatives')

    ax[1].set_ylabel('Slope')

    ax[1].set_xlabel('Year+2')

    plt.suptitle('Slopes for {}'.format(column),size=25)

    plt.subplots_adjust(hspace=.5)

    plt.show()
outback_df = make_data(data,make='Subaru',model='Outback')

outback_df
#check for correlation among variables to Price

outback_df.corr()
#TODO boxplot for year/mileage, year/price

outback_df.plot.scatter(y='Year', x='Mileage', xlim=(-100,300000))
def vehicle_neighbors(df, sortby, myCar, adj):

    df = df.append(pd.Series(myCar,index=df.columns),ignore_index=True)

    car_idx = len(df.index)-1

    df = df.sort_values(by=sortby)

    df = df.reset_index()



    slice_idx = df.loc[df['index']==car_idx].index

    slice_idx = slice_idx.values[0]

    sorted_df = df.loc[(slice_idx-adj):(slice_idx+adj),:]

    

    return(sorted_df)



def good_deal_plot(df, myCarArray):

    if len(myCarArray) != 5:

        print("Too Few Arguments in Array")

        print("Expects [Price, Year, Mileage, State, Model]")

        return(None)

    myCar = myCarArray

    df['myCar'] = 0

    myCar.append(1)

    df2 = df.append(pd.Series(myCarArray,index=df.columns),ignore_index=True)

    df2 = drop_outliers(df2,'Mileage')

    df2 = drop_outliers(df2, 'Price')

    fig = px.scatter_3d(df2, x='Year', y='Mileage', z='Price', color='myCar',opacity=0.5)



    fig.show()

    
#Price, Year, Mileage, State, Model

myCarArray = [14000,2014,68000,'WA','Outback2.5i']
#TODO add multiple vehicles at once

#TODO clean up viz

good_deal_plot(outback_df,myCarArray)
from scipy import stats

def plotter(df, column,**kwargs):

    figsize = kwargs['figsize']

    ylim = kwargs['ylim']

    slope_pos = kwargs['slope_pos']

    

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_ylim(ylim)

    #remove outliers

    df = drop_outliers(df,column)

    ax.scatter(df[column],df.Price)

    #function calls

    x_trend, y_trend = trendline(df, column)

    

    slope_x, slope_y, slope = get_slope(x_trend,y_trend,column)

    ax.plot(x_trend, y_trend, color='red') #trendline

    ax.plot(slope_x,slope_y,color='black',linestyle='--') #slope



    for x,y in zip(x_trend,y_trend):

        label = '{:.0f}'.format(y)

        label = '$ '+label[:2]+','+label[2:]

        plt.annotate(label,(x,y),textcoords='offset points',xytext=(40,50),ha='center') #add avg vals to plot



    plt.annotate('Slope: {:.1f} $ per {}'.format(slope,column), slope_pos, size=20) #add slope value

    plt.title('Price Vs. {} w/ Trendline'.format(column))

    plt.xlabel(column)

    plt.ylabel('Price')

    plt.show()



plotter(outback_df, 'Year', figsize=(25,10),ylim=(0,50000),slope_pos=(2010,45000))

plotter(outback_df, 'Mileage', figsize=(25,10),ylim=(0,50000),slope_pos=(0,45000))

slope_plotter(outback_df,'Year')

slope_plotter(outback_df,'Mileage') #todo: fix mileage axis label
#outback_df.query('Mileage == 80000').describe()
#model_cat = outback_df.Model.unique()

#outback_df.Model = outback_df.Model.astype('category')

#print(model_cat)

#print(outback_df.Model.cat.categories)

#outback_df.Model.hist()
#outback_df.Model = outback_df.Model.cat.codes

#outback_df.plot.scatter(x='Model',y='Price',ylim=(0,50000))
