import os

import numpy as np

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
def weather_data(data=None):

    """

    param: data a pandas dataframe 

    

    return

    data per month

    """

    

    cols=data[1].columns # take all columns of first row of list data

    

    month = data[2].iloc[0] # take month

    m_time = data[2].iloc[1:-1] # take all data from this month

    

    # We create series pandas for dayofmonth

    if month[0] == 'Jan':

        s=pd.date_range(start='2020-01-01', periods= len(m_time) , freq='D')

    if month[0] == 'Feb':

        s=pd.date_range(start='2020-02-01', periods= len(m_time) , freq='D')

        

    if month[0] == 'Mar':

        s=pd.date_range(start='2020-03-01', periods= len(m_time) , freq='D')

        

    if month[0] == 'Apr':

        s=pd.date_range(start='2020-03-01', periods= len(m_time) , freq='D')

        

        

        

    v = [] # initialize

    v.append(np.array(s)) # append s

    

    # take all data for  'Temperature (Â° F)', 'Dew Point (Â° F)', 'Humidity (%)',

    #   'Wind Speed (mph)', 'Pressure (Hg)'

    for i in range(3, 8):

        c = np.array(data[i].iloc[1:-1,1]) 

        

        v.append(c)

        

    last_feature = np.array(data[8].iloc[1:-1]) # columns precipitation

    

    v.append(last_feature)

    

    vdt = pd.DataFrame() # initialize 

    for i in range(len(cols)):

        vdt[cols[i]] = v[i]

        

    return vdt # pandas dataframe go below to see a result
def read_main_folder(pardir=None):

    """

    param: pardir: the main path folder

    

    return 

    list of file in each folder

    

    """



    folder_file = []

    for dirname, _, filenames in os.walk(pardir):

        gv = []

        for filename in filenames:

            gv.append(os.path.join(dirname, filename))

    

        if gv == []: # to avoid an empty list

            pass

        else:

            folder_file.append(gv)

    

    return folder_file # see the result below
def concatenate_and_save_data(folder=None):

    """

    param: folder: take a path html file

    

    return

    read each html file after concatenate all data per month and then save to csv file

    

    """

    

    value = []

    cities = ['cairo-egypt','new-york','tehraniran','tokyo-japan','milan-italy']

    for f in folder:

        dc = pd.read_html(f)

        bc = weather_data(data=dc)

    

        value.append(bc)

        

    con_data = pd.concat(value)

    print(f)

        

    for item in cities:

        if item in f:

            city = item #input('Give a city name correspond to this data §§\n')

    

            datasets = 'data_'

    

            save = datasets + city + '_weather.csv'

        

            con_data.to_csv(save, index=False)

            print('Saving data is done for {} !\n'.format(item))

    

    #return con_data
# take html file

html_file = '/kaggle/input/tokyo-japan/Tokyo Japan Weather History _ Weather Underground (2020-04-04 17_37_31).html'

html_1 = '/kaggle/input/tehraniran/Tehran Iran Weather History _ Weather Underground (2020-04-04 17_34_08).html'

html_2 = '/kaggle/input/new-york/New York City NY Weather History _ Weather Underground (2020-03-24 11_04_39).html'
if 'new-york' in html_2:

    print('yes')

else:

    print('no')
df = pd.read_html(html_file) # extract unstructured data from html file using pandas for Tokyo-Japan

du = pd.read_html(html_2) # extract unstructured data from html file using pandas for New York City 

ds = pd.read_html(html_1)# extract unstructured data from html file using pandas for Tehran-Iran
# see the type of our unstructured data

type(df)
# size

len(df) # 
# eda

df[1].head(3) # the first row 
df[1].columns # columns of first row
df[2].head(3) # the second row that are a month
#see name of month

df[2].iloc[0]
df[3:] # this is a data for all columns above that contains Max, Avg and Min value. 

#For us, we are taking only Avg value of each columns. To get all this data we need to use weather_data function. 

# Go to create function section above. to explain the function
# result for Tokyo Japan

weather_data(data=df).head(3) # this is for weather data function
# result for New York City

weather_data(data=du).head(3)
# result for Tehran Iran

weather_data(data=ds).head(3)
# now we have a weather data function. we cannot take data for each html file like that we need to define

#a function that can read all file to put it in the list. go above. 
dir_parent = '/kaggle/input'

docs = read_main_folder(pardir=dir_parent)
docs # see the last function above
dir_parent = '/kaggle/input'

docs = read_main_folder(pardir=dir_parent)
# save data for each subfolders

for fk in docs:

    concatenate_and_save_data(folder=fk)