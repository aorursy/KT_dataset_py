# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

import magic



# Any results you write to the current directory are saved as output.
#Validation function

def gunvalidation(directory):

    #check format of files 

    def fileformat(file):

        print (f'File format is {magic.from_file(file)}')

        

    #check for missing data in files

    def missinglatloncol(file):

            if "latitude" in pd.read_csv(file).columns:

                print (f'Lat/Lon column present')

            else:

                print (f'Lat/Lon column missing')

                

    #percent of missing values in gunstolen column

    def missingdata(file):

        if 'latitude' in (pd.read_csv(file).columns):

            mis = round((pd.read_csv(file).isnull().sum().sum()/(len(pd.read_csv(file))*len(pd.read_csv(file).columns))*100))

            print(f'{mis} % of total data missing')

            

    for file in glob.glob(directory):

        print (f'Validation for ...{file[-20:]}')

                

        fileformat(file)

        missinglatloncol(file)

        missingdata(file)

        

        print('\n')
gunvalidation('/kaggle/input/repository/jamesqo-gun-violence-data-17dc307/intermediate/*.csv')
#Pull relevant details out of each dataframe and add it to a cleaned dataframe

#only interested in FL killed/injured by month and year

cleaned_data = pd.DataFrame()

for file in glob.glob("/kaggle/input/repository/jamesqo-gun-violence-data-17dc307/intermediate/*.csv"):

    if 'stage2' in file:

        df1 = pd.read_csv(file)

        data = {'FL killed': df1[df1['state']=='Florida']['n_killed'].sum(),

                'FL injured': df1[df1['state']=='Florida']['n_injured'].sum(),

                'Month': pd.to_datetime(df1[df1['state']=='Florida']['date']).dt.month.mean(),

                'Year': pd.to_datetime(df1[df1['state']=='Florida']['date']).dt.year.mean()}

        df_temp = pd.DataFrame(data, index = [0])

    cleaned_data = cleaned_data.append(df_temp, ignore_index=True).drop_duplicates().reset_index(drop=True)
cleaned_data