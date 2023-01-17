# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import dataset 



df=pd.read_csv('/kaggle/input/airquality/Air_Quality.csv',parse_dates=['Date'],index_col='Date')

df.head()

# EDA : describe data quickly by pandas-profiling



from pandas_profiling import ProfileReport

profile=ProfileReport(df, title='EDA of Air-Quality',html={'style':{'full_width':True}})

profile.to_widgets()
#visualize data 



# Import missingno as msno

import missingno as msno

import matplotlib.pyplot as plt



# Plot amount of missingness

msno.bar(df) # you can see pandas-profilin count part



plt.show()
# Plot nullity matrix of airquality with frequency 'M'

msno.matrix(df, freq='M') # this part actually displays the missingness types, also more visible version of pandas-profiling matrix.



plt.show()
### Forward Fill

# Impute airquality DataFrame with ffill method

ffill_imputed = df.copy(deep=True)



ffill_imputed.fillna(method='ffill',inplace=True)



# Plot the imputed DataFrame ffill_imp in red dotted style 

ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))



# Plot the airquality DataFrame with title

df['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))



plt.show()
### Back Fill Fill

# Impute airquality DataFrame with bfill method

bfill_imputed = df.copy(deep=True)



bfill_imputed.fillna(method='bfill',inplace=True)



# Plot the imputed DataFrame bfill_imp in red dotted style 

bfill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))



# Plot the airquality DataFrame with title

df['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))



plt.show()
# Interpolate the NaNs quadratically



quadratic_imput=df.copy(deep=True)



quadratic_imput.interpolate(method='quadratic', inplace=True)



quadratic_imput['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))



df['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

# Interpolate the NaNs by nearest method



nearest_imput=df.copy(deep=True)



nearest_imput.interpolate(method='nearest', inplace=True)



nearest_imput['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))



df['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))
# Interpolate the NaNs by linear method



linear_imput=df.copy(deep=True)



linear_imput.interpolate(method='linear', inplace=True)



linear_imput['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))



df['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))
# Set nrows to 3 and ncols to 1

fig, axes = plt.subplots(6, 1, figsize=(30, 20))



# Create a dictionary of interpolations

interpolations = {'Airquality': df, 'Back-fill':bfill_imputed, 'Forward-fill':ffill_imputed,

                  'Linear Interpolation': linear_imput, 'Quadratic Interpolation': quadratic_imput, 

                  'Nearest Interpolation': nearest_imput}



# Loop over axes and interpolations

for ax, df_key in zip(axes, interpolations):

  # Select and also set the title for a DataFrame

  interpolations[df_key].Ozone.plot(color='red', marker='o', 

                                 linestyle='dotted', ax=ax)

  df.Ozone.plot(title=df_key + ' - Ozone', marker='o', ax=ax)

  

plt.show()
