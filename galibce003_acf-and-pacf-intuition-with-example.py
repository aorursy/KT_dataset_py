# Necessary modules



import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Dataset



df = pd.read_csv('../input/icecream-production/ice.csv', parse_dates = True)
# First few Rows



df.head()
# Last few Rows



df.tail()
# Shape



df.shape
# Data type



df.dtypes
# Changing the data type of date column



df['DATE'] = pd.to_datetime(df['DATE'])
df.dtypes
# Set the date column as Index



df.set_index('DATE', inplace = True)
df.head()
plt.figure(figsize = (15, 5))



plt.plot(df['Icecream'], color = 'red')



plt.title("Icecream Production over Time\n", fontsize = 15)

plt.ylabel("Production\n", fontsize = 12)



for i in range(1972, 2021):

    plt.axvline(pd.to_datetime(str(i) + '-01-01'), color = 'black', linestyle = '--', alpha = 0.5)

    

plt.show()
acf_plot = plot_acf(df.Icecream, lags = 100)
pacf_plot = plot_pacf(df.Icecream)