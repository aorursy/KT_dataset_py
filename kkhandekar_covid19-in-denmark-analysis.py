import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation Libraries

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib import cm

import seaborn as sns

import warnings

import re



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-darkgrid')

pd.set_option('display.max_columns', 50)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.0f}'.format
url = '../input/denmark/corona_denmark.csv'

data = pd.read_csv(url, header='infer')

#Converting the Date column to Index and then droping the original Date Column

data.index = pd.DatetimeIndex(data['Date'])

data.drop(['Date'], axis=1, inplace=True)  
data.head()
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    

ax.plot(data['Confirmed Cases'].resample('D').sum(), color='red',linewidth=2.5) 

ax.set_ylabel('Number of Confirmed Cases')

ax.set_title('Denmark - Rise in Confirmed Covid19 Cases', fontsize=16)

fig.tight_layout()

plt.show()