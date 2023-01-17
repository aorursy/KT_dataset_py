import os 



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from IPython.display import display 

%matplotlib inline

Data = pd.read_csv('/kaggle/input/palm-data/palm_ffb.csv', delimiter=',')
Data.head()
#Ignore the Date column as the focus on external factors



Df = Data.iloc[:,1:]
Df.head()
Df.describe() #Summary Statistic of the palm data
!pip install pingouin
import pingouin as pg

corr = pg.pairwise_corr(Df, columns=[['FFB_Yield'], ['SoilMoisture', 'Average_Temp', 'Min_Temp', 'Max_Temp', 'Precipitation','Working_days','HA_Harvested']], method='pearson')

corr.sort_values(by=['p-unc'])[['X', 'Y', 'n', 'r', 'p-unc']].head()
Df.corr().round(2) # Convert the Data into a correlation matrix
import seaborn as sns

import matplotlib.pyplot as plt

corrs = Df.corr()

mask = np.zeros_like(corrs)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4)

plt.title('Correlation matrix')