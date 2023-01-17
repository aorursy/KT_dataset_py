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
import pandas as pd



#data = "../input/indonesia-health-indicators/health-indicators-for-indonesia-1.csv"

data  = "../input/health-indicator-indonesia/health_indicators_for_indonesia_1.csv"

df_data = pd.read_csv(data)
features = df_data[df_data.columns[0:]]

features.head()
labels = df_data['low']

labels.describe(include='all')
features.values.shape
used_data = df_data[['gho_code','gho_display','dhsmicsgeoregion_display','year_code','numeric']]

clean_data= used_data.dropna()



antenatal =clean_data.loc[clean_data['gho_code'] == 'anc1']

antenatal_wj = antenatal[antenatal.dhsmicsgeoregion_display.str.contains('west java', regex= True, na=False)]

antenatal_ck  = antenatal[antenatal.dhsmicsgeoregion_display.str.contains('central kalimantan', regex= True, na=False)]
antenatal_wj
antenatal_ck
import matplotlib.pyplot as plt

plt.plot(antenatal_wj['year_code'] ,antenatal_wj['numeric'])

plt.plot(antenatal_ck['year_code'] ,antenatal_ck['numeric'])

plt.title('Grafik Perubahan Antenatal Care')

plt.ylabel('numeric')

plt.xlabel('year_code')

plt.legend(['Jawa Barat', 'Kalimantan Tengah'], loc='lower left')

plt.show()
antenatal_wj['numeric']