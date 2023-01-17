%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

from sklearn.preprocessing import StandardScaler



sns.set(style="white")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
with open('../input/Accidents0515.csv', 'r') as f:

    df = pd.read_csv(f)
df = df.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude',

               'Accident_Index', 'Police_Force', 'Local_Authority_(District)',

               'Local_Authority_(Highway)', 'LSOA_of_Accident_Location',

               'Did_Police_Officer_Attend_Scene_of_Accident',

               '1st_Road_Number', '2nd_Road_Number' ], axis=1)

df.head()
# format dates to retqin month only

def to_month(date):

    month = datetime.strptime(date, '%d/%m/%Y')

    return int(datetime.strftime(month, '%m'))



df['Date'] = df['Date'].apply(to_month)
# format time to hours only

def to_hour(time):

    try:

        hour = datetime.strptime(str(time), '%H:%M')

        return int(datetime.strftime(hour, '%H'))

    except Exception:

        return 0



df['Time'] = df['Time'].apply(to_hour)
df.head()

df.dropna(inplace=True)
data = df.values
scl = StandardScaler()

scl_data = scl.fit_transform(data)
#Computing the correlation matrix

corr = df.corr()



#Generating a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



#Setting up the matplotlib figure

f, ax = plt.subplots(figsize=(8,8))



#Generating a custom diverging colormap

cmap = sns.diverging_palette(220,10, as_cmap=True)



#Drawing the heatmap with the mask

sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



plt.title("Correlation Matrix")

plt.show()