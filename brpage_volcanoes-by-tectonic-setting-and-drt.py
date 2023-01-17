# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt



from pandas import set_option

set_option("display.max_rows", None)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/database.csv")
df.head()
df = df.drop(['Number', 'Activity Evidence', 'Last Known Eruption', 'Latitude', 'Longitude', 'Elevation (Meters)'], axis=1)
df.info()
df['Tectonic Setting'] = df['Tectonic Setting'].replace("Unknown", np.nan)
mask1 = df['Tectonic Setting'].notnull()

df = df[mask1]

df.info()
df['Type'].value_counts()
df['Type'] = df['Type'].str.replace(r"\(.*\)", "")
df['Type'] = df['Type'].str.replace("?", "")
df['Type'] = df['Type'].replace("Unknown", np.nan)
mask2 = df['Type'].notnull()

df = df[mask2]

df.info()
df['Dominant Rock Type'] = df['Dominant Rock Type'].str.replace("No Data", "")
df['Dominant Rock Type'] = df['Dominant Rock Type'].replace("", np.nan)
mask3 = df['Dominant Rock Type'].notnull()

df = df[mask3]

df.head(5)
df['Tectonic Setting'].value_counts()
df['Dominant Rock Type'].value_counts()
for i, value in enumerate(df['Dominant Rock Type'].sort_values().unique()):

    df.loc[df['Dominant Rock Type'] == value, 'DRT'] = i
df.head()
#This is the sorted unique order for each new ocurrence of Dominant Rock Type.

DRT_names = ['A_BA', 'B_PB', 'Dac', 'Fd', 'P_TP', 'Ph', 'Rhy', 'TaBTa', 'TbTB', 'TTd']

DRT_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D', 'black']



tec_set = df['Tectonic Setting'].values

DRT = df['DRT'].values
mpl.rcParams.update({'font.size': 10})

mpl.rcParams['figure.figsize'] = (15.0, 15.0)

inline_rc = dict(mpl.rcParams)



for tectonic, tec in enumerate(np.unique(tec_set)):

    ax = plt.subplot(4, 3, tectonic+1)

    hist = np.histogram(DRT[tec_set == tec], bins=np.arange(len(DRT_names)+1))

    plt.bar(np.arange(len(hist[0])), hist[0], color=DRT_colors, align='center')

    ax.set_xticks(np.arange(len(hist[0])))

    ax.set_xticklabels(DRT_names)

    ax.set_title(tec)