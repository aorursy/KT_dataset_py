# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')

df.head(5)
df.describe().round(decimals=2, out=None)

df1 = df[(df['Latitude'] >= -60) & (df['Latitude'] <= 11)]

df2 = df1[(df1['Longitude'] >= -90) & (df1['Longitude'] <= -30)]

df2.count()

#df3 = df2[df2['Type']=='Earthquake']

#df3.count()

df4 = df2[df2['Magnitude type'] == 'MWW']