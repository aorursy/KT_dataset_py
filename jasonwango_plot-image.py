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
#

df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

df
#print(df['Name'].head())

#print(df['Genre'].head())


combined_df = df.groupby(['Platform','Genre']).count()

sorted_df = combined_df.sort_values('Name',ascending=False )

#print(sorted_df['Name'])
import matplotlib.pyplot as plt

from matplotlib import style



style.use('fivethirtyeight')



fig, ax = plt.subplots(figsize=(16, 9))



sorted_df.loc['PS4']['Name'].plot(label='PS4')

sorted_df.loc['PSV']['Name'].plot(label='PSV')

sorted_df.loc['PS3']['Name'].plot(label='PS3')

sorted_df.loc['PS2']['Name'].plot(label='PS2')

sorted_df.loc['PS']['Name'].plot(label='PS')

plt.xlabel('PS GENRE TITLES')

legend = ax.legend(loc='upper right')

plt.show()
Platform_Release_df = df.groupby(['Platform','Year_of_Release']).count()

sorted_df = Platform_Release_df.sort_values('Name',ascending=False )

#print(sorted_df['Name'])
fig, ax = plt.subplots(figsize=(16, 9))



Platform_Release_df['Title'] = Platform_Release_df['Name']

Platform_Release_df.loc['PS4']['Title'].plot(label='PS4')

Platform_Release_df.loc['PSV']['Title'].plot(label='PSV')

Platform_Release_df.loc['PS3']['Title'].plot(label='PS3')

Platform_Release_df.loc['PS2']['Title'].plot(label='PS2')

Platform_Release_df.loc['PS']['Title'].plot(label='PS')

legend = ax.legend(loc='best')

plt.xlabel('PS Life CIRCLE')



plt.show()





Wii_df = df.loc[df['Platform'] == 'Wii']

#print(Wii_df)

sum_Wii_df = Wii_df.groupby(['Platform','Publisher']).sum()

sorted_Wii_df = sum_Wii_df.sort_values('Global_Sales',ascending=False )['Global_Sales']

print('sales average :' + str(sorted_Wii_df.mean()))
high_Wii_df = sum_Wii_df[sum_Wii_df.Global_Sales >= sum_Wii_df.Global_Sales.mean()]['Global_Sales']

low_Wii_df = sum_Wii_df[sum_Wii_df.Global_Sales <= sum_Wii_df.Global_Sales.mean()]['Global_Sales']

high_Wii_df.loc['Wii','Others']=low_Wii_df.sum()

print(high_Wii_df)



style.use('seaborn-pastel')

fig, ax = plt.subplots(figsize=(15, 15))

sorted_Wii_df = high_Wii_df.sort_values(0,ascending=False )





sorted_Wii_df.plot(kind='pie',fontsize=17,labels=sorted_Wii_df.index.get_level_values(1),title='Wii Publisher pie chart')

plt.show()
