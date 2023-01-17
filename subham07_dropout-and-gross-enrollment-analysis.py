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
import matplotlib.pyplot as plt



df1 = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')

print(df1.head())
data_wb=df1[df1['State_UT']=='West Bengal']

print(data_wb.head())
primary_boys=data_wb['Primary_Boys'].values.astype(np.float32);

primary_girls=data_wb['Primary_Girls'].values.astype(np.float32);



upper_primary_boys=data_wb['Upper Primary_Boys'].values.astype(np.float32);

upper_primary_girls=data_wb['Upper Primary_Girls'].values.astype(np.float32);



secondary_boys=data_wb['Secondary _Boys'].values.astype(np.float32);

secondary_girls=data_wb['Secondary _Girls'].values.astype(np.float32);



higher_secondary_boys=data_wb['HrSecondary_Boys'].values.astype(np.float32);

higher_secondary_girls=data_wb['HrSecondary_Girls'].values.astype(np.float32);
years=data_wb['year'].values;

print(years)



years_ind=[1,2,3]
# set width of bar

barWidth = 0.10



#set figure

fig=plt.figure()

ax=fig.add_axes([5,5,2,2])



# set height of bar

bars1 = primary_boys

bars2 = primary_girls

bars3 = upper_primary_boys

bars4 = upper_primary_girls

bars5 = secondary_boys

bars6 = secondary_girls

bars7 = higher_secondary_boys

bars8 = higher_secondary_girls

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]

r5 = [x + barWidth for x in r4]

r6 = [x + barWidth for x in r5]

r7 = [x + barWidth for x in r6]

r8 = [x + barWidth for x in r7]

 

# Make the plot

ax.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='primary_boys')

ax.bar(r2, bars2, color='pink', width=barWidth, edgecolor='white', label='primary_girls')

ax.bar(r3, bars3, color='blue', width=barWidth, edgecolor='white', label='upper_primary_boys')

ax.bar(r4, bars4, color='orange', width=barWidth, edgecolor='white', label='upper_primary_girls')

ax.bar(r5, bars5, color='black', width=barWidth, edgecolor='white', label='secondary_boys')

ax.bar(r6, bars6, color='grey', width=barWidth, edgecolor='white', label='secondary_girls')

ax.bar(r7, bars7, color='green', width=barWidth, edgecolor='white', label='higher_secondary_boys')

ax.bar(r8, bars8, color='yellow', width=barWidth, edgecolor='white', label='higher_secondary_girls')



# Add xticks on the middle of the group bars

plt.xlabel('YEAR', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], years)



plt.ylabel('Dropout Ratio')

 

# Create legend & Show graphic

plt.legend()

plt.show()

import matplotlib.pyplot as plt



df2 = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')

print(df2.head())
data_wb=df2[df2['State_UT']=='West Bengal']

print(data_wb.head())
data_wb['Primary_Boys']=data_wb['Primary_Boys'].values.astype(np.float32);

data_wb['Primary_Girls']=data_wb['Primary_Girls'].values.astype(np.float32);



data_wb['Upper_Primary_Boys']=data_wb['Upper_Primary_Boys'].values.astype(np.float32);

data_wb['Upper_Primary_Girls']=data_wb['Upper_Primary_Girls'].values.astype(np.float32);



data_wb['Secondary_Boys']=data_wb['Secondary_Boys'].values.astype(np.float32);

data_wb['Secondary_Girls']=data_wb['Secondary_Girls'].values.astype(np.float32);



data_wb['Higher_Secondary_Boys']=data_wb['Higher_Secondary_Boys'].values.astype(np.float32);

data_wb['Higher_Secondary_Girls']=data_wb['Higher_Secondary_Girls'].values.astype(np.float32);
import plotly.express as px

fig = px.bar(data_wb, x="Year", y=["Primary_Boys","Primary_Girls", "Upper_Primary_Boys","Upper_Primary_Girls" ,"Secondary_Boys","Secondary_Girls", "Higher_Secondary_Boys","Higher_Secondary_Girls"], title="Gross Enrollment Year-Wise for all ")

fig.show()
import numpy as np

df4=df2[df2.columns[[0,1,10,13]]]



print(df4.head())

df4=df4.replace('NR',0)

df4=df4.replace('@',0)





df4['Secondary_Total']=df4['Secondary_Total'].values.astype(np.float32);

df4['Higher_Secondary_Total']=df4['Higher_Secondary_Total'].values.astype(np.float32);





sec_total=df4["Secondary_Total"].values

high_sec_total=df4["Higher_Secondary_Total"].values



per_change=np.divide((high_sec_total-sec_total),sec_total)



print(per_change)

print(len(per_change))

print(len(df4))



df4['percentage_change']=per_change



print(df4.head())
import plotly.express as px

df4_1=df4[df4["Year"]=='2013-14']

fig = px.bar(df4_1, x='State_UT', y='percentage_change')

fig.show()
import plotly.express as px

df4_2=df4[df4["Year"]=='2014-15']

fig = px.bar(df4_2, x='State_UT', y='percentage_change')

fig.show()
import plotly.express as px

df4_3=df4[df4["Year"]=='2015-16']

fig = px.bar(df4_3, x='State_UT', y='percentage_change')

fig.show()