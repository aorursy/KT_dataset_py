# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sf_Dataset = pd.read_csv('../input/sf-library-usage-data/Library_Usage.csv')

sf_Dataset
sf_Dataset.columns
new_sf_Dataset = sf_Dataset.sample(20000)

new_sf_Dataset
# Bar chart showing average score for racing games by platform

plt.figure(figsize=(16, 9))



sns.barplot(y=new_sf_Dataset['Total Checkouts'], x=new_sf_Dataset['Age Range'])

# Add label for horizontal axis

#plt.xlabel("")

# Add label for vertical axis

plt.title("Average Score for Age Range, by Total Checkouts")
# Bar chart showing average score for racing games by platform

plt.figure(figsize=(16, 9))

sns.barplot(y=new_sf_Dataset['Total Renewals'], x=new_sf_Dataset['Age Range'])

# Add label for horizontal axis

#plt.xlabel("")

# Add label for vertical axis

plt.title("Average Score for Age Range, by Total Checkouts")
# Bar chart showing average score for racing games by platform

plt.figure(figsize=(20, 9))

sns.barplot(x=new_sf_Dataset['Total Renewals'], y=new_sf_Dataset['Patron Type Definition'])

# Add label for horizontal axis

#plt.xlabel("")

# Add label for vertical axis

plt.title("Average Score for Patron Type Definition, by Total Renewals")
# Bar chart showing average score for racing games by platform

plt.figure(figsize=(20, 9))

sns.barplot(x=new_sf_Dataset['Total Checkouts'], y=new_sf_Dataset['Patron Type Definition'])

# Add label for horizontal axis

#plt.xlabel("")

# Add label for vertical axis

plt.title("Average Score for Patron Type Definition, by Total Renewals")
new_sf_Dataset
#new_sf_Dataset = new_sf_Dataset.dropna() 

df = new_sf_Dataset[['Year Patron Registered','Total Checkouts','Total Renewals']]

#df /// values='Year Patron Registered'



ddf = df.set_index(['Year Patron Registered'])

plt.figure(figsize=(16,9))

sns.lineplot(data=ddf)
drop_None_Value = new_sf_Dataset.drop(index=new_sf_Dataset[new_sf_Dataset['Circulation Active Year'] == 'None'].index)

new_sf_Dataset = drop_None_Value



df = new_sf_Dataset[['Circulation Active Year','Total Checkouts','Total Renewals']]

#df // values='Year Patron Registered'

ddf = df.set_index(['Circulation Active Year'])

plt.figure(figsize=(16,9))

sns.lineplot(data=ddf)
#lis = new_sf_Dataset['Circulation Active Year']

#for i in lis:

#    print(i)

#new_sf_Dataset['Circulation Active Year'] = new_sf_Dataset["Circulation Active Year"].astype(str).astype(int)

#new_sf_Dataset['Year Patron Registered']
#new_sf_Dataset.columns

new_sf_Dataset
#sns.lmplot(x="Total Checkouts", y="Total Renewals", hue="Outside of County", data=new_sf_Dataset)

sns.regplot(x=new_sf_Dataset['Total Checkouts'], y=new_sf_Dataset['Total Renewals'])
new_sf_Dataset
plt.figure(figsize=(16, 9))



sns.barplot(x=new_sf_Dataset['Circulation Active Month'], y=new_sf_Dataset['Total Checkouts'])
plt.figure(figsize=(16, 9))



sns.barplot(x=new_sf_Dataset['Circulation Active Month'], y=new_sf_Dataset['Total Renewals'])
##new_sf_Dataset.index.name= 'id'

new_sf_Dataset.columns
#new_sf_Dataset['Circulation Active Year'] = new_sf_Dataset['Circulation Active Year'].astype(str).astype(int)

#sns.lmplot(x='Circulation Active Year', y="Total Renewals", hue="Outside of County",data=new_sf_Dataset)

test_df = new_sf_Dataset.sample(2000)

sns.swarmplot(x='Outside of County',y='Total Checkouts',data=test_df)



test_df = new_sf_Dataset.sample(2000)

sns.swarmplot(x='Outside of County',y='Total Renewals',data=test_df)
new_sf_Dataset
df = new_sf_Dataset[['Patron Type Definition','Total Checkouts','Age Range']]



heatmap_Data = pd.pivot_table(df, values='Total Checkouts', index=['Patron Type Definition'],columns=['Age Range'])

plt.figure(figsize=(16,9))

sns.heatmap(data=heatmap_Data, annot=True)
new_sf_Dataset
df = new_sf_Dataset[['Patron Type Definition','Total Checkouts','Circulation Active Month']]



heatmap_Data = pd.pivot_table(df, values='Total Checkouts', index=['Patron Type Definition'],columns=['Circulation Active Month'])

plt.figure(figsize=(16,9))

sns.heatmap(data=heatmap_Data, annot=True)
#Outside of County

#sns.lmplot(x="Age Range", y="Provided Email Address", hue="Outside of County", data=new_sf_Dataset)

#sns.swarmplot(x=new_sf_Dataset['Outside of County'],y=new_sf_Dataset['Provided Email Address'])

#df = new_sf_Dataset[['Year Patron Registered','Total Checkouts','Total Renewals']]



#df

#heatmap_Data = pd.pivot_table(df, values='Year Patron Registered', index=['Total Checkouts'], 

                             # columns='Total Renewals')



#plt.figure(figsize=(16,9))



#Heatmap showing average arrival delay for each airline by month

#sns.heatmap(data=heatmap_Data, annot=True)
