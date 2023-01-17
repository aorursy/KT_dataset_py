# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/us-police-shootings/shootings.csv")
df10 = pd.read_excel("/kaggle/input/populous/population1.xlsx")
df
df10
df['Year'] = df['date'].apply(lambda x:x[0:4])
df['Month'] = df['date'].apply(lambda x :x[5:7])
df
sns.countplot(x='manner_of_death',data = df)
sns.countplot(x='signs_of_mental_illness',data = df)
df1 = pd.DataFrame(df['gender'].value_counts(normalize=True),
                   index=['M','F'])
plot = df1.plot.pie(subplots=True, autopct='%1.1f%%', figsize=(5,5))
plt.xlabel("Gender distribution(shot)")

my_colors = 'yellow'
df['flee'].value_counts().plot(kind='barh', figsize=(10,5) , color=my_colors)
plt.xlabel("Count")
plt.ylabel("Method of fleeing")

sns.countplot(x='body_camera',data = df)
plt.figure(figsize=(18,4))
sns.countplot(x='arms_category',data = df)
sns.countplot(x='Month',data = df)
plt.xlabel("Deaths every month")
sns.countplot(x='Year',data = df)
plt.xlabel("Deaths every year")

sns.distplot(df['age'], hist=True, kde=True,bins=int(180/5), color = 'darkblue')
plt.figure(figsize=(15,4))
sns.countplot(x='state',data = df)
plt.xlabel("Deaths grouped by states")
df['city'].value_counts().nlargest(10)
df9 = pd.DataFrame(df['city'].value_counts(normalize=True).nlargest(10))
plot = df9.plot(subplots=True,kind ='bar', figsize=(5,5))
plot
plt.xlabel("Top 10 cities")
df1 = df.pivot_table(values = 'age', index=['state'], columns='manner_of_death',aggfunc='count')
df1 = df1.fillna(0)
df2 = df.pivot_table( values = 'age',index=['state'],aggfunc=np.mean)
df2 = df2.fillna(0)
df3 = df.pivot_table( values = 'Year',index=['state'], columns='race',aggfunc='count')
df3 = df3.fillna(0)

all_data = pd.merge(df1, df3, 'left', on = ["state"] ) 
all1_data = pd.merge(all_data, df2, 'left', on = ["state"] ) 
cm = sns.light_palette("green", as_cmap=True)
cm1 = sns.light_palette("purple", as_cmap=True)
cm2 = sns.light_palette("blue", as_cmap=True)
cm3 = sns.light_palette("m", as_cmap=True)
cm4 = sns.light_palette("orange", as_cmap=True)
cm5 = sns.light_palette("gold", as_cmap=True)
cm6 = sns.light_palette("peru", as_cmap=True)




cm = (all1_data.style
  .background_gradient(cmap=cm, subset=['age']) 
  .background_gradient(cmap=cm1, subset=['White']) 
  .background_gradient(cmap=cm2, subset=['Hispanic']) 
  .background_gradient(cmap=cm3, subset=['Native']) 
  .background_gradient(cmap=cm4, subset=['shot']) 
  .background_gradient(cmap=cm5, subset=['Asian']) 
  .background_gradient(cmap=cm6, subset=['Black']) 
  .highlight_max(subset=['age','White','Other','Native','Hispanic','Black','Asian','shot'])
  .set_caption('This is a custom caption.')
  .format({'total_amt_usd_pct_diff': "{:.2%}"}))
cm
df10[['state','People shot per million']].sort_values(['People shot per million'], ascending=False).head(10).style.background_gradient(subset='People shot per million', cmap='BuGn')