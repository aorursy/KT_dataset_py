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
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

from brewer2mpl import qualitative
df= pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")

df.shape
df.head()
df.info()
df.isnull().sum()
df.columns.tolist()
df_na= ( df.isnull().sum() / len(df) ) * 100

df_na= df_na.drop(df_na[df_na == 0].index).sort_values(ascending= False)
f, ax= plt.subplots(figsize=(12, 8))

plt.xticks(rotation='90')

sns.barplot(x=df_na.index, y=df_na.values)

ax.set(title='Missing Values Plot', ylabel='% Missing')

df.Platform.unique()
#df.Platform.value_counts()

ssc = df.Platform.value_counts()

f, ax= plt.subplots(figsize=(12, 8))

plt.xticks(rotation='90')

sns.barplot(x=ssc.values, y=ssc.index, orient='h')

ax.set(title='Consoles by count', ylabel='Count')

f.tight_layout()


df_clean= df.dropna(axis=0)

df_clean.shape
ssc = df_clean.Platform.value_counts()

f, ax= plt.subplots(figsize=(12, 8))

plt.xticks(rotation='90')

sns.barplot(x=ssc.values, y=ssc.index, orient='h')

ax.set(title='Consoles by count after dropping NAs', ylabel='Count')

f.tight_layout()
#df['User_Score']= df['User_Score'].convert_objects(convert_numeric=True)

df_clean.User_Score= df_clean.User_Score.astype('float')

#df.User_Score.dtype

#df['User_Score'] = video['User_Score'].convert_objects(convert_numeric= True)
sns.jointplot(x='User_Score', y='Critic_Score', data=df_clean, kind='hex', cmap='coolwarm', size=7)

sns.jointplot(x='Critic_Score', y='Critic_Count', data=df_clean, kind='hex', cmap='plasma', size=7)
stats=['Year_of_Release','NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 

       'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 

       'Rating']

corrmat = df_clean[stats].corr()



f, ax = plt.subplots(figsize=(10, 7))

plt.xticks(rotation='90')

plt.title('correlation between columns')

sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
play= df_clean[(df_clean['Platform']== 'PS2') | (df_clean['Platform']== 'PS3')

              | (df_clean['Platform']== 'PS')| (df_clean['Platform']== 'PS4')]

play.shape
sales_Play= play.groupby(['Year_of_Release', 'Platform'])['Global_Sales'].sum()

sales_Play.unstack().plot(kind='bar',stacked=True, colormap= 'Oranges',  grid=False)

ax.set(title='Playststation Global over the year', ylabel='Cumulative count')
sales_Play= play.groupby(['Genre', 'Platform'])['Global_Sales'].sum()

sales_Play.unstack().plot(kind='bar',stacked=True, colormap= 'Oranges',  grid=False)
sales_Play= play.groupby(['Rating', 'Platform'])['Global_Sales'].sum()

sales_Play.unstack().plot(kind='bar',stacked=True, colormap= 'Oranges',  grid=False)
xb= df_clean[(df_clean['Platform']== 'X360') | (df_clean['Platform']== 'XOne')

              | (df_clean['Platform']== 'XB')]

xb.shape
sales_xb= xb.groupby(['Year_of_Release', 'Platform'])['Global_Sales'].sum()

sales_xb.unstack().plot(kind='bar',stacked=True, colormap= 'Vega20',  grid=False)
sales_xb= xb.groupby(['Genre', 'Platform'])['Global_Sales'].sum()

sales_xb.unstack().plot(kind='bar',stacked=True, colormap= 'Vega20',  grid=False)
sales_xb= xb.groupby(['Rating', 'Platform'])['Global_Sales'].sum()

sales_xb.unstack().plot(kind='bar',stacked=True, colormap= 'Vega20',  grid=False)
nintendo= df_clean[(df_clean['Platform']== 'DS') | (df_clean['Platform']== 'Wii')

              | (df_clean['Platform']== 'GC')| (df_clean['Platform']== 'GBA')

                  |(df_clean['Platform']== '3DS') | (df_clean['Platform']== 'WiiU')]

nintendo.shape
nintendo_sales= nintendo.groupby(['Year_of_Release', 'Platform'])['Global_Sales'].sum()

nintendo_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Set1',  grid=False)
nintendo_sales= nintendo.groupby(['Genre', 'Platform'])['Global_Sales'].sum()

nintendo_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Set1',  grid=False)
nintendo_sales= nintendo.groupby(['Rating', 'Platform'])['Global_Sales'].sum()

nintendo_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Set1',  grid=False)
current_gen= df_clean[(df_clean['Platform']== 'Wii') | (df_clean['Platform']== 'X360') | 

                                                        (df_clean['Platform']== 'PS3')]

current_gen.shape
current_gen_sales= current_gen.groupby(['Year_of_Release', 'Platform'])['Global_Sales'].sum()

current_gen_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Set2',  grid=False)
current_gen_sales= current_gen.groupby(['Genre', 'Platform'])['Global_Sales'].sum()

current_gen_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Set2',  grid=False)
current_gen_sales= current_gen.groupby(['Rating', 'Platform'])['Global_Sales'].sum()

current_gen_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Set2',  grid=False)
current_gen_sales= current_gen.groupby(['Year_of_Release', 'Platform'])['NA_Sales'].sum()

current_gen_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Blues',  grid=False)
current_gen_sales= current_gen.groupby(['Year_of_Release', 'Platform'])['JP_Sales'].sum()

current_gen_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Blues',  grid=False)
current_gen_sales= current_gen.groupby(['Year_of_Release', 'Platform'])['EU_Sales'].sum()

current_gen_sales.unstack().plot(kind='bar',stacked=True, colormap= 'Blues',  grid=False)