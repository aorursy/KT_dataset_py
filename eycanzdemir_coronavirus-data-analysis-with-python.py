import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/covid19-turkey-current-report/covd.csv" , sep=';', encoding='latin-1')
df
df.drop(['TotalIntubation','TotalTest','TotalCase','TotalDeath','TotalHealing','TotalIntensive Care'],axis=1,inplace=True)

df  
new_df = df.loc[80:]

new_df
new_df.index
new_df.columns
plt.figure(figsize=(18,8))              #Size of Chart



sns.set_style('whitegrid')              #Style and drawing processes of our chart

p1 = sns.pointplot(x=new_df.Date,       #x-axis data

                   y=new_df.DailyDeath, #y-axis data

                   color="#22b2da",     #drawing color

                   alpha=0.5)           #transparency

plt.xticks(rotation= 90)                #We wrote the x-axis posts at a 90 degree angle
plt.figure(figsize=(18,8))

sns.set_style('whitegrid')

p1 = sns.pointplot(x=new_df.Date,

                   y=new_df.DailyDeath,

                   color="#22b2da",

                   alpha=5)

plt.grid(linestyle='--')

plt.xticks(rotation= 90)
plt.savefig(fname="covid_grafik.png",  #our graphic name

            facecolor="#f0f9e8",       #background color of the chart

            dpi=600,                   #graphic resolution

            quality=95)                #the quality of the chart(1-95)