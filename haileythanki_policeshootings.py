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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
pd.options.display.max_rows=None
df = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv", parse_dates=["date"])
df
df.info()
df.describe(include="O")
df.describe()
!pip install calmap

import calmap

daywise_incidents = df.groupby(df["date"])["id"].count()

plot, axis = calmap.calendarplot(daywise_incidents, monthticks=2, daylabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fillcolor="grey", linewidth=1, fig_kws=dict(figsize=(20,20)))

plot.colorbar(axis[0].get_children()[1], ax=axis, cmap=plt.cm.get_cmap("Blues",9), orientation="horizontal", label = "Number of incidents per day")

plt.show()
df["date"] = pd.to_datetime(df.date)
df.info()
df["month_year"] = df["date"].dt.to_period("M")
df.head(5)
month_year_vc = df.month_year.value_counts()

month_year_vc
plt.figure(figsize=(18,10))

sns.set(font_scale=0.7, palette="viridis")

sns.lineplot(x=month_year_vc.index.astype(str), y=month_year_vc.values)

plt.title("Number of shootings over the years", fontsize=20)

plt.xlabel("Month and year", fontsize=10)

plt.ylabel("Number of shootings", fontsize=10)

plt.xticks(rotation=45) 

plt.ylim(60,120)

plt.yticks(np.arange(60,120,2))

plt.show()
df.manner_of_death.unique()
plt.figure(figsize=(12,8))

labels = ['shot', 'shot and tasered']

count_shot = df[df.manner_of_death=="shot"].id.count()

percentage_shot = (float(count_shot)/5416)*100

percentage_shot_tased = 100 - percentage_shot

percentages = [percentage_shot, percentage_shot_tased]

explode=(0.1,0)

plt.rcParams['font.size'] = 15

plt.pie(percentages, explode=explode, labels=labels, autopct='%1.0f%%', shadow=False, startangle=0, pctdistance=1.2, labeldistance=1.4)

plt.axis('equal')

plt.title("Manner of death", fontsize=20)

plt.legend(frameon=False, bbox_to_anchor=(1.5,0.8), fontsize=15)

plt.show()
armed_vc = df.armed.value_counts()

print(sum((armed_vc[armed_vc.values<80]).values))

armed_vc.loc["misc"] = sum((armed_vc[armed_vc.values<80]).values)

armed_vc.drop((armed_vc[armed_vc.values<80]).index, inplace=True, axis=0)

print(armed_vc)

plt.figure(figsize=(12,8))

plt.xticks(rotation=45, fontsize=12)

sns.set(font_scale=1, palette="viridis")

sns.barplot(data=df, x=armed_vc.index, y=armed_vc.values)

plt.show()
f = df[df.gender=="F"].id.count()

m = df[df.gender=="M"].id.count()

perc_f = (f/(f+m))*100

perc_m = 100-perc_f

print("The percentage of female victims are: ",perc_f,"%")

print("The percentage of male victims are: ",perc_m,"%")
ill = df[df.signs_of_mental_illness==True].id.count()

not_ill = df[df.signs_of_mental_illness==False].id.count()

perc_ill = (ill/(ill + not_ill))*100

perc_not_ill = 100 - perc_ill

print("Percentage of victims showing signs of mental illness:",perc_ill,"%")

print("Percentage of victims not showing any signs of mental illness:",perc_ill,"%")
plt.figure(figsize=(5,8))

sns.set(font_scale=1, palette="viridis")

sns.countplot(data=df, x="gender", hue="signs_of_mental_illness")

plt.xticks(fontsize=10)

plt.show()
plt.figure(figsize=(12,12))

sns.violinplot(hue=df.gender, y=df.age, x=df.signs_of_mental_illness, split=True, inner="quartile")

plt.yticks(np.arange(0,100,2))

plt.show()
plt.figure(figsize=(12,8))

sns.set(font_scale=1, palette="viridis")

sns.countplot(data=df, x="manner_of_death", hue="flee")

plt.xticks(fontsize=10)

plt.show()
df.race.value_counts()
plt.figure(figsize=(12,10))

labels = ["White", "Black", "Asian" , "Native American", "Hispanic", "Other", "unknown"]

perc_W = (float(df[df.race=="W"].id.count())/5416)*100

perc_B = (float(df[df.race=="B"].id.count())/5416)*100

perc_A = (float(df[df.race=="A"].id.count())/5416)*100

perc_N = (float(df[df.race=="N"].id.count())/5416)*100

perc_H = (float(df[df.race=="H"].id.count())/5416)*100

perc_O = (float(df[df.race=="O"].id.count())/5416)*100

perc_U = (float(df[df.race==None].id.count())/5416)*100

percentages = [perc_W, perc_B, perc_A, perc_N, perc_H, perc_O, perc_U]

explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1)

plt.rcParams['font.size'] = 15

plt.pie(percentages, autopct="%1.0f%%", shadow=False, startangle=0, pctdistance=1.06, labeldistance=1.1,rotatelabels=True)

plt.axis('equal')

plt.title("Race of Victims", fontsize=20)

plt.legend(frameon=False, bbox_to_anchor=(1.5,0.8), fontsize=15, labels=labels)

plt.show()
df.body_camera.value_counts()
((((df["id"].groupby([df["body_camera"],df["race"]])).count())/5416)*100)
plt.figure(figsize=(12,9))

(((((df["id"].groupby([df["body_camera"],df["race"]])).count())/5416)*100).rename("percentage").reset_index().pipe((sns.barplot, "data"), x="body_camera", y="percentage", hue="race"))

plt.yticks(np.arange(0,44,2))

plt.show()

((df["id"].groupby([df["body_camera"],df["threat_level"]])).count())
plt.figure(figsize=(12,9))

(((((df["id"].groupby([df["body_camera"],df["threat_level"]])).count())/5416)*100).rename("percentage").reset_index().pipe((sns.barplot, "data"), x="body_camera", y="percentage", hue="threat_level"))

plt.yticks(np.arange(0,60,2))

plt.show()

df.state.value_counts()
plt.figure(figsize=(12,8))

sns.set(font_scale=1, palette="viridis")

sns.countplot(data=df, x="state")

plt.xticks(fontsize=10)

plt.xticks(rotation=45, fontsize=9)

plt.show()
df[df.state=="CA"].city.value_counts()
df[df.state=="TX"].city.value_counts()
df[df.state=="FL"].city.value_counts()
data = df[(df.signs_of_mental_illness == True) & df.race.isin(['W', 'B', 'H'])]

fig, ax = plt.subplots(figsize=(40,10))

sns.boxplot(x="state", y="age", data=data, hue='race')

ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('Age', fontsize=20)

ylabels= ['{:,.0f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)

ax.legend(fontsize = 14, title='Races')

plt.show()
threat_level = pd.get_dummies(df.threat_level, prefix='threat_level:')

mental_illness = pd.get_dummies(df.signs_of_mental_illness, prefix='mental_illness:')

flee = pd.get_dummies(df.flee, prefix='flee:')

df_sub = pd.concat([threat_level, mental_illness, flee], axis=1)

sns.set(style="white")

corr = df_sub.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

figure, ax = plt.subplots(figsize=(15,15 ))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

xlabels = [x for x in df_sub.columns]

ax.set_xticklabels(xlabels, rotation=90, fontsize=12)

ylabels= [x for x in df_sub.columns]

ax.set_yticklabels(ylabels,fontsize=12)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
df["armed_tf"] = True

df.loc[df.armed=="unarmed","armed_tf"]=False

armed = pd.get_dummies(df["armed_tf"], prefix='armed:')

mental_illness = pd.get_dummies(df.signs_of_mental_illness, prefix='mental_illness:')

race = pd.get_dummies(df.race, prefix='race:')

df_sub = pd.concat([ race, armed, mental_illness], axis=1)

sns.set(style="white")

corr = df_sub.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

figure, ax = plt.subplots(figsize=(15,15 ))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

xlabels = [x for x in df_sub.columns]

ax.set_xticklabels(xlabels, rotation=90, fontsize=12)

ylabels= [x for x in df_sub.columns]

ax.set_yticklabels(ylabels,fontsize=12)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()