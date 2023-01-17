import numpy as np 
import pandas as pd 
import seaborn as sns
import  plotly.plotly  as py
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

inspections = pd.read_csv("../input/inspections.csv")
violations = pd.read_csv("../input/violations.csv")
inspections.head()
violations.head()
# violations['violation_code'].unique()
inspections.info()
violations.info()
# groupby activity_day to create new column with the name count
count_ins = inspections.groupby(['activity_date']).size().reset_index(name='count')
x = pd.DataFrame(count_ins['activity_date'])
y = pd.DataFrame(count_ins['count'])
# create separate df 
timePlot = pd.concat([x,y], axis=1)
timePlot.activity_date = pd.to_datetime(timePlot.activity_date)
timePlot.set_index('activity_date', inplace=True)
# show the first 10 raws, what we get
timePlot.head(10)

inspections.activity_date.max()
inspections.activity_date.min()
timePlot.plot.area(figsize=(20,6), linewidth=5, fontsize=15, stacked=False)
plt.xlabel('Activity Date', fontsize=15)
pd.crosstab(index=inspections['pe_description'], columns='count').sort_values(by=['count'],ascending=False)
inspections.groupby(['pe_description']).size()
# Number of Food Inspections by Risk
sns.set(rc={'figure.figsize':(14.7,8.27)})
g = sns.countplot(x="pe_description", data=inspections, order = inspections['pe_description'].value_counts().index, palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.tick_params(labelsize=10)
inspections['risk_level'] = inspections['pe_description'].apply(lambda x: x.split(")")[1])
print(inspections['risk_level'].head(5),'\n')
rsk = inspections.loc[inspections['pe_description'] == "RESTAURANT (0-30) SEATS HIGH RISK"]
rsk.head()
pd.crosstab(index=rsk['facility_name'], columns='count').sort_values(by=['count'],ascending=False).head(20)
pd.crosstab(index=inspections['risk_level'], columns='count').sort_values(by=['count'],ascending=False)
cntRisk = inspections.groupby(['risk_level']).size().reset_index(name='count')
cntRisk.head()

cntRisk['percent'] =  cntRisk['count']/cntRisk['count'].sum()
cntRisk.head()
# pie plot with percent
plot = cntRisk.plot.pie(y='percent', figsize=(5, 5))
# bar plot 
ax = sns.barplot(x="percent", y="risk_level", data=cntRisk)
# chart with sorting
inspections['risk_level'] = inspections['risk_level'].str.replace('SEATS', 'RESTAURANTS')
ax = inspections['risk_level'].value_counts().plot(kind='barh',colormap='Blues_r', figsize=(15,7),
                                         fontsize=13);
ax.set_alpha(0.5)
ax.set_title("Risk Level", fontsize=18)
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=10,
color='dimgrey')

# invert for largest on top 
ax.invert_yaxis()
pd.crosstab(index=inspections['employee_id'], columns='count').sort_values(by=['count'],ascending=False)
count_id = inspections.groupby(['employee_id']).size().reset_index(name='count')
x = pd.DataFrame(count_id['employee_id'])
y = pd.DataFrame(count_id['count'])
count_id = pd.concat([x,y], axis=1)
count_id['count'].describe()
pd.crosstab(index=inspections['facility_name'], columns='count').sort_values(by=['count'],ascending=False)
inspections['facility_name'].value_counts().nlargest(20).plot(kind="bar", figsize=(15, 6), fontsize=12, colormap='Blues_r')
subway = inspections.loc[inspections['facility_name'] == 'SUBWAY']
subway.head()
pd.crosstab(index=subway['risk_level'], columns='count').sort_values(by=['count'],ascending=False)
# inspections['facility_city'].count()
pd.crosstab(index=inspections['facility_city'], columns='count').sort_values(by=['count'],ascending=False)
# Fix bad data
inspections['facility_city'] = inspections['facility_city'].str.replace('Kern', 'KERN')
inspections['facility_city'] = inspections['facility_city'].str.replace('NORTHRISGE', 'NORTHRIDGE')
inspections['facility_city'] = inspections['facility_city'].str.replace('Rowland Heights', 'ROWLAND HEIGHTS')
inspections['facility_city'] = inspections['facility_city'].str.replace('WINNEKA', 'WINNETKA')
print(inspections['facility_city'].describe())
df_merge = inspections.merge(right=violations.reset_index(), how='left', on='serial_number', suffixes=('', '_codes_'))
df_merge.head()

pd.crosstab(index=df_merge['violation_description'], columns='count').sort_values(by=['count'],ascending=False)
# df_merge['violation_description'].value_counts().nlargest(10).plot(kind="bar", figsize=(15, 6), fontsize=12, colormap='Blues_r')
# sns.plotting_context(font_scale=5.5)
sns.set(rc={'figure.figsize':(14.7,10.27)})
top10 = sns.countplot(y="violation_description", data=df_merge, order = df_merge['violation_description'].value_counts().nlargest(10).index, palette="Blues_d")
top10.tick_params(labelsize=20)
#top10.set_xticklabels(top10.get_xticklabels(),rotation=45)

df_merge['facility_zip'].value_counts().nlargest(40).plot(kind="bar", figsize=(15, 6), fontsize=12, colormap='tab20c')