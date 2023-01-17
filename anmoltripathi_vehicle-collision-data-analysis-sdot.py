import pandas as pd

import numpy as np

import seaborn as sns

import statsmodels

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster

from pandas_profiling import ProfileReport

import plotly.express as px

import pprint

import plotly.graph_objs as go

import matplotlib as mpl

import calendar
sns.palplot(sns.color_palette("GnBu_d",10))

sns.set_palette(sns.color_palette("GnBu_d",10))
pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 20)
df = pd.read_csv("../input/seattle-sdot-collisions-data/Collisions.csv", low_memory=False, parse_dates=["INCDATE"])

df_original = df.copy()

df.head()
df.shape
corrMatrix = df.corr()

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corrMatrix, annot=True, ax=ax)

plt.show()
df.drop(["INCKEY","COLDETKEY"],axis=1,inplace=True)
df["UNDERINFL"].unique()
def for_UNDERINFL(x):

    if x in ['Y','1']:

        return "Y"

    elif x in ['N','0']:

        return "N"

    else:

        return x

    

df["UNDERINFL"] = df["UNDERINFL"].apply(for_UNDERINFL)
temp = df[["SEVERITYCODE","SEVERITYDESC"]]

temp[~(temp.duplicated())].style.hide_index()
df["SEVERITYCODE"].fillna('0', inplace=True)
df["ADDRTYPE"].fillna('Not Mentioned', inplace=True)
df["MONTH"] = df["INCDATE"].dt.month

df["YEAR"] = df["INCDATE"].dt.year
sns.countplot(df["SEVERITYCODE"])
df["SEVERITYCODE"].value_counts(normalize=True)*100
fig, ax = plt.subplots(figsize=(20,5))

sns.countplot(df["INCDATE"].dt.year,ax=ax)

ax.set_xlabel('Year', fontsize=18)

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=15)

ax.set_ylabel('Collision Count', fontsize=18)

plt.title("Collision count YoY", fontsize=18)

# plt.savefig('./blog/1.png')
fig, ax = plt.subplots(figsize=(10,5))

sns.countplot(df["INCDATE"].dt.month,ax=ax)

ax.set_xlabel('Months', fontsize=18)

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=15)

ax.set_ylabel('Collision Count', fontsize=18)

plt.title("Collision count MoM", fontsize=18)

# plt.savefig('./blog/2.png')
df["INCTIME"] = pd.to_datetime(df["INCDTTM"])

for idx, dt in enumerate(df["INCDTTM"]):

    if ':' not in dt:

        df["INCTIME"][idx] = np.nan



b = [0,4,8,12,16,20,24]

l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']

df["TIMEOFDAY"] = pd.cut(df["INCTIME"].dt.hour, bins=b, labels=l, include_lowest=True)
1 - (df["TIMEOFDAY"].isnull().sum() / len(df))
pd.DataFrame({'TIME' : ['00:00 to 04:00', '04:00 to 08:00', '08:00 to 12:00', '12:00 to 16:00', '16:00 to 20:00', '20:00 to 00:00']},

             index=['Late Night', 'Early Morning','Morning','Noon','Eve','Night'])
df_TOD = pd.DataFrame(df['OBJECTID'].groupby(df['TIMEOFDAY'].astype('object')).count())

df_TOD = df_TOD.sort_values(by='OBJECTID',ascending=False)

df_TOD["cumpercentage"] = df_TOD["OBJECTID"].cumsum()/df_TOD["OBJECTID"].sum()*100





fig, ax = plt.subplots(figsize=(10,5))

ax2 = ax.twinx()

ax.bar(df_TOD.index, df_TOD["OBJECTID"])

ax2.plot(df_TOD.index, df_TOD["cumpercentage"], color="C9", marker="D", ms=7)

ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())



ax.tick_params(axis="y", colors="C0")

ax2.tick_params(axis="y", colors="C1")



plt.title("Pareto Diagram: Vehicle collisions on Time", fontsize=18)



# plt.savefig('./blog/4.png')

plt.show()
pd.DataFrame(df['OBJECTID'].groupby(df['TIMEOFDAY']).count())
fig, ax = plt.subplots(figsize=(10,5))

sns.countplot(df["TIMEOFDAY"], hue=df['SEVERITYDESC'],ax=ax)
pd.crosstab(df.TIMEOFDAY, df.SEVERITYCODE)
fig, ax = plt.subplots(figsize=(30,20))

df_temp = df.copy()

df_temp = df_temp[df_temp["SEVERITYCODE"]!='0']

(pd.crosstab(df_temp.TIMEOFDAY, df_temp.SEVERITYDESC).apply(lambda r: r/r.sum(), axis=1)*100).plot.pie(subplots=True, ax=ax, autopct='%1.0f%%', pctdistance=0.5,labeldistance=1.2, radius=1.2, legend=None)

# plt.savefig('./blog/5.png')
(pd.crosstab(df.TIMEOFDAY, df.SEVERITYDESC).apply(lambda r: r/r.sum(), axis=1)*100)
sns.set_palette(sns.color_palette("tab10",10))

speed = df["SPEEDING"]

speed = speed.fillna("N")

speed.value_counts(normalize=True).plot.pie(subplots=True, autopct='%1.0f%%')

# plt.savefig('./blog/6.png')
((df[df["SPEEDING"]=="Y"].groupby("SEVERITYDESC")["OBJECTID"].count().sort_values(ascending=False) / df[df["SPEEDING"]=="Y"].groupby("SEVERITYCODE")["OBJECTID"].count().sum())*100).plot.pie(subplots=True,figsize=(8,8), autopct='%1.0f%%' , title="Distribution of Speeding Collisions by Severity")

# plt.savefig('./blog/7.png')
(df[df["SPEEDING"].isnull()]["SEVERITYDESC"].value_counts(normalize=True)*100).plot.pie(subplots=True,figsize=(8,8), autopct='%1.0f%%', title="Distribution of Non Speeding Collisions by Severity")

# plt.savefig('./blog/8.png')
fig, ax = plt.subplots(figsize=(10,5))

sns.countplot(df["TIMEOFDAY"], hue=df['SPEEDING'],ax=ax)

ax.set_xlabel('Day Time', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Count', fontsize=18)

plt.title("Collision count by Day Time", fontsize=18)

for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),

                 textcoords='offset points')

# plt.savefig('./blog/9.png')
sns.set_palette(sns.color_palette("Oranges_r",10))

fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(y=(df.groupby('TIMEOFDAY')['SPEEDING'].apply(lambda x: x[x == 'Y'].count()) /df.groupby('TIMEOFDAY')['OBJECTID'].count())*100, x=list(df.groupby('TIMEOFDAY').groups.keys()),ax=ax)

ax.set_xlabel('Day Time', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Count %', fontsize=18)

plt.title("Collision count by Day Time", fontsize=18)

for p in ax.patches:

             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),

                 textcoords='offset points')

# plt.savefig('./blog/10.png')
fig, ax = plt.subplots(figsize=(10,5))

sns.set_palette(sns.color_palette("tab10_r",10))

df[(df["SPEEDING"]=="Y")].groupby("SDOT_COLDESC")["OBJECTID"].count().sort_values(ascending=False).head(10).plot.barh(ax=ax)

ax.set_xlabel('Collision Count', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Type', fontsize=18)

plt.title("Collision type by count of Speeding collisions", fontsize=18)

# plt.savefig('./blog/11.png')
fig, ax = plt.subplots(figsize=(10,5))

sns.set_palette(sns.color_palette("coolwarm_r",10))

df[df["SPEEDING"]=="Y"].groupby("LOCATION")["OBJECTID"].count().sort_values(ascending=False).head(5).plot.barh() 

ax.set_xlabel('Collision Count', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Location', fontsize=18)

plt.title("Location by count of Speeding collisions", fontsize=18)

# plt.savefig('./blog/12.png')
m = folium.Map(location=[df["Y"].mean(), df["X"].mean()], zoom_start=12)

folium.Marker((47.69034175,-122.3290808),popup="MOST SPEEDING COLLISIONS: BATTERY ST TUNNEL").add_to(m)    

m
sns.set_palette(sns.color_palette("coolwarm",10))

fig, ax = plt.subplots(figsize=(10,5))

sns.countplot(df["TIMEOFDAY"], hue=df['INATTENTIONIND'],ax=ax)

ax.set_xlabel('Day Time', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Count', fontsize=18)

plt.title("Collision count by Day time for Inattentive driving collisions", fontsize=18)

# plt.savefig('./blog/14.png')
sns.set_palette(sns.color_palette("bone",10))

fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(y=((df.groupby('TIMEOFDAY')['INATTENTIONIND'].apply(lambda x: x[x == 'Y'].count()) /df.groupby('TIMEOFDAY')['OBJECTID'].count())*100).sort_values(), x=list(df.groupby('TIMEOFDAY').groups.keys()),ax=ax)

for p in ax.patches:

             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 5),

                 textcoords='offset points')

ax.set_xlabel('Day Time', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Count %', fontsize=18)

plt.title("% of Collision count by Day time for Inattentive driving collisions", fontsize=18)

# plt.savefig('./blog/15.png')
sns.set_palette(sns.color_palette("brg",10))

fig, ax = plt.subplots(figsize=(10,5))

(df[df["INATTENTIONIND"]=="Y"].groupby("COLLISIONTYPE")["OBJECTID"].count().sort_values()).plot.barh()

ax.set_xlabel('Collision Count', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision type', fontsize=18)

plt.title("Collision type by count for Inattentive driving collisions", fontsize=18)

# plt.savefig('./blog/16.png')
(df[df["INATTENTIONIND"]=="Y"].groupby("COLLISIONTYPE")["OBJECTID"].count().sort_values(ascending=False))[0]/(df[df["INATTENTIONIND"]=="Y"].groupby("COLLISIONTYPE")["OBJECTID"].count().sum()) * 100
fig, ax = plt.subplots(figsize=(10,5))

sns.set_palette(sns.color_palette("GnBu",2))

sns.countplot(df["TIMEOFDAY"], hue=df['UNDERINFL'],ax=ax)

ax.set_xlabel('Day Time', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Count', fontsize=18)

plt.title("Collision count by Day time for under influenced driving collisions", fontsize=18)

# plt.savefig('./blog/17.png')
pd.crosstab(df.TIMEOFDAY, df.UNDERINFL).apply(lambda x: (x/x.sum())*100, axis=0)
fig, ax = plt.subplots(figsize=(10,5))

df[df["UNDERINFL"]=="Y"].groupby("MONTH")["OBJECTID"].count().plot(ax=ax)

ax.set_xlabel('Month', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Collision Count', fontsize=18)

plt.title("Collision count by Month for under influenced driving collisions", fontsize=18)

# plt.savefig('./blog/19.png')
fig, ax = plt.subplots(figsize=(10,4))

df[df["UNDERINFL"]=="Y"].groupby("LOCATION")["OBJECTID"].count().sort_values(ascending=False).head(10).plot.barh(ax=ax)

ax.set_xlabel('Collision Count', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Location', fontsize=18)

plt.title("Location by Collision count for under influenced driving collisions", fontsize=18)

# plt.savefig('./blog/20.png')
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(35,13))

sns.set_palette(sns.color_palette("winter"))

col_type = list(df['COLLISIONTYPE'].unique())

(col_type).remove(np.nan)



for i in range(0,2):

    for j in range(0,5):

        sns.countplot(df[df['COLLISIONTYPE'] == col_type[0]]['TIMEOFDAY'], ax=ax[i,j])

        ax[i,j].set_title(col_type[0])

        col_type.pop(0)

# plt.savefig('./blog/21.png')
pd.crosstab(df.COLLISIONTYPE, df.TIMEOFDAY).apply(lambda r: (r/r.sum())*100, axis=0)
sns.set_palette(sns.color_palette("hot"))

tempdf = df[(df["COLLISIONTYPE"]=="Left Turn")|(df["COLLISIONTYPE"]=="Right Turn")]

fig, ax = plt.subplots(figsize=(7,7))

ax.pie(tempdf["COLLISIONTYPE"].value_counts(), textprops={'color':"white", 'fontsize': 14}, autopct='%1.0f%%')



ax.legend(tempdf["COLLISIONTYPE"],

          title="Collision Type (Turns)",

          loc="upper center",

          bbox_to_anchor=(1, 0, 0.5, 1))

# plt.savefig('./blog/24.png')

plt.show()

((df[df["COLLISIONTYPE"]=="Left Turn"]["OBJECTID"].count() - df[df["COLLISIONTYPE"]=="Right Turn"]["OBJECTID"].count()) / df[(df["COLLISIONTYPE"]=="Right Turn")|(df["COLLISIONTYPE"]=="Left Turn")]["OBJECTID"].count()) * 100
df[df["COLLISIONTYPE"]=="Left Turn"].groupby("ADDRTYPE")["OBJECTID"].count()
df[df["COLLISIONTYPE"]=="Right Turn"].groupby("ADDRTYPE")["OBJECTID"].count()
sns.set_palette(sns.color_palette("Paired"))

x = pd.DataFrame(df[(df["COLLISIONTYPE"]=='Other')&(df["TIMEOFDAY"]=='Late Night')].groupby('SDOT_COLDESC')['OBJECTID'].count()).sort_values(by='OBJECTID', ascending=False)
others_values = x["OBJECTID"].tail(12).sum()

x.drop(x.tail(12).index,inplace=True)

x = x.append(pd.DataFrame(data={"OBJECTID":others_values}, index=["Others Remaining"]))
x.plot(kind='pie',subplots=True,figsize=(10,10), autopct='%1.0f%%',textprops={'color':"grey", 'fontsize': 14}, explode=(0.1,0.1,0,0,0,0))

plt.axes().set_facecolor("white")

plt.axes().get_legend().remove()

plt.title("Collision type distribution for 'Late Night'")

# plt.savefig('./blog/25.png')

plt.show()
fig, ax = plt.subplots(figsize=(10,10))

sns.countplot(y=df['SDOT_COLDESC'],ax=ax,orient="h")
fig, ax = plt.subplots(figsize=(10,5))

df['SDOT_COLDESC'].value_counts(normalize=True).head(5).plot.barh(title="% of accidents by collision type",ax=ax)

ax.set_xlabel('% of accidents', fontsize=18)

ax.tick_params(axis="x", labelsize=12)

ax.tick_params(axis="y", labelsize=12)

# ax.set_ylabel('Collision Count', fontsize=18)

# plt.title("Collision count by Month for under influenced driving collisions", fontsize=18)

# plt.savefig('./blog/26.png')
df1 = df.copy()



df1["COLLISIONTYPE"].fillna("Unknown",inplace=True)

df1["SDOT_COLDESC"].fillna("Unknown",inplace=True)

fig = px.sunburst(

    data_frame=df1[~df1["TIMEOFDAY"].isnull()],

    path= ["TIMEOFDAY",'COLLISIONTYPE',"SDOT_COLDESC"], 

    color="TIMEOFDAY",

    color_discrete_sequence=px.colors.qualitative.Pastel,

    maxdepth=-1,                        

    branchvalues="total", 

    title="Breakdown of collision type by time",

)

fig.update_traces(textinfo='label+percent entry') # percent parent

fig.update_layout(margin=dict(t=30, b=10, r=10, l=10),

                  width=1000, height=1000)

fig.show()
pd.crosstab(df.COLLISIONTYPE, df.TIMEOFDAY).apply(lambda r: ((r/r.sum())*100).round(2), axis=1)
f_df = df.groupby('ADDRTYPE')[['FATALITIES']].sum()

f_df['INJURIES'] = df.groupby('ADDRTYPE')[['INJURIES']].sum()

f_df['SERIOUSINJURIES'] = df.groupby('ADDRTYPE')[['SERIOUSINJURIES']].sum()

f_df['ACDNTCOUNT'] = df.groupby('ADDRTYPE')[['OBJECTID']].count()

f_df = f_df.reset_index()

f_df
sns.set_palette('twilight_shifted_r')

a = df.groupby('ADDRTYPE')[['FATALITIES']].sum()

f_df1 = a.apply(lambda x : 100 * x/float(x.sum()))

a['P_INJURIES'] = df.groupby('ADDRTYPE')[['INJURIES']].sum()

f_df1 = a.apply(lambda x : 100 * x/float(x.sum()))

a['P_SERIOUSINJURIES'] = df.groupby('ADDRTYPE')[['SERIOUSINJURIES']].sum()

f_df1 = a.apply(lambda x : 100 * x/float(x.sum()))

a['P_ACDNTCOUNT'] = df.groupby('ADDRTYPE')[['OBJECTID']].sum()

f_df1 = a.apply(lambda x : 100 * x/float(x.sum()))

#f_df1 = f_df1.drop(columns=['FATALITIES', 'INJURIES'])

f_df1.reset_index(inplace=True)

f_df1.rename(columns={'FATALITIES':'P_FATALITIES'}, inplace=True)

f_df1



fig, ax = plt.subplots(figsize=(10,10))

df2=pd.melt(f_df1,id_vars=['ADDRTYPE'],var_name='DESC', value_name='PERCENTAGE OF TOTAL')

df2

sns.barplot(x='ADDRTYPE', y='PERCENTAGE OF TOTAL', hue='DESC', data=df2, ax=ax)

plt.xticks(rotation=90)

# plt.savefig('./blog/29.png')
layout = go.Layout(

    title="Location wise Type of Injuries",

    xaxis=dict(

        title="ACCIDENT ADDRESS TYPE",

        linecolor='#A9A9A9',

        showgrid=True),

    yaxis=dict(

        title="NUMBER OF ACCIDENTS",

        showgrid=False),

    barmode='group'

)



fig = go.Figure(data=[

    go.Bar(name='TOTAL #OF ACCIDENT', x=f_df['ADDRTYPE'], y=f_df['ACDNTCOUNT']),

    go.Bar(name='INJURIES', x=f_df['ADDRTYPE'], y=f_df['INJURIES']),

    go.Bar(name='SERIOUSINJURIES', x=f_df['ADDRTYPE'], y=f_df['SERIOUSINJURIES']),

    go.Bar(name='FATALITIES', x=f_df['ADDRTYPE'], y=f_df['FATALITIES'])

    

], layout= layout)

fig.show()
pd.crosstab(df["COLLISIONTYPE"],df["ADDRTYPE"])
df1 = df[df['INCDATE'].dt.year > 2016]

a= df1.groupby(['INTKEY','LOCATION'])['OBJECTID'].count() #,

a = a.to_frame()

a.reset_index(inplace=True)

a.rename(columns={'OBJECTID':'# OF ACCIDENTS'}, inplace=True)

bins = [0, 10, 20, 30, 40,50,120]

labels = ['Below 10', 'Between 10 and 20','Between 20 and 30','Between 30 and 40','Between 40 and 50',

         'Above 50']

a['BINNED'] = pd.cut(a['# OF ACCIDENTS'], bins, labels=labels)

a = a.sort_values(by=['# OF ACCIDENTS'],ascending=False)

b = a.head(20)

fig = px.bar(b, x="# OF ACCIDENTS", y="LOCATION", color="# OF ACCIDENTS")

fig.show()
fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(10,8))

sns.countplot(y=df['ROADCOND'], ax=ax[0],orient="h")

sns.countplot(y=df['LIGHTCOND'], ax=ax[1],orient="h")

# plt.savefig('./blog/33.png')
df["LIGHTCOND"].unique()
df[(df["SEVERITYCODE"]=="1")]["LIGHTCOND"].value_counts()
fig, ax = plt.subplots(nrows=2,ncols=4,figsize=(20,10))

ax[0,0].pie(df[(df["SEVERITYCODE"]=="1")]["ROADCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})

ax[0,1].pie(df[(df["SEVERITYCODE"]=="2")]["ROADCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})

ax[0,2].pie(df[(df["SEVERITYCODE"]=="2b")]["ROADCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})

ax[0,3].pie(df[(df["SEVERITYCODE"]=="3")]["ROADCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})



ax[0,0].set_title("Property Damage Only Collision",color="grey")

ax[0,1].set_title("Injury Collision",color="grey")

ax[0,2].set_title("Serious Injury Collision",color="grey")

ax[0,3].set_title("Fatal Collision",color="grey")





ax[1,0].pie(df[(df["SEVERITYCODE"]=="1")]["LIGHTCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})

ax[1,1].pie(df[(df["SEVERITYCODE"]=="2")]["LIGHTCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})

ax[1,2].pie(df[(df["SEVERITYCODE"]=="2b")]["LIGHTCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})

ax[1,3].pie(df[(df["SEVERITYCODE"]=="3")]["LIGHTCOND"].value_counts(), textprops={'color':"white", 'fontsize': 14})





ax[0,3].legend(["Dry","Wet","Unknown","Ice","Snow/Slush","Other","Standing Water","Sand/Mud/Dirt","Oil"],

          title="Road Condition",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))

ax[1,3].legend(["Daylight","Dark-Street Light On","Unknown","Dawn","Dark-No Street Lights","Dark-Street Light Off","Other","Dark-Unknown Lighting"],

          title="Light Condition",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))

# plt.savefig('./blog/34.png')

plt.show()
a = df[df['YEAR']!=2020].groupby(['MONTH'])['OBJECTID'].count()/df[df['YEAR']!=2020].groupby(['MONTH']).size().nunique()

a = a.to_frame()

a.reset_index(inplace=True)

a.rename(columns = {'OBJECTID':'2013-2019'}, inplace=True)

b= df[df['YEAR']==2020].groupby(['MONTH'])['OBJECTID'].count()

b = b.to_frame()

b.reset_index(inplace=True)

b.rename(columns = {'OBJECTID':'2020'}, inplace=True)

# merging dataframes

df_compare = pd.merge(a, b, on='MONTH', how='left')
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_compare['MONTH'], y=df_compare['2013-2019'],mode='lines+markers', 

                         name='Before 2020'))

fig.add_trace(go.Scatter(x=df_compare['MONTH'], y=df_compare['2020'],mode='lines+markers',

                        name='2020'))

fig.update_layout(title="Impact of COVID-19 on Road Accidents",

                  xaxis_title='Month',

                  yaxis_title='Average Number of Accidents'

                 )

# plt.savefig('./blog/35.png')

fig.show()
sf_map = folium.Map(location=[df["Y"].mean(), df["X"].mean()],  zoom_start=12, control_scale=True, min_zoom=11)



df_filtered = df[(df["YEAR"]>= 2014)]

HeatMap(data=df_filtered[["X","Y","VEHCOUNT"]].groupby(['Y', 'X']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(sf_map)



sf_map