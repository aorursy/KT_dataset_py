import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
tst = pd.read_csv("../input/covid19/testing.csv")
fig = px.scatter(tst, x="state", y="samples", color="positive")
fig.show()
tst = pd.read_csv("../input/covid19/testing.csv")
fig = px.scatter(tst, x="state", y="positive", color="samples", marginal_y="rug", marginal_x="histogram")
fig.show()
#df = pd.read_csv(r'e:\ram\covid\covid.csv')
df = pd.read_csv("../input/covid19/covid.csv")
df['dlyconf']=1
df['dlycured']=1
df['dlydeaths']=1
cnt = df.state.count()
cnt


for i in range(cnt -1):
    if i == 0 :
        df.iloc[0,df.columns.get_loc('dlyconf')]= df.iloc[0,df.columns.get_loc('Confirmed')]
        df.iloc[0,df.columns.get_loc('dlycured')]= df.iloc[0,df.columns.get_loc('Cured')]
        df.iloc[0,df.columns.get_loc('dlydeaths')]= df.iloc[0,df.columns.get_loc('Deaths')]
    elif (i>0) & (df.iloc[i, df.columns.get_loc('state')] == df.iloc[i-1, df.columns.get_loc('state')]):
        df.iloc[i,df.columns.get_loc('dlyconf')]= df.iloc[i,df.columns.get_loc('Confirmed')] - df.iloc[i-1,df.columns.get_loc('Confirmed')]
        df.iloc[i,df.columns.get_loc('dlycured')]= df.iloc[i,df.columns.get_loc('Cured')] - df.iloc[i-1,df.columns.get_loc('Cured')]
        df.iloc[i,df.columns.get_loc('dlydeaths')]= df.iloc[i,df.columns.get_loc('Deaths')] - df.iloc[i-1,df.columns.get_loc('Deaths')]
    elif (df.iloc[i+1, df.columns.get_loc('state')] != df.iloc[i, df.columns.get_loc('state')]):
        df.iloc[i+1,df.columns.get_loc('dlyconf')] =  df.iloc[i+1, df.columns.get_loc('Confirmed')]
        df.iloc[i+1,df.columns.get_loc('dlycured')] =  df.iloc[i+1, df.columns.get_loc('Cured')]
        df.iloc[i+1,df.columns.get_loc('dlydeaths')] =  df.iloc[i+1, df.columns.get_loc('Deaths')]
i+=1   


grouped= df.groupby(['state'])
df['stwsconf'] = grouped['dlyconf'].transform('sum')
df['stwscured'] = grouped['dlycured'].transform('sum')
df['stwsdeaths'] = grouped['dlydeaths'].transform('sum')

grouped= df.groupby(['date'])
df['dtwsconf'] = grouped['dlyconf'].transform('sum')
df['dtwscured'] = grouped['dlycured'].transform('sum')
df['dtwsdeaths'] = grouped['dlydeaths'].transform('sum')


top10stws = df.groupby('state').apply(lambda x: x.nlargest(10,'stwsconf')).reset_index(drop=True)
cols1 = ['state', 'stwsconf', 'stwscured', 'stwsdeaths']
stws = top10stws[cols1].drop_duplicates(subset = ["state"])
top10st = stws.nlargest(10,'stwsconf')


#creating cured percentage column 
top10st['stwscuredperc']=1
cnt =top10st.state.count()
cnt

for i in range(cnt -1):
    top10st.iloc[i,top10st.columns.get_loc('stwscuredperc')]= round((top10st.iloc[i,top10st.columns.get_loc('stwscured')] /top10st.iloc[i,top10st.columns.get_loc('stwsconf')])*100,2)

#creating death percentage column 
top10st['stwsdeathperc']=1
cnt =top10st.state.count()
cnt
for i in range(cnt -1):
    top10st.iloc[i,top10st.columns.get_loc('stwsdeathperc')]= round((top10st.iloc[i,top10st.columns.get_loc('stwsdeaths')] /top10st.iloc[i,top10st.columns.get_loc('stwsconf')])*100,2)


top10dtws = df.groupby('date').apply(lambda x: x.nlargest(10,'dtwsconf')).reset_index(drop=True)
cols2 = [ 'date', 'dtwsconf', 'dtwscured', 'dtwsdeaths']
dtws = top10dtws[cols2].drop_duplicates(subset = ['date'])
top10dt = dtws.nlargest(10,'dtwsconf')

#df.to_csv(r'e:\ram\covid\covid_23052020.csv')


tot = df.dlyconf.sum()
tot1 = top10st.stwsconf.sum()
totperc = round((tot1/tot)*100, 2)

x = list(top10st.state)
y = list(top10st.stwsconf)

x_pos = [i for i, _ in enumerate(x)]

fig, ax = plt.subplots()
rects1 = ax.bar(x_pos,y, color='r')
plt.xlabel("Top 10 States " + str(tot1) + " out of "+ str(tot) + " i.e "+str(totperc)+"%")
plt.ylabel("Total Covid +ve")
plt.title("Top 10 States - Covid19 - covering "+ str(totperc)+ " %" )

plt.xticks(x_pos, x)
plt.xticks(rotation=30)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                height,
        ha='center', va='bottom')
autolabel(rects1)

plt.show()

plt.figure(figsize=(16,9))
labels = top10st[["state","stwscured","stwscuredperc"]].to_string(index=False)
labels1 = top10st[["state","stwscured","stwscuredperc"]]
ax = sns.barplot(x='stwscured', y='stwsconf', data=top10st, label=labels)
ax.set_xlabel('Total Cured')
ax.set_ylabel('Total Confirmed')
plt.xticks(rotation=45)
table = ax.table(cellText=labels1.values, colLabels=labels1.columns, loc='top', colWidths = [0.25, 0.25,0.25])
table.set_fontsize(14)
table.scale(1, 1.5)  
plt.text(0,25000,  labels,   bbox=dict(facecolor='red', alpha=0.1) , fontsize=12)
plt.show()
plt.figure(figsize=(16,9))
labels = top10st[["state","stwscured"]].to_string(index=False)
plt.text(0, 47,  labels,  bbox=dict(facecolor='red', alpha=0.1) )
ax = sns.barplot(x='state', y='stwscuredperc', data=top10st)
ax.set_xlabel('States')
ax.set_ylabel('Total Cured %')
ax.set_title('Top 10 States - Total % of cured')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(16,9))
labels = top10st[["state","stwsdeaths"]].to_string(index=False)
plt.text(4, 6,  labels,  bbox=dict(facecolor='red', alpha=0.1) )
ax = sns.barplot(x='state', y='stwsdeathperc', data=top10st)
ax.set_xlabel('States')
ax.set_ylabel('Total death %')
ax.set_title('Top 10 States - Total % of deaths')
plt.xticks(rotation=45)
plt.show()
tot = df.dlyconf.sum()
tot1 = top10dt.dtwsconf.sum()
totperc = round((tot1/tot)*100, 2)
ax = sns.barplot(x='date', y='dtwsconf', data=top10dt)
ax.set_xlabel('Top 10 dates on which maximum spread')
ax.set_ylabel('Total Confirmed')
ax.set_title('Top 10 dates. No.of Cases: ' + str(tot1) + ' out of '+ str(tot)+" i.e., "+str(totperc)+"%")
plt.xticks(rotation=45)
plt.show()
tot = df.dlycured.sum()
tot1 = top10dt.dtwscured.sum()
totperc = round((tot1/tot)*100, 2)
ax = sns.barplot(x='date', y='dtwscured', data=top10dt)
ax.set_xlabel('Top 10 dates on which max no. of  covid19 cases cured')
ax.set_ylabel('Total Cured cases')
ax.set_title('Top 10 dates. No.of Cured: ' + str(tot1) + ' out of '+ str(tot)+" i.e., "+str(totperc)+"%")
plt.xticks(rotation=45)
plt.show()
tot = df.dlydeaths.sum()
tot1 = top10dt.dtwsdeaths.sum()
totperc = round((tot1/tot)*100, 2)
ax = sns.barplot(x='date', y='dtwsdeaths', data=top10dt)
ax.set_xlabel('Top 10 dates on which max no. of  covid19 deaths')
ax.set_ylabel('Total Deaths')
ax.set_title('Top 10 dates. No.of Deaths: ' + str(tot1) + ' out of '+ str(tot)+" i.e., "+str(totperc)+"%")
plt.xticks(rotation=45)
plt.show()
top50dt = dtws.nlargest(50,'dtwsconf')
plt.figure(figsize=(16,9))
sns.scatterplot(y="date", x="dtwsconf", data=top50dt,  palette="hot")
txthead = "Last 50 days position of New Cases-Datewise"
plt.text(1000, 47.2,  txthead,  bbox=dict(facecolor='y', alpha=0.4), fontsize=25)
plt.show()
top50dt = dtws.nlargest(50,'dtwsconf')
plt.figure(figsize=(16,9))
sns.scatterplot(y="date", x="dtwsconf", data=top50dt)
sns.barplot(y="date", x="dtwscured", data=top50dt)
txthead = "Last 50 days position of New Cases with Cured Cases-Datewise"
plt.text(1000, 47.2,  txthead,  bbox=dict(facecolor='b', alpha=0.4), fontsize=15)
plt.show()
plt.figure(figsize=(16,9))
labels=top10st["state"].map(str)
plt.pie(top10st.stwsconf, labels=labels, autopct="%0.2f%%")
txthead = "Covid19 Cases - TOP 10 States with %"
plt.text(0, 1.2,  txthead,  bbox=dict(facecolor='g', alpha=0.1), fontsize=25)
plt.show()
plt.figure(figsize=(16,9))
labels=top10st["state"].map(str)
plt.pie(top10st.stwscured, labels=labels, autopct="%0.2f%%")
txthead = "Covid19 Cases Cured - TOP 10 States with %"
plt.text(0, 1.2,  txthead,  bbox=dict(facecolor='b', alpha=0.1), fontsize=25)
plt.show()
plt.figure(figsize=(16,9))
labels=top10st["state"].map(str)
plt.pie(top10st.stwsdeaths, labels=labels, autopct="%0.2f%%")
txthead = "Covid19 Deaths - TOP 10 States with %"
plt.text(0, 1.2,  txthead,  bbox=dict(facecolor='r', alpha=0.7), fontsize=25)
plt.show()
dfnew = df.groupby(['state'], as_index=False).sum()
plt.figure(figsize=(16,9))
labels = dfnew["state"]
plt.pie(dfnew.Confirmed, labels=labels,  autopct="%0.2f%%")
txthead = "Covid19 Cases - Statewise with %"
plt.text(0, 1.2,  txthead,  bbox=dict(facecolor='m', alpha=0.5), fontsize=25)
plt.show()
plt.figure(figsize=(16,9))
labels = dfnew["state"]
plt.pie(dfnew.Cured, labels=labels,  autopct="%0.2f%%")
txthead = "Cured Cases -Statewise with % "
plt.text(0, 1.2,  txthead,  bbox=dict(facecolor='g', alpha=0.2), fontsize=25)
plt.show()
plt.figure(figsize=(16,9))
labels = dfnew["state"]
plt.pie(dfnew.Deaths, labels=labels,  autopct="%0.2f%%")
txthead = "Deaths-Statewise with % "
plt.text(0, 1.2,  txthead,  bbox=dict(facecolor='m', alpha=0.3), fontsize=25)
plt.show()
import folium
dfnew = df.fillna(-1).groupby(['state'], as_index=False).max()
dfnew.Confirmed.sum()
lat=list(dfnew["latitude"])
lon=list(dfnew["longitude"])
st=list(dfnew["state"])
con=list(dfnew["Confirmed"])
con1=list(round((dfnew["Confirmed"]/dfnew["Confirmed"].sum())*100,2)+1)
cur=list(dfnew["Cured"])
cur1=list(round((dfnew["Cured"]/dfnew["Cured"].sum())*100,2)+1)
map0=folium.Map(location=[21, 87.21666744], zoom_start=5)
fg=folium.FeatureGroup("my map")
for lat,lon,st,con,cur,con1,cur1 in zip(lat,lon,st,con,cur,con1,cur1):
    #fg.add_child(folium.Marker(location=[lat,lon], popup="<b>"+st+"</b>"))
    fg.add_child(folium.CircleMarker(location=[lat, lon], radius=con1+1,
    popup=str(st) + " Confirmed: " + str(con) + " , Cured: " + str(cur), 
    tooltip=str(st) + " Conf: " + str(con) + " , Cured: " + str(cur),
    fill=True,  # Set fill to True
    color='red',
    fill_opacity=1.0)).add_to(map0)
    fg.add_child(folium.CircleMarker(location=[lat, lon], radius=cur1-5,
    popup="",
    tooltip=str(st) + " Conf: " + str(con) + " , Cured: " + str(cur),
    fill=True,  # Set fill to True
    color='green',
    fill_opacity=1.0)).add_to(map0)
    #txthead = str(con)
    #plt.text(lat, lon,  txthead,  bbox=dict(facecolor='y', alpha=0.4), fontsize=10)

map0.add_child(fg)
map0
