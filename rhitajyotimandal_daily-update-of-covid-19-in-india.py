import pandas as pd

import plotly.express as px
df=pd.read_html('https://www.mohfw.gov.in/',index_col=0)[0]

#x=df.iloc[33,0]

df=df[:33]

#x=int(x.split(' ')[0].lstrip('*'))

x=0
df['State/UT']=df['Name of State / UT']

df['Confirmed']=df['Total Confirmed cases (Including 111 foreign Nationals)'].astype(int)

df['Cured']=df['Cured/Discharged/Migrated'].astype(int)

df['Death']=df['Deaths ( more than 70% cases due to comorbidities )'].astype(int)

new_cols=['State/UT','Confirmed','Cured','Death']

df=df[new_cols]
l=[]

for i in range(len(df)):

    a=int(df['Confirmed'][i])-(int(df['Cured'][i])+int(df['Death'][i]))

    l.append(a)

    

df['Active']=l

if 'Active' not in new_cols:

    new_cols.insert(2,'Active')

df=df[new_cols]
d=pd.DataFrame({"Statistics":['Confirmed','Active','Cured','Death'],

  "Values":[sum(df['Confirmed'])+x,sum(df['Active'])+x,sum(df['Cured']),sum(df['Death'])]})
fig=px.pie(d,names=d.Statistics[1:],values=d.Values[1:],

       title='Current Situation in India (Total confirmed: {})'.format(d.Values[0]))

fig.update_traces(marker=dict(colors=['#263fa3', '#2fcc41','#cc3c2f'],line=dict(color='#FFFFFF', width=2)))
rate=pd.DataFrame()

rate['Active/Confirmed']=df.Active/df.Confirmed

rate['Cured/Confirmed']=df.Cured/df.Confirmed

rate['Death/Confirmed']=df.Death/df.Confirmed



df=df.join(rate)
fig=px.scatter(df,x='Active/Confirmed',y='Confirmed',size='Active',color='State/UT',size_max=50,

               title='Current Active Cases (Bubble size: No. of Active Cases)')

fig.update_xaxes(title="Active to Confirmed Ratio")
fig=px.scatter(df,x='Cured/Confirmed',y='Confirmed',size='Cured',color='State/UT',size_max=50,

               title='Current Cured Cases (Bubble size: No. of Cured)')

fig.update_xaxes(title="Cured/Discharged to Confirmed Ratio")
fig=px.scatter(df,x='Death/Confirmed',y='Confirmed',size='Death',color='State/UT',size_max=50,

               title='Current Deaths (Bubble size: No. of Deaths)')

fig.update_xaxes(title="Deaths to Confirmed Ratio")