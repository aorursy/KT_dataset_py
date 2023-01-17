!pip install chart_studio
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot,init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
st = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
sp = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
st.set_index(["id"], inplace=True)
#Remove the first column and restore the original table
print ("row,  col")
print (st.shape[0],st.shape[1])
d_cols = [c for c in st.columns if 'd_' in c]
st1=st.loc[['HOBBIES_1_001_CA_1_validation',
            'FOODS_3_199_WI_2_validation',
            'HOUSEHOLD_2_150_TX_3_validation']][d_cols] 
fig1=st1.T.iplot(xTitle="Time",yTitle="Sales Num",title='3 items choosen randomly by 1913 days',fontsize='14')
calendar.head()
print ("row,col")
print (calendar.shape[0],calendar.shape[1])
dep_cols=[i for i in st['dept_id'].unique()]
dep_s=st.groupby(['dept_id']).sum().T.reset_index().rename(columns={'index': 'd'})\
                               .merge(calendar,how='left',on='d').set_index('date')
fig=dep_s[dep_cols].iplot()
fig2=dep_s[dep_cols].cumsum().plot(figsize=(20,10))
fig2.set_xlabel("Time")
fig2.set_ylabel("Sales Cumulative")
cat_cols=[i for i in st['cat_id'].unique()]
cat_s=st.groupby(['cat_id']).sum().T.reset_index().rename(columns={'index': 'd'})\
                               .merge(calendar,how='left',on='d').set_index('date')
cat_s[cat_cols].iplot( )
cat_s['nonevent']=cat_s.apply(lambda x: x.FOODS if x.event_name_1 is np.nan  else 0 ,axis=1)
cat_s['event']=cat_s.apply(lambda x: x.FOODS if x.event_name_1 is not np.nan else 0,axis=1)
f1=float(cat_s['event'].sum()/(len(cat_s['event'][cat_s.event>0])))
f2=float(cat_s['nonevent'].sum()/(len(cat_s['nonevent'][cat_s.nonevent>0])))
print(cat_s['nonevent'].sum())
print(len(cat_s['nonevent'][cat_s.nonevent>0]))
print("Foods on holidsys average sales:",f1)
print("Foods on non_event days average sales:", f2)
cat_s['nonevent_H']=cat_s.apply(lambda x: x.HOBBIES if x.event_name_1 is np.nan  else 0 ,axis=1)
cat_s['event_H']=cat_s.apply(lambda x: x.HOBBIES  if x.event_name_1 is not np.nan else 0,axis=1)
H1=float(cat_s['event_H'].sum()/(len(cat_s['event_H'][cat_s.event_H>0])))
H2=float(cat_s['nonevent_H'].sum()/(len(cat_s['nonevent_H'][cat_s.nonevent_H>0])))
print("HOBBIES on holidsys average sales:",H1)
print("HOBBIES on non_event days average sales:", H2)
cat_s['nonevent_w']=cat_s.apply(lambda x: x.HOUSEHOLD if x.event_name_1 is np.nan  else 0 ,axis=1)
cat_s['event_w']=cat_s.apply(lambda x: x.HOUSEHOLD  if x.event_name_1 is not np.nan else 0,axis=1)
w1=float(cat_s['event_w'].sum()/(len(cat_s['event_w'][cat_s.event_w>0])))
w2=float(cat_s['nonevent_w'].sum()/(len(cat_s['nonevent_w'][cat_s.nonevent_w>0])))
print("HOUSEHOLD on holidsys average sales:",w1)
print("HOUSEHOLD on non_event days average sales:", w2)
trace1 = go.Scatter(
                    x = cat_s.index,   
                    y = cat_s.nonevent,   
                    mode = "markers",     
                    name = "food nonevent",  
                    marker = dict(color = 'rgba(16, 112, 2, 0.3)'))
                   
trace2 = go.Scatter(
                    x = cat_s.index,
                    y = cat_s.event,
                    mode = "markers", 
                    name = "food event",
                    marker = dict(color = 'rgba(80, 26, 80, 1)'),
                    text= cat_s.event_name_1)
data = [trace1, trace2]
layout = dict(title = 'Foods',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False) ) 
fig = dict(data = data, layout = layout)
iplot(fig)
t3 = go.Scatter(
                    x = cat_s.index,   
                    y = cat_s.nonevent_H,   
                    mode = "markers",     
                    name = "Hobbies nonevent", 
                    marker = dict(color = 'rgba(160, 11, 2, 0.3)'))
                   
t4 = go.Scatter(
                    x = cat_s.index,
                    y = cat_s.event_H,
                    mode = "markers", 
                    name = "Hobbies event",
                    marker = dict(color = 'rgba(8, 26, 180, 1)'),
                    text= cat_s.event_name_1)
data2 = [t3, t4]
layout2 = dict(title = 'Hobbies',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False) ) 
fig2 = dict(data = data2, layout = layout2)
iplot(fig2)
t5 = go.Scatter(
                    x = cat_s.index,   
                    y = cat_s.nonevent_w,   
                    mode = "markers",     
                    name = "Household nonevent", 
                    marker = dict(color = 'rgba(80, 260, 18, 0.3)'))
                  
t6 = go.Scatter(
                    x = cat_s.index,
                    y = cat_s.event_w,
                    mode = "markers", 
                    name = "Household event",
                    marker = dict(color = 'rgba(160, 110, 20, 1)'),
                    text= cat_s.event_name_1)
data3 = [t5, t6]
layout3 = dict(title = 'Household',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False) ) 
fig3 = dict(data = data3, layout = layout3)
iplot(fig3)
#type1_cols=[i for i in cat_s['event_type_1'].unique()]
#type2_cols=[i for i in cat_s['event_type_2'].unique()]
#for i in range(len(type1_cols)):
tt=cat_s.groupby(['event_type_1']).sum()
tt.head()
tt[cat_cols].plot.bar(figsize=(20,10),title="Diffrent catagery under different types of festival")
fig3=cat_s[cat_cols].cumsum().plot(figsize=(20,10),title='Category Selling Trend')
fig3.set_xlabel("Time")
fig3.set_ylabel("Sales Cumulative")
store_cols=[i for i in st['store_id'].unique()]
store_s=st.groupby(['store_id']).sum().T.reset_index().rename(columns={'index': 'd'})\
                               .merge(calendar,how='left',on='d').set_index('date')
fig4=store_s[store_cols].cumsum().plot(figsize=(20,10),title='Stores Selling Trend')
fig4.set_xlabel("Time")
fig3.set_ylabel("Sales Cumulative")
state_cols=[i for i in st['state_id'].unique()]
state_s=st.groupby(['state_id']).sum().T.reset_index().rename(columns={'index': 'd'})\
                               .merge(calendar,how='left',on='d').set_index('date')
state_s[state_cols].iplot()
state_s.head()
state_s['is_snap_ca']=state_s.apply(lambda x: x.CA if x.snap_CA is 1 else 'NaN',axis=1)
CA_s1 = go.Scatter(
                    x = state_s.index,   
                    y = state_s.CA,   
                    mode = "markers",     
                    name = "CA", 
                    marker = dict(color = 'rgba(168, 86, 0, 0.5)'))
                  
CA_s2 = go.Scatter(
                    x = state_s.index,
                    y = state_s.is_snap_ca,
                    mode = "markers", 
                    name = "SANP_CA",
                    marker = dict(color = 'rgba(0, 86, 168, 0.8)'),
                    text= state_s.CA)
CAdata = [CA_s1 , CA_s2]
layout_ca = dict(title = 'Snap_CA',
              xaxis= dict(title= 'Date',ticklen= 2,zeroline= False) ) 
ca_fig = dict(data = CAdata, layout = layout_ca)
iplot(ca_fig)
state_s['is_snap_tx']=state_s.apply(lambda x: x.TX if x.snap_TX is 1 else 'NaN',axis=1)
TX_s1 = go.Scatter(
                    x = state_s.index,   
                    y = state_s.TX,   
                    mode = "markers",     
                    name = "TX", 
                    marker = dict(color = 'rgba(168, 86, 48, 0.5)'))
                  
TX_s2 = go.Scatter(
                    x = state_s.index,
                    y = state_s.is_snap_tx,
                    mode = "markers", 
                    name = "SANP_TX",
                    marker = dict(color = 'rgba(48, 86, 168, 0.8)'),
                    text= state_s.TX)
TXdata = [TX_s1 , TX_s2]
layout_tx = dict(title = 'Snap_TX',
              xaxis= dict(title= 'Date',ticklen= 2,zeroline= False) ) 
TX_fig = dict(data = TXdata, layout = layout_tx)
iplot(TX_fig)
state_s['is_snap_WI']=state_s.apply(lambda x: x.WI if x.snap_WI is 1 else 'NaN',axis=1)
WI_s1 = go.Scatter(
                    x = state_s.index,   
                    y = state_s.WI,   
                    mode = "markers",     
                    name = "WI", 
                    marker = dict(color = 'rgba(68, 186, 48, 0.5)'))
                  
WI_s2 = go.Scatter(
                    x = state_s.index,
                    y = state_s.is_snap_WI,
                    mode = "markers", 
                    name = "SANP_WI",
                    marker = dict(color = 'rgba(48, 36, 64, 0.8)'),
                    text= state_s.WI)
WIdata = [WI_s1 , WI_s2]
layout_WI = dict(title = 'Snap_WI',
              xaxis= dict(title= 'Date',ticklen= 2,zeroline= False) ) 
WI_fig = dict(data = WIdata, layout = layout_WI)
iplot(WI_fig)
fig5=state_s[state_cols].cumsum().plot(figsize=(20,10),title='State Selling Trend')
fig5.set_xlabel("Time")
fig5.set_ylabel("Sales Cumulative")
!pip install calmap
cat_s.head()
import calmap
import matplotlib.pylab as plt
cat_s.index = pd.to_datetime(cat_s.index)
for i in cat_s[cat_cols]:
    calmap.calendarplot(cat_s[i], monthticks=1,
                    cmap='RdBu_r',
                    fillcolor='grey', linewidth=0,
                    fig_kws=dict(figsize=(20, 16)),
                    subplot_kws={'title':i})
st1.head()
sp1=sp.loc[sp['item_id']=='HOBBIES_1_001']
sp_1=sp1.loc[sp1['store_id']=='CA_1']
sp_1.head()
st_h1=st.loc[['HOBBIES_1_001_CA_1_validation']][d_cols] 
st2=st_h1.T.reset_index().rename(columns={'index': 'd'})\
                               .merge(calendar,how='left',on='d').set_index('date')
st2.head()
!pip install isoweek
from isoweek import Week
def week2date(yw):
    stryw=str(yw)
    year=int('20'+stryw[1:3])
    week=int(stryw[3:5])
    dt6 = Week(year, week).monday()
    return dt6
sp_1['date']=sp_1.apply(lambda x: week2date(x.wm_yr_wk),axis=1)
sp_1.head()
s1 = go.Scatter(
                    x = st2.index,   
                    y = st2.HOBBIES_1_001_CA_1_validation,   
                    mode = "lines+markers",     
                    name = "HOBBIES_1_001_CA_1_Sales",  
                    marker = dict(color = 'rgba(166, 112, 2, 0.5)'))
                   
s2 = go.Scatter(
                    x = sp_1.date,
                    y = sp_1.sell_price,
                    mode = "lines", 
                    name = "sell price",
                    marker = dict(color = 'rgba(80, 26, 180, 1)'))
data_p= [s1, s2]
layout_p = dict(title = 'HOBBIES_1_001_CA_1 Prices on Sales',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= True) ) 
fig_p = dict(data = data_p, layout = layout_p)
iplot(fig_p)