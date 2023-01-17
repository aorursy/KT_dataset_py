## Import required packages

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

import pandas as pd

import tensorflow as tf

import geopandas as geopd

import datetime as dt



from mpl_toolkits.axes_grid1 import make_axes_locatable



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Input, SimpleRNN, GRU

from sklearn.model_selection import train_test_split

from keras.models import load_model

from keras.callbacks import ModelCheckpoint

import tensorflow as tf 
## Load data

train=pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test=pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

train['Date']=pd.to_datetime(train['Date'])

test['Date']=pd.to_datetime(test['Date'])

train['Days']=((train['Date'] - dt.datetime(2020,1,22)).dt.total_seconds()/(24*60*60)).apply(int)

test['Days']=((test['Date'] - dt.datetime(2020,1,22)).dt.total_seconds()/(24*60*60)).apply(int)
## Transform to dB scale, base 10

train['ConfirmedCases_dB']=10*np.log10(train['ConfirmedCases'])

train['Fatalities_dB']=10*np.log10(train['Fatalities'])

train.loc[np.where(train.loc[:, 'ConfirmedCases_dB']==-np.inf)[0],'ConfirmedCases_dB']=np.nan # remove -inf

train.loc[np.where(train.loc[:, 'Fatalities_dB']==-np.inf)[0],'Fatalities_dB']=np.nan # remove -inf



## Get unique countries and dates

countryUnique=np.unique(train['Country_Region'])

dateUnique=np.unique(train['Date'])
## Populate a geopandas world map with CC and F data

world = geopd.read_file(geopd.datasets.get_path('naturalearth_lowres'))



world['ConfirmedCases_dB']=0

world['Fatalities_dB']=0

for world_country_idx in range(0,len(world)):

#for world_country_idx in range(1,2):

    world_country_name=world.iloc[world_country_idx]['name']

    country_name=[]

    country_idx=np.where(world_country_name==countryUnique)[0]

    if country_idx.shape[0]>0:

        #print(np.max(train.loc[np.where(train.loc[:, 'Country/Region']==world_country_name)[0],'ConfirmedCases_dB']))

        world.loc[world_country_idx,'ConfirmedCases_dB']=np.max(train.loc[np.where(train.loc[:, 'Country_Region']==world_country_name)[0],'ConfirmedCases_dB'])

        world.loc[world_country_idx,'Fatalities_dB']=np.max(train.loc[np.where(train.loc[:, 'Country_Region']==world_country_name)[0],'Fatalities_dB'])

    else:

        if world_country_name=='United States of America':country_name='US'

        elif world_country_name=='Dem. Rep. Congo':country_name='Congo (Kinshasa)'

        elif world_country_name=='Congo':country_name='Congo (Brazzaville)'

        elif world_country_name=='Dominican Rep.':country_name='Dominican Republic'

        elif world_country_name=='Côte d\'Ivoire':country_name='Cote d\'Ivoire'

        elif world_country_name=='Central African Rep.':country_name='Central African Republic'

        elif world_country_name=='Eq. Guinea':country_name='Equatorial Guinea'

        elif world_country_name=='Gambia':country_name='Gambia, The'

        elif world_country_name=='South Korea':country_name='Korea, South'

        elif world_country_name=='Taiwan':country_name='Taiwan*'

        elif world_country_name=='Bosnia and Herz.':country_name='Bosnia and Herzegovina'

        if country_name!=[]:

            world.loc[world_country_idx,'ConfirmedCases_dB']=np.max(train.loc[np.where(train.loc[:, 'Country_Region']==country_name)[0],'ConfirmedCases_dB'])

            world.loc[world_country_idx,'Fatalities_dB']=np.max(train.loc[np.where(train.loc[:, 'Country_Region']==country_name)[0],'Fatalities_dB'])

        #else:

            #print(world_country_name)
## Plot a world map with the most recent data



fig, ax = plt.subplots(2,1,figsize=(20,10))

divider = make_axes_locatable(ax[0])

cax = divider.append_axes("right", size="5%", pad=0.1)



world.plot(column='ConfirmedCases_dB', cmap='jet',ax=ax[0],legend=True, cax=cax, vmin=0, vmax=55);

ax[0].set_title('Confirmed Cases (dB)')

divider = make_axes_locatable(ax[1])

cax = divider.append_axes("right", size="5%", pad=0.1)

world.plot(column='Fatalities_dB', cmap='jet',ax=ax[1],legend=True, cax=cax, vmin=0, vmax=55);

ax[1].set_title('Fatalities (dB)')
train['Province_State']=train['Province_State'].fillna('None')

train_grouped=train.groupby(['Country_Region','Province_State'])



## Plot confirmed cases and fatalities by unique combination of province/state and country/region

numRows, numCols = 37, 8

fig, ax = plt.subplots(numRows,numCols,figsize=(20,55))

fig.tight_layout(pad=2.5)



for idx in range(0,len(train_grouped)):

    row, col = np.divmod(idx,numCols)

    

    days=train_grouped['Days'].apply(np.array)[idx]

    cc=train_grouped['ConfirmedCases_dB'].apply(np.array)[idx]

    f=train_grouped['Fatalities_dB'].apply(np.array)[idx]

    if train_grouped['Province_State'].apply(list)[idx][0] == 'None':

        title=train_grouped['Country_Region'].apply(list)[idx][0]

    else:

        title=train_grouped['Country_Region'].apply(list)[idx][0]+'\n'+train_grouped['Province_State'].apply(list)[idx][0]

    

    # Plotting

    sn.scatterplot(x=days,y=cc,ax=ax[row,col])

    sn.scatterplot(x=days,y=f,ax=ax[row,col])

    ax[row,col].set_title(title)

    ax[row,col].set_ylim([-5,55])

    ax[row,col].set_xlim([np.min(days),np.max(days)])

    ax[row,col].set_ylabel('CC / F (dB)')

    ax[row,col].set_xlabel('',visible=False)

    ax[row,col].grid(1)
train['Province_State']=train['Province_State'].fillna('None')

train_grouped=train.groupby(['Country_Region','Province_State'])



## Plot confirmed cases and fatalities by unique combination of province/state and country/region

numRows, numCols = 37, 8

fig, ax = plt.subplots(numRows,numCols,figsize=(20,55))

fig.tight_layout(pad=2.5)



for idx in range(0,len(train_grouped)):

    row, col = np.divmod(idx,numCols)

    

    days=train_grouped['Days'].apply(np.array)[idx][1:len(train_grouped['Days'].apply(np.array)[idx])]

    cc=10*np.log10(np.diff(train_grouped['ConfirmedCases'].apply(np.array)[idx]))

    f=10*np.log10(np.diff(train_grouped['Fatalities'].apply(np.array)[idx]))

    if train_grouped['Province_State'].apply(list)[idx][0] == 'None':

        title=train_grouped['Country_Region'].apply(list)[idx][0]

    else:

        title=train_grouped['Country_Region'].apply(list)[idx][0]+'\n'+train_grouped['Province_State'].apply(list)[idx][0]

    

    # Plotting

    sn.scatterplot(x=days,y=cc,ax=ax[row,col])

    sn.scatterplot(x=days,y=f,ax=ax[row,col])

    ax[row,col].set_title(title)

    ax[row,col].set_ylim([-5,45])

    ax[row,col].set_xlim([np.min(days),np.max(days)])

    ax[row,col].set_ylabel('diff CC / F (dB)')

    ax[row,col].set_xlabel('',visible=False)

    ax[row,col].grid(1)
## Data windowing



day_limit=57

window_size=14



train_x_cc=[]

train_y_cc=[]

train_x_f=[]

train_y_f=[]

test_x_cc=[]

test_y_cc=[]

test_x_f=[]

test_y_f=[]



train_x_days=[]

train_y_days=[]

test_x_days=[]

test_y_days=[]



train_idx=[]

test_idx=[]



#for idx in range(0,1):

for idx in range(0,len(train_grouped)):

    

    days=train_grouped['Days'].apply(np.array)[idx]

    cc=10*np.log10(np.diff(train_grouped['ConfirmedCases'].apply(np.array)[idx]))

    f=10*np.log10(np.diff(train_grouped['Fatalities'].apply(np.array)[idx]))

    

    ## Make up for the missing difference day

    cc=np.insert(cc,0,-np.inf)

    f=np.insert(f,0,-np.inf)

    

    for window_start in range(0,day_limit-window_size):

        window_end=window_start+window_size

        days_window=days[window_start:window_end]

        cc_window=cc[window_start:window_end]

        f_window=f[window_start:window_end]

        

        ## Median replace

        #train_cc_median=np.median(cc_window[cc_window>=0])

        #train_f_median=np.median(f_window[f_window>=0])    

        #cc_window[cc_window<0]=train_cc_median

        #f_window[f_window<0]=train_f_median

        #    

        #f_window[(np.isnan(f_window))|(f_window==-np.inf)]=0

        #f_window_end=f[window_end]

        #if (f_window_end==-np.inf):

        #    f_window_end=0



        train_x_cc.append(cc_window)

        train_x_f.append(f_window)

        train_x_days.append(days_window)

        train_y_cc.append(cc[window_end])

        #train_y_f.append(f_window_end)

        train_y_f.append(f[window_end])

        train_y_days.append(days[window_end])

        train_idx.append(idx)

            

            

    for window_start in range(day_limit-window_size,len(days)-window_size):

        window_end=window_start+window_size

        days_window=days[window_start:window_end]

        cc_window=cc[window_start:window_end]

        f_window=f[window_start:window_end]

        

        ## Median replace

        #test_cc_median=np.median(cc_window[cc_window>=0])

        #test_f_median=np.median(f_window[f_window>=0])

        #cc_window[cc_window<0]=test_cc_median

        #f_window[f_window<0]=test_f_median

        #    

        #f_window[(np.isnan(f_window))|(f_window==-np.inf)]=0

        #f_window_end=f[window_end]

        #if (f_window_end==-np.inf):

        #    f_window_end=0



        test_x_cc.append(cc_window)

        test_x_f.append(f_window)

        test_x_days.append(days_window)

        test_y_cc.append(cc[window_end])

        #test_y_f.append(f_window_end)

        test_y_f.append(f[window_end])

        test_y_days.append(days[window_end])

        test_idx.append(idx)

        

    

train_x_cc = np.asarray(train_x_cc)

train_y_cc = np.asarray(train_y_cc)

train_x_f = np.asarray(train_x_f)

train_y_f = np.asarray(train_y_f)



test_x_cc = np.asarray(test_x_cc)

test_y_cc = np.asarray(test_y_cc)

test_x_f = np.asarray(test_x_f)

test_y_f = np.asarray(test_y_f)



train_x_days = np.asarray(train_x_days)

train_y_days = np.asarray(train_y_days)

test_x_days = np.asarray(test_x_days)

test_y_days = np.asarray(test_y_days)



train_idx = np.asarray(train_idx)

test_idx = np.asarray(test_idx)
# Preprocessing

scale_factor=60



## Remove all with training target as 0, but yet have training vector mean >1

train_keep=(train_y_cc!=-np.inf) & (np.ma.masked_invalid(train_x_cc).mean(axis=1).filled(0)>1)

train_x_cc=train_x_cc[train_keep]

train_y_cc=train_y_cc[train_keep]

train_x_f=train_x_f[train_keep]

train_y_f=train_y_f[train_keep]





train_x_cc[np.isnan(train_x_cc)]=0

train_y_cc[np.isnan(train_y_cc)]=0

train_x_f[np.isnan(train_x_f)]=0

train_y_f[np.isnan(train_y_f)]=0



test_x_cc[np.isnan(test_x_cc)]=0

test_y_cc[np.isnan(test_y_cc)]=0

test_x_f[np.isnan(test_x_f)]=0

test_y_f[np.isnan(test_y_f)]=0



train_x_cc[train_x_cc==-np.inf]=0

train_y_cc[train_y_cc==-np.inf]=0

train_x_f[train_x_f==-np.inf]=0

train_y_f[train_y_f==-np.inf]=0



test_x_cc[test_x_cc==-np.inf]=0

test_y_cc[test_y_cc==-np.inf]=0

test_x_f[test_x_f==-np.inf]=0

test_y_f[test_y_f==-np.inf]=0



#train_X=train_x_cc.reshape(train_x_cc.shape[0],train_x_cc.shape[1],1)/scale_factor

#train_y=train_y_cc.reshape(train_y_cc.shape[0],1)/scale_factor

#test_X=test_x_cc.reshape(test_x_cc.shape[0],test_x_cc.shape[1],1)/scale_factor

#test_y=test_y_cc.reshape(test_y_cc.shape[0],1)/scale_factor



train_X=np.concatenate((train_x_cc.reshape(train_x_cc.shape[0],train_x_cc.shape[1],1)/scale_factor,

                        train_x_f.reshape(train_x_f.shape[0],train_x_f.shape[1],1)/scale_factor),axis=2)

train_y=np.concatenate((train_y_cc.reshape(train_y_cc.shape[0],1)/scale_factor,

                        train_y_f.reshape(train_y_f.shape[0],1)/scale_factor),axis=1)

test_X=np.concatenate((test_x_cc.reshape(test_x_cc.shape[0],test_x_cc.shape[1],1)/scale_factor,

                       test_x_f.reshape(test_x_f.shape[0],test_x_f.shape[1],1)/scale_factor),axis=2)

test_y=np.concatenate((test_y_cc.reshape(test_y_cc.shape[0],1)/scale_factor,

                       test_y_f.reshape(test_y_f.shape[0],1)/scale_factor),axis=1)
model = Sequential()

model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dropout(0.2))

model.add(Dense(train_y.shape[1]))



model.compile(loss='mean_squared_error', optimizer='adam')

callback=ModelCheckpoint('20200331_model_cc_v2.h5', monitor='val_loss', save_best_only=True)



model.fit(train_X,train_y,epochs=5,batch_size=16,validation_split=0.4,callbacks=[callback],verbose=1)

model=tf.keras.models.load_model('20200331_model_cc_v2.h5')
test_y_predictions=model.predict(test_X)
## Plot confirmed cases and fatalities by unique combination of province/state and country/region

numRows, numCols = 37, 8

fig, ax = plt.subplots(numRows,numCols,figsize=(20,55))

fig.tight_layout(pad=2.5)



for idx in range(0,len(train_grouped)):

    row, col = np.divmod(idx,numCols)

    

    days=train_grouped['Days'].apply(np.array)[idx][1:len(train_grouped['Days'].apply(np.array)[idx])]

    cc=10*np.log10(np.diff(train_grouped['ConfirmedCases'].apply(np.array)[idx]))

    f=10*np.log10(np.diff(train_grouped['Fatalities'].apply(np.array)[idx]))

    if train_grouped['Province_State'].apply(list)[idx][0] == 'None':

        title=train_grouped['Country_Region'].apply(list)[idx][0]

    else:

        title=train_grouped['Country_Region'].apply(list)[idx][0]+'\n'+train_grouped['Province_State'].apply(list)[idx][0]

    

    # Plotting

    sn.scatterplot(x=days,y=cc,ax=ax[row,col])

    sn.scatterplot(x=days,y=f,ax=ax[row,col])

    

    sn.scatterplot(x=test_y_days[np.where(test_idx==idx)[0]],

                   y=np.squeeze(test_y_predictions[np.where(test_idx==idx)[0]])[:,0]*scale_factor,ax=ax[row,col])

    sn.scatterplot(x=test_y_days[np.where(test_idx==idx)[0]],

                   y=np.squeeze(test_y_predictions[np.where(test_idx==idx)[0]])[:,1]*scale_factor,ax=ax[row,col])

    

    sn.lineplot(x=[day_limit, day_limit],y=[-5,45],color='r',ax=ax[row,col])

    

    ax[row,col].set_title(title)

    ax[row,col].set_ylim([-5,45])

    ax[row,col].set_xlim([np.min(days),np.max(days)])

    ax[row,col].set_ylabel('diff CC / F (dB)')

    ax[row,col].set_xlabel('',visible=False)

    ax[row,col].grid(1)
#for idx in range(0,len(train_grouped)):

#    

#idx=509

#test_y_predictions=model.predict(test_X)

#plt.plot(test_x_days[idx],test_X[idx]*45)

#plt.plot(test_y_days[idx],test_y[idx]*45,'bx')

#plt.plot(test_y_days[idx],test_y_predictions[idx]*45,'ro')

#plt.ylim([-5,45])

#plt.show()
## Data windowing



test_x_cc_extrp=[]

test_x_f_extrp=[]



test_x_days_extrp=[]

test_idx_extrp=[]



#for idx in range(0,1):

for idx in range(0,len(train_grouped)):

    

    days=train_grouped['Days'].apply(np.array)[idx][1:len(train_grouped['Days'].apply(np.array)[idx])]

    cc=10*np.log10(np.diff(train_grouped['ConfirmedCases'].apply(np.array)[idx]))

    f=10*np.log10(np.diff(train_grouped['Fatalities'].apply(np.array)[idx]))

    

    for window_start in range(day_limit-window_size,day_limit-window_size+1):

    #for window_start in range(day_limit-window_size+10,day_limit-window_size+1+10):

        window_end=window_start+window_size

        days_window=days[window_start:window_end]

        cc_window=cc[window_start:window_end]

        f_window=f[window_start:window_end]

        train_cc_median=np.median(cc_window[cc_window>=0])

        train_f_median=np.median(f_window[f_window>=0])



        cc_window[cc_window<0]=train_cc_median

        f_window[f_window<0]=train_f_median



        test_x_cc_extrp.append(cc_window)

        test_x_f_extrp.append(f_window)

        test_x_days_extrp.append(days_window)

        test_idx_extrp.append(idx)



test_x_cc_extrp = np.asarray(test_x_cc_extrp)

test_x_f_extrp = np.asarray(test_x_f_extrp)



test_x_days_extrp = np.asarray(test_x_days_extrp)

test_idx_extrp = np.asarray(test_idx_extrp)
# Preprocessing



test_x_cc_extrp[np.isnan(test_x_cc_extrp)]=0

test_x_f_extrp[np.isnan(test_x_f_extrp)]=0
days_to_predict_until=100



#test_X=test_x_cc_extrp.reshape(test_x_cc_extrp.shape[0],test_x_cc_extrp.shape[1],1)/scale_factor

test_X=np.concatenate((test_x_cc_extrp.reshape(test_x_cc_extrp.shape[0],test_x_cc_extrp.shape[1],1)/scale_factor,

                       test_x_f_extrp.reshape(test_x_f_extrp.shape[0],test_x_f_extrp.shape[1],1)/scale_factor),axis=2)



test_X_extrp=[]

test_X_extrp_days=[]

test_extrp_idx=[]

test_y_extrp_days=[]

test_y_extrp=[]

for idx in range(0,test_X.shape[0]):

#for idx in range(0,1):

#for idx in range(48,49):

    test_X_extrp.append(test_X[idx])

    test_X_extrp_days.append(test_x_days_extrp[idx])

    while np.max(test_X_extrp_days[len(test_X_extrp_days)-1])<= days_to_predict_until-1:

        new_days=test_X_extrp_days[len(test_X_extrp_days)-1]+1

        

        features_to_predict=test_X_extrp[len(test_X_extrp_days)-1].reshape(1,test_X_extrp[len(test_X_extrp_days)-1].shape[0],

                                                                     test_X_extrp[len(test_X_extrp_days)-1].shape[1])

        end_features=model.predict(features_to_predict)

        end_features[end_features<0]=0

        new_features=features_to_predict.reshape(features_to_predict.shape[1],features_to_predict.shape[2])[1:window_size+1]

        new_features=np.concatenate((new_features,end_features),axis=0)

        

        test_X_extrp.append(new_features)

        test_X_extrp_days.append(new_days)

        test_extrp_idx.append(idx)

        

        test_y_extrp_days.append(np.max(new_days))

        test_y_extrp.append(end_features)

        

test_X_extrp=np.asarray(test_X_extrp)

test_X_extrp_days=np.asarray(test_X_extrp_days)

test_extrp_idx=np.asarray(test_extrp_idx)



test_y_extrp_days=np.asarray(test_y_extrp_days)

test_y_extrp=np.asarray(test_y_extrp)
## Plot confirmed cases and fatalities by unique combination of province/state and country/region

numRows, numCols = 37, 8

fig, ax = plt.subplots(numRows,numCols,figsize=(20,55))

fig.tight_layout(pad=2.5)



for idx in range(0,len(train_grouped)):

    row, col = np.divmod(idx,numCols)

    

    days=train_grouped['Days'].apply(np.array)[idx][1:len(train_grouped['Days'].apply(np.array)[idx])]

    cc=10*np.log10(np.diff(train_grouped['ConfirmedCases'].apply(np.array)[idx]))

    f=10*np.log10(np.diff(train_grouped['Fatalities'].apply(np.array)[idx]))

    if train_grouped['Province_State'].apply(list)[idx][0] == 'None':

        title=train_grouped['Country_Region'].apply(list)[idx][0]

    else:

        title=train_grouped['Country_Region'].apply(list)[idx][0]+'\n'+train_grouped['Province_State'].apply(list)[idx][0]

    

    # Plotting

    sn.scatterplot(x=days,y=cc,ax=ax[row,col])

    sn.scatterplot(x=days,y=f,ax=ax[row,col])

    

    sn.scatterplot(x=test_y_extrp_days[np.where(test_extrp_idx==idx)[0]],

                   y=test_y_extrp[np.where(test_extrp_idx==idx)[0],0,0]*scale_factor,ax=ax[row,col])

    sn.scatterplot(x=test_y_extrp_days[np.where(test_extrp_idx==idx)[0]],

                   y=test_y_extrp[np.where(test_extrp_idx==idx)[0],0,1]*scale_factor,ax=ax[row,col])

    

    

    #sn.scatterplot(x=np.max(test_X_extrp_days[np.where(test_extrp_idx==idx)[0]],axis=1)+1,

    #               y=np.squeeze(model.predict(test_X_extrp[np.where(test_extrp_idx==idx)[0]]))*scale_factor,ax=ax[row,col])

    sn.lineplot(x=[day_limit, day_limit],y=[-5,45],color='g',ax=ax[row,col])

    

    ax[row,col].set_title(title)

    ax[row,col].set_ylim([-5,45])

    #ax[row,col].set_xlim([np.min(days),np.max(days)])

    ax[row,col].set_xlim([np.min(days),110])

    ax[row,col].set_ylabel('diff CC / F (dB)')

    ax[row,col].set_xlabel('',visible=False)

    ax[row,col].grid(1)
# Submission

last_timestamp=train['Date'].loc[day_limit-1]

last_timestamp_idx=np.squeeze(np.asarray(np.where(train['Date']==last_timestamp)))

last_cumulative_cc=np.asarray(train['ConfirmedCases'].iloc[last_timestamp_idx])

last_cumulative_f=np.asarray(train['Fatalities'].iloc[last_timestamp_idx])



result=[]

for idx in range(0,len(last_timestamp_idx)):

    cumulative=[last_cumulative_cc[idx],last_cumulative_f[idx]]

    for gg in np.where(test_extrp_idx==idx)[0]:

        cumulative=cumulative+10**(test_y_extrp[gg][0,:]/10)

        result.append(cumulative)



result=np.round(result)

# Kaggle submission

submission=pd.DataFrame({'ForecastId':np.arange(0,test_y_extrp.shape[0])+1,

                     'ConfirmedCases':result[:,0],'Fatalities':result[:,1]})

submission= submission.astype(int)

submission.to_csv('Submission_1.csv', index=False)