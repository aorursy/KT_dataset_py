import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
#print(os.listdir('/kaggle/input'))

DATA_FOLDER = "/kaggle/input/coronavirus-2019ncov"

#print(os.listdir(DATA_FOLDER))

data_df = pd.read_csv(os.path.join(DATA_FOLDER, "covid-19-all.csv"))

#cn_geo_data = os.path.join(GEO_DATA, "china.json")

#wd_geo_data = os.path.join(WD_GEO_DATA, "world-countries.json")
#print(f"Rows: {data_df.shape[0]}, Columns: {data_df.shape[1]}")

#data_df.head()

#data_df.tail()

#print(f"Date - unique values: {data_df['Date'].unique()}")
data_df['date'] = pd.to_datetime(data_df['Date'])

import datetime as dt

dt_string = dt.datetime.now().strftime("%Y-%m-%d")

last_update=data_df['date'].unique().max()

print(f"Kernel last updated on {dt_string} based on data until {last_update}")
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))

#missing_data(data_df)
#print(f"Countries/Regions:\n {data_df['Country/Region'].unique()}")

#print(f"Province/State:\n {data_df['Province/State'].unique()}")

#print(f"Last Update:\n {data_df['Last Update'].unique()}")

#print(f"Last Update:\n {data_df['DateTime'].unique()}")
data_cn = data_df.loc[data_df['Country/Region']=="Mainland China"]

data_cn = data_cn.sort_values(by = ['Province/State','date'], ascending=False)

data_not_cn=data_df.loc[~(data_df['Country/Region']=="Mainland China")]

data_not_cn=data_not_cn.sort_values(by = ['Province/State','date'], ascending=False)

data_countries = data_df.groupby(['Country/Region','date']).sum().reset_index()

data_countries = data_countries.sort_values(by = ['date','Country/Region'], ascending=True)

#data_not_cn

if False:

    data_cn['Country/Region']=data_cn['Province/State']

    data_countries=pd.concat([data_countries,data_cn])

    data_countries = data_countries.sort_values(by = ['date','Country/Region'], ascending=True)

data_countries

#filtered_data_last = data_cn.drop_duplicates(subset = ['Province/State'],keep='first')

#filtered_data_last = data_not_cn.drop_duplicates(subset = ['Province/State'],keep='first')
def plot_count(feature, value, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    g = sns.barplot(df[feature], df[value],  palette='Set3')

    g.set_title("Number of {}".format(title))

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    plt.show()    

#plot_count('Province/State', 'Confirmed', 'Confirmed cases (last updated)', filtered_data_last, size=4)
def plot_time_variation(df, y='Confirmed', hue='Province/State', size=1,title='Title', logscale=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    #missing_data(df)

    #print(y,hue,df)

    g = sns.lineplot(x="date", y=y, hue=hue, data=df)

    plt.xticks(list(df['date'].unique()),rotation=90)

    if logscale:

        #plt.set_yscale('log')

        plt.yscale('log')

    #plt.xticks(rotation=90)

    #ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    #print(df[y].max())

    try:

        plt.yticks(np.arange(0, df[y].max(), step=2000))

    except:

        pass

    plt.grid()

    plt.title(f'{title}: {y} cases grouped by {hue}')

    plt.show()  
#data_cn

#plot_time_variation(data_not_cn, size=4)

plot_time_variation(data_countries, hue='Country/Region',size=4, title='Curve base')
#plot_time_variation(data_cn.loc[~(data_cn['Province/State']=='Hubei')], size=4)
def normalize(a):

    return a/a[-1]



def mse(a,b):

    return np.sqrt(np.square(a-b).sum())



def mse_normalized(a,b):

    return mse(normalize(a),normalize(b))



def mse_normalized_average(a,b):

    return mse_normalized(a,b)/overlap(a,b)



def mse_average(a,b):

    return mse(a,b)/overlap(a,b)



def mae(a,b):

    return (np.abs(a-b).sum())



def mae_normalized(a,b):

    return mae(normalize(a),normalize(b))



def hamming(a,b):

    diff=a-b

    return np.where(diff==0,0,1).sum()



def overlap(a,b):

    diff=a*b

    return np.where(diff==0,0,1).sum()



def get_min_dist(a,b,min_overlap=5,dist_func=mse): #e.g. length 10 and 20

#    min_shift=-len(b)+min_overlap #

    min_shift=len(a)-len(b)+1 #-9, so that there is at least one future value, which is b[len(a)+min_shift+1]=b[10+9+1]=b[20]

    max_shift=len(a)-min_overlap #

    #print(min_shift,max_shift)

    shifts=list(range(min_shift,max_shift+1,1))

    dists=np.ones(len(shifts))

    for i,shift in enumerate(shifts):

        a_shifted=a

        b_shifted=b

        if shift <= 0:

            a_shifted=np.pad(a, (-shift, 0), 'constant', constant_values=(0))

        else:

            b_shifted=np.pad(b, (shift, 0), 'constant', constant_values=(0))

        #print(a_shifted)

        #print(b_shifted)

        #print(b_shifted[0:len(a_shifted)])

        dist=dist_func(a_shifted,b_shifted[0:len(a_shifted)])

        #print(dist)

        dists[i]=dist

    return np.array([shifts,dists])



#a=np.arange(10)

#b=np.arange(5)

#get_min_dist(a,b,min_overlap=3,dist_func=overlap)
min_length=10

min_overlap=5

#country="Germany"

country="Spain"

normalize_plots=True

k=5

days_to_eval=5

#temps=[0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,500,1000,10000,100000,1000000]

temps=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000]

#too low values seem to be too extreme, since then the just do max, so that we introduce a min_prob factor, in relation to unit distribution (all equal probability)

min_prob_factor=0.1

#print("compute predictions for "+country)

dist_function=mse_normalized_average

#dist_function=mse_normalized

#dist_function=mse_average

plot_past_days=14

def filter_regions(data,cat='Confirmed'):



    tss={}

    dates=data['date'].unique() #should already be sorted

    regions=data['Country/Region'].unique()

    #print(dates)



    for c in regions:

        ts=data[(data['Country/Region']==c)]

        ts=ts[['date',cat]]

        #filter out the first days where there are 0 cases

        ts=ts[~(data[cat]==0)]

        ts=ts[cat].to_numpy()

        #ts=[data[(data['Country/Region']==c) & (data['date']==date)]['Confirmed'] for date in dates]

        #filter 

        if len(ts)>=min_length:

            tss[c]=ts

        else:

            print(f"remove {c} since only {len(ts)} days")

        #print(c,ts)

    regions=list(tss.keys())

    return regions, tss, dates



data=data_countries

regions, tss, dates=filter_regions(data,cat='Confirmed')
# get curve as panda data frame

def get_index_df(a,shift,name,weight=1.0, pivot_date=None):

    if pivot_date is None:

        df=pd.DataFrame({'date':list(range(shift,shift+len(a))),'Number':a,'name':[name]*len(a),'weight':[weight]*len(a)})

    else:

        start_date=pivot_date+pd.Timedelta(days=shift)

#        df=pd.DataFrame({'date':pd.date_range(start_date, periods=len(a), freq='D'),'Number':a,'name':[name]*len(a),'weight':[weight]*len(a)},dtype=['datetime64',

#                                                                                                                                                     'float64',

#                                                                                                                                                     'object',

#                                                                                                                                                     'float64'])

        #print(type(a))

        df=pd.DataFrame({'date':pd.date_range(start_date, periods=len(a), freq='D'),'Number':a,'name':[name]*len(a),'weight':[weight]*len(a)})

    df["Number"] = pd.to_numeric(df["Number"])

    #print(missing_data(df))

    return df

def get_first_last_date(country,df,cat='Confirmed'):

    ts=df[(df['Country/Region']==country)]

    ts=ts[['date',cat]]

    #filter out the first days where there are 0 cases

    ts=ts[~(data[cat]==0)]

    return (ts['date'].min(),ts['date'].max())

    
def get_all_nn(country,data,tss,days_to_cut_off,normalize_plots,dist_func, cat='Confirmed'):

    do_normalize=normalize_plots

    pivot_date,last_date=get_first_last_date(country,data,cat=cat)

    viss=[]

    dists=[]

    shifts=[]

    errors=[]

    timeseries=[]

    ts=tss[country]

    ts=ts[0:len(ts)-days_to_cut_off]

    for region in regions:

        if country == region:

            continue

        dist=get_min_dist(ts,tss[region],dist_func=dist_func,min_overlap=min_overlap)

        min_index=np.argmin(dist[1])

        shift=dist[0,min_index]

        error=dist[1,min_index]

        ts_nn=tss[region]

        days_ahead=len(ts_nn)-len(ts)+int(shift)

        if do_normalize:

            last_value_a=ts[-1]

            #print(int(len(tss[country])-shift))

            last_value_b=ts_nn[len(ts)-int(shift)-1]

            ts_nn=ts_nn/last_value_b*last_value_a

        vis=get_index_df(ts_nn,int(shift),"%s ahead=%s dist=%s"%(region,int(days_ahead),error),pivot_date=pivot_date,weight=error)

        viss.append(vis)

        dists.append(dists)

        shifts.append(shift)

        errors.append(error)

        timeseries.append(ts)

    errors=np.array(errors)

    return errors,shifts,dists,viss,timeseries



#def get_k_nn(country,data,timeseries,k,normalize_plots,dist_func):

#    errors,shifts,dists,viss=get_all_nn(country,data,timeseries,normalize_plots,dist_func)

def get_k_nn(k,errors,shifts,dists,viss,timeseries):

    top_ind=errors.argsort()[:k]

    errors=errors[top_ind]

    dists=[dists[ind] for ind in top_ind]

    shifts=[shifts[ind] for ind in top_ind]

    viss=[viss[ind] for ind in top_ind]

    timeseries=[timeseries[ind] for ind in top_ind]

    return errors,shifts,dists,viss,timeseries



#std_start_date=

def plot_nn_ts(viss,size=4,future_limit_in_days=5,last_date=None,start_date=pd.Timestamp('2020-03-01'),title='Plot'):

    #print(viss[0])

    #print(viss[2])

    #for col in viss[0].columns:

    #    print(viss[0][col].dtype)

    #missing_data(viss[0])

    #plot_time_variation(viss[0], y='Number',hue='name',size=4,title=title)

    viss=pd.concat(viss)

    if future_limit_in_days is not None:

        viss=viss[(viss['date']<last_date+pd.Timedelta(days=future_limit_in_days))]

    viss=viss[(viss['date']>=start_date)]

    #print(viss)

    plot_time_variation(viss, y='Number',hue='name',size=4,title=title)



import math

    

def weighted_avg_and_std(values, weights):

    """

    from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Return the weighted average and standard deviation.



    values, weights -- Numpy ndarrays with the same shape.

    """

    average = np.average(values, weights=weights)

    # Fast and numerically precise:

    variance = np.average((values-average)**2, weights=weights)

    return (average, math.sqrt(variance),weights)



def bounded_softmax(x, temperature):

    p=softmax(x,temperature)

    min_prob=min_prob_factor*1.0/len(x)

    p=p*(1.0-min_prob_factor)+min_prob

    #p=p/sum(p)

    return p



def softmax(x,temperature):

#    return np.exp(x/temperature)/sum(np.exp(x/temperature))

    x=x/temperature

    x=x-max(x)

    return np.exp(x)/sum(np.exp(x))



#not used anymore

def predict(temperature,ts_length,errors,shifts,timeseries):

    means=[]

    stds=[]

    futureday=0

    i=0

    for futureday in range(100):

        tss=[(errors[i],timeseries[i][int(ts_length-shifts[i]+futureday)]) for i,ts in enumerate(timeseries) if len(timeseries[i])>ts_length-shifts[i]+futureday]

        if len(tss)>0:

            e,v=zip(*tss)

            e=np.array(e)

            v=np.array(v)

            weights=softmax(e,temperature)

            mean,std=weighted_avg_and_std(v,weights)

            means.append(mean)

            stds.append(std)

        else:

            break

    return (means,stds,weights)



def predict2(country,temperature,viss,last_date_given):

    #print(viss)

    viss=pd.concat(viss)

    viss=viss.reset_index(drop=True)

    viss['date']=pd.to_datetime(viss['date'])

    #print(viss)

    max_date=viss['date'].max()

    first_pred_date=last_date_given+pd.Timedelta(days=1)

    prediction=[]

    #print(first_pred_date)

    #print(max_date)

    for future_day in pd.date_range(start=first_pred_date, end=max_date, freq='D'):

        viss_day=viss[(viss['date']==future_day)]

        values=viss_day['Number'].to_numpy()

        weights=viss_day['weight'].to_numpy()

        normalized_weights=bounded_softmax(weights,temperature)

        #print(future_day,temperature,values,weights,normalized_weights)

        #print(viss_day)

        #print(weights)

        #print(normalized_weights)

        prediction.append(weighted_avg_and_std(values,normalized_weights))

        #print(viss_day,values,weights,mean,std)

    prediction=np.array(prediction).T



    viss=[]

    viss.append(get_index_df(prediction[0]+prediction[1],0,country+' upper confidence bound',pivot_date=first_pred_date))

    viss.append(get_index_df(prediction[0],0,country+' pred',pivot_date=first_pred_date))

    viss.append(get_index_df(prediction[0]-prediction[1],0,country+' lower confidence bound',pivot_date=first_pred_date))

    return viss,prediction



def eval_k_on_train(k,temperature,days_to_eval,loss_function,country,data,tss,normalize_plots,dist_function,log=False,cat='Confirmed'):

    ts=tss[country]

    pivot_date,last_date=get_first_last_date(country,data,cat=cat)

    predictions=[]

    if log:

        print("date real pred  <="+country)

    for days_to_cut_off in range(1,days_to_eval+1,1):

        before_prediction_date=last_date-pd.Timedelta(days=days_to_cut_off)

        errors,shifts,dists,viss,timeseries=get_k_nn(k,*get_all_nn(country,data,tss,days_to_cut_off,normalize_plots,dist_function))

        _,prediction=predict2(country,temperature,viss,before_prediction_date)

        predictions.append(prediction[0,0])

        if log:

            print("%s %s %s"%(before_prediction_date+pd.Timedelta(days=1),ts[-days_to_cut_off],prediction[0,0]))

        

    predictions=np.array(predictions)

    return loss_function(np.array(ts[-days_to_eval:]),predictions)







#for temperature in temps:

#    eval_k_on_train(k,temperature,days_to_eval,mse,country,data,tss,normalize_plots,mse)

losses=[eval_k_on_train(k,temperature,days_to_eval,mse,country,data,tss,normalize_plots,dist_function,True) for temperature in temps]

print("Losses for different temperature (choose min): %s"%losses)

#min_index = losses.index(min(losses)) 

min_index = np.nanargmin(np.array(losses)) 

temperature=temps[min_index]

print("chosen temperature %s with loss %s"%(temperature,losses[min_index]))





ts=tss[country]

pivot_date,last_date=get_first_last_date(country,data,cat='Confirmed')

#errors,shifts,dists,viss,timeseries=get_all_nn(country,data,tss,normalize_plots,mse)

errors,shifts,dists,viss,timeseries=get_k_nn(k,*get_all_nn(country,data,tss,0,normalize_plots,dist_function))



#print(viss)



country_vis=get_index_df(ts,0,country,pivot_date=pivot_date)





pred,pred_series=predict2(country,temperature,viss,last_date)

print("weights of regions for mixing (weights for linear combination) for first future day (using temperature %s):\n %s"%(temperature,pred_series[2,0]))

plot_nn_ts(viss+[country_vis]+[pred[1]],size=4,last_date=last_date, title=f'Prediction for {country} based on {k} neighbors', start_date=last_date-pd.Timedelta(days=plot_past_days))

#print(pred_series)

#viss=pred

#viss=viss+pred
plot_nn_ts([country_vis]+pred,size=4,last_date=last_date, title=f'Prediction for {country} based on {k} neighbors', start_date=last_date-pd.Timedelta(days=plot_past_days))

table=pred[1]

table['lower confidence bound']=pred[2]['Number']

table['upper confidence bound']=pred[0]['Number']

table.head(10)
def predict_and_plot(country,cat='Confirmed',dist_function=mse_normalized_average):

    #for temperature in temps:

    #    eval_k_on_train(k,temperature,days_to_eval,mse,country,data,tss,normalize_plots,mse)

    losses=[eval_k_on_train(k,temperature,days_to_eval,mse,country,data,tss,normalize_plots,dist_function) for temperature in temps]

    print(losses)

    #min_index = losses.index(min(losses)) 

    min_index = np.nanargmin(np.array(losses)) 

    temperature=temps[min_index]

    #print("chosen temperature %s with loss %s"%(temperature,losses[min_index]))



    ts=tss[country]

    pivot_date,last_date=get_first_last_date(country,data,cat=cat)

    #print("dfadsf",pivot_date,last_date)

    #errors,shifts,dists,viss,timeseries=get_all_nn(country,data,tss,normalize_plots,mse)

    errors,shifts,dists,viss,timeseries=get_k_nn(k,*get_all_nn(country,data,tss,0,normalize_plots,dist_function))



    country_vis=get_index_df(ts,0,country,pivot_date=pivot_date)





    pred,pred_series=predict2(country,temperature,viss,last_date)

    print("weights of regions for mixing (weights for linear combination) for first future day (using temperature %s):\n %s"%(temperature,pred_series[2,0]))

    plot_nn_ts(viss+[country_vis]+[pred[1]],size=4,last_date=last_date, title=f'Prediction for {country} based on {k} neighbors', start_date=last_date-pd.Timedelta(days=plot_past_days))



    plot_nn_ts([country_vis]+pred,size=4,last_date=last_date, title=f'Prediction for {country} based on {k} neighbors', start_date=last_date-pd.Timedelta(days=plot_past_days))



    table=pred[1]

    table['lower confidence bound']=pred[2]['Number']

    table['upper confidence bound']=pred[0]['Number']

    return table.head(10)
predict_and_plot('Germany')
predict_and_plot('France')
predict_and_plot('US')
predict_and_plot('Austria')
try:

    predict_and_plot('UK')

except:

    pass

try:

    predict_and_plot('United Kingdom')

except:

    pass
predict_and_plot('Spain',dist_function=mse_normalized_average)



predict_and_plot('Germany',dist_function=mse_average)

predict_and_plot('France',dist_function=mse_average)

predict_and_plot('US',dist_function=mse_average)
predict_and_plot('Austria',dist_function=mse_average)
try:

    predict_and_plot('UK',dist_function=mse_average)

except:

    pass

try:

    predict_and_plot('United Kingdom',dist_function=mse_average)

except:

    pass
plot_time_variation(data_countries, y='Deaths', hue='Country/Region',size=4, title='Curve base')
min_length=7

min_overlap=3

regions, tss, dates=filter_regions(data,cat='Deaths')

predict_and_plot('Spain',cat='Deaths',dist_function=mse_normalized_average)
predict_and_plot('Germany',cat='Deaths',dist_function=mse_normalized_average)
predict_and_plot('France',cat='Deaths',dist_function=mse_normalized_average)
predict_and_plot('US',cat='Deaths',dist_function=mse_normalized_average)