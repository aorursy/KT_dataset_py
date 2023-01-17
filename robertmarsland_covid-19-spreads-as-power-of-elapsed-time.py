import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
        
#Load the data and notes. 
#I take the US data from the covidtracking.com website, which has daily updates.
data_live = pd.read_csv('http://covidtracking.com/api/states/daily.csv')
data_live['date'] = pd.to_datetime(data_live['date'],format='%Y%m%d')
data_live['elapsed'] = (data_live['date'] - data_live['date'].iloc[-1])/timedelta(days=1)
#These are notes on data quality, and other relevant info for each state
info = pd.read_csv('https://covidtracking.com/api/states/info.csv',index_col=0)

#This is the 'training' data from the Kaggle project, which I use for all other countries
data_global = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
data_global.columns = ['Id','subregion','state','date','positive','death']
data_global['date'] = pd.to_datetime(data_global['date'],format='%Y-%m-%d')
data_global['elapsed'] = (data_global['date'] - data_live['date'].iloc[-1])/timedelta(days=1)
data_global = data_global.fillna(value='NaN')

#Make lookup table for predictions
id_list = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv',index_col=0)
id_list['Date'] = pd.to_datetime(id_list['Date'],format='%Y-%m-%d')
id_list['elapsed'] = (id_list['Date'] - data_live['date'].iloc[-1])/timedelta(days=1)
id_list = id_list.fillna(value='NaN')
id_list = id_list.reset_index().set_index(['Country_Region','Province_State','elapsed'])
id_list = id_list.sort_index()

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv',index_col=0)


#Make script for basic plotting
def BasicPlots(data,info=info,metric='positive',regions=['MA']):
    #Make table of states from data, using the column in 'metric', with date as index
    table = data.pivot_table(index='date',values=metric,columns='state')
    #Pick out a few states we are interested in and plot
    table[regions].plot()
    plt.gca().set_yscale('log')
    plt.gca().set_ylabel(metric)
    plt.show()
    
    #Print notes if using US data
    for region in regions:
        if region in info.index.values:
            print(region)
            print(info['notes'].loc[region])
            print('------------------')

    #Make another table with purely numeric data in index (instead of datetime) so we can regress
    table = data.pivot_table(index='elapsed',values=metric,columns='state')
    plt.plot(np.log10(table.sum(axis=1)).index.values,np.log10(table.sum(axis=1)),'o',label='Data')
    sns.regplot(np.log10(table.sum(axis=1)).index.values[:-4],np.log10(table.sum(axis=1)).iloc[:-4],label='Regression')
    plt.legend()
    plt.gca().set_xlabel('Days Elapsed since March 4')
    plt.gca().set_ylabel('log10 '+metric)
    plt.gca().set_title('Total over all regions')
    plt.show()
    
    return table

def PowerLaw(data,metric='positive',regions=['NY'],subregion=None,
             start_cutoff=4,start_shift=2,params=[4,0.1],t0=None,ymin=100,xmin=10):
    end_time = 0
    if subregion is None:
        table = data.pivot_table(index='elapsed',values=metric,columns='state',aggfunc=np.sum)
        for region in regions:
            if t0 is None:
                start_time = table.loc[table[region]>=start_cutoff].index.values[0]
            else:
                start_time = t0
                start_shift = 0
            table.index = table.index.values-start_time+start_shift
            table[region].plot(marker='o',label=region)
            end_time = np.max([end_time,np.max(table.index.values)])
    else:
        data_region = data.copy()
        data_region = data_region.loc[data['state']==regions[0]]
        table = data.pivot_table(index='elapsed',values=metric,columns='subregion',aggfunc=np.sum)
        if t0 is None:
            start_time = table.loc[table[subregion]>=start_cutoff].index.values[0]
        else:
            start_time = t0
            start_shift = 0
        table.index = table.index.values-start_time+start_shift
        table[subregion].plot(marker='o',label=subregion)
        end_time = np.max([end_time,np.max(table.index.values)])

    plt.legend()
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_xlabel('Days since beginning of spread')
    plt.gca().set_ylabel(metric)

    z = params[0]
    c = params[1]
    days = np.linspace(1,end_time*1.01,100)
    plt.plot(days,c*days**z,'k',linewidth='2')
    plt.gca().set_xlim((xmin,end_time*1.01))
    plt.gca().set_ylim((ymin,np.nanmax(table.values.reshape(-1))*1.2))
    plt.show()
    
    return table
table = PowerLaw(data_live,regions=['NY'],params=[4,0.11])
table = PowerLaw(data_live,regions=['WA','MA','IL','CA'],params=[4,0.01])
table = PowerLaw(data_global,regions=['Italy'],params=[3.6,0.22])
table = PowerLaw(data_global,regions=['Spain'],params=[4.5,0.009])
def cost_p(params,data):
    t0,C,z = params
    prediction = np.log(C)+z*np.log(data.index.values-t0)
    
    return 0.5*((np.log(data.values)-prediction)**2).sum()

def cost_f(params,data,p_params):
    t0,C,z = p_params
    Delta,r = params
    prediction = np.log(r*C)+z*np.log(data.index.values-t0-Delta)
    
    return 0.5*((np.log(data.values)-prediction)**2).sum()

def jac_p(params,data):
    t0,C,z = params
    prediction = np.log(C)+z*np.log(data.index.values-t0)
    
    return np.asarray([((z/(data.index.values-t0))*(np.log(data.values)-prediction)).sum(),
                       -((1/C)*(np.log(data.values)-prediction)).sum(),
                      -(np.log(data.index.values-t0)*(np.log(data.values)-prediction)).sum()])

def jac_f(params,data,p_params):
    t0,C,z = p_params
    Delta,r = params
    prediction = np.log(r*C)+z*np.log(data.index.values-t0-Delta)
    
    return np.asarray([((z/(data.index.values-t0-Delta))*(np.log(data.values)-prediction)).sum(),
                       -((1/r)*(np.log(data.values)-prediction)).sum()])
region = 'Italy'
start_cutoff=4
p0=5e2
z0 = 4
Delta0 = 5
r0 = 0.1
f0 = 50

table = data_global.pivot_table(index='elapsed',values='positive',columns='state',aggfunc=np.sum)
t00 = table.loc[table[region]>=start_cutoff].index.values[0]
C0 = table[region].max()/(np.max(table.index.values)-t00)**z0
train = table[region].loc[table[region]>p0]
out = minimize(cost_p,[t00,C0,z0],args=(train,),jac=jac_p,bounds=((None,int(train.index.values[0])-1),(1e-6,None),(0,10)))
t0,C,z = out.x
table = PowerLaw(data_global,metric='positive',regions=[region],params=[z,C],t0=t0,xmin=1)

table = data_global.pivot_table(index='elapsed',values='death',columns='state',aggfunc=np.sum)
train = table[region].loc[table[region]>f0]
out = minimize(cost_f,[Delta0,r0],args=(train,[t0,C,z]),jac=jac_f,bounds=((0,12),(1e-6,1)))
Delta,r = out.x
table = PowerLaw(data_global,metric='death',regions=[region],params=[z,C*r],t0=t0+Delta,ymin=1,xmin=1)
start_cutoff=4 #Initial estimate of t0 is when the number of cases exceeds this number
p0=5e2 #For confirmed cases model, only include times after the number of cases crosses this point
f0=50 #For fatality model, only include times after the number of fatalities crosses this point
Delta0 = 5 #Initial estimate of time lag between t0 and first fatality
r0 = 0.1 #Initial estimate of death rate

#Set up the variables
p_valid = True
f_valid = True
failed_regions = []
p = data_global.pivot_table(index='elapsed',values='positive',columns=['state','subregion'],aggfunc=np.sum)
f = data_global.pivot_table(index='elapsed',values='death',columns=['state','subregion'],aggfunc=np.sum)
params_table = pd.DataFrame(columns=['Country_Region','State_Province','t0','t0_abs','C','z','Delta t','r'])
param_id = 0

#Loop through regions
for region in set(id_list.reset_index()['Country_Region']):
    #These regions need different initial conditions for optimizer to converge nicely
    if region in ['Australia','Japan','Germany','Canada','United Kingdom','France','Iceland']:
        z0 = 9
        f0 = 10
    #These initial conditions work well everywhere else
    else:
        z0 = 4.5
        f0 = 50
    p_region = p.T.loc[region].T
    f_region = f.T.loc[region].T
    
    #Now loop through the "subregions" (states or provinces)
    for subregion in p_region.keys():
        p_train = p_region[subregion]
        f_train = f_region[subregion]
        
        #Only use places where the number of cases eventually exceeds twice the minimum threshold, and where there are at least three data points
        #South Korea and China have already saturated, so I'm going to keep the estimate at the current number of cases
        #South Africa and Ecuador aren't working, and I haven't tracked down the problem yet.
        if np.max(p_train)>2*p0 and np.sum(p_train>p0)>3 and region not in ['Japan','Holy See','Diamond Princess','Greenland','Korea, South','China','South Africa','Ecuador','Syria'] and subregion not in ['Missouri','Wisconsin','North Carolina','Quebec','New South Wales']:
            t00 = p_train.loc[p_train>=start_cutoff].index.values[0]
            C0 = p_train.max()/(np.max(p_train.index.values)-t00)**z0
            p_train = p_train.loc[p_train>p0]
            out = minimize(cost_p,[t00,C0,z0],args=(p_train,),jac=jac_p,bounds=((None,int(p_train.index.values[0])-1),(1e-6,None),(0,10)))
            t0,C,z = out.x
            p_valid = True #out.success
        else:
            p_valid = False

        #If the spreading model was successfully trained, now try to learn the fatality rate and time delay
        if np.max(f_train)>2*f0 and p_valid and np.sum(f_train>f0)>2:
            f_train = f_train.loc[f_train>f0]
            out = minimize(cost_f,[Delta0,r0],args=(f_train,[t0,C,z]),jac=jac_f,bounds=((1,12),(1e-6,1)))
            Delta,r = out.x
            f_valid = True #out.success
        else:
            f_valid = False

        #Save the data
        if p_valid:
            params_table.loc[param_id,'Country_Region']=region
            params_table.loc[param_id,'State_Province']=subregion
            params_table.loc[param_id,'t0'] = t0
            params_table.loc[param_id,'t0_abs'] = (timedelta(days=t0)+data_live['date'].iloc[-1]).isoformat()[:10]
            params_table.loc[param_id,'C'] = C
            params_table.loc[param_id,'z'] = z
            if subregion is not 'NaN':
                table = PowerLaw(data_global,metric='positive',regions=[region],params=[z,C],t0=t0,subregion=subregion,ymin=1,xmin=1)
            else:
                table = PowerLaw(data_global,metric='positive',regions=[region],params=[z,C],t0=t0)
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]
                submission.loc[pred_id,'ConfirmedCases'] = C*((t-t0)**z)
        else:
            failed_regions.append([region,subregion])
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]
                if t in p_train.index.values:
                    submission.loc[pred_id,'ConfirmedCases'] = p_train.loc[t]
                else:
                    submission.loc[pred_id,'ConfirmedCases'] = np.max(p_train)
                
        if f_valid:
            params_table.loc[param_id,'Delta t'] = Delta
            params_table.loc[param_id,'r'] = r
            if subregion is not 'NaN':
                table = PowerLaw(data_global,metric='death',regions=[region],params=[z,C*r],t0=t0+Delta,ymin=1,subregion=subregion)
            else:
                table = PowerLaw(data_global,metric='death',regions=[region],params=[z,C*r],t0=t0+Delta,ymin=1)
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]
                submission.loc[pred_id,'Fatalities'] = r*C*((t-t0-Delta)**z)
        else:
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]
                if t in f_train.index.values:
                    submission.loc[pred_id,'Fatalities'] = f_train.loc[t]
                else:
                    submission.loc[pred_id,'Fatalities'] = np.max(f_train)
                
        param_id += 1
submission.to_csv('submission.csv')
params_table.to_csv('params.csv',index=False)
params_table = pd.read_csv('params.csv',index_col=[0,1])
zlist = params_table['z']
zlist = zlist.loc[~np.isnan(zlist)]
plt.hist(zlist)
plt.show()

rlist = params_table['r']
rlist = rlist.loc[~np.isnan(rlist)]
plt.hist(rlist)
plt.show()

Dellist = params_table['Delta t']
Dellist = Dellist.loc[~np.isnan(Dellist)]
plt.hist(Dellist)
plt.show()
params_table = pd.read_csv('params.csv')
params_table['State_Province'] = params_table['State_Province'].fillna(value='NaN')
params_table = params_table.set_index(['Country_Region','State_Province']).sort_index()
zlist = params_table['z']
rlist = params_table['r']
Dellist = params_table['Delta t']

z = zlist.mean()
r0 = rlist.mean()
Delta = Dellist.mean()

start_cutoff=4
p = data_global.pivot_table(index='elapsed',values='positive',columns=['state','subregion'],aggfunc=np.sum)
f = data_global.pivot_table(index='elapsed',values='death',columns=['state','subregion'],aggfunc=np.sum)

#Make predictions for regions with insufficient data, based on global averages
for region in set(id_list.reset_index()['Country_Region']):
    p_region = p.T.loc[region].T
    f_region = f.T.loc[region].T
    
    #Now loop through the "subregions" (states or provinces)
    for subregion in p_region.keys():
        p_train = p_region[subregion]
        f_train = f_region[subregion]
        z = zlist.mean()
        
        #Find all the regions that did not meet our criteria before (and that are not on the excluded list)
        if not(np.max(p_train)>2*p0 and np.sum(p_train>p0)>3) and region not in ['Japan','Holy See','Diamond Princess','Greenland','Korea, South','China','South Africa','Ecuador','Syria'] and subregion not in ['Missouri','Wisconsin','North Carolina','Quebec','New South Wales']:
            #Estimate start time if possible
            if (p_train>start_cutoff).sum()>1:
                t0 = p_train.loc[p_train>=start_cutoff].index.values[0]
                C = p_train.max()/(np.max(p_train.index.values)-t0)**z
            #If not enough cases to estimate start time, assume infection is contained
            else:
                t0 = -80
                z = 0
                C = p_train.max()
            p_valid = False 
        else:
            p_valid = True

        #Find all the regions that did not meet our criteria before (and that are not on the excluded list)
        if not(np.max(f_train)>2*f0 and p_valid and np.sum(f_train>f0)>2) and region not in ['Japan','Holy See','Diamond Princess','Greenland','Korea, South','China','South Africa','Ecuador','Syria'] and subregion not in ['Missouri','Wisconsin','North Carolina','Quebec','New South Wales']:
            #If there is spread and there are fatalities, estimate fatality rate from data
            if (region, subregion) in params_table.index.tolist():
                z = float(params_table.loc[[region,subregion],'z'])
                t0 = float(params_table.loc[[region,subregion],'t0'])
                C = float(params_table.loc[[region,subregion],'C'])
                t0_abs = (timedelta(days=t0)+data_live['date'].iloc[-1]).isoformat()
            if f_train.max() > 1 and z>0 and np.max(f_train.index.values)-t0-Delta > 0:
                r = f_train.max()/(C*(np.max(f_train.index.values)-t0-Delta)**z)
            else:
                r = r0
            f_valid = False 
        else:
            f_valid = True

        #Save the data and plot
        if not p_valid:
            t0_abs = (timedelta(days=t0)+data_live['date'].iloc[-1]).isoformat()
            new_params = pd.DataFrame(np.asarray([t0,t0_abs,C,z,np.nan,np.nan])[np.newaxis,:],index=pd.MultiIndex.from_tuples([(region,subregion)]),columns=['t0','t0_abs','C','z','Delta t','r'])
            params_table = params_table.append(new_params)
            if subregion is not 'NaN':
                table = PowerLaw(data_global,metric='positive',regions=[region],params=[z,C],t0=t0,subregion=subregion,ymin=1,xmin=1)
            else:
                table = PowerLaw(data_global,metric='positive',regions=[region],params=[z,C],t0=t0,ymin=1,xmin=1)
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]
                submission.loc[pred_id,'ConfirmedCases'] = C*((t-t0)**z)
                
        if not f_valid:
            params_table.loc[region,subregion] = np.asarray([t0,t0_abs,C,z,Delta,r])
            if subregion is not 'NaN':
                table = PowerLaw(data_global,metric='death',regions=[region],params=[z,C*r],t0=t0+Delta,subregion=subregion,ymin=1,xmin=1)
            else:
                table = PowerLaw(data_global,metric='death',regions=[region],params=[z,C*r],t0=t0+Delta,ymin=1,xmin=1)
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]
                submission.loc[pred_id,'Fatalities'] = r*C*((t-t0-Delta)**z)
submission.to_csv('submission.csv')
params_table.to_csv('params.csv',index=False)
params_table=pd.read_csv('params.csv')
region = 'US'
params_region = params_table.set_index(['Country_Region','State_Province']).loc[region]
params_region['t0'] = pd.to_datetime(params_region['t0'],format='%Y-%m-%d')
Delta_region = timedelta(days=params_region['Delta t'].mean())
r_region = params_region['r'].mean()
params_region['Delta t'] = params_region['Delta t'].fillna(value=0)
for item in params_region.index:
    params_region.loc[item,'Delta t'] = timedelta(days=params_region.loc[item,'Delta t'])

dates = [datetime.today()+timedelta(days=k) for k in range(40)]
case_predictions = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])
death_predictions = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])
for subregion in case_predictions.keys():
    params = params_region.loc[subregion]
    for t in case_predictions.index:
        case_predictions.loc[t,subregion] = params['C']*((t-params['t0'])/timedelta(days=1))**params['z']
        if params['r'] is not np.nan:
            death_predictions.loc[t] = params['r']*params['C']*((t-params['t0']-params['Delta t'])/timedelta(days=1))**params['z']
        else:
            death_predictions.loc[t] = r_region*params['C']*((t-params['t0']-Delta_region)/timedelta(days=1))**params['z']
case_predictions = case_predictions.fillna(value=0)
death_predictions = death_predictions.fillna(value=0)

case_predictions.sum(axis=1).plot(label='Cases')
death_predictions.sum(axis=1).plot(label='Fatalities')
plt.gca().set_yscale('log')
plt.gca().set_title('US Total Predictions')
plt.legend()
plt.show()
table = BasicPlots(data_live,metric='positive',regions=['MA','WI','MI','TX','IL'])
table = BasicPlots(data_global,metric='positive',regions=['Italy','Spain','France'])
table = BasicPlots(data_live,metric='hospitalized',regions=['MA','NY'])
table = BasicPlots(data_live,metric='death',regions=['MA','WI','MI','TX','IL'])
states=['MA','WI','NY','CA']
table = BasicPlots(data_live,metric='totalTestResults',regions=states)

table1 = data_live.pivot_table(index='date',values='totalTestResults',columns='state')
table2 = data_live.pivot_table(index='date',values='positive',columns='state')
table = table2/table1
table[states].plot()
plt.gca().set_ylabel('Positive fraction')
plt.show()

(table2.sum(axis=1)/table1.sum(axis=1)).plot(marker='o')
plt.gca().set_ylabel('Positive fraction')
plt.gca().set_ylim((0,.2))
plt.show()
