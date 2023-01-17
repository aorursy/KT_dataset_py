!pip install psypy



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import psypy.psySI as si
CSV_TBL = "../input/plantlab-data/sensor_readings.csv"

df = pd.read_csv(CSV_TBL,parse_dates=['datetime'],dayfirst=True)
df.tail()
df.set_index('datetime',inplace=True)

df = df.loc['2019'].copy()



window = df.loc['2019-08-01':].copy()



int_min = 10

window.reset_index(inplace=True)

window['dt_short'] = window.apply(lambda x:dt.datetime(

    year=x.datetime.year,month=x.datetime.month,day=x.datetime.day,

    hour=x.datetime.hour,minute=int(x.datetime.minute/int_min)*int_min),axis=1)

del window['datetime']

window.rename(columns={'dt_short':'datetime'},inplace=True)



#create pivot table

# datetime, value 

tsdata = pd.pivot_table(window,values='value',columns='sensorid',index='datetime',aggfunc='mean')

tsdata['ENTH01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[1],axis=1)

tsdata['ENTH04'] = tsdata.apply(lambda x: si.state("DBT", x.TA04+273.15, "RH", x.HA04/100, 101325)[1],axis=1)

tsdata['MOIST01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[4],axis=1)

tsdata['deltaE'] = tsdata['ENTH04'] - tsdata['ENTH01']



import seaborn as sb

import matplotlib.pyplot as plt

sb.set(rc={'figure.figsize':(16, 8)})
window = tsdata.loc['2019-08-09':'2019-08-16']
def plot_parameters(window,parameters,x_axis=False):

    N = len(parameters)

    f, axes = plt.subplots(N,1,figsize=(16,6*N))

    i = 0

    for p in parameters:

        window[p].plot(linewidth=0.5,ax=axes.flat[i]).set_title(parameters[p])

        axes.flat[i].get_xaxis().set_visible(x_axis)

        i = i + 1

    plt.tight_layout()
parameters = {'TA01':'Temp C',

              'HA01':'rel hum %',

              'ENTH01':'Enthalpy kJ/kg',

              'deltaE':'HVAC work kJ/kg',

             }

plot_parameters(window,parameters)
window = tsdata.loc['2019-08-17':'2019-08-25']
parameters = {'LED01':'LED Blue %',

              'LED02':'LED White %',

              'LED03':'LED Red %',

              'TA01':'Temp C',

              'HA01':'rel hum %',

              'ENTH01':'Enthalpy kJ/kg',

              'deltaE':'HVAC work kJ/kg',

             }

plot_parameters(window,parameters)
window['CD01'].plot(linewidth=0.5);

window['CD02'].plot(linewidth=0.5);
lastweek = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(days=9),'%Y-%m-%d')

window = tsdata.loc[lastweek:]
parameters = {'LED01':'LED Blue %',

              'LED02':'LED White %',

              'LED03':'LED Red %',

              'TA01':'Temp C',

              'HA01':'rel hum %',

              'deltaE':'HVAC work kJ/kg',

             }

plot_parameters(window,parameters)
last48hrs = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(hours=48),'%Y-%m-%d')

window = tsdata.loc[last48hrs:]
parameters = {'LED01':'LED Blue %',

              'LED02':'LED White %',

              'LED03':'LED Red %',

              'TA01':'Temp C',

              'HA01':'rel hum %',

              'deltaE':'HVAC work kJ/kg',

             }

plot_parameters(window,parameters,x_axis=True)