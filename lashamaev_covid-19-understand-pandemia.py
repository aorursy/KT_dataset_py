# run each cell by Shift+Enter



import pandas as pd    

import numpy as np    



df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv') 



def shorten(x):

    return x[:3]



df['weekday'] = pd.to_datetime(df['Date'], format="%m/%d/%y").dt.day_name()

df['weekday'] = df['weekday'].apply(shorten)



countries = list(df['Country/Region'].unique())





def add_col(xf, xs, col, shift=20):

    xs = xs[shift:].reset_index(drop=True)

    xs = xs.append(pd.Series(np.zeros(shift)))

    xs = xs.replace(0, np.NaN)

    xf = xf.reset_index(drop=True)

    xs = xs.reset_index(drop=True)

    xf[col] = xs.replace(0, np.NaN)

    return xf





def correction_late_ones(conf):

    daily = conf[1:] - conf[:-1]

    for i in range(1,len(daily)-1):                  # clean problem with late confirmation. we think that confirmation can late for 1 day, not longer

        if (daily[i-1]>5)and(daily[i]>5)and(daily[i]>1.1*1.1*daily[i-1])and(1.1*daily[i-1]<daily[i-2]):  # strange minimum and peak

            twodays = daily[i] + daily[i-1]               # two days sum  

            print('found late confirmation', daily[i-3], daily[i-2],daily[i-1],daily[i],daily[i+1])

            ratio = np.cbrt(daily[i+1]/daily[i-2])        # geometric mean of three days 

            daily[i-1] = int(1*twodays/(1+ratio))                

            daily[i] = twodays - daily[i-1]          # sum of two days are the same 

            print('changed to confirmation', daily[i-3], daily[i-2],daily[i-1],daily[i],daily[i+1])



    newconf =  np.insert(np.cumsum(daily),0,0)    

    print('Map of corrections ',newconf-conf)

    return [newconf,np.sum(np.abs(newconf-conf))]
print('Current data set is "covid_19_clean_complete.csv" from kaggle.com by Devakumar KP')

print('data from ',df[:1]['Date'].to_string(index=False),' till ', df[-1:]['Date'].to_string(index=False))

# select big coiuntries

countriez = sorted(list(df[df['Confirmed']>300][50:]['Country/Region'].unique()))



data = []

for country in countriez:

    print(' ')

    print('COUNTRY ',country)

    region_df = df[df['Country/Region']==country]

    if len(region_df['Province/State'].unique())>1:

        region_df = region_df[region_df['Province/State']==country]

        

    if len(region_df)>30:

        conf = np.array(region_df['Confirmed'])

        actual_value=conf[-1]



        print('REPORT ON CORRECTIONS ###############')

    

        for i in range(1,len(conf)):                    # clean March 22-23rd problem

            if (conf[i]>300)and(i>1):

                if conf[i] == conf[i-1]:

                     print('found no change in value of CONFIRMED',  region_df[i:i+1]['Date'].to_string(index=False))

                     conf[i-1] = int(np.sqrt(conf[i-2]*conf[i]))   # geometric mean used to keep the daily rate stable



        for i in range(3):

            [newconf, a]  = correction_late_ones(conf)

            conf = newconf

            if a < 1:

                break

    

        region_df['Confirmed'] = conf

        

        daily = conf[1:] - conf[:-1]

        region_df['Daily'] =  np.insert(daily,0,0)

        region_df['Daily_shifted'] =  np.append(daily,daily[-1])

        region_df['Percent'] = 100*(region_df['Daily_shifted']/region_df['Confirmed'])





        region_df = region_df[['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Daily', 'weekday','Percent']]



        data.append(region_df)

data = pd.concat(data)



data['Percent'] = data['Percent'].replace(-np.inf, np.NaN)

data['Percent'] = data['Percent'].replace(np.inf, np.NaN)

data['Percent'] = data['Percent'].fillna(0.0)



data['Percent'] = data['Percent'].astype(int)

data['Percent'] = data['Percent'].replace(0,np.NaN)





data.to_excel('output.xlsx')         
data[(data['Country/Region']=='Italy')&(data['Confirmed']>30)]
data[(data['Country/Region']=='France')&(data['Confirmed']>30)]
data[(data['Country/Region']=='Germany')&(data['Confirmed']>30)]