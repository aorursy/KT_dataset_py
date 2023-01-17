Italy=True
# Download the data and do some cleaning up

%matplotlib notebook

import matplotlib.pyplot as plt



import pandas as pd

import io

import requests

import numpy as np



url="https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv"



source = requests.get(url).content

dataframe = pd.read_csv(io.StringIO(source.decode('utf-8')))



if Italy==True:

    dataframe=dataframe[dataframe['Country/Region']=='Italy']

    

df=dataframe.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

# Transpose df so date columns are rows

df=df.transpose()



df.reset_index(inplace=True)

df.rename(columns={"index": "Date"},inplace=True)

df.Date=pd.to_datetime(df.Date, infer_datetime_format=True)  

Total=df.drop(columns='Date').sum(axis=1)

df['Total']=Total
# Exploratory data analysis





ax=df.plot(x='Date',y='Total',kind='line',legend=False,title='Italy: Total COVID-19 cases',label='Infected')





plt.legend(frameon=False)



plt.xlabel('Date')

plt.ylabel('Total')

plt.tight_layout()

plt.show()
def seir_model_with_soc_dist(init_vals, params, t):

    S_0, E_0, I_0, R_0 = init_vals

    S, E, I, R = [S_0], [E_0], [I_0], [R_0]

    alpha, beta, gamma, rho = params

    dt = t[1] - t[0]

    for _ in t[1:]:

        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt

        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt

        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt

        next_R = R[-1] + (gamma*I[-1])*dt

        S.append(next_S)

        E.append(next_E)

        I.append(next_I)

        R.append(next_R)

    return np.stack([S, E, I, R]).T
data=df[["Date","Total"]]

realtotal=np.array(df.Total.values)

days=np.array((df.Date - df.Date.iloc[0]).dt.days)









N = 60.48e6

infected0=100



recovery=0.30



S_0 = 1-(infected0/N) # Susceptible

E_0 = infected0/N  # Exposed

I_0 = infected0/N

R_0 = (recovery*infected0)/N



# Define parameters

t_max = 100

dt = 1/t_max

t = np.linspace(0, t_max, int(t_max/dt) + 1)





init_vals = S_0, E_0, I_0, R_0 



incubation_period=5 # [days]

latent_time=2 # [days]

alpha = 1.0/incubation_period 



gamma = 1.0/latent_time



R0=3.9

beta = gamma*R0



params = alpha, beta, gamma



rho_array=np.array([0.5,0.4])

fig=plt.figure()







# Run simulation

for rho in rho_array:

    params = alpha, beta, gamma, rho



    results = seir_model_with_soc_dist(init_vals, params, t)



    plt.plot(t,100*results[:,2],label='Infected, $\\rho$={}'.format(rho));

    plt.plot(t,100*results[:,1],label='Exposed',linestyle='--',color=plt.gca().lines[-1].get_color())

    





plt.ylim(0,0.2)

plt.plot(days,100*realtotal/N,'.',color='black',label='Real data')

plt.xlabel('Time [days]')

plt.ylabel('Population fraction (%)')

plt.title('Baseline covid-19 SEIR model with social distancing $\\alpha$={}, $\\beta$={}, $\\gamma$={}'.format(alpha,beta,gamma))



plt.legend(frameon=False)

plt.tight_layout()