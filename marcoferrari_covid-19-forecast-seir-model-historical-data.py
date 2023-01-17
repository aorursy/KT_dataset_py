import pandas as pd

import numpy as np

import datetime as dt

from scipy.integrate import solve_ivp

from scipy.optimize import minimize

import plotly.graph_objects as go

import plotly.express as px

from IPython.display import Image
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df.head()
country = 'Italy'



# Confirmed cases

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

cols = df.columns[4:]

infected = df.loc[df['Country/Region']==country, cols].values.flatten()



# Deaths

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

deceased = df.loc[df['Country/Region']==country, cols].values.flatten()



# Recovered

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

recovered = df.loc[df['Country/Region']==country, cols].values.flatten()
dates = cols.values

x = [dt.datetime.strptime(d,'%m/%d/%y').date() for d in dates]



fig = go.Figure(data=go.Scatter(x=x, y=infected,

                               mode='lines+markers',

                               name='Infected'))

fig.add_trace(go.Scatter(x=x, y=deceased,

                    mode='lines+markers',

                    name='Deceased'))

fig.add_trace(go.Scatter(x=x, y=recovered,

                    mode='lines+markers',

                    name='Recovered'))

fig.update_layout(title='COVID-19 spread in Italy',

                   xaxis_title='Days',

                   yaxis_title='Number of individuals')

fig.show()
infected
infected_clean = infected[30:]

deceased_clean = deceased[30:]

recovered_clean = recovered[30:]
Image('../input/seir-model/SEIR.png')
def SEIR_q(t, y, beta, gamma, sigma, alpha, t_quarantine):

    """SEIR epidemic model.

        S: subsceptible

        E: exposed

        I: infected

        R: recovered

        

        N: total population (S+E+I+R)

        

        Social distancing is adopted when t>=t_quarantine.

    """

    S = y[0]

    E = y[1]

    I = y[2]

    R = y[3]

    

    if(t>t_quarantine):

        beta_t = beta*np.exp(-alpha*(t-t_quarantine))

    else:

        beta_t = beta

    dS = -beta_t*S*I/N

    dE = beta_t*S*I/N - sigma*E

    dI = sigma*E - gamma*I

    dR = gamma*I

    return [dS, dE, dI, dR]
N = 100

beta, gamma, sigma, alpha = [2, 0.4, 0.1, 0.5]

t_q = 10

y0 = np.array([99, 0, 1, 0])

sol = solve_ivp(SEIR_q, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q))



fig = go.Figure(data=go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, with intervention',

                               line=dict(color=px.colors.qualitative.Plotly[0])))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, with intervention',

                        line=dict(color=px.colors.qualitative.Plotly[1])))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, with intervention',

                        line=dict(color=px.colors.qualitative.Plotly[2])))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, with intervention',

                        line=dict(color=px.colors.qualitative.Plotly[3])))





beta, gamma, sigma, alpha = [2, 0.4, 0.1, 0.0]

t_q = 10

y0 = np.array([99, 0, 1, 0])

sol = solve_ivp(SEIR_q, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q))



fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, no intervention',

                               line=dict(color=px.colors.qualitative.Plotly[0], dash='dash')))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, no intervention',

                        line=dict(color=px.colors.qualitative.Plotly[1], dash='dash')))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, no intervention',

                        line=dict(color=px.colors.qualitative.Plotly[2], dash='dash')))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, no intervention',

                        line=dict(color=px.colors.qualitative.Plotly[3], dash='dash')))



fig.update_layout(title='SEIR epidemic model',

                 xaxis_title='Days',

                 yaxis_title='Percentage of population')

fig.show()
def fit_to_data(vec, t_q, N, test_size):

    beta, gamma, sigma, alpha = vec

    

    sol = solve_ivp(SEIR_q, [0, t_f], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)

    

    split = np.int((1-test_size) * infected_clean.shape[0])

    

    error = (

        np.sum(

            5*(deceased_clean[:split]+recovered_clean[:split]-sol.y[3][:split])**2) +    

        np.sum(

            (infected_clean[:split]-np.cumsum(sol.y[1][:split]+sol.y[2][:split]))**2)

    ) / split

    

    return error
N = 60e6 / (10/1.1)

N = np.int(N)

t_q = 7 # quarantine takes place

t_f = infected_clean.shape[0]

y0 = [N-infected_clean[0], 0, infected_clean[0], 0]

t_eval = np.arange(0,t_f,1)

test_size = 0.1



opt = minimize(fit_to_data, [2, 1, 0.8, 0.3], method='Nelder-Mead', args=(t_q, N, test_size))

beta, gamma, sigma, alpha = opt.x

sol = solve_ivp(SEIR_q, [0, t_f], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)
fig = go.Figure(data=go.Scatter(x=x[30:], y=np.cumsum(sol.y[1]+sol.y[2]), name='E+I',

                               marker_color=px.colors.qualitative.Plotly[0]))

fig.add_trace(go.Scatter(x=x[30:], y=infected_clean, name='Infected', mode='markers', 

                         marker_color=px.colors.qualitative.Plotly[0]))

fig.add_trace(go.Scatter(x=x[30:], y=sol.y[3], name='R', mode='lines', 

                         marker_color=px.colors.qualitative.Plotly[1]))

fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean+recovered_clean, name='Deceased+recovered', 

                         mode='markers', 

                         marker_color=px.colors.qualitative.Plotly[1]))

fig.add_trace(go.Scatter(x=[x[37], x[37]], y=[0, 100000], name='Quarantine', mode='lines',

                        marker_color='darkgrey'))

fig.update_layout(title='''Model's predictions vs historical data''',

                   xaxis_title='Days',

                   yaxis_title='Number of individuals')



fig.show()
test_size = 0



opt = minimize(fit_to_data, [2, 1, 0.8, 0.3], method='Nelder-Mead', args=(t_q, N, test_size))

beta, gamma, sigma, alpha = opt.x

sol = solve_ivp(SEIR_q, [0, t_f], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)
fig = go.Figure(data=go.Scatter(x=x[30:], y=np.cumsum(sol.y[1]+sol.y[2]), name='E+I',

                               marker_color=px.colors.qualitative.Plotly[0]))

fig.add_trace(go.Scatter(x=x[30:], y=infected_clean, name='Infected', mode='markers', 

                         marker_color=px.colors.qualitative.Plotly[0]))

fig.add_trace(go.Scatter(x=x[30:], y=sol.y[3], name='R', mode='lines', 

                         marker_color=px.colors.qualitative.Plotly[1]))

fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean+recovered_clean, name='Deceased+recovered', 

                         mode='markers', 

                         marker_color=px.colors.qualitative.Plotly[1]))

fig.add_trace(go.Scatter(x=[x[37], x[37]], y=[0, 100000], name='Quarantine', mode='lines',

                        marker_color='darkgrey'))

fig.update_layout(title='''Model's predictions vs historical data''',

                   xaxis_title='Days',

                   yaxis_title='Number of individuals')



fig.show()
days_ahead = 45

new_x = x[30:] + [x[-1]+dt.timedelta(days=day) for day in range(1, days_ahead)]

t_eval = np.arange(0,t_f+days_ahead,1)

sol = solve_ivp(SEIR_q, [0, t_f+days_ahead], y0, args=(beta, gamma, sigma, alpha, t_q), t_eval=t_eval)



peak = new_x[np.argmax(sol.y[2])]



fig = go.Figure(data=go.Scatter(x=new_x, y=sol.y[1], name='E'))

fig.add_trace(go.Scatter(x=new_x, y=sol.y[2], name='I'))

fig.add_trace(go.Scatter(x=new_x, y=sol.y[3], name='R'))

fig.add_trace(go.Scatter(x=[peak, peak], y=[0, 5e4], name='Predicted peak', mode='lines',

             line=dict(color=px.colors.qualitative.Plotly[3], dash='dot')))

fig.update_layout(title='''Model's predictions''',

                   xaxis_title='Days',

                   yaxis_title='Number of individuals')

fig.show()
fig = go.Figure(data=go.Scatter(x=new_x, y=np.cumsum(sol.y[1]+sol.y[2]), name='Infected'))

fig.add_trace(go.Scatter(x=new_x, y=sol.y[3], name='Deceased+recovered'))



fig.update_layout(title='''Model's predictions''',

                   xaxis_title='Days',

                   yaxis_title='Number of individuals')

fig.show()
death_rate=.457



fig = go.Figure(data=go.Scatter(x=new_x, y=sol.y[3]*death_rate, name='Deceased (predicted)',

                               line=dict(color=px.colors.qualitative.Plotly[2])))

fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean, name='Historical', mode='markers',

                        marker_color=px.colors.qualitative.Plotly[3]))

fig.update_layout(title='Predicted deaths and historical data',

                   xaxis_title='Days',

                   yaxis_title='Number of individuals')

fig.show()
R_0 = beta / gamma

incubation = 1 / sigma



print('Estimated reproductive number: {:.2f}'.format(R_0))

print('Estimated mean incubation period: {:.2f}'.format(incubation))
def SEIR_q_stop(t, y, beta, gamma, sigma, alpha, t_quarantine, t_stop):

    """SEIR epidemic model.

        S: subsceptible

        E: exposed

        I: infected

        R: recovered

        

        N: total population (S+E+I+R)

        

        Social distancing is adopted when t>t_quarantine and t<=t_stop.

    """

    S = y[0]

    E = y[1]

    I = y[2]

    R = y[3]

    

    if(t>t_quarantine and t<=t_stop):

        beta_t = beta*np.exp(-alpha*(t-t_quarantine))

    else:

        beta_t = beta

    dS = -beta_t*S*I/N

    dE = beta_t*S*I/N - sigma*E

    dI = sigma*E - gamma*I

    dR = gamma*I

    return [dS, dE, dI, dR]
N = 100

beta, gamma, sigma, alpha = [2, 0.4, 0.1, 0.5]

t_q = 10

t_stop = 30

y0 = np.array([99, 0, 1, 0])

sol = solve_ivp(SEIR_q_stop, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q, t_stop))



fig = go.Figure(data=go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, interrupted',

                               line=dict(color=px.colors.qualitative.Plotly[0])))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, interrupted',

                        line=dict(color=px.colors.qualitative.Plotly[1])))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, interrupted',

                        line=dict(color=px.colors.qualitative.Plotly[2])))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, interrupted',

                        line=dict(color=px.colors.qualitative.Plotly[3])))



t_stop = 200

sol = solve_ivp(SEIR_q_stop, [0, 100], y0, t_eval=np.arange(0, 100, 0.1), args=(beta, gamma, sigma, alpha, t_q, t_stop))



fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, continuous',

                               line=dict(color=px.colors.qualitative.Plotly[0], dash='dash')))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, continuous',

                        line=dict(color=px.colors.qualitative.Plotly[1], dash='dash')))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, continuous',

                        line=dict(color=px.colors.qualitative.Plotly[2], dash='dash')))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, continuous',

                        line=dict(color=px.colors.qualitative.Plotly[3], dash='dash')))



fig.update_layout(title='SEIR epidemic model - effect of social-distancing',

                 xaxis_title='Days',

                 yaxis_title='Percentage of population')

fig.show()