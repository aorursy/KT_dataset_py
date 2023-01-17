import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import pycountry
import plotly.express as px
%matplotlib inline

sns.set(style="darkgrid")
# load hungarian data from csv and create another chart
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
df_hun = df.query('Country=="Hungary"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df_hun['Last Update']=df_hun['Last Update'].dt.date
df_hun
df_hun.drop(df_hun.index[3], inplace=True)
current_dates = df_hun['Last Update'].tolist()

# create python lists for matplotlib
confirmed_values = df_hun.Confirmed.tolist()
recovered_values = df_hun.Recovered.tolist()
deaths_values = df_hun.Deaths.tolist()
days_values = len(confirmed_values)
# SEIR model infection process for Hungary day-by-day
days = 270 # days to run
N = 9700000
S =  9700000 # susceptible population
E = 1.0 # exposed population - infected
I = 0.0 # infectious population - infected
R = 0.0 # recovered population
D = 0.0 # deceased population

r = 10.0 # number of contacts per day
beta = 0.5 # probability of disease transmission per contact
epsilon = 0.2 # per capita rate of progression to infectious state (1/incubation time)
mu = 0.03 # mortality rate
gamma = 1 - mu # per capita recovery rate

infections, recoverings, deaths = [], [], []

for d in range(days):
    
    # intézkedések hatására csökkenő r (kontaktszám)
    if (d ==  0): r = 10.0
    if (d == 10): r = 5.0
    if (d == 18): r = 4.0
    if (d == 25): r = 3.0
    if (d == 40): r = 2.5
    
    infections.append(np.floor(E+I))
    deaths.append(np.floor(D))
    recoverings.append(np.floor(R))
    
    S2 = S - r * beta * S * I / N
    E2 = E + r * beta * S * I / N - epsilon * E
    I2 = I + epsilon * E - gamma * I - mu * I
    R2 = R + gamma * I
    D2 = D + mu * I
    
    S = S2; E = E2; I = I2; R = R2; D = D2;       
# create chart with current data from csv and predicted data from theoretical model
pred_dates = pd.date_range(start='2020-03-04', periods=days).to_pydatetime().tolist()

plt.figure(figsize=(20,10)) 
plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

plt.plot(current_dates, confirmed_values, color='blue', label='confirmed')
plt.plot(current_dates, deaths_values, color='red', label='deaths')
plt.plot(current_dates, recovered_values, color='green', label='recovered')

plt.plot(pred_dates, infections, color='blue', linestyle='--', label='pred-confirmed')
plt.plot(pred_dates, deaths, color='red', linestyle='--', label='pred-deaths')
plt.plot(pred_dates, recoverings, color='green', linestyle='--', label='pred-recovered')

plt.legend()
plt.show()
from ipywidgets import interact
import numpy as np

from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import DatetimeTickFormatter
from bokeh.plotting import figure
output_notebook()
# SEIR model infection process for Hungary day-by-day

def calculate_epidemic_data(r, beta, days, epsilon, mu):
    N = 9700000
    S =  9700000 # susceptible population
    E = 1.0 # exposed population - infected
    I = 0.0 # infectious population - infected
    R = 0.0 # recovered population
    D = 0.0 # deceased population
    
    r = r # number of contacts per day
    beta = beta # probability of disease transmission per contact
    epsilon = epsilon # per capita rate of progression to infectious state (1/incubation time)
    mu = mu # mortality rate
    gamma = 1 - mu # per capita recovery rate

    infections, recoverings, deaths = [], [], []

    for d in range(days):

        infections.append(np.floor(E+I))
        deaths.append(np.floor(D))
        recoverings.append(np.floor(R))

        S2 = S - r * beta * S * I / N
        E2 = E + r * beta * S * I / N - epsilon * E
        I2 = I + epsilon * E - gamma * I - mu * I
        R2 = R + gamma * I
        D2 = D + mu * I

        S = S2; E = E2; I = I2; R = R2; D = D2;  
        
    return infections
days_values
x = pd.date_range(start='2020-03-04', periods=60).to_pydatetime().tolist()
y = calculate_epidemic_data(r=5.0, beta=0.5, days=60, epsilon=0.2, mu=0.03) 

p = figure(title="number of infected people", plot_height=500, plot_width=1000, background_fill_color='#efefef')
infection_lines = p.multi_line(xs=[df_hun['Last Update'].tolist(), x], ys=[df_hun.Confirmed.tolist(), y], line_color=["red", "blue"])
scatterplot = p.circle(df_hun['Last Update'].tolist(), df_hun.Confirmed.tolist(), color="red")
p.xaxis.formatter=DatetimeTickFormatter(hours=["%d %B %Y"], days=["%d %B %Y"], months=["%d %B %Y"], years=["%d %B %Y"])
def update(r=5.2, beta=0.5, days=60):
    infection_lines.data_source.data['xs'] = [df_hun['Last Update'].tolist(), pd.date_range(start='2020-03-04', periods=days).to_pydatetime().tolist()]
    infection_lines.data_source.data['ys'] = [df_hun.Confirmed.tolist(), calculate_epidemic_data(r, beta, days, epsilon, mu)]
    push_notebook()
show(p, notebook_handle=True)
interact(update, r=(1,15, 0.1), beta=(0, 1, 0.01), days=(30,210,10))
x = pd.date_range(start='2020-03-04', periods=40).to_pydatetime().tolist()
df_pred = pd.DataFrame(x, columns=["Date"])
df_pred['Date']=df_pred['Date'].dt.date
for r in np.arange(1.0, 15.0, 0.10):
    y_hat = calculate_epidemic_data(r, beta=0.6, days=40, epsilon=0.2, mu=0.03)
    df_pred[r] = pd.Series(y_hat)
df_pred
df_pred.dtypes
df2 = df_hun.merge(df_pred,left_on='Last Update', right_on='Date')
df2.drop(columns=['Deaths', 'Recovered', 'Date', 'Last Update'], inplace=True)
df2
df3=df2.sub(df2['Confirmed'], axis=0)
df3=df3.drop(columns='Confirmed')
df3=df3.apply(np.square)
df3=df3.sum(axis=0)
df3.min(axis=0)
df3.idxmin(axis=0)