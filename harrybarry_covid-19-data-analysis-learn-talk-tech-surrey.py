# Default Kaggle Python 3 environment message



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from scipy import integrate, optimize



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read in the data

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")



# Fill in empty cells under 'Provice_State'

train.Province_State.fillna("None", inplace=True)



# Display the first 5 rows and provide a summary table

display(train.head(5))

display(train.describe())



# Print some additional stats

print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
# Aggregate confirmed cases and fatalities across all countries

confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})



overall_total_date = confirmed_total_date.join(fatalities_total_date)



# Plot both confirmed and fatalities together

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

overall_total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)



# Plot just the fatalities

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
# Exclude China in our training data plots

confirmed_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

overall_total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)



# Plot both confirmed and fatalities together

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

overall_total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)



# Plot just the fatalities

fatalities_total_date_noChina.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases excluding China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

overall_total_date_China = confirmed_total_date_China.join(fatalities_total_date_China)



# Plot confirmed and fatalities together

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

overall_total_date_China.plot(ax=ax1)

ax1.set_title("China confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)



# Plot fatalities only

fatalities_total_date_China.plot(ax=ax2, color='orange')

ax2.set_title("China fatalities", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
# Italy

confirmed_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



# Spain 

confirmed_total_date_Spain = train[train['Country_Region']=='Spain'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Spain = train[train['Country_Region']=='Spain'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)



# South Korea

confirmed_total_date_KOR = train[train['Country_Region']=='Korea, South'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_KOR = train[train['Country_Region']=='Korea, South'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_KOR = confirmed_total_date_KOR.join(fatalities_total_date_KOR)



# TODO: Add Canada

confirmed_total_date_Canada = train[train['Country_Region']=='Canada'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Canada = train[train['Country_Region']=='Canada'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Canada = confirmed_total_date_Canada.join(fatalities_total_date_Canada)



# TODO: Add US

confirmed_total_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_US = confirmed_total_date_US.join(fatalities_total_date_US)







# Create the subplots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))



# Plot the confirmed totals together

ax1.plot(confirmed_total_date_Italy, label='Italy')

ax1.plot(confirmed_total_date_Spain, label='Spain')

ax1.plot(confirmed_total_date_KOR, label='South Korea')

ax1.plot(confirmed_total_date_Canada, label='Canada')

ax1.plot(confirmed_total_date_US, label='US')



ax1.set_title("Confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

ax1.legend(loc="upper left")





# Plot the fatalities totals together

ax2.plot(fatalities_total_date_Italy, label='Italy')

ax2.plot(fatalities_total_date_Spain, label='Spain')

ax2.plot(fatalities_total_date_KOR, label='South Korea')

ax2.plot(fatalities_total_date_Canada, label='Canada')

ax2.plot(fatalities_total_date_US, label='US')



ax2.set_title("Fatalities", size=13)

ax2.set_ylabel("Number of fatalities", size=13)

ax2.set_xlabel("Date", size=13)

ax2.legend(loc="upper left")
# Italy since first confirmed case

confirmed_total_date_Italy = train[(train['Country_Region']=='Italy') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[(train['Country_Region']=='Italy') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



# Spain since first confirmed case

confirmed_total_date_Spain = train[(train['Country_Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Spain = train[(train['Country_Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)



# South Korea since first confirmed case

confirmed_total_date_KOR = train[(train['Country_Region']=='Korea, South') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_KOR = train[(train['Country_Region']=='Korea, South') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_KOR = confirmed_total_date_KOR.join(fatalities_total_date_KOR)



# Canada since first confirmed case

confirmed_total_date_Canada = train[(train['Country_Region']=='Canada') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Canada = train[(train['Country_Region']=='Canada') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Canada = confirmed_total_date_Canada.join(fatalities_total_date_Canada)



# US since first confirmed case

confirmed_total_date_US = train[(train['Country_Region']=='US') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_US = train[(train['Country_Region']=='US') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_US = confirmed_total_date_US.join(fatalities_total_date_US)





# Extract 'Confirmed Cases' list and remove timestamps

italy_confirmed = [i for i in total_date_Italy.ConfirmedCases['sum'].values]

spain_confirmed = [i for i in total_date_Spain.ConfirmedCases['sum'].values]

kor_confirmed = [i for i in total_date_KOR.ConfirmedCases['sum'].values]

canada_confirmed =[i for i in total_date_Canada.ConfirmedCases['sum'].values]

us_confirmed = [i for i in total_date_US.ConfirmedCases['sum'].values]



# Extract 'Fatalities' list and remove timestamps

italy_fatalities = [i for i in total_date_Italy.Fatalities['sum'].values]

spain_fatalities = [i for i in total_date_Spain.Fatalities['sum'].values]

kor_fatalities = [i for i in total_date_KOR.Fatalities['sum'].values]

canada_fatalities = [i for i in total_date_Canada.Fatalities['sum'].values]

us_fatalities = [i for i in total_date_US.Fatalities['sum'].values]





# Create the subplots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))



# Plot the confirmed totals together

ax1.plot(italy_confirmed, label='Italy')

ax1.plot(spain_confirmed, label='Spain')

ax1.plot(kor_confirmed, label='South Korea')

ax1.plot(canada_confirmed, label='Canada')

ax1.plot(us_confirmed, label='US')



ax1.set_title("Total COVID-19 infections from the first confirmed case", size=13)

ax1.set_ylabel("Infected cases", size=13)

ax1.set_xlabel("Days", size=13)

ax1.legend(loc="upper left")





# Plot the fatalities totals together

ax2.plot(italy_fatalities, label='Italy')

ax2.plot(spain_fatalities, label='Singapore')

ax2.plot(kor_fatalities, label='Korea')

ax2.plot(canada_fatalities, label='Canada')

ax2.plot(us_fatalities, label='US')



ax2.set_title("Total COVID-19 fatalities from the first confirmed case", size=13)

ax2.set_ylabel("Number of fatalities", size=13)

ax2.set_xlabel("Days", size=13)

ax2.legend(loc="upper left")
pop_italy = 60461826.

pop_spain = 46754778.

pop_korea = 51269185.

pop_canada = 37742154.

pop_us = 331002651.





frac_Italy_ConfirmedCases = total_date_Italy.ConfirmedCases/pop_italy*100.

frac_Spain_ConfirmedCases = total_date_Spain.ConfirmedCases/pop_spain*100.

frac_KOR_ConfirmedCases = total_date_KOR.ConfirmedCases/pop_korea*100.



frac_Canada_ConfirmedCases = total_date_Canada.ConfirmedCases/pop_canada*100.

frac_US_ConfirmedCases = total_date_US.ConfirmedCases/pop_us*100.



plt.figure(figsize=(15,10))

plt.subplot(2, 3, 1)

frac_Italy_ConfirmedCases.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Fraction % of population infected")

plt.ylim(0, 0.5)



plt.subplot(2, 3, 2)

frac_Spain_ConfirmedCases.plot(ax=plt.gca(), title='Spain')

plt.ylim(0, 0.5)



plt.subplot(2, 3, 3)

frac_KOR_ConfirmedCases.plot(ax=plt.gca(), title='South Korea')

plt.ylim(0, 0.5)



plt.subplot(2, 3, 4)

frac_Canada_ConfirmedCases.plot(ax=plt.gca(), title='Canada')

plt.ylim(0, 0.5)



plt.subplot(2, 3, 5)

frac_US_ConfirmedCases.plot(ax=plt.gca(), title='US')

plt.ylim(0, 0.5)
# Susceptible equation

def fa(N, a, b, beta):

    fa = -beta*a*b

    return fa



# Infected equation

def fb(N, a, b, beta, gamma):

    fb = beta*a*b - gamma*b

    return fb



# Recovered/deceased equation

def fc(N, b, gamma):

    fc = gamma*b

    return fc



# Runge-Kutta method of 4th order for 3 dimensions (susceptible a, infected b and recovered r)

def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):

    a1 = fa(N, a, b, beta)*hs

    b1 = fb(N, a, b, beta, gamma)*hs

    c1 = fc(N, b, gamma)*hs

    ak = a + a1*0.5

    bk = b + b1*0.5

    ck = c + c1*0.5

    a2 = fa(N, ak, bk, beta)*hs

    b2 = fb(N, ak, bk, beta, gamma)*hs

    c2 = fc(N, bk, gamma)*hs

    ak = a + a2*0.5

    bk = b + b2*0.5

    ck = c + c2*0.5

    a3 = fa(N, ak, bk, beta)*hs

    b3 = fb(N, ak, bk, beta, gamma)*hs

    c3 = fc(N, bk, gamma)*hs

    ak = a + a3

    bk = b + b3

    ck = c + c3

    a4 = fa(N, ak, bk, beta)*hs

    b4 = fb(N, ak, bk, beta, gamma)*hs

    c4 = fc(N, bk, gamma)*hs

    a = a + (a1 + 2*(a2 + a3) + a4)/6

    b = b + (b1 + 2*(b2 + b3) + b4)/6

    c = c + (c1 + 2*(c2 + c3) + c4)/6

    return a, b, c



def SIR(N, b0, beta, gamma, hs):

    

    """

    N = total number of population

    beta = transition rate S->I

    gamma = transition rate I->R

    k =  denotes the constant degree distribution of the network (average value for networks in which 

    the probability of finding a node with a different connectivity decays exponentially fast

    hs = jump step of the numerical integration

    """

    

    # Initial condition

    a = float(N-1)/N -b0

    b = float(1)/N +b0

    c = 0.

    hs = 0.1



    sus, inf, rec= [],[],[]

    for i in range(10000): # Run for a certain number of time-steps

        sus.append(a)

        inf.append(b)

        rec.append(c)

        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)



    return sus, inf, rec
# User input parameters for the model

N = 7800*(10**6)

beta = 0.1

recovery_days = 28



# Other set parameters

b0 = 0

hs = 0.1



sus, inf, rec = SIR(N, b0, beta, 1/recovery_days, hs)



f = plt.figure(figsize=(8,5)) 

plt.plot(sus, 'b.', label='susceptible');

plt.plot(inf, 'r.', label='infected');

plt.plot(rec, 'c.', label='recovered/deceased');

plt.title("SIR model")

plt.xlabel("time", fontsize=10);

plt.ylabel("Fraction of population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,5000)

plt.savefig('SIR_example.png')

plt.show()
# Fraction of population confirmed cases

frac_Canada_ConfirmedCases = total_date_Canada.ConfirmedCases/pop_canada*100.

canada_confirmed = [i for i in frac_Canada_ConfirmedCases['sum'].values]



# Number of days since first confirmed case

num_days = list(range(1,len(canada_confirmed) + 1))



# Input parameters of the model

N = 37*(10**6)

beta = 0.25

recovery_days = 14



# Defined parameters

b0 = 0

hs = 0.1

gamma = 1 / recovery_days



# Create the model

sus, inf, rec = SIR(N, b0, beta, gamma, hs)



ydata = np.array(canada_confirmed, dtype=float)

xdata = np.array(num_days, dtype=float)



inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0



def sir_model(y, x, beta, gamma):

    sus = -beta * y[0] * y[1] / N

    rec = gamma * y[1]

    inf = -(sus + rec)

    return sus, inf, rec



def fit_odeint(x, beta, gamma):

    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]



# Fit the data to the model

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)





# Plot the result

plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SIR model for Canada infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1], " and recovery_days = ", 1/popt[1])

import scipy

import seaborn as sns

def plot_exponential_fit_data(d_df, title, delta):

    d_df = d_df.sort_values(by=['Date'], ascending=True)

    d_df['x'] = np.arange(len(d_df)) + 1

    d_df['y'] = d_df['sum']



    x = d_df['x'][:-delta]

    y = d_df['y'][:-delta]



    c2 = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y,  p0=(40, 0.1))

    #y = Ae^(Bx)

    A, B = c2[0]

    print(f'(y = Ae^(Bx)) A: {A}, B: {B}')

    x = range(1,d_df.shape[0] + 1)

    y_fit = A * np.exp(B * x)

    size = 3

    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))

    g = sns.scatterplot(x=d_df['x'][:-delta], y=d_df['y'][:-delta], label='Confirmed cases (included for fit)', color='red')

    g = sns.scatterplot(x=d_df['x'][-delta:], y=d_df['y'][-delta:], label='Confirmed cases (validation)', color='blue')

    g = sns.lineplot(x=x, y=y_fit, label='Predicted values', color='green')

    plt.xlabel('Days since first case')

    plt.ylabel(f'cases')

    plt.title(f'Confirmed cases & predicted evolution: {title}')

    plt.xticks(rotation=90)

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()
# Plot excluding the last 10 days

plot_exponential_fit_data(total_date_Canada.ConfirmedCases, 'Canada', 10)
# South Korea since first confirmed case

confirmed_total_date_KOR = train[(train['Country_Region']=='Korea, South') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_KOR = train[(train['Country_Region']=='Korea, South') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_KOR = confirmed_total_date_KOR.join(fatalities_total_date_KOR)



# BC confirmed cases and fatalities

confirmed_total_date_BC = train[(train['Country_Region']=='Canada') & (train['Province_State']=='British Columbia') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_BC = train[(train['Country_Region']=='Canada') & (train['Province_State']=='British Columbia') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_BC= confirmed_total_date_BC.join(fatalities_total_date_BC)



# Quebec confirmed cases and fatalities

confirmed_total_date_QC = train[(train['Country_Region']=='Canada') & (train['Province_State']=='Quebec') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_QC = train[(train['Country_Region']=='Canada') & (train['Province_State']=='Quebec') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_QC= confirmed_total_date_QC.join(fatalities_total_date_QC)



# Ontario confirmed cases and fatalities

confirmed_total_date_ON = train[(train['Country_Region']=='Canada') & (train['Province_State']=='Ontario') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_ON = train[(train['Country_Region']=='Canada') & (train['Province_State']=='Ontario') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_ON= confirmed_total_date_ON.join(fatalities_total_date_ON)
# Extract 'Confirmed Cases' list and remove timestamps



bc_confirmed = [i for i in total_date_BC.ConfirmedCases['sum'].values]

qc_confirmed = [i for i in total_date_QC.ConfirmedCases['sum'].values]

on_confirmed = [i for i in total_date_ON.ConfirmedCases['sum'].values]

kor_confirmed = [i for i in total_date_KOR.ConfirmedCases['sum'].values]



# Extract 'Fatalities' list and remove timestamps

bc_fatalities = [i for i in total_date_BC.Fatalities['sum'].values]

qc_fatalities = [i for i in total_date_QC.Fatalities['sum'].values]

on_fatalities = [i for i in total_date_ON.Fatalities['sum'].values]

kor_fatalities = [i for i in total_date_KOR.Fatalities['sum'].values]





# Create the subplots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))



# Plot the confirmed totals together

ax1.plot(bc_confirmed, label='BC')

ax1.plot(qc_confirmed, label='Quebec')

ax1.plot(on_confirmed, label='Ontario')

ax1.plot(kor_confirmed, label='South Korea')



ax1.set_title("Total COVID-19 infections from the first confirmed case", size=13)

ax1.set_ylabel("Infected cases", size=13)

ax1.set_xlabel("Days", size=13)

ax1.legend(loc="upper left")





# Plot the fatalities totals together

ax2.plot(bc_fatalities, label='BC')

ax2.plot(qc_fatalities, label='Quebec')

ax2.plot(on_fatalities, label='Ontario')

ax2.plot(kor_fatalities, label='Korea')



ax2.set_title("Total COVID-19 fatalities from the first confirmed case", size=13)

ax2.set_ylabel("Number of fatalities", size=13)

ax2.set_xlabel("Days", size=13)

ax2.legend(loc="upper left")