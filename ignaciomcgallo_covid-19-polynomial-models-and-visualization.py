import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from scipy.integrate import odeint

from statistics import mean
corona=pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv",delimiter=",")
corona.head()
corona['Date'] = pd.to_datetime(corona['Date'],format='%Y-%m-%d')
max_date = max(corona['Date'])

corona["Time"] = max_date - corona['Date']
corona.head()
corona.shape
corona.Fatalities.describe()
(corona[corona.Fatalities>15]).count()
corona.isnull().sum()
top_country=corona["Country/Region"].value_counts().head(20)

top_country
aux1 = corona.groupby("Date").ConfirmedCases.sum()

aux2 = corona.groupby("Date").Fatalities.sum()



y1 = aux1

x1 = aux1.index



y2 = aux2

x2 = aux2.index



plt.figure(figsize=(15,8))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="Total Confirmed Cases")

plt.plot(x2,y2,label="Total Fatalities")



plt.legend(fontsize=10)

plt.ylabel("Cases",fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.title('Global confirmed cases and fatalities',fontsize=24)

plt.xticks(rotation=90)

plt.show()
china_c = corona[corona["Country/Region"]=="China"].groupby("Date").ConfirmedCases.sum()

china_f = corona[corona["Country/Region"]=="China"].groupby("Date").Fatalities.sum()
y1 = china_c

x1 = china_c.index



y2 = china_f

x2 = china_f.index



plt.figure(figsize=(15,5))





plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="Confirmed Cases in China")

plt.plot(x2,y2,label="Fatalities in China")



plt.legend(fontsize=10)

plt.ylabel("Cases",fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.title('China confirmed cases and fatalities',fontsize=24)

plt.xticks(rotation=90)









plt.show()
without_c = corona[corona["Country/Region"]!="China"].groupby("Date").ConfirmedCases.sum()

without_f = corona[corona["Country/Region"]!="China"].groupby("Date").Fatalities.sum()
y1 = without_c

x1 = without_c.index



y2 = without_f

x2 = without_f.index



plt.figure(figsize=(15,8))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="Confirmed Cases in China")

plt.plot(x2,y2,label="Fatalities in China")



plt.legend(fontsize=10)

plt.ylabel("Cases",fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.title('Global (without China) confirmed cases and fatalities',fontsize=24)

plt.xticks(rotation=90)

plt.show()
spain_c = corona[corona["Country/Region"]=="Spain"].groupby("Date").ConfirmedCases.sum()

spain_f = corona[corona["Country/Region"]=="Spain"].groupby("Date").Fatalities.sum()



italy_c = corona[corona["Country/Region"]=="Italy"].groupby("Date").ConfirmedCases.sum()

italy_f = corona[corona["Country/Region"]=="Italy"].groupby("Date").Fatalities.sum()



germany_c = corona[corona["Country/Region"]=="Germany"].groupby("Date").ConfirmedCases.sum()

germany_f = corona[corona["Country/Region"]=="Germany"].groupby("Date").Fatalities.sum()

y1 = spain_c

x1 = spain_c.index



y2 = italy_c

x2 = italy_c.index



y3 = germany_c

x3 = germany_c.index





plt.figure(figsize=(15,5))

plt.subplot(1,2,1)



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="Spain")

plt.plot(x2,y2,label="Italy")

plt.plot(x3,y3,label="Germany")





plt.legend(fontsize=10)

plt.ylabel("Cases",fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.title('Confirmed cases',fontsize=24)

plt.xticks(rotation=90)





y11 = spain_f

x11 = spain_f.index



y22 = italy_f

x22 = italy_f.index



y33 = germany_f

x33 = germany_f.index



plt.subplot(1,2,2)



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x11,y11,label="Spain")

plt.plot(x22,y22,label="Italy")

plt.plot(x33,y33,label="Germany")





plt.legend(fontsize=10)

plt.ylabel("Cases",fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.title('Fatalities',fontsize=24)

plt.xticks(rotation=90)

plt.show()



updated = corona[corona["Date"] == max(corona["Date"])]

updated_f = updated.groupby("Country/Region")["ConfirmedCases","Fatalities"].sum().reset_index()
import folium

from folium.plugins import HeatMap

m=folium.Map([30.5928,114.3055],zoom_start=3)

HeatMap(corona[['Lat','Long']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
import plotly.express as px



fig = px.choropleth(updated_f,locations='Country/Region', color='ConfirmedCases',

                    locationmode='country names', hover_name="Country/Region",

                           color_continuous_scale="Viridis",

                           range_color=(0, 60000),

                           labels={'Country/Region':'ConfirmedCases'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv", delimiter=",")

test.head()
test["Date"] = pd.to_datetime(test['Date'],format='%Y-%m-%d')
max(test.Date.unique())
X2 = np.arange(len(corona.groupby("Date")),len(corona.groupby("Date"))+15).reshape(-1,1)
degrees = [1,2,3]



X = np.arange(0, len(corona.groupby("Date"))).reshape(-1,1)

y = corona.groupby("Date").ConfirmedCases.sum().values



# Polynomial Regression-nth order

plt.figure(figsize=(15,10))

plt.scatter(corona.groupby("Date").ConfirmedCases.sum().index, y, s=10, alpha=0.3)



for degree in degrees:

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    model.fit(X, y)

    y_plot = model.predict(X)

    score = model.score(X, y)

    plt.plot(corona.groupby("Date").ConfirmedCases.sum().index, y_plot, label="n = %d" % degree + '; $R^2$: %.2f' % score)

    

    

plt.plot(corona.groupby("Date").ConfirmedCases.sum().index, corona.groupby("Date").ConfirmedCases.sum().values, label="Confirmed cases")

    

plt.legend(loc='lower right')

plt.xlabel("Date")

plt.ylabel("Cases")

plt.title("Confirmed cases")

plt.xlim(min(corona.groupby("Date").ConfirmedCases.sum().index), max(corona.groupby("Date").ConfirmedCases.sum().index))

plt.xticks(fontsize=14,rotation=90)

plt.show()
degree = 2



X = np.arange(0, len(corona.groupby("Date"))).reshape(-1,1)

y = corona.groupby("Date").ConfirmedCases.sum().values



# Polynomial Regression-2nd order

plt.figure(figsize=(15,10))

plt.scatter(X, y, s=10, alpha=0.3)





model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X, y)

y_plot = model.predict(X)

score = model.score(X, y)

plt.plot(X, y_plot, label="model, n = %d" % degree + '; $R^2$: %.2f' % score)

    

y_plot2 = model.predict(X2)

plt.plot(X2, y_plot2, label="Predictions")



    

plt.plot(X, corona.groupby("Date").ConfirmedCases.sum().values, label="Confirmed cases")

    



plt.legend(loc='lower right')

plt.xlabel("Days")

plt.ylabel("Cases")

plt.title("Confirmed cases")

plt.show()
# # Total population, N.

N = 47000000

# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 ,Tr = 1, 3.5, 10

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

beta, gamma = R0/Tr, 1./Tr

# A grid of time points (in days)

t = np.linspace(0, 120, 120)



# The SIR model differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



# Initial conditions vector

y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, beta, gamma))

S, I, R = ret.T



# Plot the data on three separate curves for S(t), I(t) and R(t)

fig = plt.figure(figsize=(15,5),facecolor='w')

ax = fig.add_subplot(111, axisbelow=True)

ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

ax.set_xlabel('Time /days')

ax.set_ylabel('Population')

ax.set_ylim(0,47000000)

ax.yaxis.set_tick_params(length=0)

ax.xaxis.set_tick_params(length=0)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')

legend = ax.legend()

legend.get_frame().set_alpha(0.5)

for spine in ('top', 'right', 'bottom', 'left'):

    ax.spines[spine].set_visible(False)

plt.show()
corona=corona[(corona["Date"]<"2020-03-12")]
degree=2

scores=[]



for country in corona['Country/Region'].unique():

    

    country_train = corona[corona['Country/Region']==country]

    country_test = test[test['Country/Region']==country]

        

    X = np.array(range(len(country_train))).reshape((-1,1))

    y = country_train['ConfirmedCases']

  



    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    model.fit(X, y)

    

    print("Country:", country)

    score = model.score(X, y)

    print("Score:",score)

        

    predict_x = (np.array(range(len(country_test)))+51).reshape((-1,1))

    test.loc[test['Country/Region']==country,'ConfirmedCases'] = model.predict(predict_x)

    

    scores.append(score)

    
mean(scores)
degree=2

scores=[]



for country in corona['Country/Region'].unique():

    

    country_train = corona[corona['Country/Region']==country]

    country_test = test[test['Country/Region']==country]

        

    X = np.array(range(len(country_train))).reshape((-1,1))

    y = country_train['Fatalities']

   



    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    model.fit(X, y)

    

    print("Country:", country)

    score = model.score(X, y)

    print("Score:",score)

        

    predict_x = (np.array(range(len(country_test)))+51).reshape((-1,1))

    test.loc[test['Country/Region']==country,'Fatalities'] = model.predict(predict_x)

    

    scores.append(score)
mean(scores)
sol = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

sol['Fatalities'] = test['Fatalities'].round(0).astype(int)

sol['ConfirmedCases'] = test['ConfirmedCases'].round(0).astype(int)

sol.to_csv('submission.csv',index=False)