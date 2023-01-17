import pandas as pd 

import numpy as np 

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt 

from IPython.display import Image
corona_dataset_csv = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

corona_dataset_csv.head(10)
corona_dataset_csv.drop(['Lat','Long'],axis=1,inplace=True)
corona_dataset_csv.head(10)
corona_dataset_aggregated = corona_dataset_csv.groupby("Country/Region").sum()
corona_dataset_aggregated.head()
corona_dataset_aggregated.loc['China'].plot()

corona_dataset_aggregated.loc['Italy'].plot()

corona_dataset_aggregated.loc['Turkey'].plot()

plt.legend()
corona_dataset_aggregated.loc['China'].plot()
corona_dataset_aggregated.loc['China'][:3].plot()
names = ["China", "Italy", "Turkey"]

for name in names:

    data_= corona_dataset_aggregated.loc[name].values

    data__ = corona_dataset_aggregated.loc[name].keys()

    data_diff = []

    for i in range(len(data_)-1):

         data_diff.append(abs(data_[i+1]-data_[i]))

        

    d = dict( Daily_Increase = data_diff, Numbers = data__ )



    df= pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))

    df.plot(title=name)

corona_dataset_aggregated.loc['China'].diff().max()
corona_dataset_aggregated.loc['Italy'].diff().max()
corona_dataset_aggregated.loc['Turkey'].diff().max()
countries = list(corona_dataset_aggregated.index)

max_infection_rates = []

for country in countries :

    max_infection_rates.append(corona_dataset_aggregated.loc[country].diff().max())

corona_dataset_aggregated['max infection rate'] = max_infection_rates
corona_dataset_aggregated.head()
corona_data = pd.DataFrame(corona_dataset_aggregated['max infection rate'])
corona_data.head(-10)

#We will use this new dataframe soon.
world_happiness_report = pd.read_csv("../input/world-happiness/2019.csv")

world_happiness_report.head()
columns_to_dropped = ['Overall rank','Score','Generosity','Perceptions of corruption']

world_happiness_report.drop(columns_to_dropped,axis=1 , inplace=True)
world_happiness_report.head()
world_happiness_report.set_index(['Country or region'],inplace=True)

world_happiness_report.head()
#Time to use our infection rates DataFrame

corona_data.head()
world_happiness_report.head()
data = world_happiness_report.join(corona_data).copy()
data.head()
X = data['GDP per capita']

Y = data['max infection rate']



X_bar = np.mean(X) #finding the average for all X values

Y_bar = np.mean(Y) #finding the average for all Y values



top = np.sum((X - X_bar) * (Y - Y_bar))



bot = np.sqrt(np.sum(np.power(X - X_bar, 2)) * np.sum(np.power(Y - Y_bar, 2)))



print("Correlation of GDP per Capita and Max Infection Rate:", top/bot)



X = data['Freedom to make life choices']

Y = data['max infection rate']



X_bar = np.mean(X) #finding the average for all X values

Y_bar = np.mean(Y) #finding the average for all Y values



top = np.sum((X - X_bar) * (Y - Y_bar))



bot = np.sqrt(np.sum(np.power(X - X_bar, 2)) * np.sum(np.power(Y - Y_bar, 2)))



print("Correlation of Freedom to make life choices and Max Infection Rate:", top/bot)
data.corr()
Image(filename='../input/images/img/LSM.png') 
X = data['GDP per capita']

Y = data['max infection rate']



# Mean X and Y

x_bar = np.mean(X)

y_bar = np.mean(Y)



top = np.sum((X - x_bar) * (Y - y_bar))

bot = np.sum(np.power(X - x_bar, 2)) 



m = top / bot

c = y_bar - (m * x_bar)

 

# Print coefficients

print(m, c)
# Plotting Values and Regression Line

max_x = np.max(X) 

min_x = np.min(X) 

# Calculating line values x(independent values) and line(output)

independent = np.linspace(min_x, max_x,)

line = c + m * independent 



plt.plot(independent, line, color='#52b920', label='Regression Line')



# Ploting Scatter Points

plt.scatter(X, Y, c='#ef4423', label='Countries')

 

plt.xlabel('GDP per Capita')

plt.ylabel('Max Infection Rate')

plt.legend()

plt.show()
Image(filename='../input/images/img/RMSD.png') 
pred = []

for i in X.values:

    pred.append(i * m + c)

numer = np.sum(np.power(pred -Y , 2))

denom = len(pred)



RMSD = np.sqrt(numer/denom)

print("\n","root-mean-square-error(RMSE):",RMSD,"\n")
x = data['GDP per capita']

y = data['max infection rate']

sns.regplot(x,np.log(y))
x = data['Social support']

y = data['max infection rate']

sns.regplot(x,np.log(y))
x = data['Healthy life expectancy']

y = data['max infection rate']

sns.regplot(x,np.log(y))
x = data['Freedom to make life choices']

y = data['max infection rate']

sns.regplot(x,np.log(y))

corona_dataset_csv2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

corona_dataset_csv2.drop(['Lat','Long'],axis=1,inplace=True)

corona_dataset_aggregated2 = corona_dataset_csv.groupby("Country/Region").sum()

corona_dataset_aggregated2.loc['China'].plot()

corona_dataset_aggregated2.loc['Italy'].plot()

corona_dataset_aggregated2.loc['Turkey'].plot()

plt.legend() 
countries2 = list(corona_dataset_aggregated.index)

max_death_rates = []

for country in countries2 :

    max_death_rates.append(corona_dataset_aggregated2.loc[country].diff().max())

corona_dataset_aggregated2['Max Death Rate'] = max_death_rates
corona_data2 = pd.DataFrame(corona_dataset_aggregated2['Max Death Rate'])
corona_data2.head()
data2 = world_happiness_report.join(corona_data2).copy()

data2.head()
data2.corr()
x = data2['GDP per capita']

y = data2['Max Death Rate']

sns.regplot(x,np.log(y))
x = data2['Social support']

y = data2['Max Death Rate']

sns.regplot(x,np.log(y))
x3 = data2['Healthy life expectancy']

y3 = data2['Max Death Rate']

sns.regplot(x,np.log(y))
x = data2['Freedom to make life choices']

y = data2['Max Death Rate']

sns.regplot(x,np.log(y))