import numpy as np

import pandas as pd

import plotly.express as px

import os
covid = pd.read_csv("/kaggle/input/fortbrasil/questao1_coronavirus.csv",sep=";")

print("\nDimension of InicialData:",covid.shape)

covid.head()
covid.describe()
covid.isna().mean()
covid[covid.estimated_population_2019.isna()]
covid[covid.estimated_population_2019.isna()].describe()
covid[(covid.estimated_population_2019.isna())& (covid.new_confirmed>0)]["new_confirmed"].plot.hist(title = "Histogram of new_confirmed >0 for population NaN")
state = covid.state.unique()

for state_i in state:

    covid["estimated_population_2019"][covid.state ==state_i]= covid[covid.state ==state_i]["estimated_population_2019"].fillna(

        covid[covid.state ==state_i]["estimated_population_2019"].min())

covid.isna().mean()
print("\n Proportion of negative values in new_confirmed variable in %:",np.mean(covid.new_confirmed<0)*100)

print("\n Proportion of negative values in new_deaths variable in %:",np.mean(covid.new_deaths<0)*100)
covid.new_confirmed[covid.new_confirmed<0] = 0

covid.new_deaths[covid.new_deaths<0] = 0
BrasilCum = covid.groupby("date").sum().cumsum().reset_index()

BrasilCum["rateDeath"] = BrasilCum["new_deaths"]/BrasilCum["new_confirmed"]*100

populationBrasil = covid.groupby(["state","city"])["estimated_population_2019"].mean().sum()

print("\n The total population of Brazil in 2019 was:",populationBrasil)

BrasilCum["rateInfected"] = BrasilCum["new_confirmed"]/populationBrasil*100



StateCum = covid.groupby(["date","state"]).sum().groupby('state').cumsum().reset_index().set_index("state").drop("estimated_population_2019",axis=1)

StateCum["rateDeath"] = StateCum["new_deaths"]/StateCum["new_confirmed"]*100

populationByState = covid.groupby(["state","city"])["estimated_population_2019"].mean().groupby("state").sum().to_frame()

StateCum = StateCum.join(populationByState).reset_index()

StateCum["rateInfected"] = StateCum["new_confirmed"]/StateCum["estimated_population_2019"]*100
CumCasesBrasil = px.line(BrasilCum, x="date", y="new_confirmed",title="Cumulative cases of covid in Brazil",

                        labels={'new_confirmed': 'Cumulative cases'})

CumCasesBrasil.show()

CumCasesState = px.line(StateCum, x="date", y="new_confirmed",color = "state" ,title="Cumulative cases of covid in states of Brazil ",

                        labels={'new_confirmed': 'Cumulative cases'})

CumCasesState.show()
CumDeathBrasil = px.line(BrasilCum, x="date", y="new_deaths",title="Cumulative deaths of covid in Brazil",

                        labels={'new_deaths': 'Cumulative death'})

CumDeathBrasil.show()

CumDeathState = px.line(StateCum, x="date", y="new_deaths",color = "state" ,title="Cumulative deaths of covid in states of Brazil ",

                        labels={'new_deaths': 'Cumulative death'})

CumDeathState.show()
DeathRateBrasil = px.line(BrasilCum, x="date", y="rateDeath",title="Rate death of covid in Brazil",

                        labels={'rateDeath': 'Rate death %'})

DeathRateBrasil.show()

DeathRateState = px.line(StateCum, x="date", y="rateDeath",color = "state" ,title="Rate death of covid in states of Brazil ",

                        labels={'rateDeath': 'Rate death %'})

DeathRateState.show()
InfectedRateBrasil = px.line(BrasilCum, x="date", y="rateInfected",title="% of Infected of covid in Brazil",

                        labels={'rateInfected': 'Infected %'})

InfectedRateBrasil.show()

InfectedRateState = px.line(StateCum, x="date", y="rateInfected",color = "state" ,title="% of Infected of covid in states of Brazil ",

                        labels={'rateInfected': 'Infected %'})

InfectedRateState.show()