import numpy as np 

import pandas as pd 

from pathlib import Path



import cufflinks as cf



## (from https://www.kaggle.com/hotessy/sir-solver-hyperopt/)

from sir_solver_hyperopt import Learner
pd.set_option('display.max_rows', 500)

pd.set_option('use_inf_as_na', True)

cf.set_config_file(offline=True, theme='solar');
path = Path("../input/novel-corona-virus-2019-dataset/")
recovered_df = (pd.read_csv(path/'time_series_covid_19_recovered.csv')

                .drop(columns=['Lat', 'Long'])

                .groupby('Country/Region')

                .sum())



deaths_df = (pd.read_csv(path/'time_series_covid_19_deaths.csv')

             .drop(columns=['Lat', 'Long'])

             .groupby('Country/Region')

             .sum())



confirmed_df = (pd.read_csv(path/'time_series_covid_19_confirmed.csv')

                .drop(columns=['Lat', 'Long'])

                .groupby('Country/Region')

                .sum())
class MyLearner(Learner):

    

    def __init__(self, country):

        self.first_case_date = (confirmed_df.filter(items=[country], axis=0).iloc[0] > 0).idxmax()

        super().__init__(country)

    

    def load_confirmed(self, country):

        return confirmed_df.filter(items=[country], axis=0).iloc[0][self.first_case_date:]



    def load_recovered(self, country):

        return recovered_df.filter(items=[country], axis=0).iloc[0][self.first_case_date:]



    def load_dead(self, country):

        return deaths_df.filter(items=[country], axis=0).iloc[0][self.first_case_date:]
learner = MyLearner('India')
pd.DataFrame(data=[learner.infected, learner.recovered, learner.dead]).T.head()
first_case_date = (learner.infected > 0).idxmax()

first_recovery_date = (learner.recovered > 0).idxmax()



i_0 = learner.infected[first_case_date:].rolling(14).max().dropna()[0]

r_0 = learner.recovered[first_recovery_date:].rolling(14).max().dropna()[0]

d_0 = learner.dead[first_recovery_date:].rolling(14).max().dropna()[0]



print(i_0, r_0 + d_0)
learner.train(s_0=1e4, i_0=1, r_0=1, weight=0.3, max_evals=50)
print(learner.country.upper())

print(f"Γ = {learner.Γ}")

print(f"β (standardised) = {learner.β}")

print(f"Reproduction Rate (standardised) = {learner.β/learner.Γ}")
fig, data = learner.plot()
data.to_csv(f'{learner.country}.csv')
fig.show()