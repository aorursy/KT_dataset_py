import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import os
data = pd.read_csv("/kaggle/input/air-pollution-at-so-paulo-brazil-since-2013/cetesb.csv/cetesb.csv",

                   parse_dates=['time'])
x = data.groupby([data.time.dt.year, data.time.dt.month]).MP10.mean().unstack()

months = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]



plt.figure(figsize=(12, 4))

sns.heatmap(x, annot=True, xticklabels=months, cmap='plasma')

plt.title("Particulado mensal médio em São Paulo (PM10), em $\mu/m^3$")

plt.xlabel("mês")

plt.ylabel("ano")

plt.show()
x = data.groupby([data.time.dt.dayofweek, data.time.dt.month]).MP10.mean().unstack()

months = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

dayofweeks = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']

plt.figure(figsize=(12, 4))

sns.heatmap(x, annot=True, xticklabels=months, yticklabels=dayofweeks, cmap='plasma')

plt.title("Particulado por dia de semana em São Paulo (PM10), em $\mu/m^3$")

plt.xlabel("mês")

plt.ylabel("dia da semana")

plt.show()
x = data.groupby([data.time.dt.dayofweek, data.time.dt.hour]).MP10.mean().unstack()

#months = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

dayofweeks = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']

plt.figure(figsize=(12, 4))

sns.heatmap(x, annot=True, yticklabels=dayofweeks, cmap='plasma')

plt.title("Particulado por dia de semana em São Paulo (PM10), em $\mu/m^3$")

plt.xlabel("hora do dia")

plt.ylabel("dia da semana")

plt.show()