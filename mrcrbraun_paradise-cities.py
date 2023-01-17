import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

os.listdir("../input/")

dataset = pd.read_csv('../input/Entities.csv', keep_default_na=False)

dataset.loc[:,'country_codes'].describe()

count_status = pd.value_counts(dataset['status'][dataset['status']!='']).sort_values(ascending=False)

count_countries = pd.value_counts(dataset['countries'][dataset['countries']!='']).sort_values(ascending=False)

count_codes = pd.value_counts(dataset['country_codes'][dataset['country_codes']!='']).sort_values(ascending=False)

count_jurisdiction = pd.value_counts(dataset['jurisdiction'][dataset['jurisdiction']!='']).sort_values(ascending=False)

count_jurisdiction_des = pd.value_counts(dataset['jurisdiction_description'][dataset['jurisdiction_description']!='']).sort_values(ascending=False)

count_service_provider = pd.value_counts(dataset['service_provider'][dataset['service_provider']!='']).sort_values(ascending=False)

count_company_type = pd.value_counts(dataset['company_type'][dataset['company_type']!='']).sort_values(ascending=False)

# Quick review of data

count_status[0:10].plot(kind = 'bar')

plt.title("Current Status")

plt.xlabel("Status")

plt.ylabel("# of Occurances")

count_countries[:15].plot(kind = 'bar')

plt.title("Countries")

plt.xlabel("Country")

plt.ylabel("# of Occurances")
count_jurisdiction[:15].plot(kind = 'bar')

plt.title("Jurisdictions")

plt.xlabel("jurisdiction")

plt.ylabel("# of Occurances")
count_service_provider.plot(kind = 'bar')

plt.title("Service Provider")

plt.xlabel("Provider")

plt.ylabel("# of Occurances")

''' Majority of work done using Pygal World Map which is currently unsupported'''