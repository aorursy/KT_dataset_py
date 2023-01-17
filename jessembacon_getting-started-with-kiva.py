from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
import random
def makeDf(file):
    return file[:-4], pd.read_csv(os.path.join('../input',file))
dataFiles = {}
for file in [f for f in os.listdir('../input') if f.endswith('csv')]:
    f_name, df = makeDf(file)
    dataFiles[f_name] = df
dataFiles.keys()
dataFiles['kiva_loans'].head(3)
dataFiles['loan_theme_ids'].where(dataFiles['loan_theme_ids']['id'] == 653068).dropna()
kiva_startup_loans = dataFiles['loan_theme_ids'].where(dataFiles['loan_theme_ids']['Loan Theme Type'] == 'Startup').dropna()
theme_type_frequency = Counter(dataFiles['loan_theme_ids']['Loan Theme Type'])
#sorted(theme_type_frequency)
#plt.figure(figsize=(96,12))
plt.figure(figsize=(12,96))
x = range(len(theme_type_frequency))
y = theme_type_frequency.values()
#bar = plt.bar(theme_type_frequency.keys(), y)
bar = plt.barh(x,list(y))
#plt.xticks(rotation=90)
plt.yticks(x, theme_type_frequency.keys())
plt.savefig('Kiva_Theme_Distributions.png', orientation='landscape', transparent=False, )
themes = theme_type_frequency.keys()
dataFiles['kiva_loans'].describe()
dataFiles['kiva_mpi_region_locations'].describe()
dataFiles['loan_theme_ids'].describe()
loan_ids = list(kiva_startup_loans['id'])
len(loan_ids)


loan_sizes = dataFiles['kiva_loans'].groupby(['activity']).mean()[['loan_amount','term_in_months']]
loan_sizes.head(3)
loan_sizes.describe()
average_monthly_loan_payment = loan_sizes['loan_amount'].mean() / loan_sizes.term_in_months.mean()
print(average_monthly_loan_payment)
partners = list(set(dataFiles['kiva_loans']['partner_id'].dropna().astype(int)))
countries = list(set(dataFiles['kiva_loans']['country'].dropna()))
countries = sorted(countries)
df = dataFiles['kiva_loans'].where(dataFiles['kiva_loans']['partner_id'] == partners[2]).dropna()
df
loans_per_partner = {}
for p in partners:
    try:
        df = dataFiles['kiva_loans'].where(dataFiles['kiva_loans']['partner_id'] == p).dropna()
        loans_per_partner[p] = dict(Counter(list(df['country'])))
    except IndexError as e:
        print('Partner Number: {}'.format(p))
        print(e)
loans_per_partner
df = dataFiles['kiva_loans'].where(dataFiles['kiva_loans']['partner_id'] == 462).dropna()
df.where(df['country'] == 'Israel').dropna()
loans_per_partner[462]
country_markers = (list(range(len(countries))))
def returnXY(d):
    x = []
    y = list(d.values())
    for c in d.keys():
        x.append(countries.index(c))
    return(x,y)
plt.figure(figsize=(96,96))
plt.xticks(country_markers, countries, rotation=90)
plt.yticks(partners)
for partner in loans_per_partner:
    try:
        x,y = returnXY(loans_per_partner[partner])
        if len(x) >= 1:
            plt.scatter(x, [partner for x in range(len(x))], y,)
    except:
        print(partner)
plt.grid(True)
plt.show()
