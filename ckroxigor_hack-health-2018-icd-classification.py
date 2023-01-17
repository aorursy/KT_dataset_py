# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# References: https://www.kaggle.com/berhag/co2-emission-forecast-with-python-seasonal-arima
# References: https://github.com/statsmodels/statsmodels/issues/4465

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns
import editdistance

sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

file_name = os.listdir("../input/home-medical-visits-healthcare/")[0]
df = pd.read_csv('../input/home-medical-visits-healthcare/Hack2018_ES.csv', parse_dates=[9])


print("The dataset has {} rows with {} features each".format(len(df), len(df.columns)))
df.head()
def conversion(text):
    if text.startswith("amigdalitis aguda"):
        return "amigdalitis aguda"
    return text

def remove(text):
    for w in ['anonymxxx']:
        text = text.replace(w, ' ')
    return text

def clean_patologia(text):
    text = text.lower()
    text = remove(text)
    text = text.split(" con ")[0]
    text = text.split(" -")[0]
    text = text.split("llamar")[0]
    text = text.split("pc smc")[0]
    text = text.strip()
    text = conversion(text)
    text = ' '.join([t for t in text.split(' ') if len(t) > 0])
    return text

print("Before: {}".format(len(df.patologia.unique())))

cleaned = df.patologia.apply(clean_patologia)
print("After: {}".format(len(cleaned.unique())))
', '.join(cleaned.unique().tolist())
patologies = pd.read_csv('../input/classes-patologia/patologias.csv')
mapping = dict()
for cat, row in patologies.transpose().iterrows():
    row = row[pd.notnull(row)].values
    for value in row:
        mapping[value.lower()] = cat
def map_to_classes(name):
    if name in mapping:
        return mapping[name]
    for key in mapping:
        if editdistance.eval(name, key) < 3:
            return mapping[key]
    return np.nan

cleaned.apply(map_to_classes).value_counts()

df['cleaned'] = cleaned
df['E_class'] = cleaned.apply(map_to_classes)
print("{} unclassigied pathologies".format(len(df[pd.isnull(df.E_class)])))
df[pd.isnull(df.E_class)].cleaned.value_counts()
df.loc[pd.isnull(df.E_class),'E_class'] = 'OTROS'
df.to_csv('Hack2018_ES_cleaned.csv', index=False)
plt.figure(figsize=(12,5))
fig = sns.countplot(df.E_class, order = df.E_class.value_counts().index)
fig.set_title("#Visits per pathology category")
fig.set_xlabel("Pathology")
plt.xticks(rotation=90)
plt.show()