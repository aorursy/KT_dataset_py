# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
# Читаем данные из csv
# Переменная df является типом DataFrame
my_filepath = "../input/ebolaaa/ebola.csv"
df = pd.read_csv(my_filepath)
df.head()
grouped_df = df[df.Indicator=='Cumulative number of confirmed Ebola cases'].groupby('Country').count()['value']
#grouped_df = df[str(df.Indicator)=='Cumulative number of confirmed Ebola cases'].groupby('Country')[['Country']['value']]
grouped_df.head()
grouped_df.plot.hist(by=['value'])
#Создаем csv файл
grouped_df.to_csv("output.csv", index=True)