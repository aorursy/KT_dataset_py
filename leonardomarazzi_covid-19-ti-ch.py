# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

file_path = "../input/sarscov2-ch-ti/cases.csv"

covid_data =pd.read_csv(file_path, index_col=0)

print("Setup Complete")
covid_data.tail()
sns.set_style("whitegrid")

plt.figure(figsize=(25,6))

plt.title("totale casi in Ticino")

plt.xlabel("giorni")

plt.ylabel("Totale casi")

sns.lineplot(y =covid_data.confirmed_cases[1:] ,x = covid_data.index[1:len(covid_data)])
print("la percentuale di nuovi casi di oggi è: ", 100*(covid_data['confirmed_cases'][len(covid_data)-1]-covid_data['confirmed_cases'][len(covid_data)-2])/covid_data['confirmed_cases'][len(covid_data)-1], "%")
