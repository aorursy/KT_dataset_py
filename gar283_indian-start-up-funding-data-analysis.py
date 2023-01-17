# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/startup_funding.csv")
# checking data
print(data.head())

print(data.info())
city = data['CityLocation']
print(city.value_counts().head())
city.value_counts().head(10).plot.bar()
print(data['InvestmentType'].value_counts())
data.InvestorsName.value_counts().head(10)
data['AmountInUSD'][data.InvestorsName == 'Ratan Tata']
# converting object to numeric for arthmetic calculations
data["AmountInUSD"] = data["AmountInUSD"].str.replace(",","")
data["AmountInUSD"] = pd.to_numeric(data["AmountInUSD"])

print(data["AmountInUSD"].dropna().sort_values().max())
data[data.AmountInUSD == 1400000000.0]

print(data["AmountInUSD"].dropna().sort_values().min())
data[data.AmountInUSD == 16000.0]
print(data["AmountInUSD"].dropna().sort_values().mean())
#data[data.AmountInUSD == 16000.0]
data