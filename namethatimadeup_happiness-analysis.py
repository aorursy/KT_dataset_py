# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../input/2017.csv")
data = data.rename(index=str, columns={"Happiness.Score": "happyscore", "Economy..GDP.per.Capita.": "GDP", "Trust..Government.Corruption.": "GovtTrust","Health..Life.Expectancy.":"LifeExpectancy"})
%pylab inline
pylab.rcParams['figure.figsize'] = (12, 8)
plt.plot(data.happyscore, data.GDP)
plt.plot(data.happyscore, data.GovtTrust * 2)
plt.plot(data.happyscore, data.LifeExpectancy)
plt.xlabel("Happiness")
plt.ylabel("GDP Per Capita, Trust in Govt, & Life Expectancy")
plt.legend(["GDP", "Government Trust", "Life Expectancy"])
plt.figure(figsize=(20,20))
plt.show()
data

