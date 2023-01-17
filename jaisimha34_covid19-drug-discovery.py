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
import pandas as pd 

master_results = pd.read_csv("../input/coronavirus-covid19-drug-discovery/master_results_table.csv")

mers_results = pd.read_csv("../input/coronavirus-covid19-drug-discovery/mers_results.csv")

sars_results = pd.read_csv("../input/coronavirus-covid19-drug-discovery/mers_results.csv")
import seaborn as sns 

master_results.head()
mers_results.head()
sars_results.head(20)
import matplotlib.pyplot as plt 

plt.figure(figsize=(100,50))

sns.countplot(sars_results['Unnamed: 0'])
sns.countplot(sars_results["Connectivity"])
sns.countplot(sars_results['P_value'])
plt.scatter(sars_results['Connectivity'],sars_results['P_value'])
plt.scatter(mers_results['Connectivity'],sars_results['P_value'])