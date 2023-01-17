## Import Statements

import pandas as pd                  # DataFrame for analysis
import numpy as np                   # for numerical precision in quantitative operations
import seaborn as sns                # for more stylish plotting
import matplotlib.pyplot as plt      # for basic plotting
from scipy import stats              #for linear regression (least-squares fit)
import datetime as dt                # for datetime objects

sns.set_style("darkgrid",
              {'axes.grid' : False})# setting a dark grid style for all visualizations
                                    # removing annoying grid lines

# for visualizations to appear in this notebook
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Ignoring warnings.
import warnings
warnings.filterwarnings("ignore")
# Loading Data into DataFrame
global_attacks = pd.read_excel("../input/GSAF5.xls")
global_attacks.head()
# Removing whitespace from column (header) names
global_attacks.columns = global_attacks.columns.str.strip()
global_attacks.columns
useful_columns = ['Date', 'Year', 'Type', 'Type', 'Country', 'Area', 'Location', 'Activity',
                  'Sex', 'Age', 'Fatal (Y/N)', 'Time', 'Species']

new_shark_data = global_attacks[useful_columns]
new_shark_data.head()
new_shark_data['Count'] = 1
attacks_per_country = new_shark_data.groupby("Country").count()
attacks_per_country = attacks_per_country.sort_values(by ='Count', ascending=False)['Count']
#plt.barh(attacks_per_country.index[:10], attacks_per_country[:10])

attacks_per_country[:10].plot(kind="barh", figsize=(10,5))

plt.title("Top 10 Countries with Shark Attacks\n 1960-2017", size=16)
plt.ylabel("Country", size=15)
plt.xlabel("Total Shark Attacks", size=15)
plt.show()
