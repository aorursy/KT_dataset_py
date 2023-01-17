# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/seattleWeather_1948-2017.csv')
df.describe()
def process_hours(df):
    cut_points = [29,39,49,59,69,1000]
    label_names = ["30-39","40-49","50-59","60-69",'70+']
    df["TMAX categories"] = pd.cut(df["TMAX"],
                                             cut_points,labels=label_names)
    return df
data = process_hours(df)
data
new_df = data[['RAIN', 'TMAX categories']]
new_df
contingency_table = pd.crosstab(
    new_df['RAIN'],
    new_df['TMAX categories'], margins = True
)
contingency_table
F_observed = contingency_table.iloc[0][0:4].values
T_observed = contingency_table.iloc[1][0:4].values
barWidth = .25
r1 = np.arange(len(list(F_observed)))
r2 = [x + barWidth for x in r1]

plt.bar(r1, F_observed, color='#48D1CC', width=barWidth, edgecolor='white', label='True')
plt.bar(r2, T_observed, color='#ff4500', width=barWidth, edgecolor='white', label='False')

plt.xlabel('Does it rain ?', fontweight = 'bold')
plt.xticks([r + barWidth for r in range(len(F_observed))], ['30-39', '40-49', '50-59', '60-69', '70+'])

plt.legend()
plt.show

new_cool = np.array([contingency_table.iloc[0][0:5].values,
                  contingency_table.iloc[1][0:5].values])
new_cool

from scipy import stats
chi = stats.chi2_contingency(new_cool)[0:3]
print('Chi-Square Statistics is ' + str(chi[0]))
print('The degrees of freedom is ' + str(chi[2]))
print('Assuming that the maximum weather per day is not associated whether it rains \n The probability of getting this data would be ' + str(chi[1]))
