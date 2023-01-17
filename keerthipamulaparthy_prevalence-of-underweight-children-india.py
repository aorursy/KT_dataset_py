# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

import seaborn as sns
sns.set(style="whitegrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Indicators.csv")

df_india_underweightchildren = df[(df.CountryName=='India')&(df.IndicatorCode=='SH.STA.MALN.ZS')]
fig = plt.figure()
plt.plot(df_india_underweightchildren.Year, df_india_underweightchildren.Value, 'o-', color='y')
plt.xlabel('Years')
plt.ylabel('Percentage of children under 5')
plt.title('Prevalence of underweight children, under 5 years of age in India')
fig.savefig('underweightchildren.png')



