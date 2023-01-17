# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import warnings
import string
import time
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15
data_2015 = pd.read_csv('../input/2015.csv')
data_2016 = pd.read_csv('../input/2016.csv')
data_2017 = pd.read_csv('../input/2017.csv')
data_2017.head()
sns.distplot(data_2015['Happiness Score'], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.ylabel("Percentile")
plt.title("Happiness Score 2015")
plt.show()
sns.distplot(data_2016['Happiness Score'], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.ylabel("Percentile")
plt.title("Happiness Score 2016")
plt.show()
sns.distplot(data_2017['Happiness.Score'], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.ylabel("Percentile")
plt.title("Happiness Score 2017")
plt.show()
countries = ['Switzerland', 'Iceland', 'Denmark', 'Norway']

for country in countries:
    happiness_score = [data_2015[data_2015['Country'] == country]['Happiness Score'].values,
                      data_2016[data_2016['Country'] == country]['Happiness Score'].values,
                      data_2017[data_2017['Country'] == country]['Happiness.Score'].values]
    #print(happiness_score)
    years = [2015,2016,2017]
    
    ax = sns.pointplot(x=years, y=happiness_score)
    plt.xlabel("Years")
    plt.ylabel("Happiness Score")
    plt.title("Happiness Trend of " + country)
    plt.show()
## Countries whose happiness score increased/decersed during 3 year span

countries = data_2017['Country'].values
countries_with_increasing_happiness = []
countries_with_decreasing_happiness = []

for country in countries:
    happiness_score = [data_2015[data_2015['Country'] == country]['Happiness Score'].values,
                      data_2016[data_2016['Country'] == country]['Happiness Score'].values,
                      data_2017[data_2017['Country'] == country]['Happiness.Score'].values]
    
    if ((happiness_score[0] > happiness_score[1]) and (happiness_score[1] > happiness_score[2])):
        countries_with_decreasing_happiness.append(country)
        
    if ((happiness_score[0] < happiness_score[1]) and (happiness_score[1] < happiness_score[2])):
        countries_with_increasing_happiness.append(country)
    
print("Countries with Increasing happiness score are :\n" + str(len(countries_with_increasing_happiness)))
print(countries_with_increasing_happiness)
print("Countries with decreasing happiness score are :\n" + str(len(countries_with_decreasing_happiness)))
print(countries_with_decreasing_happiness)
