# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
fifa = pd.read_csv('../input/fifa_ranking.csv')
fifa.head()
fifa.info()
def country_ranking_over_yrs(country1,country2):

    fifa_country = fifa[(fifa.country_full==country1) | (fifa.country_full==country2)]
    fifa_country['rank_date'] = pd.to_datetime(fifa_country['rank_date'])
    fifa_country['rank_year'] = fifa_country['rank_date'].dt.year
    fig,ax = pyplot.subplots(figsize=(8,5))
    sns.lineplot(x='rank_year',y='rank',data=fifa_country,hue='country_full',ax=ax)    
    #return fifa_country.head()
    
country_ranking_over_yrs('Brazil','Argentina')

