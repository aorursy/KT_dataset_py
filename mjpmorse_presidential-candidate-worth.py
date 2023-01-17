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

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import matplotlib.cm as cm

import seaborn as sns



import pandas_profiling

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,

                               AutoMinorLocator)



#import reverse_geocoder as rg



import os, sys

import calendar



import warnings

warnings.filterwarnings('ignore')



plt.rc('font', size=18)        

plt.rc('axes', titlesize=22)      

plt.rc('axes', labelsize=18)      

plt.rc('xtick', labelsize=12)     

plt.rc('ytick', labelsize=12)     

plt.rc('legend', fontsize=12)   



plt.rcParams['font.sans-serif'] = ['Verdana']



# function that converts to thousands

# optimizes visual consistence if we plot several graphs on top of each other

def format_1000(value, tick_number):

    return int(value / 1_000)



pd.options.mode.chained_assignment = None

pd.options.display.max_seq_items = 500

pd.options.display.max_rows = 500

pd.set_option('display.float_format', lambda x: '%.5f' % x)
Candidate_wealth = {'Michael Bloomberg':61800000000,'Donald Trump':3100000000,'Elizabeth Warren':12000000,'Joe Biden':9000000,

                    'Bernie Sanders':2500000, 'Amy Klobuchar':2000000,'Pete Buttigieg':100000}

y_ticks = ('$1 Thousand','$10 Thousand','$100 Thousand','$1 Million','$10 Million','$100 Million','$1 Billion','$10 Billion','$100 Billion')



sns.set()

plt.tight_layout()

fig, ax = plt.subplots()

ax.bar(Candidate_wealth.keys(),Candidate_wealth.values(),color = list('rgbymc'),log=True,align='edge')

plt.xticks(rotation=45) 

locs, labels = plt.yticks()

plt.yticks(locs,y_ticks)

plt.title('2020 Presidential Candidates by Net Worth, Log Plot')

plt.show()

plt.tight_layout()

#plt.savefig('presidential_worth',dpi=300)
