# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



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



US_Accidents_Dec19 = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")
US_Accidents_Dec19.columns
def has_null(df):

    missing = [(c, df[c].isna().mean()*100) for c in df]

    missing = pd.DataFrame(missing, columns=["column_name", "percentage"])

    missing = missing[missing.percentage > 0]

    display(missing.sort_values("percentage", ascending=False))
has_null(US_Accidents_Dec19)
accidents_by_state = US_Accidents_Dec19.groupby('State')['ID'].count().sort_values(ascending=False)
plt.xticks(rotation='vertical')

plt.bar(x = accidents_by_state.index[accidents_by_state.values>100000],height=accidents_by_state.values[accidents_by_state.values>100000])
licensed_drivers = pd.read_excel("../input/licensesdriverbystate/dl1.xlsx",index_col='State')
us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Palau': 'PW',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY',

}





licensed_drivers.rename(index={'New York2':'New York'},inplace=True)

licensed_drivers.rename(index= us_state_abbrev,inplace=True)

licensed_drivers.drop(index='United States, total',inplace=True)
normalized_accident_by_state = accidents_by_state/licensed_drivers['Number of licensed drivers1']
normalized_accident_by_state.dropna(inplace=True)

normalized_accident_by_state.sort_values(ascending=False,inplace=True)
plt.xticks(rotation='vertical')

plt.xlabel('State')

plt.ylabel('Accident per registered driver')

plt.bar(x = normalized_accident_by_state.index[normalized_accident_by_state.values>.01],height=normalized_accident_by_state.values[normalized_accident_by_state.values>.01])


fig, (ax1, ax2) = plt.subplots(1, 2)

fig.tight_layout(pad=3.0)

ylabels = ['Number of accidents per driver','Total Number of Accidents']

for num,ax in enumerate(fig.axes):

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.ylabel(ylabels[num])

    plt.xlabel('State')

#plt.ylabel('Accident per registered driver')

ax1.bar(x = normalized_accident_by_state.head(5).index,height=normalized_accident_by_state.head(5).values)

ax2.bar(x = accidents_by_state.head(5).index,height=accidents_by_state.head(5).values)