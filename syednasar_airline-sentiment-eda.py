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
# read the data



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

from matplotlib import gridspec

%matplotlib inline

matplotlib.rcParams.update({'font.size': 12})
from sklearn import preprocessing

import brewer2mpl
in_file_train = '../input/Tweets.csv'



print("Loading data...\n")

data = pd.read_csv(in_file_train)

data.head()
data.airline_sentiment.unique()
data.count()
data.groupby("airline_sentiment").size()
set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors



font = {'family' : 'sans-serif',

        'color'  : 'teal',

        'weight' : 'bold',

        'size'   : 18,

        }

plt.rc('font',family='serif')

plt.rc('font', size=16)

plt.rc('font', weight='bold')

#plt.style.use('seaborn-poster')

#plt.style.use('bmh')

#plt.style.use('ggplot')

plt.style.use('seaborn-dark-palette')

#plt.style.use('presentation')

print (plt.style.available)



# Get current size

fig_size = plt.rcParams["figure.figsize"]

 

# Set figure width to 6 and height to 6

fig_size[0] = 6

fig_size[1] = 6

plt.rcParams["figure.figsize"] = fig_size
#array(['neutral', 'positive', 'negative'], dtype=object)



neutral = sum(data.loc[:, 'airline_sentiment'] == 'neutral')

positive = sum(data.loc[:, 'airline_sentiment'] == 'positive')

negative = sum(data.loc[:, 'airline_sentiment'] == 'negative')



#functional , non_functional = sum(df2.loc[:, 'OutcomeType'] == 0), sum(df2.loc[:, 'status_group'] == 1)

print(negative, neutral, positive)
from matplotlib import rcParams

rcParams['font.size'] = 12

#print (rcParams.keys())

rcParams['text.color'] = 'black'



piechart = plt.pie(

    (negative, neutral, positive),

    labels=('negative', 'neutral', 'positive'),

    shadow=False,

    colors=('teal', 'crimson', 'cyan'),

    explode=(0.08,0.08,0.08), # space between slices 

    startangle=90,    # rotate conter-clockwise by 90 degrees

    autopct='%1.1f%%',# display fraction as percentages

)



plt.axis('equal')   

plt.title("Twitter Outcomes", y=1.08,fontdict=font)

plt.tight_layout()

plt.savefig('Outcomes-train.png', bbox_inches='tight')