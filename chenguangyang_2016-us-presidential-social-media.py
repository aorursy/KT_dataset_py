# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Any results you write to the current directory are saved as output



import numpy

import pandas

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
def get_data(first, last):

    """ get data for person named first, last"""

    df1 = pandas.read_csv('../input/{}.csv'.format(first), low_memory=False,

                 header=0, sep=',', encoding = "ISO-8859-1", parse_dates = ['timestamp'],

                     usecols=['timestamp','network','author','likes'])

    df2 = pandas.read_csv('../input/{}.csv'.format(last), low_memory=False,

                 header=0, sep=',', encoding = "ISO-8859-1", parse_dates = ['timestamp'],

                     usecols=['timestamp','network','author','likes'])

    return pandas.concat([df1, df2]).drop_duplicates()
donald = get_data("Donald", "Trump")

hillary = get_data("Hillary", "Clinton")

donald.info()

print("-"*80)

hillary.info()
donald.head()
donald = donald[donald['likes'] > 0]

hillary = hillary[hillary['likes'] > 0]

donald.groupby('network')['likes'].describe()

hillary.groupby('network')['likes'].describe()
hillary['year'] = hillary['timestamp'].apply(lambda t:t.year)

hillary['month'] = hillary['timestamp'].apply(lambda t:t.month)

hillary_agg = hillary.groupby([hillary['year'], hillary['month']])['likes'].sum()
donald['year'] = donald['timestamp'].apply(lambda t:t.year)

donald['month'] = donald['timestamp'].apply(lambda t:t.month)

donald_agg = donald.groupby([donald['year'], donald['month']])['likes'].sum()
frames = [hillary_agg, donald_agg]

data = pandas.concat(frames, axis = 1)
data.columns = ['Hillary', 'Trump']

data.plot(figsize = [9,6])
donald_network=donald.groupby([donald['network']])['likes'].sum()

hillary_network=hillary.groupby([hillary['network']])['likes'].sum()
network_data = pandas.concat([donald_network, hillary_network], axis=1)

network_data.columns=['Trump','Hillary']
network_data.head()
network_data.plot(kind='bar')
donald_by_author = donald.groupby(['author']).likes.sum().sort_values(ascending=False)
donald_by_author.head(10)
hillary_by_author = hillary.groupby('author').likes.sum().sort_values(ascending=False)
hillary_by_author.head(10)