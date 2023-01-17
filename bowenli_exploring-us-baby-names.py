%%sh
# location of data files
ls /kaggle/input
# imports
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", message="axes.color_cycle is deprecated")
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sqlite3
# explore sqlite contents
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
# helper method to load the data
def load(what='NationalNames'):
    assert what in ('NationalNames', 'StateNames')
    cols = ['Name', 'Year', 'Gender', 'Count']
    if what == 'StateNames':
        cols.append('State')
    df = pd.read_sql_query("SELECT {} from {}".format(','.join(cols), what),
                           con)
    return df
df = load(what='NationalNames')
df.head(5)
# top ten names across full history
glob_freq = (df.groupby('Name')
             .agg({'Count': 'sum'})  #, 'Year': ['min', 'max']})
             .sort_values('Count', ascending=False))
#glob_freq[['Count']].head(10).plot(kind='bar')

glob_freq.head(10).plot(kind='bar')
glob_freq[['Count']].head(10).plot(kind='bar')
# A random sample of unpopular names
glob_freq.query('Count <= 10').sample(10, random_state=2)
# visualize post WW2 baby boom:
population = df[['Year', 'Count']].groupby('Year').sum()
population.plot()
# "Jackie" peaks during Kennedy presidency (thanks to the very popular first lady, Jackie Kennedy)
df.query('Name=="Jackie"')[['Year', 'Count']].groupby('Year').sum().plot()
# Are more male babies born?
tmp = df.groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'")
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'")
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join['Male Excess'] = join['Male'] - join['Female']
join.plot()
# Common names that are shared between girls and boys
tmp = df.groupby(['Gender', 'Name']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'")
        .set_index("Name")
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'")
          .set_index("Name")
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='inner')
join['Frequency'] = join['Male'] + join['Female']
join['FemalePct'] = join['Female'] / join['Frequency'] * 100.0
join['MalePct'] = join['Male'] / join['Frequency'] * 100.0
(join[['Frequency', 'FemalePct', 'MalePct']]
 .query('(FemalePct > 10) & (MalePct) > 10')
 .sort_values('Frequency', ascending=False)
 .head(10))
df2 = load(what='StateNames')
df2.head(5)
# Evolution of baby births in the 10 largest states
tmp = df2.groupby(['Year', 'State']).agg({'Count': 'sum'}).reset_index()
largest_states = (tmp.groupby('State')
                  .agg({'Count': 'sum'})
                  .sort_values('Count', ascending=False)
                  .index[:10].tolist())
tmp.pivot(index='Year', columns='State', values='Count')[largest_states].plot()
# in what states do people give similar names to their babies?

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def most_similar_states(state, year=2014, top=3):
    """
    Returns the list of states where babies have the most similar names to a given state

    Details
    ============
    The state-by-state data is first converted to an TF-IDF matrix:
       https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    Then, cosine similarity is computed betweent the given state 
    and every other state:
       https://en.wikipedia.org/wiki/Cosine_similarity
    Finally, the list of distances is sorted, and the states with the
    largest similarity are returned along with their cosine similarity metric.   

    Arguments
    =============
    state      :   input state that will be compared with all other states
    year       :   (2014) use baby names from this year
    top        :   (3) return this many most-similar states
    
    """
    # compute a matrix where rows = states and columns = unique baby names
    features = pd.pivot_table(df2.query('Year=={}'.format(year)),
                              values='Count', index='State', columns='Name',
                              aggfunc=np.sum)
    all_states = features.index.tolist()
    if state not in all_states:
        raise ValueError('Unknown state: {}'.format(state))
    idx = all_states.index(state)
    # compute a TF-IDF matrix
    model = TfidfTransformer(use_idf=True, smooth_idf=True, norm='l2')
    tf = model.fit_transform(features.fillna(0))
    sims = pd.DataFrame.from_records(cosine_similarity(tf[idx:idx+1], tf),
                                     columns=all_states).T
    return (sims
            .rename(columns={0: 'similarity'})
            .sort_values('similarity', ascending=False)
            .head(top+1).tail(top))   # remove first entry, since self-similarity is always 1.0
# states where baby names in 2014 are most similar to those in Texas:
most_similar_states(state='CA', year=2014, top=4)
# we get: California, Arizona, Nevada, and Florida
# states where baby names in 2010 are most similar to those in Massachusetts:
most_similar_states(state='MA', year=2010, top=4)
# we get: Connecticut, Pennsylvania, Virginia, and Rhose Island
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