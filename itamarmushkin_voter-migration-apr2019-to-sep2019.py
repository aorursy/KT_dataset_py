# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import cvxpy as cvx

import matplotlib.pyplot as plt



from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os

import cvxpy as cvx

from matplotlib import pyplot as plt



DATA_PATH = '/kaggle/input/israeli-elections-2015-2013/'

df_apr_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per settlement 2019a.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()

df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per settlement 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()



df_apr = df_apr_raw.drop(df_apr_raw.index.difference(df_sep_raw.index))

df_sep = df_sep_raw.drop(df_sep_raw.index.difference(df_apr_raw.index))

df_sep.drop('סמל ועדה', axis=1, inplace=True) # new column added in Sep 2019



df_sep = df_sep[df_sep.columns[5:]] # removing "metadata" columns

df_apr = df_apr[df_apr.columns[5:]]



print("{} votes in April 2019, vs {} in September".format(df_apr_raw['כשרים']['מעטפות חיצוניות'], 

                                                                df_sep_raw['כשרים']['מעטפות חיצוניות']))
def party_votes(df, year):

    return df.sum().div(df.sum().sum()).sort_values(ascending=False).rename('party votes '+str(year))



parties_apr, parties_sep = party_votes(df_apr, 'Apr'), party_votes(df_sep, 'Sep')
thresh = 0.01

def threshold_parties(series, threshold):

    s = series[series>threshold]

    year = s.name.split(' ')[-1]

    print("in {}, the following {} parties (out of {}) got {:.3} of votes: \n{}\n".format(year, len(s), len(series), 

                                                                                sum(s), s.index.values))

    return s



parties_apr, parties_sep = threshold_parties(parties_apr, thresh), threshold_parties(parties_sep, thresh)





df_apr.loc[:,'אחר'] = df_apr[df_apr.columns.difference(parties_apr.index)].sum(axis=1)

df_apr = df_apr[parties_apr.index.append(pd.Index(['אחר']))]

df_sep.loc[:,'אחר'] = df_sep[df_sep.columns.difference(parties_sep.index)].sum(axis=1)

df_sep = df_sep[parties_sep.index.append(pd.Index(['אחר']))]
df_apr.loc[:,'לא הצביעו'] = (df_sep.sum(axis=1)-df_apr.sum(axis=1)).clip(lower=0) # "added" voters (new voters + increased turnout)

df_sep.loc[:,'לא הצביעו'] = (df_apr.sum(axis=1)-df_sep.sum(axis=1)).clip(lower=0) # "removed" voteres (decreased turnout)



parties_apr, parties_sep = party_votes(df_apr, 'Apr'), party_votes(df_sep, 'Sep')

parties_apr.to_frame('April').join(parties_sep.rename('September'), how='outer').round(2)
df_apr.head()
df_sep.head()
coefficients = cvx.Variable(shape=(df_apr.shape[1], df_sep.shape[1]))

constraints=[0<=coefficients, coefficients<=1, cvx.sum(coefficients,axis=1)==1]



mse = cvx.sum_squares(df_apr.values*coefficients-df_sep.values) 

mse = mse / df_apr.shape[0]

objective=cvx.Minimize(mse)

prob=cvx.Problem(objective, constraints)



mse = prob.solve(verbose = True, solver='OSQP')

coeff_mat = coefficients.value
naive_estimation = df_sep.apply(lambda x: x.sum() * parties_sep, axis=1)

naive_mse = naive_estimation.subtract(df_sep).apply(np.square).sum(axis=1).mean()

print("the R2 score of our solution (calculated manually) is {:.3}".format(1 - mse / naive_mse))
def display_df(_df): # TThanks, Dean Langsam!

    _display_df = _df.join(pd.DataFrame(columns=_df.index.difference(_df.columns)))

    _display_df = _display_df.append(pd.DataFrame(index=_display_df.columns.difference(_display_df.index), data=0, columns=_display_df.columns)).fillna(0)

    _display_df = _display_df.sort_index(axis=0).sort_index(axis=1).T.round(1)

    return _display_df.style.background_gradient(cmap=plt.get_cmap('Accent_r'))
transfer_matrix = pd.DataFrame(data=coefficients.value, 

                               index=df_apr.columns, columns = df_sep.columns).applymap(lambda x: round(x,3))

display_df(transfer_matrix)
vote_transfers = (transfer_matrix.T * parties_apr * 120).sort_index(axis=1).T

display_df(vote_transfers)
transfer_threshold=1

links=np.where(vote_transfers > transfer_threshold)



# labels_english = ['other', 'Avoda_15', 'Joint', 'Lapid','Kahlon_15', 'Bait',

#                   'Shas_15', 'Liberman_15', 'Gimel_15' ,'Meretz_15', 'Yachad', 'other_15', 'no_15',

#                  'Likud_19', 'Kaholavan', 'Shas_19', 'Gimel_19', 'Hadash', 'Avoda_19', 'Liberman_19',

#                   'UYamin', 'Meretz_19', 'Kahlon_19', 'Raam-Balad', 'NYamin','Zehut','other_19','no_19']



labels_hebrew = vote_transfers.index.to_list()+vote_transfers.columns.to_list()

labels_hebrew = [x[::-1] for x in labels_hebrew]



data = dict(

    type='sankey',

    node = dict(pad = 15, 

                thickness = 20, 

                line = dict(color = "black",width = 0.5),

                color='black',

                label=labels_hebrew),

    link = dict(source=links[0],

                target=links[1]+max(links[0])+1,

                value=[vote_transfers.values[f[0],f[1]] for f in zip(links[0],links[1])]),

    orientation = 'h'

)



layout =  dict(

    title = "Shift in votes between parties, from Apr 2019 to Sep 2019 elections",

    font = dict(size = 14)

)



fig = dict(data=[data], layout=layout)

iplot(fig,validate=False)