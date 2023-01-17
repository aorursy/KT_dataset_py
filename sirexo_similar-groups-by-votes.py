import pandas as pd

import matplotlib as mpl

import sklearn.decomposition
mpl.style.use('seaborn-darkgrid')

mpl.rc('font', size=16)
votes = pd.read_csv('../input/votes.csv', parse_dates=['time'], dtype={'question': str})
pca = sklearn.decomposition.PCA(n_components=1, random_state=0)



def reduce_manifold(g):

    f = g.groupby(['group', 'voting_id'])['vote'].mean().unstack().fillna(0)

    return pd.DataFrame(pca.fit_transform(f), index=f.index)





# Lithuanian parliament is reelected every 4 years somewher in October.

terms = votes.set_index('time').resample('4AS-OCT')



for term in terms.groups.keys():

    # Get all the votes from one term.

    frame = terms.get_group(term)

    title = '%s - %s Parliament' % (frame.index.min().year, frame.index.max().year)

    

    # In order to make graphs readable, we reduce number of parliamentary groups to 5 major ones.

    frame = frame[frame['group'].isin(frame['group'].value_counts().index[:5])]

    

    # Group all votes by day, unstack all votings and reduce dimensionality to one using PCA. 

    agg = frame.groupby(frame.index.to_period('D')).apply(reduce_manifold).unstack()

    agg.columns = agg.columns.droplevel(0)

    

    # Reduce noise by smoothing data using 14 days window.

    agg = agg.rolling(14, win_type='hamming', center=True).mean()

    ax = agg.plot(figsize=(16, 6))

    ax.set_title(title)

    ax.set_ylabel('vote similarity')