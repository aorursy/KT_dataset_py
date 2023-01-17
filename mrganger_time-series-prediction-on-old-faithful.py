import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

np.warnings.filterwarnings('ignore')



df = pd.read_csv('../input/faithful.csv', index_col=[0])

df.head()
sns.distplot(df.waiting, bins=10);
from scipy.stats import kendalltau

def offset(k):

    return df.waiting.iloc[:-k], df.waiting.iloc[k:]



def kendalltau_df(offsets, pval):

    pvals = [kendalltau(*offset(i)).pvalue for i in offsets]

    df = pd.DataFrame({'Offset (cycles)': offsets})

    df1,df2 = df.copy(),df.copy()

    df1['p-value'] = pvals

    df1[''] = 'Kendall Tau'

    df2['p-value'] = 0.05

    df2[''] = 'p = {}'.format(pval)

    return pd.concat([df1,df2])



    

offsets = np.arange(1,10)

plt.figure(figsize=[12, 6])

sns.lineplot(data=kendalltau_df(offsets, 0.05), x='Offset (cycles)', y='p-value', hue='')

plt.xlabel('Offset (Cycles)')

plt.ylabel('p-value');

plt.title('Kendall Tau test for independence of waiting times offset by 1 or more cycles.');
code = (df.waiting > 67).values.astype(int)

print(''.join(code.astype(str)))
def code_set(code, n):

    return pd.Series(list(code[(np.arange(len(code)-n)[:,None] + np.arange(n)).ravel()].reshape(-1,n).astype(str))).str.join('')



def entropy(code_set):

    from scipy.stats import entropy as h

    return h(code_set.value_counts().values, base=2)



def entropy_df(title, *codes):

    offsets = np.arange(1,10)

    ents = np.array([[entropy(code_set(c, i))/i for i in offsets] for c in codes])

    df = pd.DataFrame({'Sequence Length': offsets[None,:][[0]*len(codes)].flatten(), 'Normalized Entropy (bits)':ents.flatten()})

    df['Sequence'] = title

    return df

    

def fake_code(code):

    from scipy.stats import bernoulli as ber

    return ber.rvs(code.mean(), size=len(code))



plt.figure(figsize=[12,6])

edf = pd.concat([entropy_df('Wait Times', code), entropy_df('Random', *[fake_code(code) for _ in range(15)])])

sns.lineplot(data=edf, x='Sequence Length', y='Normalized Entropy (bits)', hue='Sequence', ci='sd')

plt.title('Entropy of subsequences of discretized waiting times for Old Faithful, compared with a random sequence.');
code_set(code, 4).value_counts().to_frame().rename({0:'count'}, axis=1)
def marginals(code, n):

    vc = code_set(code, n).value_counts().to_frame().reset_index()

    vc['prefix'], vc['suffix'] = vc['index'].str.slice(0,3), vc['index'].str.slice(3,4)

    vc = vc.set_index(['prefix', 'suffix'])[0].unstack('suffix').fillna(0)

    vc['prob'] = vc.sum(axis=1)

    vc[['0','1']] = vc[['0','1']].div(vc.prob, axis=0)

    vc.prob /= vc.prob.sum()

    return vc.sort_values('prob', ascending=False)



def accuracy(marg):

    return (marg[['0','1']].max(axis=1)*marg.prob).sum()

    

m = marginals(code, 4)

print("Accuracy: {:0.2f}".format(accuracy(m)))

print("Random Guessing: {:0.2f} ± {:0.2f}".format(code.mean(), code.std()/np.sqrt(len(code))))

m
from sklearn.linear_model import LinearRegression



wait = df.waiting.values

times = wait[np.arange(len(wait)-4)[:,None] + np.arange(4)].reshape(-1,4)

X = times[:,:3]

y = times[:,3]



reg = LinearRegression()

reg.fit(X,y)

qs = [0,0.1,0.33,0.5,0.66,0.9,1.0]

te = pd.DataFrame(np.abs(reg.predict(X)-y), columns=['true_error']).quantile(qs)

rd = pd.DataFrame(np.abs((wait[:,None]-wait).flatten()), columns=['rand_error']).quantile(qs)

pd.concat([te,rd], axis=1)