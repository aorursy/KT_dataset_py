from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import statsmodels.api as sm

import scipy.stats as ss

import itertools
df = pd.read_json("../input/nnDataSet.json").T

df.Party.replace({'I':'Indep'}, inplace=True)

df['contrib'] = df.Contributions.apply(lambda x: int(x.replace('$','').replace(',','')))/1e6

df.head()
fig, ax = plt.subplots(figsize=(8, 8))

sns.boxplot(x='Vote', y='contrib', data=df, ax=ax); ax.set_ylabel("Contribution ($, Millions)");

ax.set_title("Contributions by Expected Vote");
fig, ax = plt.subplots(figsize=(8, 8))

sns.boxplot(x='Party', y='contrib', data=df, ax=ax); ax.set_ylabel("Contribution ($, Millions)");

ax.set_title("Contributions by Expected Vote");
res = pd.crosstab(df.Party, df.Vote)

res = res.div(res.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(7, 6))

ax = sns.heatmap(res, annot=True, ax=ax); ax.set_title("Voting Split by Party");
def cramers_corrected_stat(confusion_matrix):

    """ calculate Cramers V statistic for categorical-categorical association.

        uses correction from Bergsma and Wicher, 

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
cols = ["Party", "Vote", "contrib"]

corrM = np.zeros((len(cols),len(cols)))

# there's probably a nice pandas way to do this

for col1, col2 in itertools.combinations(cols, 2):

    idx1, idx2 = cols.index(col1), cols.index(col2)

    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))

    corrM[idx2, idx1] = corrM[idx1, idx2]
corr = pd.DataFrame(corrM, index=cols, columns=cols)

fig, ax = plt.subplots(figsize=(7, 6))

ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
equation = "Vote_i ~ -1 + C(Party)"# + contrib"# + C(Party)*contrib"

# statsmodels/patsy doesn't like categorical for the output variable

df['Vote_i'] = df.Vote.replace({'Yes':1, 'No':0, 'Unknown':np.NaN})

subset = (df.Party != 'Indep') & (df.Vote != 'Unknown')

model = sm.Logit.from_formula(equation, data=df, subset=subset).fit()

model.summary2()
# Print the probability returned for each party

for k, p in model.params.items():

    odds = np.exp(p)

    prob = odds/(1+odds)

    print(f"{k}: {prob*100:.2f}%")
# keeping it simple, here's the confusion matrix for the predictions:

a = pd.DataFrame(model.pred_table(), index=['Yes','No'],columns=['Yes','No'], )

a.index.name = 'Actual'; a.columns.name = 'Predicted'

a
results = model.predict(df[(df.Vote=='Unknown') & (df.Party != 'Indep')])

results = results.to_frame(name='Prob')

results.head()
vals = df.Vote.value_counts()

yes = vals['Yes']

no = vals['No']



# Monte Carlo

N = 100000

win_percents = []

for _ in range(N):

    results['Draw'] = np.random.random(results.shape[0])

    n_yes = sum(results.Draw <= results.Prob)

    n_no = results.shape[0] - n_yes

    win_percents.append((yes + n_yes)/(yes + no + n_yes + n_no))
plt.hist(win_percents);

yes_chance = sum(1 for x in win_percents if x > 0.5)/N

print(f"Chance of 'Yes' winning: {yes_chance*100}%")
df[(df.Party=='Democrat') & (df.Vote=='Yes')]
df[(df.Party=='Republican') & (df.Vote=='No')]