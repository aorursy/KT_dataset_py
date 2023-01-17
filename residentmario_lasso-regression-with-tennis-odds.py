import pandas as pd

import numpy as np

matches = pd.read_csv("../input/t_odds.csv")

matches.head()

book = matches.loc[:, [c for c in matches.columns if'odd' in c]].dropna()



bookies = set([c.split("_")[0] for c in matches.columns if'odd' in c])



def deltafy(srs):

    return pd.Series(

        {b: srs[b + '_player_1_odd'] - srs[b + '_player_2_odd'] for b in bookies}

    )



book = book.apply(deltafy, axis='columns')

book = book.assign(

    winner=(matches.player_1_score > matches.player_2_score).astype(int).loc[book.index]

)
book.head(3)
bookie_odds = book.iloc[:, :15].values

outcomes = book.iloc[:, 15:].values
from sklearn.linear_model import Lasso



def get_lasso_model(alpha):

    clf = Lasso(alpha=alpha, normalize=True)

    clf.fit(bookie_odds, outcomes)

    return clf
clf = get_lasso_model(0.00001)
clf.coef_
predicted_outcomes = (clf.predict(bookie_odds) > 0.5).astype(int)



# Sort outcomes by average exchange score.

sorted_outcomes = pd.DataFrame(

    {'outcomes': outcomes.flatten(), 

     'predicted_outcomes': predicted_outcomes.flatten()}

).loc[

    np.argsort(np.average(bookie_odds, axis=1))

].reset_index(drop=True)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



sample = sorted_outcomes.sample(50, random_state=5).sort_index().reset_index(drop=True)

plt.scatter(sample.index.values, 

            sample.outcomes,

            color=sample.apply(

lambda srs: 'green' if srs.outcomes == srs.predicted_outcomes else 'red', axis='columns')

           )

plt.gca().set_title('Predictions versus Reality')
print("Our model is correct ~{0:.1f}% of the time.".format(

    (sorted_outcomes.outcomes == sorted_outcomes.predicted_outcomes).sum() / 

    len(sorted_outcomes) * 100

))
get_lasso_model(0.001).coef_
book.columns[np.argmax(_)]