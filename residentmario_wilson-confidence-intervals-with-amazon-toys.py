import pandas as pd

toys = pd.read_csv("../input/amazon_co-ecommerce_sample.csv")

toys.head()
import numpy as np



def get_scores(l):

    try:

        if pd.isnull(l):

            return np.array([])

    except:

        ret = np.array(l)[1::4]

        ret = [str(s).strip() for s in ret]

        try:

            ret = [float(s) for s in ret]

        except ValueError:

            ret = []

            

        return ret

    

review_scores = toys.customer_reviews.str.split("//").map(get_scores).values
review_scores[:5].tolist()
pos_neg_review_scores = list(

    map(

        lambda sc: [s >= 4 for s in sc], review_scores.tolist()

    )

)
import scipy.stats as st

import numpy as np



def wilson_confidence_interval(X, c):

    n = len(X)

    

    z_score = st.norm.ppf(1 - ((1 - c) / 2))

    p_hat = np.array(X).astype(int).sum() / n

    

    correction_1 = z_score * z_score / (2*n)

    correction_2 = z_score * np.sqrt((p_hat*(1-p_hat) + z_score * z_score/(4 * n))/n) / (1 + z_score * z_score /n)

    additive_part = correction_1 + correction_2



    return (p_hat - additive_part, p_hat + additive_part)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
cis = [wilson_confidence_interval([True]*n + [False]*n, 0.95) for n in range(1, 1000)]
plt.plot(range(1, 1000), np.array(cis)[:, 0])

plt.plot(range(1, 1000), np.array(cis)[:, 1])
plt.plot(range(1, 51), np.array(cis)[:50, 0])
toy_recommendability_cis = np.array(

    [wilson_confidence_interval(pos_neg_review_scores[n], 0.95) for n in range(len(pos_neg_review_scores))]

)
pd.Series(toy_recommendability_cis[:, 1] - toy_recommendability_cis[:, 0]).plot.hist(bins=20)
toys.iloc[np.nanargmax(toy_recommendability_cis)].number_of_reviews
(len(toys.iloc[np.nanargmax(toy_recommendability_cis)].customer_reviews.split("//")) - 1) / 4