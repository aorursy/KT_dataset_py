import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from scipy.optimize import minimize



%matplotlib inline
MIN_VOTES = 100
df = pd.read_csv("/kaggle/input/board-games/bgg_GameItem.csv", index_col="bgg_id")

df.drop(

    index=df.index[(df.compilation == 1) | (df.num_votes < MIN_VOTES)],

    columns=['game_type', 'compilation', 'bga_id', 'dbpedia_id', 'luding_id', 'spielen_id', 'wikidata_id', 'wikipedia_id'],

    inplace=True,

)

df.shape
top = df.sort_values('avg_rating', ascending=False).head(5)

top.T
for n, (bgg_id, name) in enumerate(top.name.items()):

    print(f"{n + 1}. {{{{% game {bgg_id} %}}}}{name}{{{{% /game %}}}}")
data = df[

    (df.year >= 1900)

    & (df.year <= 2020)

    & df.avg_rating.notna()

    & df.bayes_rating.notna()

    & (df.bayes_rating != 5.5)

]

print(len(data))

df['num_dummy_votes'] = data.num_votes * (data.avg_rating - data.bayes_rating) / (data.bayes_rating - 5.5)

df.num_dummy_votes.describe()
df[df.index == 199478][["name", "num_votes", "avg_rating", "bayes_rating", "num_dummy_votes"]]
num = 20

df.num_dummy_votes.quantile([n / num for n in range(num + 1)])
ax = df.num_dummy_votes[(df.num_dummy_votes >= 1000) & (df.num_dummy_votes <= 2500)].hist(bins=200)

ax
fig = ax.get_figure()

fig.savefig("num_dummies_hist.svg")
def bayes(avg_rating, num_rating, dummy_value, num_dummy):

    return (avg_rating * num_rating + dummy_value * num_dummy) / (num_rating + num_dummy)
def correlations(data, start, end, step, dummy_value):

    for num_dummy in range(start, end + 1, step):

        col = "bayes_rating_adj"

        data[col] = bayes(data.avg_rating, data.num_votes, dummy_value, num_dummy)

        corr = data[["bayes_rating", col]][data.num_votes >= MIN_VOTES].corr('spearman')['bayes_rating'][col]

        yield num_dummy, corr



dummy_value = 5.5

start, end, step = 1000, 2_500, 1

dummy_values, correlations = zip(*correlations(df, start, end, step, dummy_value))



print(f"Best value: {np.max(correlations):.10f} with {dummy_values[np.argmax(correlations)]} dummy ratings")



plt.plot(dummy_values, correlations)

plt.savefig("num_dummies_corr.svg")
def target_corr(x, data=df):

    num_dummy = x[0]

    dummy_value = x[1] if len(x) > 1 else 5.5

    temp = pd.DataFrame({

        'original': data.bayes_rating,

        'estimated': bayes(data.avg_rating, data.num_votes, dummy_value, num_dummy)

    })

    return -temp.corr('spearman')['original']['estimated']



def target_rmse(x, data=df):

    num_dummy = x[0]

    dummy_value = x[1] if len(x) > 1 else 5.5

    return np.linalg.norm(data.bayes_rating - bayes(data.avg_rating, data.num_votes, dummy_value, num_dummy))
x0 = np.array([1500])

for fun in (target_corr, target_rmse):

    print(f"Optimizing {fun}")

    result = minimize(

        fun=fun, 

        x0=x0, 

        method='Nelder-Mead', 

        options={

            'xatol': 1e-12, 

            'maxiter': 10_000,

            'maxfev': 10_000,

            'disp': True,

        },

    )

    print(f"Best value: {result.fun:.10f} with {result.x[0]:.1f} dummy ratings")

    if len(x0) > 1:

        print(f"Best dummy value: {result.x[1]:.5f}")
x0 = np.array([1500, 5.5])

for fun in (target_corr, target_rmse):

    print(f"Optimizing {fun}")

    result = minimize(

        fun=fun, 

        x0=x0, 

        method='Nelder-Mead', 

        options={

            'xatol': 1e-12, 

            'maxiter': 10_000,

            'maxfev': 10_000,

            'disp': True,

        },

    )

    print(f"Best value: {result.fun:.10f} with {result.x[0]:.1f} dummy ratings")

    if len(x0) > 1:

        print(f"Best dummy value: {result.x[1]:.5f}")
average = (df.avg_rating * df.num_votes).sum() / df.num_votes.sum()

print(f"Average rating: {average:.5f}")
num_dummies = 1_600

df["bayes_rating_adj"] = bayes(df.avg_rating, df.num_votes, average, num_dummies)

df["rank_adj"] = df.bayes_rating_adj.rank(ascending=False)



for n, (bgg_id, name) in enumerate(df.sort_values("rank_adj").name.head(100).items()):

    print(f"{n + 1}. {{{{% game {bgg_id} %}}}}{name}{{{{% /game %}}}}")
top_ratings = df.num_votes.sort_values(ascending=False).iloc[249]

print(f"The top 250 games have at least {top_ratings} ratings")
dummy_value = 5.5

df["bayes_rating_adj"] = bayes(df.avg_rating, df.num_votes, dummy_value, top_ratings)

df["rank_adj"] = df.bayes_rating_adj.rank(ascending=False)



for n, (bgg_id, name) in enumerate(df.sort_values("rank_adj").name.head(100).items()):

    print(f"{n + 1}. {{{{% game {bgg_id} %}}}}{name}{{{{% /game %}}}}")
df["bayes_rating_adj"] = bayes(df.avg_rating, df.num_votes, average, top_ratings)

df["rank_adj"] = df.bayes_rating_adj.rank(ascending=False)



for n, (bgg_id, name) in enumerate(df.sort_values("rank_adj").name.head(100).items()):

    print(f"{n + 1}. {{{{% game {bgg_id} %}}}}{name}{{{{% /game %}}}}")