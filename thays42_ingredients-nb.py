import pandas as pd
import numpy as np

%cd ../input
train_df = pd.read_json("train.json")
train_df.head()
train_df['flat_ingredients'] = train_df.ingredients.apply(lambda x: ", ".join(x))
train_df.flat_ingredients.head()


cuisine_ingredient_corpa = train_df.groupby('cuisine')['ingredients'].sum()

full_corpus = train_df['ingredients'].sum()
from collections import Counter
brazil_corpus = cuisine_ingredient_corpa[0]
brazil_c = Counter(brazil_corpus)

#british_corpus = cuisine_ingredient_corpa[1]
#british_c = Counter(british_corpus)

brazil_tf = [(x, y, y / sum(brazil_c.values())) for x, y in brazil_c.most_common()]
brazil_tf
cuisine_ingredient_corpa
