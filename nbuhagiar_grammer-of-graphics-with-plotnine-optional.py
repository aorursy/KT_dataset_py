import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head(3)
from plotnine import *
top_wines = reviews[reviews['variety'].isin(reviews['variety'].value_counts().head(5).index)]
df = top_wines.head(1000).dropna()

(ggplot(df)
 + aes('points', 'price')
 + geom_point())
df = top_wines.head(1000).dropna()

(
    ggplot(df)
        + aes('points', 'price')
        + geom_point()
        + stat_smooth()
)
df = top_wines.head(1000).dropna()

(
    ggplot(df)
        + geom_point()
        + aes(color='points')
        + aes('points', 'price')
        + stat_smooth()
)
df = top_wines.head(1000).dropna()

(ggplot(df)
     + aes('points', 'price')
     + aes(color='points')
     + geom_point()
     + stat_smooth()
     + facet_wrap('~variety')
)
(ggplot(df)
 + geom_point(aes('points', 'price'))
)
(ggplot(df, aes('points', 'price'))
 + geom_point()
)
(ggplot(top_wines)
     + aes('points')
     + geom_bar()
)
(ggplot(top_wines)
     + aes('points', 'variety')
     + geom_bin2d(bins=20)
)
(ggplot(top_wines)
         + aes('points', 'variety')
         + geom_bin2d(bins=20)
         + coord_fixed(ratio=1)
         + ggtitle("Top Five Most Common Wine Variety Points Awarded")
)
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)\
                        .rename(columns=lambda x: x.replace(" ", "_"))
pokemon.head(3)
ggplot(pokemon) + aes("Attack", "Defense") + geom_point()
ggplot(pokemon) + aes("Attack", "Defense") + aes(color="Legendary") + geom_point()
(ggplot(pokemon) + 
 aes("Attack") + 
 geom_histogram(bins=20) + 
 facet_wrap("~Generation"))
