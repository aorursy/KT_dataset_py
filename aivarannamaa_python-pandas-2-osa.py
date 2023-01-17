import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)

pd.set_option('display.max_columns', 200)

df = pd.read_csv("../input/movies.csv")

df = df.assign(release_date=pd.to_datetime(df.release_date))

df = df.assign(release_year=df.release_date.dt.year)
df[df.runtime > 210]
df[df.runtime > 210 and not (df.title.str.contains("the"))]
df[(df.runtime > 210) & ~df.title.str.contains("the")]
df[df.runtime > 210 & ~df.title.str.contains("the")]
df.homepage
df[(df.homepage == None) | (df.homepage == np.nan)]
np.nan == np.nan
print(pd.isnull(np.nan))

print(pd.isnull(None))

print(pd.isnull(0))

print(pd.isnull(0.0))

print(pd.isnull(''))

print(pd.isnull('Tere'))
df.homepage.isnull()
df[df.homepage.isnull()]
print(pd.notnull(np.nan))

print(len(df))

print(len(df[df.homepage.notnull()]) + len(df[df.homepage.isnull()]))
df[df.spoken_languages.isnull()].release_date.dt.year.value_counts()
print(np.nan / 2)

print(max(np.nan, 8))

print(min(np.nan, 8))
df.homepage.str.upper()
df.runtime.mean()
# Kirjuta siia vastav päring
df[df.homepage.str.contains("themovie")]
df[df.homepage.str.contains("themovie").fillna(False)]
# lisa õigesse kohta .fillna("")

df[df.homepage.str.contains("themovie")]
df[(df.runtime > 210) & ~df.genres.str.contains("Drama")]
def extract_first_item(comma_separated_items):

    # NA argumendi korral tagasta NA

    if pd.isnull(comma_separated_items):

        return None

    

    items = comma_separated_items.split(", ")

    

    # arvestame ka võimalusega, et sõne oli tühi

    if len(items) == 0: 

        return None

    else:

        return items[0]



# demonstreerime funktsiooni kasutamist

extract_first_item("Action, Adventure, Fantasy, Science Fiction")
# Pane tähele, et argumendiks läheb ainult funktsiooni nimi, 

# ilma väljakutsumist tähistavate sulgudeta.

# Funktsiooni väljakutsumine õiges kohas jääb apply hooleks

df.genres.apply(extract_first_item) 
df = df.assign(main_genre=df.genres.apply(extract_first_item))
df[(df.vote_count == 0) & (df.vote_average == 0)] 
def correct_vote_average(row):

    if row.vote_count == 0 and row.vote_average == 0:

        # asendame vigased juhtumid nan-iga

        return np.nan

    else:

        # muud jäävad samaks

        return row.vote_average



df.apply(correct_vote_average, axis='columns')
df = df.assign(vote_average=df.apply(correct_vote_average, axis='columns'))
# https://stackoverflow.com/questions/47571618/how-to-split-expand-a-string-value-into-several-pandas-dataframe-rows/47571866#47571866

# https://stackoverflow.com/questions/17084579/how-to-remove-levels-from-a-multi-indexed-dataframe

df2 = (df.genres.str.split(", ", expand=True)

       .stack()

       .rename("genre")

       .reset_index(level=1, drop=True)

       .to_frame())
df.join(df2)