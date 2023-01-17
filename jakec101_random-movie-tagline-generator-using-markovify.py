import pandas as pd
import markovify as mkv
import numpy as np

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/movies_metadata.csv',low_memory = False)
df.head()
df.tagline.isnull().value_counts()
(df.tagline.str.rstrip()=='').value_counts()
df = df.drop(df[df.tagline.str.rstrip()==''].index)
df = df.dropna(subset=['tagline'])
df = df[['original_title','genres','tagline']]
df.genres.head()
df = df[df.genres.str.len()>2]
df= df.reset_index(drop=True)
df.head()
df.genres = df.genres.apply(lambda x: [j['name'] for j in eval(x)])
df.head()
genre_list = set()
for s in df['genres']:
    genre_list = set().union(s, genre_list)
    
genre_list = sorted(list(genre_list))
genre_list
genre_table = df[['tagline']]
for genre in genre_list:
    genre_table[genre] = df['genres'].str.join(" ").str.contains(genre).apply(lambda x:1 if x else 0)
genre_table
new_df = pd.DataFrame(index = genre_list)
for genre in genre_list:
    new_df.loc[genre,'Composite_Tagline'] = genre_table[genre_table[genre] == 1].tagline.str.rstrip('!.,').str.cat(sep = ". ")
new_df
text_model = mkv.Text(new_df.loc['Horror','Composite_Tagline'])
for i in range(5):
    print(text_model.make_sentence())
def random_tagline_generator(genre,taglines=1):
    text_model = mkv.Text(new_df.loc[genre,'Composite_Tagline'])
    return_string = str()
    for i in range(taglines):
        new_tagline = text_model.make_sentence()
        return_string += (new_tagline + "\n")
    return(return_string)
print(random_tagline_generator('War',5))