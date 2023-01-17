import pandas as pd
df = pd.read_csv('/kaggle/input/bts-lyrics/lyrics.csv')

# make all na fields reflect as such
df = df.fillna('NA')

# ensure date format for album release date
df['album_rd'] = pd.to_datetime(df.album_rd)

# ignore any track that does not have any lyrics or are album notes
df = df[~df['eng_track_title'].str.contains('skit', case=False) & ~df['eng_track_title'].str.contains('note', case=False)]

import re
def normalise(text, remove_punc=True):
    """method to normalise text"""
    # change text to lowercase and remove leading and trailing white spaces
    text = text.lower().strip()

    # remove punctuation
    if remove_punc:
        # remove punctuation
        text = re.sub(r'[\W]', ' ', text)
        # remove double spacing sometimes caused by removal of punctuation
        text = re.sub(r'\s+', ' ', text)

    return text

# normalise lyrics
df['lyrics'] = df['lyrics'].apply(lambda x: normalise(x))

df.drop_duplicates(subset='track_title', inplace=True)

df = df[~df['repackaged']]
