!pip install translators
import numpy as np

import pandas as pd



!ls '../input/preprocessing-the-indonesian-hate-abusive-text'
data = pd.read_csv('../input/preprocessing-the-indonesian-hate-abusive-text/preprocessed_indonesian_toxic_tweet.csv')

print("Shape: ", data.shape)

data.head(15)
import translators as ts

def back_translate(text):

    return ts.google(ts.google(str(text), 'id', 'en'), 'en', 'id').lower()



# Example :)

back_translate("haha kamu sangat lucu dehh. Dasar goblok anjing.")
data['Tweet_back_translated'] = data['Tweet'].apply(lambda x: back_translate(x))
data.head(15)
for x in data['Tweet'].values[:15]:

    print(x)
for x in data['Tweet_back_translated'].values[:15]:

    print(x)
data.to_csv('back_translated_indonesian_toxic_tweet.csv', index=False)