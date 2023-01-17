import pandas as pd

tweets = pd.read_json('../input/tweets.json')

tweets.columns
# Retain all non-retweet entries (value is 0/1 as a float).

tweets = tweets[tweets.is_retweet < 1.0]
# Some values may be treated as floats, so cast all to string.

tweets['text'] = tweets.text.astype('str')



# Remove all URLs.

url_regex = r"""https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}"""

tweets['text'] = tweets.text.str.replace(url_regex, '')



# Fix `&amp;`.

tweets['text'] = tweets.text.str.replace(r'&amp;', '&')



# Replace inverted quotes.

tweets['text'] = tweets.text.str.replace('“', '"')

tweets['text'] = tweets.text.str.replace('”', '"')



# Replace strange hyphens.

tweets['text'] = tweets.text.str.replace('–', '-')

tweets['text'] = tweets.text.str.replace('—', '-')



# Replace strange apostrophes.

tweets['text'] = tweets.text.str.replace('’', "'")

tweets['text'] = tweets.text.str.replace('‘', "'")

tweets['text'] = tweets.text.str.replace('\x92', "'")



# Replace latin space.

tweets['text'] = tweets.text.str.replace('\xa0', ' ')

# Zero width space.

tweets['text'] = tweets.text.str.replace('\u200b', ' ')



# l2r and r2l marks.

tweets['text'] = tweets.text.str.replace('\u200e', '')

tweets['text'] = tweets.text.str.replace('\u200f', '')



# Fix bad unicode.

tweets['text'] = tweets.text.str.replace('\U0010fc00', '')
# Join all lines for the model.

tweets_text = '\n'.join(tweets.text.values)
# Make a Text model using markovify.

import markovify

text_model = markovify.Text(tweets_text, state_size=5)
for i in range(10):

    print('{}: {}'.format(i, text_model.make_short_sentence(400)))