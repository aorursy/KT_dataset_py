import json

from collections import Counter



import matplotlib.pyplot as plt

%matplotlib inline



from wordcloud import WordCloud
cnt = Counter()



with open('../input/dump-tokenized.jsonl/dump-tokenized.jsonl') as f_in:

    for line in f_in:

        line = json.loads(line)

        tokens = set(line['source'].split())

        cnt.update(tokens)
wc = WordCloud(background_color='white', max_words=200,

               width=400, height=300, max_font_size=40, scale=3,

               random_state=1)

wc.generate_from_frequencies(cnt)



wc.to_image()