import matplotlib.pyplot as plt

import pandas as pd

from collections import Counter



topics_table = pd.read_csv("../input/ForumTopics.csv")



topic_words = [ z.lower() for y in

                   [ x.split() for x in topics_table['Name'] if isinstance(x, str)]

                   for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

plt.bar(range(10), [word_count_dict[w] for w in popular_words[0:10]])

plt.xticks([x + 0.5 for x in range(10)], popular_words[0:10])

plt.title("Popular Words in Kaggle Forum Topics")

plt.show()