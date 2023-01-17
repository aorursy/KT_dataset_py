!pip install git+https://github.com/ssut/py-googletrans.git
import os

import pandas as pd

from googletrans import Translator
text = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/' \

                       'jigsaw-toxic-comment-train.csv', 

                        nrows=10_000)





# TODO: Get the proportion of languages in the test set and set a randomized language per comment with np.random.choice()



translator = Translator()

for i,t in enumerate(text.comment_text[19:22]):

    try:

        encoded = translator.translate(t, dest='fr').text

        decoded = translator.translate(encoded, dest='en').text

        print(f"\nSet {i}\n"

              f"Original: {t}\n\n"

              f"Recoded: {decoded}\n")

    except: pass
import markovify as mk
doc = text.loc[text.toxic == 1, 'comment_text'].tolist()

text_model = mk.Text(doc)

for i in range(10):

    print(text_model.make_sentence())
