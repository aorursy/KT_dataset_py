import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv')
train.head()
severe_toxic = train[train.severe_toxic==1]
severe_toxic.head()
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(train.comment_text)
word_to_id = tokenizer.word_index

id_to_word = {value:key for key,value in word_to_id.items()}



texts = train.comment_text.tolist()

sequences = tokenizer.texts_to_sequences(texts)

print(' '.join(id_to_word[id] for id in sequences[1] ))

cleanText = []

for seq in sequences:

    c = ' '.join(id_to_word[id] for id in seq)

    cleanText.append(c)

train['comment_processed'] = cleanText 



severe_toxic = train[train.severe_toxic==1]
severe_toxic.head()
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

wc = WordCloud(background_color="black", 

                stopwords=STOPWORDS,width=2500, height=1800)

wc.generate(" ".join(severe_toxic.comment_processed))

plt.title("Severe Toxic", fontsize=50)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

plt.axis('off')
toxic = train[(train.toxic==1) & (train.severe_toxic==0) & (train.obscene==0) &(train.threat == 0) &(train.identity_hate == 0) &(train.insult == 0)]

wc.generate(" ".join(toxic.comment_processed))

plt.title("Toxic", fontsize=50)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

plt.axis('off')
obscene = train[(train.toxic==0) & (train.severe_toxic==0) & (train.obscene==1) &(train.threat == 0) &(train.identity_hate == 0) &(train.insult == 0)]

wc.generate(" ".join(obscene.comment_processed))

plt.title("obscene", fontsize=40)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17))

plt.axis('off')
threat = train[(train.toxic==0) & (train.severe_toxic==0) & (train.obscene==0) &(train.threat == 1) &(train.identity_hate == 0) &(train.insult == 0)]

wc.generate(" ".join(threat.comment_processed))

plt.title("Threat", fontsize=40)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17))

plt.axis('off')
ih = train[(train.toxic==0) & (train.severe_toxic==0) & (train.obscene==0) &(train.threat == 0) &(train.identity_hate == 1) &(train.insult == 0)]

wc.generate(" ".join(ih.comment_processed))

plt.title("Identity Hate", fontsize=40)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17))

plt.axis('off')
insult = train[(train.toxic==0) & (train.severe_toxic==0) & (train.obscene==0) &(train.threat == 0) &(train.identity_hate == 0) &(train.insult == 1)]

wc.generate(" ".join(insult.comment_processed))

plt.title("Insult", fontsize=40)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17))

plt.axis('off')
NotToxic = train[(train.toxic==0) & (train.severe_toxic==0) & (train.obscene==0) &(train.threat == 0) &(train.identity_hate == 0) &(train.insult == 0)]

wc.generate(" ".join(NotToxic.comment_processed))

plt.title("NotToxic", fontsize=40)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17))

plt.axis('off')