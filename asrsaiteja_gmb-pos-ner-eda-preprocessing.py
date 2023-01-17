import numpy as np # linear algebra

import pandas as pd # data processing

import os

data_paths = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_paths[filename] = os.path.join(dirname, filename)

        

print(data_paths)
# Define required imports

import matplotlib

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# ner_df = pd.read_csv(data_paths['ner.csv'])

ner_data = pd.read_csv(data_paths['ner_dataset.csv'], encoding = 'unicode_escape')

print(ner_data.shape)

ner_data.head()
ner_data.fillna(method = 'ffill', inplace = True)

ner_data.tail()
ner_data.rename(columns = {'Sentence #':'SentId'}, inplace = True)

ner_data.columns
ner_data['SentId'] = ner_data['SentId'].apply(lambda x:x.split()[-1]).astype(int)
# Save sentence length to list

sentence_len = ner_data['SentId'].value_counts().tolist()

# Plot sentence by length

plt.hist(sentence_len, bins=60)

plt.title('Number of words per sentence')

plt.xlabel('Sentence length in words')

plt.ylabel('Number of sentences')

plt.show()
# ner_tags = ner_data[ner_data['Tag'] != 'O']

entities = ner_data.groupby("Tag")['Word']

entities.describe()
postags = ner_data.groupby("POS")['Word']

postags.describe()
ner_data['entity_type'] = ner_data['Tag'].apply(lambda x:x.split('-')[-1])

ner_data['entity_type'].unique()
i = 0

plt.figure(figsize=(16,8))

for ent_type in ['geo', 'gpe', 'per', 'org',]:

    ent_type_df = ner_data[ner_data['entity_type'] == ent_type]

    ent_texts = ' '.join(ent_type_df['Word'].tolist())

    ent_wc = WordCloud(collocations = False).generate(ent_texts)

    plt.subplot(2, 2, i + 1)

    plt.imshow(ent_wc, interpolation='bilinear')

    plt.title("{}".format(ent_type).upper())

    i += 1
i = 0

plt.figure(figsize=(16,8))

for ent_type in ['tim', 'art', 'nat', 'eve']:

    ent_type_df = ner_data[ner_data['entity_type'] == ent_type]

    ent_texts = ' '.join(ent_type_df['Word'].tolist())

    ent_wc = WordCloud(collocations = False).generate(ent_texts)

    plt.subplot(2, 2, i + 1)

    plt.imshow(ent_wc, interpolation='bilinear')

    plt.title("{}".format(ent_type).upper())

    i += 1
ner_data = pd.read_csv(data_paths['ner_dataset.csv'], encoding = 'unicode_escape')
val_end = int(47959 * 0.84)

train_end = int(val_end*0.8)

train_end, val_end



sent_idx = 0

outputfile = open('train.txt', 'w')



for i, row in ner_data.iterrows():

    if str(row[0]).startswith('Sentence'):

        outputfile.write("\n")

        sent_idx += 1

        

        if sent_idx == train_end + 1:

            outputfile.close()

            outputfile = open('val.txt', 'w')

        

        if sent_idx == val_end + 1:

            outputfile.close()

            outputfile = open('test.txt', 'w')

            

    outputfile.write("{} {} {}\n".format(row[1], row[2], row[3]))

    

outputfile.close()
!ls