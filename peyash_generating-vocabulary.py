# import the dataset

import json

dataset = '../input/bangla-newspaper-dataset/data/data.json'

with open(dataset, encoding='utf-8') as f:

    data = json.load(f)
!jupyter nbextension enable --py widgetsnbextension
from collections import defaultdict



class Vocab:

    def __init__(self):

        self.word_count = defaultdict(int)

        self.n_sent = 0     

    

    def addWord(self, word):

        self.word_count[word] += 1

    

    def incr_sent(self, n):

        self.n_sent += n

    

    def export(self, filename='vocabs.txt', order='desc'):

        with open(filename, 'w') as f:

            for key in self.sort(order):

                #temp_key = key.replace(',', '')

                f.write(f'{key} \t {self.word_count[key]}\n')

    

    def unique_words(self):

        return len(self)

    

    def total_sentences(self):

        return self.n_sent

    

    def total_words(self):

        total = 0

        for key in self.word_count.keys():

            total += self.word_count[key]

        return total

    

    def sort(self, order='desc'):

        if order == 'desc':

            return dict(sorted(vocab.word_count.items(),  key=lambda x: x[1], reverse=True))

        else:

            return dict(sorted(vocab.word_count.items(),  key=lambda x: x[1]))

       

    def __len__(self):

        return len(self.word_count)

    

    def __str__(self):

        return f'\tTotal sentences: {self.n_sent}\n\tTotal tokens: {self.total_words()}\n\tUnique tokens: {self.unique_words()}'
import nltk

nltk.download('punkt')
from tqdm import tqdm_notebook as tqdm



vocab = Vocab()

print('Total News: ', len(data))



with tqdm(total=len(data)) as pbar:

    

    for dict_obj in data:

        contents = dict_obj['content']

        contents = contents.split('।')

        

        # remove the empty strings from the list

        contents = list(filter(None, contents))

        

        # increment the number of sentences

        vocab.incr_sent(len(contents))

        

        for content in contents:              

            content += '।'

            # tokenize the sentence using nltk

            content = nltk.word_tokenize(content)

            

            for word in content:

                vocab.addWord(word)

        

        pbar.update(1)

print(vocab)
# print top 25 tokens based on occurrence

sample = 25

for key in vocab.sort('desc').keys():

    print(f'{key}: {vocab.sort("desc")[key]}')

    sample -= 1

    if not sample:

        break
vocab.export('vocabs.txt')