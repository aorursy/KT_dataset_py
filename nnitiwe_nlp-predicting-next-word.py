from bs4 import BeautifulSoup

from nltk.util import ngrams

from collections import defaultdict

from nltk import trigrams

from nltk.tokenize import RegexpTokenizer

import requests



#load fetch speech text from blog

response = requests.get("https://maxsiollun.wordpress.com/great-speeches-in-nigerias-history/")

soup = BeautifulSoup(response.text,'html.parser')

sentence = soup.find_all('p',text=True)

print(sentence[1:3])
note='' #we will merge the list string values into a single string

for line in sentence[1:3]:

    note+=str(line)

#convert text to lower case

sentence=note.lower()
#convert Sentence into Tokens and extract all punctuations

tokenizer = RegexpTokenizer(r'\w+')

tk_sentence=tokenizer.tokenize(sentence)

tk_sentence
#A view of our Trigram

gram_sentence=list(ngrams(tk_sentence, 3))

gram_sentence
# Create Word Model

word_model = defaultdict(lambda: defaultdict(lambda: 0))







for sentence in tk_sentence:

    for first_word, second_word, word_label in trigrams(tk_sentence,pad_left=True,pad_right=True):

        word_model[(first_word, second_word)][word_label] += 1

dict(word_model)
#run convert the word occurance scores into probabilities

for words_train in word_model:

    total_count = float(sum(word_model[words_train].values()))

    for word_test in word_model[words_train]:

        word_model[words_train][word_test] /= total_count
#predict the next word after 'the', 'nigerian'

dict(word_model['the', 'nigerian'])
#predict the next word after 'us', 'as'

dict(word_model['us', 'as'])