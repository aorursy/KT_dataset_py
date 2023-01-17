import nltk 

nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize, sent_tokenize 

stop_words = set(stopwords.words('english'))
#Text 

text = "Harika, asha and vineetha are my good friends.Harika is getting married next year.But friendship is a sacred bond between people."  
#word_tokenizing 

word_token = word_tokenize(text)

print("Printing Word tokens")

word_token
#Now filtering Sentence



Sentence_Filter = []



for words in word_token:

    if words not in stop_words:

        Sentence_Filter.append(words)



Sentence_Filter
tagged = nltk.pos_tag(Sentence_Filter) 



print("Printing POS tag words")

tagged