#Finding different definitions and checking meaning of those for the word play.
from nltk.corpus import wordnet
for words in wordnet.synsets("Play"):
	for lemma in words.lemmas():
		print(lemma)
	print('\n')
#Finding details of word 'Fun'
word = wordnet.synsets("Fun")[0]
print(word.name())
print(word.definition())
print(word.examples())
#Hyponyms of word
word=wordnet.synsets("Program")[0]
word.hyponyms()
#Finding Antonyms and Synonyms
synonyms=[]
antonyms=[]
for words in wordnet.synsets('Good'):
    for lemma in words.lemmas():
        synonyms.append(lemma.name())
        
for words in wordnet.synsets('Good'):
    for lemma in words.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
            
        
print('synonyms of good are:',synonyms)
print('\n\n antonyms of good are:',antonyms)
#Find the similarity between ‘Car’ and ‘Bike’ words
word1= wordnet.synsets("Car","n")[0]
word2= wordnet.synsets("Bike","n")[0]
print(word1.wup_similarity(word2))

                       