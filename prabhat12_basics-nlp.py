!pip install wikipedia
# import and check

import wikipedia as wiki





try:

    not_going_to_work = wiki.summary('Dog')



except:

    print("Be specific. \n")



text = wiki.summary('German Shepherd') # German_Shepherd will work to, the underscore is present in the wikipedia link

text
from nltk.tokenize import sent_tokenize



sent_tokenize(text)
sentence = "I am bored, I need . popcorn : and { netflix * `. I am a big @ fan of Big Bang ~ theory."

sent_tokenize(sentence)
from nltk.tokenize import RegexpTokenizer



# regex for matching capital letters

tokenizer = RegexpTokenizer('[A-Z]\w+')

tokenizer.tokenize(sentence)
# getting rid of special characters used in text

tokenizer = RegexpTokenizer(r'\w+')

text_list = tokenizer.tokenize(text)

text_list
from nltk.corpus import stopwords



stopwords_eng = stopwords.words('english')

stopwords_eng
add_stop_words = ['The','is','an','a','as','ˈdɔʏtʃɐ','ˈʃɛːfɐˌhʊnt']

stopwords_eng.extend(add_stop_words)
text_list = [word for word in text_list if word not in stopwords_eng]

text_list
from wordcloud import WordCloud,ImageColorGenerator

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



str_text_list = ""

for word in text_list:

    str_text_list += word + " "



wordcloud = WordCloud(background_color="white",mode="RGBA",).generate(str_text_list)



plt.figure(figsize = (10,10))



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
mask = np.array(Image.open("../input/visual-yolo-opencv/german_shephard.jpg"))

wordcloud = WordCloud(background_color="white", mask=mask, mode="RGBA",max_words=700).generate(wiki.page('German Shephard').content)



plt.figure(figsize = (10,10))



image_colors = ImageColorGenerator(mask)

plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')

plt.axis("off")

plt.show()
from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures



biagram =  BigramCollocationFinder.from_words(text_list)

biagram.nbest(BigramAssocMeasures.likelihood_ratio, 10)
from nltk.collocations import TrigramCollocationFinder

from nltk.metrics import TrigramAssocMeasures



trigram =  TrigramCollocationFinder.from_words(text_list)

trigram.nbest(TrigramAssocMeasures.likelihood_ratio, 10)
from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer

from nltk.stem import RegexpStemmer



porter = PorterStemmer()

lancaster = LancasterStemmer()

regexp = RegexpStemmer('ing')



print("Porter Stemmer: ")

print(porter.stem("dance"))

print(porter.stem("dancing"))

print(porter.stem("danced"))

print(porter.stem("dances"))



print("\nLancaster Stemmer: ")

print(lancaster.stem("dance"))

print(lancaster.stem("dancing"))

print(lancaster.stem("danced"))

print(lancaster.stem("dances"))



print("\nRegexp Stemmer: ")

print(regexp.stem("dance"))

print(regexp.stem("dancing"))

print(regexp.stem("danced"))

print(regexp.stem("dances"))
from nltk.stem import WordNetLemmatizer





lemmatizer = WordNetLemmatizer()



print(lemmatizer.lemmatize("dance", pos="v"))

print(lemmatizer.lemmatize("dancing", pos="v"))

print(lemmatizer.lemmatize("danced", pos="v"))

print(lemmatizer.lemmatize("dances", pos="v"))
# import from the library

from nltk.corpus import wordnet
# we will explore the word gain

for synset in wordnet.synsets("gain")[:7]:

    

    print("\n Synset name :  ", synset.name())   

    print("\n meaning : ", synset.definition())       

    print("\n example : ", synset.examples()) 

    print("\n part of speech : ", synset.pos())

    

    # printing lemmas for a synset

    for i,lemma in enumerate(synset.lemmas()):

        print("\n lemma " , (i+1) , " :", lemma.name())

        

        # printing antonyms for the above lemma

        if(lemma.antonyms()):

            print(" antonyms: ")

            

            for i,antonym in enumerate(lemma.antonyms()):

                print(i+1, ": " , antonym.name())

        

    print("\n hypernyms : ", synset.hypernyms())

    print("\n hyponyms : ", synset.hyponyms())

    #print("\n root_hypernyms :", synset.root_hypernyms())

        

    # divider        

    print("\n"+"##"*20)
profit = wordnet.synset('net_income.n.01') #[Synset('profit.n.01')]

gain = wordnet.synset('gain.n.01') # [Synset('gain.n.01')]

loss = wordnet.synset('loss.n.01') # [Synset('loss.n.01')]



print("profit and gain:")

print("\tPath similarity :", profit.path_similarity(gain))

print("\tLeacock-Chodorow similarity:", profit.lch_similarity(gain))

print("\tWu-Palmer similarity: ", profit.wup_similarity(gain))



print("\nprofit and loss:")

print("\tPath similarity :", profit.path_similarity(loss))

print("\tLeacock-Chodorow similarity:", profit.lch_similarity(loss))

print("\tWu-Palmer similarity: ", profit.wup_similarity(loss))
