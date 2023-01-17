print("Hello World..!!")

#print("Hello \nWorld..!!")
name = "kaggle"
year = "2010"

print("Hello my name is " + name + ",")
print("and I am founded in " + year + ".")
from math import *

num = 625
print(sqrt(num))

# wordcloud using python with the help of Lucid programming

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
text = "One of the key areas of artificial intelligence is Natural Language Processing (NLP) or text mining as it is generally known that deals with teaching computers how to extract meaning from text."
basecloud = WordCloud().generate(text)
plt.imshow(basecloud)
plt.axis("off")
plt.show()
# The above three lines of code is used very frequently so define it as function

def plot_wordcloud(WordCloud):
    plt.imshow(basecloud)
    plt.axis("off")
    plt.show()

from wordcloud import STOPWORDS
from wordcloud import WordCloud
stopwords = set(STOPWORDS)
stopwords.add("key")
stopwords.add("known")
stopwords.add("generally")
wordcloud = WordCloud(stopwords = stopwords, relative_scaling= 1.0).generate(text)
plot_wordcloud(wordcloud)