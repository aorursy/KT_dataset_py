# Import the python libraries needed for the NLP for Real Estate article.

import numpy as np

import pandas as pd

import spacy as sp

import nltk

import re

from nltk.corpus import treebank

from nltk.stem.porter import PorterStemmer
nlp = sp.load('en_core_web_lg')
title = """OpenAI Says New AI too Risky for Full Disclosure"""

copy = """OpenAI has determined its latest research into language-processing AI models has high-risk implications if fully released to the public. A risk OpenAI has felt compelled to limit the amount of published information from this advanced research.



Before diving deeply into the research, here is a little information about OpenAI. OpenAI was created at the end of 2015 through a $1 billion backing by leading Silicon Valley people and organizations. The goal for this non-profit group clearly outlines its mission statement.



OpenAI is a non-profit artificial intelligence research company. Our goal is to advance digital intelligence in a way that is most likely to benefit humanity as a whole, unconstrained by a need to generate a financial return. Since our research is free from financial obligations, we can better focus on a positive human impact.



Monitoring the growth and implementation of AI has become a serious topic of debate around the world. OpenAI’s backing comes from people who have deep concerns on artificial intelligence’s impact on humanity.



Unprecedented Results

Today’s society is about creating and consuming content. We create content about everything in our lives, from our professional experiences, promoting ourselves and our business, to recording events and facts. One barrier to content creation is the time it takes to develop it. Even for me, the time to research and write a somewhat readable article involves a significant amount of energy.



The latest research by OpenAI solves this problem, but there are consequences.



OpenAI has developed a new language-processing model called Generative Pre-Training or GPT. The latest version of this model is GPT-2, and yes, it is much better than the first version. So much better that OpenAI only disclosed small amounts of the code and data that was used to create the model.



GPT-2 training requires the ingestion of content of over 8 million web pages to use as a dataset. The model was then given input of a few sentences that were written by a human. The task for the model is to predict the next word and following sentences based on its training it received from the content of its dataset.



The results are astonishing with the model completing paragraphs of content. Samples provided by OpenAI show how readable the paragraphs are and how easily attributable as written by humans. You can view one sample of a task here. While it took the model 10 or more times to produce the content, it is exciting, but scary at the same time.



Boom for Real Estate.

I am sure you all see how this piece of language-processing artificial intelligence would work in the field of real estate. Just a few of my thoughts.



Think about how chatbots could be more intuitive and answer questions outside of the box of MLS data.

Public remarks on property pages would be generated in seconds and could easily carry the voice of the listing agent and brokerage.

Imagine how fast press releases and new agent articles could be published? Consider all the Agent bio’s that are out in the wild.

What if a brokerage could place language filters on the bio’s so the content conveys Agent and brand messaging?

Conceive the ability to create web page content that is very hyper-local – even dynamic based on current events.

I started to think about how emails and newsletters could be written when you marry customer data profiles. Google Gmail is already doing some of this already. Talk about delivering an experience.



But wait, is there a downside to all this technology?



OpenAI says “Wait a minute”

OpenAI’s mission statement clearly articulated its goals. The research they are performing is to advance the technology in order to have a dialogue on the impact on humanity. The concern is not if “artificial intelligence” is taking us down the road to a world as depicted in the movie “Terminator”, but what would happen if the technology was in the hands of bad actors.



The founders and leaders of OpenAI are very cognizant of the perils of exploited usage. Bad actors manually crafted masses of false information during both the presidential and last elections. When speaking of bad actors, they could be anybody. They could be politicians, criminals, governments, or companies and organizations intending to promote their own agenda.



Imagine if language-processing artificial intelligence was available to the bad actors; how much high-quality misinformation could be generated over thousands of websites? Would the false information override the truth and become perceived reality? How would society be able to monitor and govern what is true or false?



These are questions that need answers or direction prior to developing a release strategy of the code. Jack Clark, policy director at OpenAI, commented in the MIT Technology Review article An AI that writes convincing prose risks mass-produced fake news by saying, “It’s very clear that if this technology matures—and I’d give it one or two years—it could be used for disinformation or propaganda,” he says. “We’re trying to get ahead of this”



Getting ahead of this is the right thing to do. OpenAI plans to have more discussions around this and will make more known to the public over the next six months. We will keep you updated on the results of this conversation.



AI Strategy Side Note:

IT Brief – New Zealand published an article this week called “Gartner debunks common AI misconceptions”. One myth Gartner debunks is that not every company needs an AI strategy. In the article, Gartner VP Alexander Linden says, “Even if the current strategy is ‘no AI’, this should be a conscious decision based on research and consideration. And, like every other strategy, it should be periodically revisited and changed according to the organization’s needs. AI might be needed sooner than expected.”



While AI might be only available through technology companies for small brokerages and MLSs, the WAV Group knows enterprise brokerages and MLSs can benefit from having a strategy. If you would like to discuss your AI strategy, call David Gumpper, President of Business Intelligence and Technology."""
pack = "package, packages, packaged"

list = pack.split(",")

corpus = []

for i in list:

    stem = PorterStemmer()

    x = stem.stem(i)

    corpus.append(x)

print("stemming - package, packages, packaged =", corpus)
pack = "penny, pennies"

list = pack.split(",")

corpus = []

for i in list:

    stem = PorterStemmer()

    x = stem.stem(i)

    corpus.append(x)

print("stemming - penny and pennies =", corpus)
article = title + " " + copy

# Regex code removes any punctuation from the article

article = re.sub('[^a-zA-Z0-9]', ' ', article)

# Lower case all characters

article = article.lower()

# Regex removes hash tags

article = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", article)

# Remove extra chartacters

article = re.sub("(\\d|\\W)+"," ",article)

# Convert into a list

article = article.split()

print(article)
corpus = []

for i in article:

    stem = PorterStemmer()

    x = stem.stem(i)

    corpus.append(x)

print("stemming - ", corpus)
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords
corpus = []

stop_words = set(stopwords.words("english"))

#for i in article:

lemma = WordNetLemmatizer()

y = [lemma.lemmatize(word) for word in article if not word in stop_words]

corpus.append(y)

print("lemmatization -", corpus)
from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
corpuscloud = "".join(str(corpus).strip('[]'))

corpuscloud = corpuscloud.replace("'", "")

corpuscloud = corpuscloud.replace(",", "")
%matplotlib inline

wordcloud = WordCloud(

    background_color='white',

    stopwords=stop_words,

    max_words=50,

    max_font_size=72, 

    random_state=42).generate(str(corpuscloud))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("article.png", dpi=1800)
!pip install rake-nltk
from rake_nltk import Metric, Rake

import operator
a = title + " " + copy

# Regex code removes any punctuation from the article

a = re.sub('[^a-zA-Z0-9]', ' ', a)

# Lower case all characters

a = a.lower()

# Regex removes hash tags

a = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", a)

# Remove extra chartacters

a = re.sub("(\\d|\\W)+"," ",a)

r = Rake()

r.extract_keywords_from_text(str(a))

r.get_ranked_phrases_with_scores()