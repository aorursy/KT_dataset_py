from nltk.tokenize import sent_tokenize
data = "Adolf Hitler (20 April 1889 – 30 April 1945) was a German politician of Austrian origin and the leader of Nazi Germany. He became Chancellor of Germany in 1933, after a democratic election in 1932. He became Führer (leader) of the German Empire in 1934. Hitler led the Nazi Party NSDAP from 1921."
print(sent_tokenize(data))
from nltk.tokenize import word_tokenize
data = "All work and no play makes jack a dull boy, all work and no play"
print(word_tokenize(data))
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
data = "Adolf Hitler (20 April 1889 – 30 April 1945) was a German politician of Austrian origin and the leader of Nazi Germany. He became Chancellor of Germany in 1933, after a democratic election in 1932. He became Führer (leader) of the German Empire in 1934. Hitler led the Nazi Party NSDAP from 1921."
words_in_data = word_tokenize(data)
stop_words = set(stopwords.words('english'))
print(stop_words)
stopwords_in_data = []
stopwords_not_in_data = []

for word in words_in_data:
    if word in stop_words:
        stopwords_in_data.append(word)
        
for word in words_in_data:
    if word not in stop_words:
        stopwords_not_in_data.append(word)
print('Stopwords present in the data are: \n' + str(stopwords_in_data))
print('\n')
print('Words that are not a part of stopwords are: \n' + str(stopwords_not_in_data))
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
words = ["game","gaming","gamed","games"]
ps = PorterStemmer()

for word in words:
    print(ps.stem(word))
import spacy
# Uploading english model
spacy_model = spacy.load('en')
data = "Adolf Hitler (20 April 1889 – 30 April 1945) was a German politician of Austrian origin and the leader of Nazi Germany. He became Chancellor of Germany in 1933, after a democratic election in 1932. He became Führer (leader) of the German Empire in 1934. Hitler led the Nazi Party NSDAP from 1921."
# Analysing the data
document = spacy_model(data)
# Checking for nouns in data
for nc in document.noun_chunks:
    print(nc.text)
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.chunk import named_entity,ne_chunk
data = doc="""According to a media report, Mumbai has the highest density of cars in India. Pune is in second place. 

The density of private cars in Mumbai has gone up by 18% in 2 years. There are 510 cars per km of road as opposed to 430 cars per km in 2016. This is almost five times that of Delhi (108 cars per km). Despite having fewer cars than Delhi, Mumbai is more congested due to low road space. Mumbai has 2,000 km of roads compared to the national capital, which has 28,000 km of roadways.

There are 10.2 lakh private cars in Mumbai. That is 28% of the total number of vehicles in the city, which stands at 36 lakh. According to RTO officials, the western suburbs have the highest number of registered cars (5 lakh). There are 3.3 lakh private cars in the island city and 1.7 lakh in the eastern suburbs.

Pune has 359 cars per km and Kolkata is the third most congested city with 319 cars per km. Chennai comes in fourth with 297 cars per km followed by Bangalore with 149 cars per km."""
words=word_tokenize(doc) # Extracting words from sentences
words[0:10] # Checking random 10 words
pos=nltk.pos_tag(words) # Checking part of speech 
pos[0:10]
name=ne_chunk(pos)
loc=[]
for na in str(name).split('\n'):
    if '/NNP'in na:
        loc.append(na)
        print(na)
import re 
i=0
town=[]
while i<len(loc): 
    pattern=r'\w+/NNP'
    s=re.findall(pattern,loc[i])
    print(s)
    if s not in town:
        town.append(s)
    i=i+1
city=[]
for l in town:
    z=str(l)
    w=z.split('/')
    az=w[0]
    rr=az[2:]
    city.append(rr)
    
print(city)