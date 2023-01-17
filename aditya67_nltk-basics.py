import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
text = 'Deidre paced on the beach behind her bungalow, unable to do anything but lecture herself over and over about how stupid she was to sleep with some random stranger.'
print(word_tokenize(text))
print(sent_tokenize(text))
stop_words = set(stopwords.words("english"))
print(stop_words)
words = word_tokenize(text)
filtered = []
filtered = [w for w in words if w not in stop_words]
print(filtered)
from nltk.stem import PorterStemmer
ps = PorterStemmer()

txt = "Deidre paced on the beach behind her bungalow, unable to do anything but lecture herself over and over about how stupid she was to sleep with some random stranger."
sent = sent_tokenize(txt)
new = []
for i in range(len(sent)):
    words = word_tokenize(sent[i])
    words = [ps.stem(w) for w in words]
    new.append(' '.join(words))
print(new)
from nltk.stem import WordNetLemmatizer
sentences = sent_tokenize(txt)
lemmtizer = WordNetLemmatizer()
new_sentence = []
for i in range(len(sentences)):
    words = word_tokenize(sentences[i])
    words = [lemmtizer.lemmatize(word) for word in words]
    new_sentence.append(' '.join(words))
print(new_sentence)
from nltk.tokenize import PunktSentenceTokenizer
# Now, let's create our training and testing data:
train_txt="Crocodiles (subfamily Crocodylinae) or true crocodiles are large aquatic reptiles that live throughout the tropics in Africa, Asia, the Americas and Australia. Crocodylinae, all of whose members are considered true crocodiles, is classified as a biological subfamily. A broader sense of the term crocodile, Crocodylidae that includes Tomistoma, is not used in this article. The term crocodile here applies to only the species within the subfamily of Crocodylinae. The term is sometimes used even more loosely to include all extant members of the order Crocodilia, which includes the alligators and caimans (family Alligatoridae), the gharial and false gharial (family Gavialidae), and all other living and fossil Crocodylomorpha."
sample_text ="Crocodiles are large aquatic reptiles which are carnivorous. Allegators belong to this same reptile species"
# Next, we can train the Punkt tokenizer like:
cust_tokenizer = PunktSentenceTokenizer(train_txt)
# Then we can actually tokenize, using:
tokenized = cust_tokenizer.tokenize(sample_text)

def process_text():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))

process_text()
from nltk.corpus import gutenberg

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])
