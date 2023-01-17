from nltk.corpus import brown
brown.readme().replace('\n', ' ')
brown.fileids()
brown.categories()
brown.sents('ca01')[0]
from nltk import FreqDist # Takes a bunch of tokens and returns the frequencies of all unique cases.
words_in_corpora = FreqDist(w.lower() for w in brown.words() if w.isalpha()) # Checking if the word is alphabetical avoids including stuff like `` and '' which are actually pretty common. Note that it also omits words such as 1 (very common), aug., 1913, $30, 13th, over-all etc. Another option would have been .isalnum().
words_in_corpora
words_in_corpora_freq_sorted = list(map(list, words_in_corpora.items())) # I use this instead of sorted() because I want to sort my dictionary into a (mutable) list in order to delete the second column as opposed to into a tuple (immutable).
words_in_corpora_freq_sorted
words_in_corpora_freq_sorted.sort(key=lambda x: x[1], reverse=True) # Using a lambda function is an alternative to using the operator library.
words_in_corpora_freq_sorted
best1500 = words_in_corpora_freq_sorted[:1500]

for list_item in best1500:
    del list_item[1]

best1500
# Since best1500 is now a list of words, it should be flattened.
import itertools
chain = itertools.chain(*best1500) # We break down the list into its individual sublists and then chain them. What chain does is that it further breaks down each sublist into its individual components so this approach can be used to flatten any list of lists.
best1500 = list(chain) # chain is of type itertools.chain so we need the cast
best1500
from nltk.corpus import stopwords

stopw = stopwords.words('english')

# Receives a list of words and removes stop words from list
def nonstop(listwords):
    return [word for word in listwords if word not in stopw]

best1500_words_corpora = nonstop(best1500) # Note how this will probably contain less than 1500 words.
best1500_words_corpora
# documents = [(nonstop(brown.words(fileid)), category) for category in brown.categories() for fileid in brown.fileids(category)]
# documents # Note how documents is a list of tuples.

# The code above generates a representation of the corpus but without removing punctuation. This is better:
documents = [([item.lower() for item in nonstop(brown.words(fileid)) if item.isalpha()], category)
             for category in brown.categories()
             for fileid in brown.fileids(category)]
documents # Note how documents is a list of tuples.
from random import shuffle

shuffle(documents)
documents
# Given a document extract features (the presence or not of the 1500 most frequent words of the corpus)
def document_features(doc):
    doc_set_words = set(doc) # Checking whether a word occurs in a set is much faster than checking whether it occurs in a list.
    features_dic = {} # Features is a dictionary
    for word in best1500_words_corpora:
        features_dic['has(%s)' % word] = (word in doc_set_words)
    return features_dic

doc_features_set = [(document_features(d),c) for (d,c) in documents]
doc_features_set
from nltk import NaiveBayesClassifier

train_set = doc_features_set[:350] # Since the total is 500
test_set  = doc_features_set[150:]

classifier = NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(15)
from nltk.classify import accuracy

print(accuracy(classifier, test_set))
# 'ca01' is under the 'news' category
classifier.classify(document_features(brown.words('ca01')))
from nltk.tokenize import RegexpTokenizer

# The test text needs to be long enough in order to contain a significant amount of the 1500 most common words in our training corpus.
text = "1 God, infinitely perfect and blessed in himself, in a plan of sheer goodness freely created man to make him share in his own blessed life. For this reason, at every time and in every place, God draws close to man. He calls man to seek him, to know him, to love him with all his strength. He calls together all men, scattered and divided by sin, into the unity of his family, the Church. To accomplish this, when the fullness of time had come, God sent his Son as Redeemer and Saviour. In his Son and through him, he invites men to become, in the Holy Spirit, his adopted children and thus heirs of his blessed life. 2 So that this call should resound throughout the world, Christ sent forth the apostles he had chosen, commissioning them to proclaim the gospel: \"Go therefore and make disciples of all nations, baptizing them in the name of the Father and of the Son and of the Holy Spirit, teaching them to observe all that I have commanded you; and lo, I am with you always, to the close of the age.\"4 Strengthened by this mission, the apostles \"went forth and preached everywhere, while the Lord worked with them and confirmed the message by the signs that attended it.\" 3 Those who with God's help have welcomed Christ's call and freely responded to it are urged on by love of Christ to proclaim the Good News everywhere in the world. This treasure, received from the apostles, has been faithfully guarded by their successors. All Christ's faithful are called to hand it on from generation to generation, by professing the faith, by living it in fraternal sharing, and by celebrating it in liturgy and prayer. 4 Quite early on, the name catechesis was given to the totality of the Church's efforts to make disciples, to help men believe that Jesus is the Son of God so that believing they might have life in his name, and to educate and instruct them in this life, thus building up the body of Christ. Catechesis is an education in the faith of children, young people and adults which includes especially the teaching of Christian doctrine imparted, generally speaking, in an organic and systematic way, with a view to initiating the hearers into the fullness of Christian life. While not being formally identified with them, catechesis is built on a certain number of elements of the Church's pastoral mission which have a catechetical aspect, that prepare for catechesis, or spring from it. They are: the initial proclamation of the Gospel or missionary preaching to arouse faith; examination of the reasons for belief; experience of Christian living; celebration of the sacraments; integration into the ecclesial community; and apostolic and missionary witness. Catechesis is intimately bound up with the whole of the Church's life. Not only her geographical extension and numerical increase, but even more her inner growth and correspondence with God's plan depend essentially on catechesis. Periods of renewal in the Church are also intense moments of catechesis. In the great era of the Fathers of the Church, saintly bishops devoted an important part of their ministry to catechesis. St. Cyril of Jerusalem and St. John Chrysostom, St. Ambrose and St. Augustine, and many other Fathers wrote catechetical works that remain models for us. The ministry of catechesis draws ever fresh energy from the councils. the Council of Trent is a noteworthy example of this. It gave catechesis priority in its constitutions and decrees. It lies at the origin of the Roman Catechism, which is also known by the name of that council and which is a work of the first rank as a summary of Christian teaching. The Council of Trent initiated a remarkable organization of the Church's catechesis. Thanks to the work of holy bishops and theologians such as St. Peter Canisius, St. Charles Borromeo, St. Turibius of Mongrovejo or St. Robert Bellarmine, it occasioned the publication of numerous catechisms. It is therefore no surprise that catechesis in the Church has again attracted attention in the wake of the Second Vatican Council, which Pope Paul Vl considered the great catechism of modern times. the General Catechetical Directory (1971) the sessions of the Synod of Bishops devoted to evangelization (1974) and catechesis (1977), the apostolic exhortations Evangelii nuntiandi (1975) and Catechesi tradendae (1979), attest to this. the Extraordinary Synod of Bishops in 1985 asked that a catechism or compendium of all Catholic doctrine regarding both faith and morals be composed. The Holy Father, Pope John Paul II, made the Synod's wish his own, acknowledging that this desire wholly corresponds to a real need of the universal Church and of the particular Churches. He set in motion everything needed to carry out the Synod Fathers' wish."

tokenizer = RegexpTokenizer(r'\w+') # Picks out sequences of alphanumeric characters as tokens and drops everything else
text_tokens = nonstop(tokenizer.tokenize(text.lower()))
text_tokens = [w for w in text_tokens if w.isalpha()]
text_tokens
text_features = document_features(text_tokens)
text_features
classifier.classify(document_features(text_tokens))