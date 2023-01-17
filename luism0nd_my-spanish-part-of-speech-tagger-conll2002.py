import nltk



tagged_sentences = nltk.corpus.conll2002.tagged_sents('esp.train')

 

print (tagged_sentences[5])

#features with 3, 4 prefixes

def features(sentence, index):

    """ sentence: [w1, w2, ...], index: the index of the word """

    return {

        'word': sentence[index],

        'is_first': index == 0,

        'is_last': index == len(sentence) - 1,

        'is_capitalized': sentence[index][0].upper() == sentence[index][0],

        'is_all_caps': sentence[index].upper() == sentence[index],

        'is_all_lower': sentence[index].lower() == sentence[index],

        'prefix-1': sentence[index][0],

        'prefix-2': sentence[index][:3],

        'prefix-3': sentence[index][:4],

        'suffix-1': sentence[index][-1],

        'suffix-2': sentence[index][-3:],

        'suffix-3': sentence[index][-4:],

        'prev_word': '' if index == 0 else sentence[index - 1],

        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],

        'has_hyphen': '-' in sentence[index],

        'is_numeric': sentence[index].isdigit(),

        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]

    }
import pprint

pprint.pprint(features(['Esta', 'es', 'una', 'oración'],0))
def untag(tagged_sentence):

    return [w for w, t in tagged_sentence]
cutoff = int(.75 * len(tagged_sentences))

training_sentences = tagged_sentences[:cutoff]

test_sentences = tagged_sentences[cutoff:]



print(len(training_sentences))

print(len(test_sentences))



def transform_to_dataset(tagged_sentences):

    X, y = [], []

    

    for tagged in tagged_sentences:

        for index in range(len(tagged)):

            X.append(features(untag(tagged),index))

            y.append(tagged[index][1])

    return X, y



X, y = transform_to_dataset(training_sentences)
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline



clf = Pipeline([

    ('vectorizer', DictVectorizer(sparse=False)),

    ('classifier', DecisionTreeClassifier(criterion='entropy'))

])



clf.fit(X[:20000], y[:20000])



print('Training completed')



X_test, y_test = transform_to_dataset(test_sentences)
print("Accuracy:", clf.score(X_test, y_test))
def pos_tag(sentence):

    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])

    return list(zip(sentence, tags))
from nltk import word_tokenize

print (pos_tag(word_tokenize('El canciller de Alemania llegó el lunes a Rusia')))
from joblib import dump, load

dump(clf, 'My_POS_Tagger_es.joblib') 