import pandas as pd

train_x = pd.read_csv("../input/train.csv", sep=',')
test_x = pd.read_csv("../input/test.csv", sep=',')
EAP, HPL, MWS = "", "", ""

for i, row in train_x.iterrows():
    a, t = row['author'], row['text']
    if a == 'EAP':
        EAP += " " + t.lower()
    elif a == 'HPL':
        HPL += " " + t.lower()
    elif a == 'MWS':
        MWS += " " + t.lower()
EAP[:50]
from nltk.tokenize import word_tokenize

EAP = word_tokenize(EAP)
HPL = word_tokenize(HPL)
MWS = word_tokenize(MWS)
EAP[:10]
from collections import Counter, defaultdict

def create_dist(text):
    c = Counter(text)

    least_common = c.most_common()[-1][1]
    total = sum(c.values())
    
    for k, v in c.items():
        c[k] = v/total

    return defaultdict(lambda: min(c.values()), c)
c_eap, c_hpl, c_mws = create_dist(EAP), create_dist(HPL), create_dist(MWS)
import decimal
from decimal import Decimal
decimal.getcontext().prec = 1000

def precise_product(numbers):
    result = 1
    for x in numbers:
        result *= Decimal(x)
    return result

def NaiveBayes(dist):
    """A simple naive bayes classifier that takes as input a dictionary of
    Counter distributions and can then be used to find the probability
    of a given item belonging to each class.
    The input dictionary is in the following form:
        ClassName: Counter"""
    attr_dist = {c_name: count_prob for c_name, count_prob in dist.items()}

    def predict(example):
        """Predict the probabilities for each class."""
        def class_prob(target, e):
            attr = attr_dist[target]
            return precise_product([attr[a] for a in e])

        pred = {t: class_prob(t, example) for t in dist.keys()}

        total = sum(pred.values())
        for k, v in pred.items():
            pred[k] = v / total

        return pred

    return predict
dist = {'EAP': c_eap, 'HPL': c_hpl, 'MWS': c_mws}
nBS = NaiveBayes(dist)
def recognize(sentence, nBS):
    return nBS(word_tokenize(sentence.lower()))
recognize("The blood curdling scream echoed across the mansion.", nBS)
def predictions(test_x, nBS):
    d = []
    for index, row in test_x.iterrows():
        i, t = row['id'], row['text']
        p = recognize(t, nBS)
        d.append({'id': i, 'EAP': p['EAP'], 'HPL': p['HPL'], 'MWS': p['MWS']})
    
    return pd.DataFrame(data=d)
submission = predictions(test_x, nBS)
submission.to_csv('submission.csv', index=False)