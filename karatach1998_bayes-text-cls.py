import os

import re



import numpy as np

import pandas as pd



from nltk.tokenize.regexp import regexp_tokenize

from nltk.stem.snowball import RussianStemmer

from sklearn.model_selection import train_test_split

from sklearn.utils import resample
ROOT = '../input/Поэты'

AUTHORS = os.listdir(ROOT)
author_texts = [(author, '\n'.join(

    line for file in os.listdir(os.path.join(ROOT, author)) for line in open(os.path.join(ROOT, author, file)).readlines() if line.startswith('\t\t')))

    for author in AUTHORS]
author = 'Лермонтов'

author_text = '\n'.join(t for a, t in author_texts if a == author)

other_text = '\n'.join(t for a, t in author_texts if a != author)
stemmer = RussianStemmer()

stem_cache = {}

regex = re.compile(r"[а-яА-Я]{3,}")



def get_stem(token):

    token = token.lower()

    stem = stem_cache.get(token)

    if stem:

        return stem

    stem_cache[token] = stem = stemmer.stem(token)

    return stem



def tokenize_and_filter(text):

    return [get_stem(x) for x in regexp_tokenize(text, r'\w+') if regex.match(x)]



author_tokens = tokenize_and_filter(author_text)

other_tokens = tokenize_and_filter(other_text)
author_train, author_test = train_test_split(author_tokens, train_size=0.6, shuffle=False)

other_train, other_test = train_test_split(other_tokens, train_size=0.6, shuffle=False)
author_train[:10], other_train[:10]
from collections import defaultdict

from functools import reduce

from operator import mul



class TextNaiveBayes:

    def __init__(self, blur_coef=0.0, p_apriori=0.5, q_apriori=0.5):

        self.blur_coef = blur_coef

        self.p_apriori, self.q_apriori = p_apriori, q_apriori

        self.bag1 = defaultdict(int)

        self.bag2 = defaultdict(int)

        self.N1, self.N2 = 0, 0

    

    def _shrink_bags(self):

        lower_bound, upper_bound = min(self.bag1.values()), max(self.bag1.values())

        shrink_zone = (upper_bound - lower_bound) / 10

        self.bag1 = defaultdict(int, filter(

            lambda t: lower_bound + shrink_zone <= t[1] <= upper_bound - shrink_zone, self.bag1.items()))

        

        lower_bound, upper_bound = min(self.bag2.values()), max(self.bag2.values())

        shrink_zone = (upper_bound - lower_bound) / 10

        self.bag2 = defaultdict(int, filter(

            lambda t: lower_bound + shrink_zone <= t[1] <= upper_bound - shrink_zone, self.bag2.items()))

    

    def fit(self, xx1, xx2):

        for x in xx1:

            self.bag1[x] += 1

        self.N1 = len(xx1)

        for x in xx2:

            self.bag2[x] += 1

        self.N2 = len(xx2)

        self._shrink_bags()

        return self

    

    def predict(self, xx):

#         blur_coef, N1, N2 = self.blur_coef, self.N1, self.N2

#         blur_coef, N1, N2 = self.blur_coef, len(self.bag1), len(self.bag2)

        blur_coef, N1, N2 = self.blur_coef, sum(self.bag1.values()), sum(self.bag2.values())

        p = reduce(mul, map(lambda x: ((self.bag1[x] + blur_coef) / (N1 + (N1 + N2) * blur_coef)) or 0.5, xx))

        q = reduce(mul, map(lambda x: ((self.bag2[x] + blur_coef) / (N2 + (N1 + N2) * blur_coef)) or 0.5, xx))

        prob = p / (p + (self.p_apriori / self.q_apriori) * q) if p + q else 0

        print(p, q, prob)

        return prob >= 0.5

    

    def predict_alternative(self, xx):

        from math import log

#         blur_coef, N1, N2 = self.blur_coef, self.N1, self.N2

#         blur_coef, N1, N2 = self.blur_coef, len(self.bag1), len(self.bag2)

        blur_coef, N1, N2 = self.blur_coef, sum(self.bag1.values()), sum(self.bag2.values())

        p = sum(map(lambda x: self.bag1[x] and log((self.bag1[x] + blur_coef) / (N1 + (N1 + N2) * blur_coef)), xx)) + self.p_apriori

        q = sum(map(lambda x: self.bag2[x] and log((self.bag2[x] + blur_coef) / (N2 + (N1 + N2) * blur_coef)), xx)) + self.q_apriori

        print(p, q)

        return p >= q

    

    def score(self, xxs, ys):

        return sum(self.predict(xx) == y for xx, y in zip(xxs, ys))
text1 = tokenize_and_filter("""

Мы все учились понемногу

Чему-нибудь и как-нибудь,

""") # Пушкин



text2 = tokenize_and_filter("""

Погиб поэт! — невольник чести —

Пал, оклеветанный молвой,

С свинцом в груди и жаждой мести,

Поникнув гордой головой!..

Не вынесла душа поэта

Позора мелочных обид,

Восстал он против мнений света

Один, как прежде... и убит!

Убит!.. к чему теперь рыданья,

Пустых похвал ненужный хор

И жалкий лепет оправданья?

Судьбы свершился приговор!

Не вы ль сперва так злобно гнали

Его свободный, смелый дар

И для потехи раздували

Чуть затаившийся пожар?

Что ж? веселитесь... — он мучений

Последних вынести не мог:

Угас, как светоч, дивный гений,

Увял торжественный венок.

""") # Лермонтов



text3 = tokenize_and_filter("""

Да! Теперь — решено. Без возврата

Я покинул родные поля.

Уж не будут листвою крылатой

Надо мною звенеть тополя.



Низкий дом без меня ссутулится,

Старый пёс мой давно издох.

На московских изогнутых улицах

Умереть, знать, судил мне Бог.



Я люблю этот город вязевый,

Пусть обрюзг он и пусть одрях.

Золотая дремотная Азия

Опочила на куполах.

""") # Есенин



tnb = TextNaiveBayes(4).fit(author_train, other_train)

print(tnb.predict(text1))

print(tnb.predict(text2))

print(tnb.predict(text3))

print(tnb.predict_alternative(text1))

print(tnb.predict_alternative(text2))

print(tnb.predict_alternative(text3))
from collections import Counter

blur_coefs = [0.0, 0.1, 0.5, 1, 1.5, 2.0]

tnbs = [TextNaiveBayes(bc).fit(

    resample(author_train, n_samples=int(0.6*len(author_train)), random_state=0),

    resample(other_train, n_samples=int(0.6*len(other_train)), random_state=0)

) for bc in blur_coefs]

text = text2

preds = Counter([tnb.predict(text) for tnb in tnbs] + [tnb.predict_alternative(text) for tnb in tnbs])

preds