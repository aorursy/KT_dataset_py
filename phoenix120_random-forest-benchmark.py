# Имена файлов с данными.
TRAIN_FILENAME = "../input/train.csv"
TEST_FILENAME = "../input/test.csv"
# Считывание файлов.
from collections import namedtuple
WordForm = namedtuple("WordForm", "word pos gram")

def get_sentences(filename, is_train):
    sentences = []
    with open(filename, "r", encoding='utf-8') as r:
        next(r)
        sentence = []
        for line in r:
            if len(line.strip()) == 0:
                if len(sentence) == 0:
                    continue
                sentences.append(sentence)
                sentence = []
                continue
            if is_train:
                word = line.strip().split("\t")[2]
                pos = line.strip().split("\t")[3].split("#")[0]
                gram = line.strip().split("\t")[3].split("#")[1]
                sentence.append(WordForm(word, pos, gram))
            else:
                word = line.strip().split("\t")[2]
                sentence.append(word)
        if len(sentence) != 0:
            sentences.append(sentence)
    return sentences
train = get_sentences(TRAIN_FILENAME, True)
test = get_sentences(TEST_FILENAME, False)
# Класс для удобной векторизации грамматических значений.
import jsonpickle
import os
from collections import defaultdict
from typing import Dict, List, Set

def process_gram_tag(gram: str):
    gram = gram.strip().split("|")
    return "|".join(sorted(gram))


def get_empty_category():
    return {GrammemeVectorizer.UNKNOWN_VALUE}


class GrammemeVectorizer(object):
    UNKNOWN_VALUE = "Unknown"

    def __init__(self, dump_filename: str):
        self.all_grammemes = defaultdict(get_empty_category)  # type: Dict[str, Set]
        self.vectors = []  # type: List[List[int]]
        self.name_to_index = {}  # type: Dict[str, int]
        self.dump_filename = dump_filename  # type: str
        if os.path.exists(self.dump_filename):
            self.load()

    def add_grammemes(self, pos_tag: str, gram: str) -> int:
        gram = process_gram_tag(gram)
        vector_name = pos_tag + '#' + gram
        if vector_name not in self.name_to_index:
            self.name_to_index[vector_name] = len(self.name_to_index)
            self.all_grammemes["POS"].add(pos_tag)
            gram = gram.split("|") if gram != "_" else []
            for grammeme in gram:
                category = grammeme.split("=")[0]
                value = grammeme.split("=")[1]
                self.all_grammemes[category].add(value)
        return self.name_to_index[vector_name]

    def init_possible_vectors(self) -> None:
        self.vectors = []
        for grammar_val, index in sorted(self.name_to_index.items(), key=lambda x: x[1]):
            pos_tag, grammemes = grammar_val.split('#')
            grammemes = grammemes.split("|") if grammemes != "_" else []
            vector = self.__build_vector(pos_tag, grammemes)
            self.vectors.append(vector)

    def get_vector(self, vector_name: str) -> List[int]:
        if vector_name not in self.name_to_index:
            return [0] * len(self.vectors[0])
        return self.vectors[self.name_to_index[vector_name]]

    def get_vector_by_index(self, index: int) -> List[int]:
        return self.vectors[index] if 0 <= index < len(self.vectors) else [0] * len(self.vectors[0])

    def get_ordered_grammemes(self) -> List[str]:
        flat = []
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            for value in sorted(list(values)):
                flat.append(category+"="+value)
        return flat
    
    def save(self) -> None:
        with open(self.dump_filename, "w") as f:
            f.write(jsonpickle.encode(self, f))

    def load(self):
        with open(self.dump_filename, "r") as f:
            vectorizer = jsonpickle.decode(f.read())
            self.__dict__.update(vectorizer.__dict__)

    def size(self) -> int:
        return len(self.vectors)

    def grammemes_count(self) -> int:
        return len(self.get_ordered_grammemes())

    def is_empty(self) -> int:
        return len(self.vectors) == 0

    def get_name_by_index(self, index):
        d = {index: name for name, index in self.name_to_index.items()}
        return d[index]

    def get_index_by_name(self, name):
        pos = name.split("#")[0]
        gram = process_gram_tag(name.split("#")[1])
        return self.name_to_index[pos + "#" + gram]

    def __build_vector(self, pos_tag: str, grammemes: List[str]) -> List[int]:
        vector = []
        gram_tags = {pair.split("=")[0]: pair.split("=")[1] for pair in grammemes}
        gram_tags["POS"] = pos_tag
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            if category not in gram_tags:
                vector += [1 if value == GrammemeVectorizer.UNKNOWN_VALUE else 0 for value in sorted(list(values))]
            else:
                vector += [1 if value == gram_tags[category] else 0 for value in sorted(list(values))]
        return vector
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

morph = MorphAnalyzer()
to_ud = converters.converter('opencorpora-int', 'ud14')

def convert_from_opencorpora_tag(tag, text):
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1]
    return pos, gram

def fill_all_variants(word, vectorizer):
    for parse in morph.parse(word):
        pos, gram = convert_from_opencorpora_tag(parse.tag, parse.word)
        gram = process_gram_tag(gram)
        vectorizer.add_grammemes(pos, gram)

vectorizer = GrammemeVectorizer("vectorizer.json")
if vectorizer.is_empty():
    for sentence in train:
        for form in sentence:
            fill_all_variants(form.word, vectorizer) 
    for sentence in test:
        for word in sentence:
            fill_all_variants(word, vectorizer)
    vectorizer.init_possible_vectors()
    vectorizer.save()
vectorizer_output = GrammemeVectorizer("vectorizer_output.json")
if vectorizer_output.is_empty():
    for sentence in train:
        for form in sentence:
            gram = process_gram_tag(form.gram)
            vectorizer_output.add_grammemes(form.pos, gram)
    vectorizer_output.init_possible_vectors()
    vectorizer_output.save()
# Получение признаков для конкретного контекста.
def get_context_features(i, parse_sentence, context_len):
    sample = []
    left = i-(context_len-1)//2
    right = i+context_len//2
    if left < 0:
        for i in range(-left):
            sample += [0 for i in range(vectorizer.grammemes_count())]
    for parse in parse_sentence[max(left, 0): min(right+1, len(sentence))]:
        word = parse.word
        pos, gram = convert_from_opencorpora_tag(parse.tag, parse.word)
        gram = process_gram_tag(gram)
        sample += vectorizer.get_vector(pos+"#"+gram)
    if right > len(sentence)-1:
        for i in range(right-len(sentence)+1):
            sample += [0 for i in range(vectorizer.grammemes_count())]
    assert len(sample) == context_len * vectorizer.grammemes_count()
    return sample
# Загрузка обучающей выборки.
import numpy as np
import os

TRAIN_SAMPLES_PATH = "samples.npy"
ANSWERS_PATH = "answers.npy"
if not os.path.exists(TRAIN_SAMPLES_PATH) or not os.path.exists(ANSWERS_PATH):
    context_len = 5
    n = sum([1 for sentence in train for word in sentence])
    samples = np.zeros((n, context_len*vectorizer.grammemes_count()), dtype='bool_')
    answers = np.zeros((n, ), dtype='int')
    index = 0
    for sentence in train:
        parse_sentence = [morph.parse(form.word)[0] for form in sentence]
        for i, form in enumerate(sentence):
            samples[index] = get_context_features(i, parse_sentence , context_len)
            gram = process_gram_tag(form.gram)
            answers[index] = vectorizer_output.get_index_by_name(form.pos+"#"+gram)
            index += 1
            if index % 100000 == 0:
                print(index)
    np.save(TRAIN_SAMPLES_PATH, samples)
    np.save(ANSWERS_PATH, answers)
else:
    samples = np.load(TRAIN_SAMPLES_PATH)
    answers = np.load(ANSWERS_PATH)
print(samples[0], answers[0])
# Выбор классификатора
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20)
# Кросс-валидация
from sklearn.model_selection import cross_val_score
X = samples[:20000]
y = answers[:20000]
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
# Загрузка тестовой выборки
TEST_SAMPLES_PATH = "test_samples.npy"
ANSWERS_PATH = "answers.npy"
if not os.path.exists(TEST_SAMPLES_PATH):
    n = sum([1 for sentence in test for word in sentence])
    test_samples = np.zeros((n, context_len*vectorizer.grammemes_count()), dtype='bool_')
    index = 0
    for i, sentence in enumerate(test):
        parse_sentence = [morph.parse(word)[0] for word in sentence]
        for i, word in enumerate(sentence):
            test_samples[index] = get_context_features(i, parse_sentence, context_len)
            index += 1
    np.save(TEST_SAMPLES_PATH, test_samples)
else:
    test_samples = np.load(TEST_SAMPLES_PATH)
# Обучение классификатора.
X = samples[:200000]
y = answers[:200000]
clf.fit(X, y)
# Предсказания.
answers = []
batch_size = 1000
n_batches = len(test_samples)//batch_size
for i in range(n_batches):
    answers += list(clf.predict(test_samples[i*batch_size: i*batch_size+batch_size]))
answers += list(clf.predict(test_samples[n_batches*batch_size:]))
# Сохранение посылки
with open("subm.csv", "w") as f: 
    f.write("Id,Prediction\n")
    for index, answer in enumerate(answers):
        f.write(str(index) + "," + vectorizer_output.get_name_by_index(answer) + "\n")
