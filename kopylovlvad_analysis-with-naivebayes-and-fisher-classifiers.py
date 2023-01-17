import text_classifier_kv as tc # package with naivebayes and fisher classifiers
from typing import List, Dict, Tuple
import csv
import os
print(os.listdir("../input"))
# print(text_classifier_kv.fisherclassifier())
# Any results you write to the current directory are saved as output.

row_limit: int = 3000
# (category, text)
train_data: List[Tuple[str, str]] = []
train_data_ham: int = 0
train_data_spam: int = 0
test_data: List[Tuple[str, str]] = []
test_data_ham: int = 0
test_data_spam: int = 0
reader = csv.reader(open('../input/spam.csv', newline='', encoding='latin-1'))
i: int = 0
for row in reader:
    i += 1
    if i == 1:
        continue
    if i < row_limit + 2:
        train_data.append((row[0], row[1]))
        if row[0] == 'ham':
            train_data_ham += 1
        else:
            train_data_spam += 1
    else:
        test_data.append((row[0], row[1]))
        if row[0] == 'ham':
            test_data_ham += 1
        else:
            test_data_spam += 1

print('Data overview:')
print('Train data size: %d' % len(train_data))
print('Train data ham: %d' % train_data_ham)
print('Train data spam: %d' % train_data_spam)
print('')
print('Test data size: %d' % len(test_data))
print('Test data ham: %d' % test_data_ham)
print('Test data spam: %d' % test_data_spam)

import text_classifier_kv as tc # package with naivebayes and fisher classifiers
from typing import List, Dict, Tuple
import csv
import os
# Any results you write to the current directory are saved as output.

row_limit: int = 3000
# (category, text)
train_data: List[Tuple[str, str]] = []
test_data: List[Tuple[str, str]] = []
reader = csv.reader(open('../input/spam.csv', newline='', encoding='latin-1'))
i: int = 0
for row in reader:
    i += 1
    if i == 1:
        continue
    if i < row_limit + 2:
        train_data.append((row[0], row[1]))
    else:
        test_data.append((row[0], row[1]))


def experiment(
    classifier,
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    show_not_equal_ham: bool = False,
    show_not_equal_spam: bool = False
) -> None:
    # main function train, test and print result
    [classifier.train(text, cat) for (cat, text) in train_data]
    stat: Dict[str, int] = {
        'equal': 0,
        'not_equal': 0,
        'not_equal_ham': 0,
        'not_equal_spam': 0
    }
    for (cat, text) in test_data:
        predict = classifier.classify(text)
        if cat == predict:
            stat['equal'] += 1
        else:
            stat['not_equal'] += 1
            if cat == 'ham':
                stat['not_equal_ham'] += 1
                if show_not_equal_ham == True:
                    print(text)
            else:
                stat['not_equal_spam'] += 1
                if show_not_equal_spam == True:
                    print(text)
    print(stat)
    ac: float = stat['equal'] / \
        ((stat['equal']+stat['not_equal'])/100)
    print('Accuracy: %f' % ac)
    return None


print('Experiment #1.1 - naivebayes')
cl = tc.naivebayes()
experiment(cl, train_data, test_data)

print('Experiment #1.2 - fisherclassifier')
cl = tc.fisherclassifier()
experiment(cl, train_data, test_data)


import text_classifier_kv as tc # package with naivebayes and fisher classifiers
from typing import List, Dict, Tuple, Callable
import csv
import os
# Any results you write to the current directory are saved as output.

row_limit: int = 3000
# (category, text)
train_data: List[Tuple[str, str]] = []
test_data: List[Tuple[str, str]] = []
reader = csv.reader(open('../input/spam.csv', newline='', encoding='latin-1'))
i: int = 0
for row in reader:
    i += 1
    if i == 1:
        continue
    if i < row_limit + 2:
        train_data.append((row[0], row[1]))
    else:
        test_data.append((row[0], row[1]))

        
def generate_text_vector(
    text_list: List[str],
    min_frequency: float=0.1,
    max_frequency: float=0.5,
    getfeatures: Callable[[str], Dict[str, int]] = tc.getwords,
) -> Callable[[str], Dict[str, int]]:
    '''
    Get a Dict of words from array of text between max and mix frequency
    '''
    text_list_len: int = len(text_list)
    text_vector: List[str] = []
    # Dict of uniq words for each text
    apcount: Dict[str, int] = {}

    for text in text_list:
        for word, count in getfeatures(text).items():
            apcount.setdefault(word, 0)
            if count > 0:
                apcount[word] += 1

    # Dict of uniq words for all texts
    # all words are between max and mix frequency
    fr_set: List[float] = []
    for word, count in apcount.items():
        frac = float(float(count) / float(text_list_len))
        fr_set.append(frac)
        if frac > min_frequency and frac < max_frequency:
            text_vector.append(word)

    def get_match_words(text: str) -> Dict[str, int]:
        new_dict: Dict[str, int] = {}
        for text, i in getfeatures(text).items():
            if text in text_vector:
                new_dict[text] = i

        return new_dict
    return get_match_words


def experiment(
    classifier,
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    show_not_equal_ham: bool = False,
    show_not_equal_spam: bool = False
) -> None:
    # main function train, test and print result
    [classifier.train(text, cat) for (cat, text) in train_data]
    stat: Dict[str, int] = {
        'equal': 0,
        'not_equal': 0,
        'not_equal_ham': 0,
        'not_equal_spam': 0
    }
    for (cat, text) in test_data:
        predict = classifier.classify(text)
        if cat == predict:
            stat['equal'] += 1
        else:
            stat['not_equal'] += 1
            if cat == 'ham':
                stat['not_equal_ham'] += 1
                if show_not_equal_ham == True:
                    print(text)
            else:
                stat['not_equal_spam'] += 1
                if show_not_equal_spam == True:
                    print(text)
    print(stat)
    ac: float = stat['equal'] / \
        ((stat['equal']+stat['not_equal'])/100)
    print('Accuracy: %f' % ac)
    return None

train_texts: List[str] = [text for _c, text in train_data]
getweatures_ignoring_common_words: Callable[[str], Dict[str, int]] 
getweatures_ignoring_common_words = generate_text_vector(
    train_texts, 
    min_frequency=0.0003,
    max_frequency=0.05
)
print('Experiment #2.1 - naivebayes with words frequency limit')
cl = tc.naivebayes(getfeatures=getweatures_ignoring_common_words)
experiment(cl, train_data, test_data)

print('Experiment #2.2 - fisherclassifier with words frequency limit')
cl = tc.fisherclassifier(getfeatures=getweatures_ignoring_common_words)
experiment(cl, train_data, test_data)

import text_classifier_kv as tc # package with naivebayes and fisher classifiers
from typing import List, Dict, Tuple, Callable
import csv
import os
# Any results you write to the current directory are saved as output.

row_limit: int = 3000
# (category, text)
train_data: List[Tuple[str, str]] = []
test_data: List[Tuple[str, str]] = []
reader = csv.reader(open('../input/spam.csv', newline='', encoding='latin-1'))
i: int = 0
for row in reader:
    i += 1
    if i == 1:
        continue
    if i < row_limit + 2:
        train_data.append((row[0], row[1]))
    else:
        test_data.append((row[0], row[1]))

        
def generate_text_vector(
    text_list: List[str],
    min_frequency: float=0.1,
    max_frequency: float=0.5,
    getfeatures: Callable[[str], Dict[str, int]] = tc.getwords,
) -> Callable[[str], Dict[str, int]]:
    '''
    Get a Dict of words from array of text between max and mix frequency
    '''
    text_list_len: int = len(text_list)
    text_vector: List[str] = []
    # Dict of uniq words for each text
    apcount: Dict[str, int] = {}

    for text in text_list:
        for word, count in getfeatures(text).items():
            apcount.setdefault(word, 0)
            if count > 0:
                apcount[word] += 1

    # Dict of uniq words for all texts
    # all words are between max and mix frequency
    fr_set: List[float] = []
    for word, count in apcount.items():
        frac = float(float(count) / float(text_list_len))
        fr_set.append(frac)
        if frac > min_frequency and frac < max_frequency:
            text_vector.append(word)

    def get_match_words(text: str) -> Dict[str, int]:
        new_dict: Dict[str, int] = {}
        for text, i in getfeatures(text).items():
            if text in text_vector:
                new_dict[text] = i

        return new_dict
    return get_match_words


def experiment(
    classifier,
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    show_not_equal_ham: bool = False,
    show_not_equal_spam: bool = False
) -> None:
    # main function train, test and print result
    [classifier.train(text, cat) for (cat, text) in train_data]
    stat: Dict[str, int] = {
        'equal': 0,
        'not_equal': 0,
        'not_equal_ham': 0,
        'not_equal_spam': 0
    }
    for (cat, text) in test_data:
        predict = classifier.classify(text)
        if cat == predict:
            stat['equal'] += 1
        else:
            stat['not_equal'] += 1
            if cat == 'ham':
                stat['not_equal_ham'] += 1
                if show_not_equal_ham == True:
                    print(text)
            else:
                stat['not_equal_spam'] += 1
                if show_not_equal_spam == True:
                    print(text)
    print(stat)
    ac: float = stat['equal'] / \
        ((stat['equal']+stat['not_equal'])/100)
    print('Accuracy: %f' % ac)
    return None

train_texts: List[str] = [text for _c, text in train_data]
getweatures_ignoring_common_words: Callable[[str], Dict[str, int]] 
getweatures_ignoring_common_words = generate_text_vector(
    train_texts, 
    min_frequency=0.0003,
    max_frequency=0.05
)


print('Experiment #3.1 - fisherclassifier with word frequency limit and cat-minimum')
cl = tc.fisherclassifier(getfeatures=getweatures_ignoring_common_words)
cl.setminimum('spam', 0.929)
experiment(cl, train_data, test_data)

print('Experiment #3.2 - fisherclassifier with cat-minimum')
cl = tc.fisherclassifier()
cl.setminimum('spam', 0.949)
experiment(cl, train_data, test_data)
