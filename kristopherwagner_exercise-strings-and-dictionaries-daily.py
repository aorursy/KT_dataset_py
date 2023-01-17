from learntools.core import binder; binder.bind(globals())

from learntools.python.ex6 import *

print('Setup complete.')
a = ""

length = 0

q0.a.check()
b = "it's ok"

length = 7

q0.b.check()
c = 'it\'s ok'

length = 7

q0.c.check()
d = """hey"""

length = 3

q0.d.check()
e = '\n'

length = 1

q0.e.check()
def is_valid_zip(zip_code):

    """Returns whether the input string is a valid (5 digit) zip code

    """

    return 5 == sum([str.isdigit(digit) for digit in zip_code])



q1.check()
q1.hint()

#q1.solution()
def word_search(doc_list, keyword):

    """

    Takes a list of documents (each document is a string) and a keyword. 

    Returns list of the index values into the original list for all documents 

    containing the keyword.



    Example:

    doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]

    >>> word_search(doc_list, 'casino')

    >>> [0]

    """

    ret = []

    for i, sentence in enumerate(doc_list):

        words = sentence.split()

        clean = [word.lower().strip(',?.') for word in words]

        if keyword in clean:

            ret.append(i)

    return ret





q2.check()
q2.hint()

q2.solution()
def multi_word_search(doc_list, keywords):

    """

    Takes list of documents (each document is a string) and a list of keywords.  

    Returns a dictionary where each key is a keyword, and the value is a list of indices

    (from doc_list) of the documents containing that keyword



    >>> doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]

    >>> keywords = ['casino', 'they']

    >>> multi_word_search(doc_list, keywords)

    {'casino': [0, 1], 'they': [1]}

    """

    ret = {}

    for word in keywords:

        ret[word] = word_search(doc_list, word)

    return ret



q3.check()
#q3.solution()