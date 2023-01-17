# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
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
    check = []
    check.append(len(zip_code) == 5)
    for c in zip_code:
        if c in '0123456789':
            check.append(True)
        else:
            check.append(False)
    return (sum(check) == 6)

q1.check()
q1.hint()
q1.solution()
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
    indices = []
    for i, doc in enumerate(doc_list):
        for word in doc.split(' '):
            if keyword.lower() == word.strip(',.\'\";:*&^%$#@!~`{}[]|\\<>-_+=').lower():
                indices.append(i)
    return indices


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
    lookup = dict([(keyword.lower(), []) for keyword in keywords])
    for i, doc in enumerate(doc_list):
        tokens = doc.split()
        normalized = [token.rstrip('.').lower() for token in tokens]
        for keyword in keywords:
            if keyword.lower() in normalized:
                lookup[keyword.lower()].append(i)
    return lookup

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    half_width = height // 2
    diamond_text = u''
    for j in range(1,half_width+1):
        diamond_text += ' ' * (half_width - j)
        diamond_text += '/' * j
        diamond_text += '\\' * j
        diamond_text += ' ' * (half_width - j)
        diamond_text += '\n'
    for j in range(half_width):
        diamond_text += ' ' * j 
        diamond_text += '\\' * (half_width - j)
        diamond_text += '/' * (half_width - j)
        diamond_text += ' ' * j
        if j < half_width-1:
            diamond_text += '\n'
    return diamond_text

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
q4.solution()
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    d = {}
    for i in set(history[:-1]):
        d.setdefault(i, [])
    for i, j in zip(history, history[1:]):
        d[i].append(j)
    p = {}
    for k in d:
        p.setdefault(k, {})
    for k in d:
        for i in set(d[k]):
            total = len(d[k])
            count = 0
            for j in d[k]:
                if i == j:
                    count += 1
            p[k].update({i: count/total})
    return p

q5.check()
q5.solution()