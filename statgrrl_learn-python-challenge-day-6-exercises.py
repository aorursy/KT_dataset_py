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
    return len(zip_code) == 5 and zip_code.isdigit()

q1.check()
#q1.hint()
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
    result = []
    for i, doc in enumerate(doc_list):
        doc = ' ' + doc.lower().replace(',', ' ').replace('.', ' ') + ' '
        if doc.find(' ' + keyword.lower() + ' ') != -1:
            result.append(i)
    return result

q2.check()
#q2.hint()
#q2.solution()
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
    result = {}
    for key in keywords:
        result[key] = word_search(doc_list, key)
    return result

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    mid = height // 2
    s = ''

    top = list(range(1, mid + 1))
    for i in top: 
        if i != 1: s += '\n'
        s += ('/' * i + '\\' * i).center(height)

    bottom = list(range(mid, 0, -1))
    for i in bottom: 
        s += '\n' + ('\\' * i + '/' * i).center(height)

    return s

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    spins = len(history)
    nums = list(set(history))
    result = {}
    for key in nums:
        occ = history.count(key)
        key_hist = list(history)
        key_follow = []
        for i in range(occ):
            occ_ind = key_hist.index(key)
            if occ_ind != len(key_hist) - 1:
                key_follow.append(key_hist[occ_ind + 1])
                key_hist.remove(key)
        key_follow_nums = list(set(key_follow))
        dict_follow = {}
        for num in key_follow_nums:
            dict_follow[num] = float(key_follow.count(num)) / len(key_follow)
        result[key] = dict_follow
    return result


q5.check()
#q5.solution()