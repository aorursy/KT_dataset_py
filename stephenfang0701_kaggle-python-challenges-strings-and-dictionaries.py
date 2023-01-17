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
help(str.isdigit)
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return len(zip_code)==5 and zip_code.isdigit()

q1.check()
help(str.translate)
help(str.maketrans)
import string
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
    result = list()
    keyword = keyword.lower()
    for i, doc in enumerate(doc_list):
        # remove punctuation, convert string into lowercase and split into a list of words
        # Alternative: s.rstrip(',.') can be used to remove the punctuation when processing each word 
        words = doc.translate(str.maketrans('', '', string.punctuation)).lower().split()
        for word in words:
            if keyword == word:
                result.append(i)
                break
    return result
q2.check()
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
    keyword_to_indice = dict()
    for keyword in keywords:
        indice = word_search(doc_list, keyword)
        keyword_to_indice[keyword] = indice
    return keyword_to_indice
q3.check()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    slash, bslash = '/', '\\'
    s = ''
    if height < 0 or height%2 == 1:
        return s
    
    num = 0
    for i in range(height):
        if i < height//2:
            num = i+1
            line = ''.join([slash*num, bslash*num]).center(height) + '\n'
        else:
            num = height-i
            line = ''.join([bslash*num, slash*num]).center(height) + '\n'
        s += line
    return s

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
from collections import Counter, defaultdict
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    counts = defaultdict(Counter)
    for i in range(len(history)-1):
        prev, roll = history[i], history[i+1]
        counts[prev][roll] += 1
    
    probs = dict()
    for prev, rolls in counts.items():
        total = sum(rolls.values())
        prob = {roll: count/total for roll, count in rolls.items()}
        probs[prev] = prob
    return probs

q5.check()