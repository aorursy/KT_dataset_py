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
    return len(zip_code) == 5 and zip_code.isdecimal()

q1.check()
#q1.hint()
#q1.solution()
import re

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
    pattern = re.compile(r'\b({0})\b'.format(keyword), flags=re.IGNORECASE)
    match_list = []
    for i, doc in enumerate(doc_list):
        if pattern.search(doc) != None:
            match_list.append(i)
    return match_list

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
    match_dict = {}
    for key in keywords:
        match_dict[key] = word_search(doc_list, key)
    return match_dict

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond = ""
    half = height//2
    for i in range(half):
        diamond = ' ' * i + '/' * (half-i) + '\\' * (half-i) + ' ' * i + '\n' + diamond
        diamond = diamond + ' ' * i + '\\' * (half-i) + '/' * (half-i) + ' ' * i + '\n'
    return diamond

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
    prob_dict = {}
    for i in range(len(history)-1):
        if history[i] not in prob_dict.keys():
            prob_dict[history[i]] = {0: 0}
        if history[i+1] not in prob_dict[history[i]].keys():
            prob_dict[history[i]][history[i+1]] = 1
            prob_dict[history[i]][0] += 1
        else:
            prob_dict[history[i]][history[i+1]] += 1
            prob_dict[history[i]][0] += 1
    
    for key in prob_dict:
        for subkey in prob_dict[key]:
            if (subkey != 0):
                prob_dict[key][subkey] = prob_dict[key][subkey] / prob_dict[key][0]
        del prob_dict[key][0]
    return prob_dict

q5.check()
#q5.solution()