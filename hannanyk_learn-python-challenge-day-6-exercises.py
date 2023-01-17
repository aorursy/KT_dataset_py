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
    if (len(zip_code) == 5) and zip_code.isdigit():
        return True
    return False

q1.check()
#q1.hint()
#q1.solution()
help(str)
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
    lower_doc_list = []
    for string in doc_list:
        lower_doc_list.append(' {} '.format(string.lower().replace('.', ' ').replace(',', ' ')))
    lower_keyword = keyword.lower()
    return [ind for ind in range(len(lower_doc_list)) if lower_doc_list[ind].find(' {} '.format(lower_keyword))>= 0]

q2.check()
#q2.hint()
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
    multi_search = {keyword: word_search(doc_list, keyword) for keyword in keywords}
    return multi_search
        

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    if height % 2:
        return print('Height must be an even integer!')
    single_height = int(height / 2)
    diamond_list = []
    for line in range(single_height):
        line += 1
        diamond_list.append((single_height - line) * ' ' + (line * '/') + (line * '\\'))
    for line in range(single_height,0,-1):
        diamond_list.append((single_height - line) * ' ' + (line * '\\') + (line * '/'))
    diamond_string = '\n'.join(diamond_list)
    return diamond_string


q4.check()
print(diamond(4))
#q4.hint()
# q4.solution()
def unique(list_object):
    unique_list = []
    for element in list_object:
        if not element in unique_list:
            unique_list.append(element)
    return unique_list
    
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    keys = unique(history)
    roulette_dict = {}
    for key in keys:
        keys_n2 = []
        for appearance in [ind for ind in range(len(history[:-1])) if history[ind] == key]:
            keys_n2.append(history[appearance + 1])
        mini_dict = {}
        if not len(unique(keys_n2)) == len(keys_n2):
            mini_dict = {key_n2: keys_n2.count(key_n2) / len(keys_n2) for key_n2 in keys_n2}
        else:
            mini_dict = {key_n2: 1/(len(keys_n2)) for key_n2 in keys_n2}
        roulette_dict[key] = mini_dict
    return roulette_dict


q5.check()
q5.solution()
