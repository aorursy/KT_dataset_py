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
    return (zip_code.isnumeric()) and (len(zip_code) == 5)

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
    indices = []
    for i , document in enumerate(doc_list):
        normal = document.lower().split()
        j = 0
        for word in normal:
            normal[j] = word.strip(".,")
            j += 1
        if keyword in normal:
            indices.append(i)
    return indices

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
    locations = {}
    for keyword in keywords:
        document_indices = word_search(doc_list, keyword)
        locations[keyword] = document_indices
    return locations

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    half_l = int(height / 2)
    start = 1
    space_start = half_l - 1
    dia = ""
    for _ in range(half_l):
        spaces = " " * space_start
        fslash = "/" * start
        bslash = "\\" * start
        space_start -= 1
        start += 1
        dia = dia + spaces + fslash + bslash + "\n"
    half_l = int(height / 2)
    start = 0
    for _ in range(half_l):
        spaces = " " * start
        fslash = "/" * half_l
        bslash = "\\" * half_l
        half_l -= 1
        start += 1
        dia = dia + spaces + bslash + fslash + "\n"
    return dia
        


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
    count = {}
    for i in range(1,len(history)):
        prev,roll = history[i-1],history[i]
        if prev not in count:
            count[prev] = {}
        if roll not in count[prev]:
            count[prev][roll] = 1
        else:
            count[prev][roll] += 1

    for key, value in count.items():
        total = sum(value.values())
        for sub_key, sub_value in value.items():
            value[sub_key] = sub_value / total
            
    
    return count


q5.check()
#q5.solution()