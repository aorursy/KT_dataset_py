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
    if (len(zip_code) == 5):
        return all(str.isdigit(char) for char in str(zip_code))
    return False

q1.check()
#q1.hint()
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
    for index, doc in enumerate(doc_list):
        if any([word.lower() == keyword.lower() for word in doc.replace('.',' ').replace(',',' ').split(' ')]):
            list.append(indices, index)
    return indices


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
    results = {}
    for keyword in keywords:
        results[keyword] = word_search(doc_list, keyword)
    
    return results

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    result_1 = ""
    result_2 = ""
    base = height // 2
    for x in range(base):
        result_1 += ' '*(base - x - 1) + '/'*(x + 1)
        result_1 += '\\'*(x + 1) + ' '*(base - x + 1) + '\n'
        
        result_2 += ' '*(x) + '\\'*(base - x)
        result_2 += '/'*(base - x) + ' '*(x) + '\n'
        
    return result_1 + result_2

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
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
    results = {}
    for index, ball in enumerate(history[:-1]):
        if (not ball in results):   
            results[ball] = [history[index + 1]]
        else:
            results[ball].append(history[index + 1])
                        
    for key, value in results.items():
        results[key] = { occurence : results[key].count(occurence) / len(results[key])  for occurence in results[key] }
    return results

q5.check()
q5.solution()