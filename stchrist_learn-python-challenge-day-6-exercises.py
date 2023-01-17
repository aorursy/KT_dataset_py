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
help(str)
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    #pass
    return (len(zip_code)==5) and zip_code.isdigit()

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
    #pass
    results = []
    for i in range(len(doc_list)):
        if keyword.lower() in doc_list[i].lower().split():
            results.append(i)
        elif keyword.lower() + " " in doc_list[i].lower():
            results.append(i)
        elif keyword.lower() + "." in doc_list[i].lower():
            results.append(i)
        elif keyword.lower() + "," in doc_list[i].lower():
            results.append(i)
    return results


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
    #pass
    results = {}
    for keyword in keywords:
        results[keyword] = word_search(doc_list, keyword)
    return results
q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    #pass
    radius = height//2
    diamond = ''
#     for i in range(height//2):
#         for top_h in range(i+1):
#             diamond += "/"
#         for top_w in range(i+1):
#             diamond += "\\"
#         diamond += '\n'
        
#     for i in range(height//2):
#         for j in range(i+1):
#             diamond += "\\"
#         for j in range(i+1):
#             diamond += "/"
#         diamond += "\n"
#     return diamond
    for row in range(height):
        if row < radius:
            l = '/'
            r = '\\'
            num_chars = row + 1
        else:
            l = '\\'
            r = '/'
            num_chars = height - row
        diamond += (l * num_chars).rjust(radius) + (r * num_chars).ljust(radius) + '\n' 
    return diamond[ :-1]

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
    #pass
    counts = {}
    for i in range(1, len(history)):
        if history[i-1] not in counts:
            counts[history[i-1]] = {}
        if history[i] not in counts[history[i-1]]:
            counts[history[i-1]][history[i]] = 0
        counts[history[i-1]][history[i]] += 1
    
    results = {}
    for k, v in counts.items():
        total = sum(v.values())
        sub_probs = {next_spin: next_count/total for next_spin, next_count in v.items()}
        results[k] = sub_probs
    
    return results
        


q5.check()
q5.solution()