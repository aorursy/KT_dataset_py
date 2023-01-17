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
    if str.isdecimal(zip_code) == True:
        if len(zip_code) == 5:
            return True
    return False

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
    results = []
    for i in range(len(doc_list)):
        myIndex = doc_list[i].lower().find(" " + keyword)
        if int(myIndex) > 0:
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
    multi_results = {}
    
    for findword in keywords:
        multi_results[findword] = word_search(doc_list, findword)
    return multi_results

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    half = height//2
    top = 1
    bottom = half
    myTop = ""
    myBottom = ""

    while top <= half:
        myTop = (myTop + ("/") * top + ("\\") * top).center(height) + "\n"
        top += 1
    
    while bottom >= 1:
        myBottom = (myBottom +  (("\\") * bottom) + (("/") * bottom)).center(height," ") + "\n"
        bottom -= 1
        
    myDiamond = myTop + myBottom
    return(myDiamond[:-1])

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
    counts = {}
    for i in range(1, len(history)):
        roll, prev = history[i], history[i-1]
        if prev not in counts:
            counts[prev] = {}
        if roll not in counts[prev]:
            counts[prev][roll] = 0
        counts[prev][roll] += 1

    # We have the counts, but still need to turn them into probabilities
    probs = {}
    for prev, nexts in counts.items():
        # The total spins that landed on prev (not counting the very last spin)
        total = sum(nexts.values())
        sub_probs = {next_spin: next_count/total
                for next_spin, next_count in nexts.items()}
        probs[prev] = sub_probs
    return probs

q5.check()
q5.solution()