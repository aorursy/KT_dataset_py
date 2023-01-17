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
    return True if len(zip_code) == 5 and zip_code.isdigit() else False

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
    res, i =[], 0
    for i, doc in enumerate(doc_list):
        if keyword.lower() in [s.strip(',.') for s in doc.lower().split()]:
            res.append(i)
        i += 1
    return res    

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
    dic = {}
    for k in keywords:
        if dic.setdefault(k) == None:
            dic[k] = word_search(doc_list, k)
    return dic

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    s = ''
    for i in range(height):
        if i < height / 2:
            s += (height // 2 - 1 - i) * ' ' + (i + 1) * '/' + (i + 1) * '\\' + (height // 2 - 1 - i) * ' '
        else:
            s += (i - height // 2) * ' ' + (height - i) * '\\' + (height - i) * '/' + (i - height // 2) * ' '
        if i < height - 1:
            s += '\n'
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
    summ = {}
    for i in range(len(history)-1):
        aft, bef = history[i+1], history[i]
        if bef not in summ:
            summ[bef] = {}
        if aft not in summ[bef]:
            summ[bef][aft] = 0
        summ[bef][aft] += 1

    prob = {}
    for bef, aft in summ.items():
        sum_all = sum(aft.values())
        local_prob = {k: v/sum_all
                for k, v in aft.items()}
        prob[bef] = local_prob
    return prob

q5.check()
#q5.solution()