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
    if len(zip_code) == 5 and zip_code.isdigit():
        return True
    else: return False

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
    l = []
    ps = [' ', '.', ',']
    for i in range(len(doc_list)):
        for p1 in ps:
            for p2 in ps:
                k = p1 + keyword + p2
                d = ' ' + doc_list[i] + ' '
                if d.upper().find(k.upper()) >= 0:
                    l.append(i)
    return l

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
    d = {}
    for k in keywords:
        d[k] = word_search(doc_list, k)
    return d

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    f1, f2 = [], []
    h2 = int(height/2)
    s = ' '
    sf = '/'
    sb = '\\'
    for i in range(1, h2 + 1):
        l1 = sf*i + sb*i
        l2 = sb*i + sf*i
        b = int((height - len(l1))/2)
        l1 = s*b + l1
        l2 = s*b + l2
        f1.append(l1)
        f2.append(l2)
    f2.reverse()
    f = ''
    for i in range(h2): f = f +f1[i] + '\n'
    for i in range(h2): f = f +f2[i] + '\n'
    return f

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
    ks = set(history)
    d={}
    for k in ks: d[k] = []
    for i in range(len(history)-1): d[history[i]].append(history[i+1])
    for k,vs in d.items():
        d2 = {}
        for v in vs:
            d2[v] = vs.count(v)/len(vs)
        d[k] = d2
    return d

q5.check()

q5.solution()