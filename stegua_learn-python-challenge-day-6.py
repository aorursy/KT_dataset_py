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
    i = 0
    for c in zip_code:
        if c < '0' or c > '9' or i > 4:
            return False
        else:
            i += 1
    return i == 5

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
    Ls = []
    for i, doc in enumerate(doc_list):
        for word in doc.replace(',','').replace('.','').split(' '):            
            if keyword.lower() == word.lower():
                Ls.append(i)
                break
    return Ls


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
    def UpLeft(l):
        return ''.join([' ' for _ in range(int(height/2)-l)] + ['/' for _ in range(l)])
    def UpRight(l):
        return ''.join(['\\' for _ in range(l)] + [' ' for _ in range(int(height/2)-l)])
    def DownLeft(l):
        return ''.join([' ' for _ in range(int(height/2)-l)] + ['\\' for _ in range(l)])
    def DownRight(l):
        return ''.join(['/' for _ in range(l)] + [' ' for _ in range(int(height/2)-l)])

    d = ''
    for l in range(1,height//2+1):
        d += UpLeft(l)+UpRight(l)+'\n'
    for l in reversed(range(1,height//2+1)):
        d += DownLeft(l)+DownRight(l)+'\n'
        
    return d[:-1]


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
    D = {}
    T = {}
    for i in range(len(history)-1):
        D[history[i]] = D.get(history[i], {})
        D[history[i]][history[i+1]] = D[history[i]].get(history[i+1], 0)+1
        T[history[i]] = T.get(history[i], 0) + 1
        
    for i in T:
        for v in D[i]:
            D[i][v] = D[i][v]/T[i]
    return D


q5.check()
q5.solution()