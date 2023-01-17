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
    #pass
    if len(zip_code) == 5:
        if zip_code.isnumeric() == True:
            return True
        else:
            return False
    else:
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
    #pass
    ret_ind = []
    for i, doc in enumerate(doc_list):
        terms = doc.split()
        norm = [term.strip('.,').lower() for term in terms]
        if keyword.lower() in norm:
            ret_ind.append(i)
    return ret_ind

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
    ret = {}
    for key in keywords:
        ind = word_search(doc_list,key)
        ret[key] = ind
    return ret

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    a = "/"
    b = "\\"
    d1 = ""
    d2 = ""
    h = int(height/2)
    for i in range(1, h + 1):
        d1 = d1 + " "*(h - i) + a*i + b*i + " "*(h - i) + "\n"
    for i in range(h, 0, -1):
        d2 = d2 + " "*(h - i) + b*i + a*i + " "*(h - i) + "\n"
    #d = '"""' + d1 + d2.rstrip('\n') + '"""'
    d = d1 + d2.rstrip('\n')
    return d

st = diamond(10)
print(st)


#q4.check()
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
    his ={}
    for i,v in enumerate(history):
        if v not in his:
            his[v] = {}
        if i < (len(history) - 1):
            his[v][history[i+1]] = 0
    for k in his:
        l = len(his[k])
        for x in his[k]:
            his[k][x] = 1 / l
    return his

#c = conditional_roulette_probs([1, 3, 1, 5, 1])
#print(c)

q5.check()
q5.solution()