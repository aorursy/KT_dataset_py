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
    if len(zip_code) != 5:
        return False
    else:
        for ch in zip_code:
            if ch not in "1234567890":
                return False
        return True

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
    solution = []
    i = 0
    for doc in doc_list:
        for word in doc.split():
            if word.lower() == keyword or word.lower() == keyword + "." or word.lower() == keyword + ",":
                solution.append(i)
        i = i + 1
    return(solution)
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
    solution_dict = {}
    for word in keywords:
        solution_dict[word] = word_search(doc_list, word)
    return solution_dict

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    s = ''
    for i in range(int(height/2)+1):
        s = s + ' ' * (int(height/2)-i) + '/' * i + '\\' * i + "\n"
    for i in range(int(height/2)+1):
        s = s + ' ' * i + '\\' * (int(height/2)-i) + '/' * (int(height/2)-i) + "\n"
    return(s[1:-1])

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
    mydict = {}
    for i in range(len(history)-1):
        if history[i] in mydict.keys():
            mydict[history[i]].append(history[i+1])
        else:
            mydict[history[i]] = [history[i+1]]
    
    result = {}
    for el in mydict:
        #mydict[el].sort()
        occ = {}
        for occurence in mydict[el]:
            if occurence in occ.keys():
                occ[occurence] += 1
            else:
                occ[occurence] = 1
        
        proba = {}
        for possibility in occ:
            proba[possibility] = occ[possibility] / sum(occ.values())
        
        result[el] = proba
    return(result)


q5.check()
q5.solution()