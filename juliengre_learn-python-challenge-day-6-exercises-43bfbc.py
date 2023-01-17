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
    return zip_code.isdecimal() and len(zip_code) == 5

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
    result = []
    for i in range(len(doc_list)):
        doc = doc_list[i].lower().replace('.', '').replace(',' , '')
        for word in doc.split():
            #print('comparing {} and {}'.format(word, keyword.lower()))
            if keyword.lower() == word:
                result.append(i)
                break
    return result            

q2.check()
#q2.hint()
q2.solution()
for i, doc in enumerate({"just a little test", 'and another one'}):
    print("i={}, doc={}".format(i,doc))
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
    res = {}
    for k in keywords:
        res[k] = word_search(doc_list, k)
    return res    

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    r = height // 2
    left = ""
    right = ""
    result = []
    for i in range(r):
        left += "/"
        right += "\\"
        result.append("{}{}".format(left.rjust(r), right))
    left = left.replace("/", "\\")
    right = right.replace("\\", "/")
    for i in range(r):
        result.append("{}{}".format(left.rjust(r), right))
        left = left[:-1]
        right = right[:-1]
    return "\n".join(result)
#diamond(10)

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
    result = {}
    prev = history[0]
    for r in history[1:]:
        if not prev in result:
            result[prev] = {}
        if not r in result[prev]:
            result[prev][r] = 1
        else:
            result[prev][r] += 1
        prev = r
    #print(result)
    for i, liste in result.items():
        total = sum(result[i].values())
        for n, nb in liste.items():
            result[i][n] = nb / total
    return result

#print(conditional_roulette_probs([1, 3, 1, 5, 1]))
q5.check()
q5.solution()