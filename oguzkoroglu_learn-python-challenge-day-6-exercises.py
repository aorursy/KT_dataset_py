# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex6 import *
print('Setup complete.')
a = ""
length = len(a)
q0.a.check()
b = "it's ok"
length = len(b)
q0.b.check()
c = 'it\'s ok'
length = len(c)
q0.c.check()
d = """hey"""
length = len(d)
q0.d.check()
e = '\n'
length = len(e)
print(length)
q0.e.check()
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return (zip_code.isdigit()) and len(zip_code) == 5

#help(str)
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
    lst = []
    x = 0
    for d in doc_list:
        for key in d.split():
            keyl = key.lower().strip(',').strip('?').strip('.')
            #print (keyl)
            if (keyl == keyword.lower()):
                lst.append(x)
                break
        x += 1                                            
    return lst

#help(str)
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
    myDict = {}
    
    for keyword in keywords:
        res = word_search(doc_list,keyword)
        myDict[keyword] = res
        
    return myDict

q3.check()
#help(dict)
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """    
    height //= 2
    
    stri = ""
    for h in range(height):
        lenRow = (h+1)*2        
        inc = ((height)*2-lenRow)//2
        stri += inc*' '        
        for _ in range(lenRow//2):
            stri += "/"
        for _ in range(lenRow//2):
            stri += "\\"
        stri += "\n"
        
    for h in range(height):
        h = (height-1)-h
        lenRow = (h+1)*2        
        inc = ((height)*2-lenRow)//2
        stri += inc*' '
        for _ in range(lenRow//2):
            stri += "\\"
        for _ in range(lenRow//2):
            stri += "/"
        stri += "\n"
            
    return stri

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
            
conditional_roulette_probs([1, 3, 1, 5, 1])
q5.check()
q5.solution()