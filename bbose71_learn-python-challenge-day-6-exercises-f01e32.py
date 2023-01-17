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
    else:
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
    result = []
    for i in range(len(doc_list)):
        words = doc_list[i].lower().split()
        for word in words :
            if word.strip(".,") == keyword.lower():
                result.append(i)
    return result
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
    results = {}
    for keyword in keywords :
        this_result = word_search(doc_list, keyword)
        results[keyword] = this_result
    return results 
q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond = ""
    for i in range(height//2):
        #diamond = diamond + (" "*((height//2)-(i+1)) + "/"*(i+1) + "\\"*(i+1) + "\n")
        line = "/"*(i+1) + "\\"*(i+1)
        diamond = diamond + line.center(height) + "\n"
    for i in range(height//2): 
            j = height//2 - i
            line = "\\"*(j) + "/"*(j)
            diamond = diamond + line.center(height) + "\n"
    return diamond

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
help(str.center)
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
    for value in list(set(history)) :
        matches = []
        temp_res = {}
        for i in range(len(history)-1):
            if history[i]==value :
                matches.append(history[i+1])
        if len(matches) >0 :
            uniq_matches = list(set(matches))
            for match in uniq_matches:
                count = 0
                for j in matches : 
                    if j == match:
                        count +=1 
                temp_res[match] = count/len(matches)
        result[value] = temp_res
    return result
        
        


q5.check()
q5.solution()