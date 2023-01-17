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
    return len(zip_code)==5 and zip_code.upper() == zip_code

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
    index_list = []
    for i in range(len(doc_list)):
        if keyword+'.' in doc_list[i].lower() or ' '+keyword in doc_list[i].lower():       
            index_list.append(i)
    return index_list

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
    return {keywords[i]: word_search(doc_list,keywords[i]) for i in range(len(keywords))}

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_string = ''
    for i in range(0,height//2):
        diamond_string = diamond_string + "{0}{1}{2}{0}\n".format(' '*(height//2-1-i),'/'*(i+1),'\\'*(i+1))
   
    for i in range(0,height//2):
        diamond_string = diamond_string + "{0}{1}{2}{0}".format(' '*(i),'\\'*(height//2-i),'/'*(height//2-i))
        if i < height//2-1:
            diamond_string = diamond_string+'\n'  
   
    return diamond_string



q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
import numpy as np
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    dictionary = {}
    once = True
    for i in range(len(history)):
        indices =  list(np.where(np.array(history) == history[i])[0])
        if history[i] in dictionary:
            continue
        
        if history[-1] == history[i] and once == True:
            indices.pop()
            once = False
        
        this = [history[index+1] for index in indices]
        if len(this) > 0:
            dictionary[history[i]] = {key: this.count(key)/len(this) for key in this}
    
    return dictionary


q5.check()
q5.solution()