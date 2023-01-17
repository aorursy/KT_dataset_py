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
    return zip_code.isdecimal() and len(zip_code) == 5

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
    ret = []
    for i,doc in enumerate(doc_list):
        processed = doc.replace(".","").replace(",","").lower().split(" ")
        if keyword.lower() in processed:
            ret.append(i)
    
    return ret


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
    ret_dict = {}
    
    for keyw in keywords:
        ret_dict[keyw] = word_search(doc_list, keyw)
    
    return ret_dict

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_str = []
    for i in range(1,(height//2+1)):
        spaces = (height - i*2)//2 
        lines = i
        line_str = " " * spaces + "/" * lines + "\\" * lines + " " * spaces + "\n"
        diamond_str.append(line_str)
        
    for i in range((height//2),0,-1):
        spaces = (height - i*2)//2 
        lines = i
        line_str = " " * spaces + "\\" * lines + "/" * lines + " " * spaces + "\n"           
        diamond_str.append(line_str)
        
    return "".join(diamond_str)

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
    ret_dict = {}
    for i in range(len(history)-1):
        crnt,nxt = history[i], history[i+1]
        if crnt not in ret_dict:
            ret_dict[crnt] = {}
            ret_dict[crnt][nxt] = 1
        else:
            if nxt not in ret_dict[crnt]:
                ret_dict[crnt][nxt] = 1
            else:
                ret_dict[crnt][nxt] += 1
                
    for key in ret_dict.keys():
        cnt = history[:-1].count(key)
        for nkey in ret_dict[key]:
            ret_dict[key][nkey] /= float(cnt)
    
    return ret_dict    

q5.check()
q5.solution()