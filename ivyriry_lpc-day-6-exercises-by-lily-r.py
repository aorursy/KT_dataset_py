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
    return zip_code.isdigit() and len(zip_code) == 5

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
    for index, doc in enumerate(doc_list):
        doc_matches = sum([word.lower().replace('.', '').replace(',', '') == keyword for word in doc.split(' ')])
        if doc_matches:
            result.append(index)
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
    return {keyword: word_search(doc_list, keyword) for keyword in keywords}

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    multiplier = 0
    mid_point = height / 2
    result = ''
    for index in range(height):
        string = ''
        if(index < mid_point):
            multiplier +=1
            string = '/'*multiplier +'\\'*multiplier
        elif index > mid_point:
            multiplier -= 1
            string = '\\'*multiplier +'/'*multiplier
        else:
            string = '\\'*multiplier +'/'*multiplier
        string = string.center(height)
        result += '\n' if len(result) > 0 else ''
        result += string 
        
    return result
    
print(diamond(4))
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
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
    linked = {}
    last_num = 0
    len_list = len(history)
    for index, number in enumerate(history):         
        if(index + 1 < len_list):
            if number in linked:
                if(history[index+1] not in linked[number]):
                    linked[number].append(history[index+1])
            else:
                linked[number] = [history[index+1]] 
    result = {}
    for key in linked:
        total = len(linked[key])
        temp = {number: 1/total for number in linked[key]}
        linked[key] = temp
    return linked
        
q5.check()
q5.solution()