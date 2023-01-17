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
#help(str)
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
    index_list = []
    for i, s in enumerate(doc_list):
        s = s.lower().split()
        cleaned_s = [sub_s.rstrip('.,') for sub_s in s]
        if keyword.lower() in cleaned_s:
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
    keyword_dict = {}
    for keyword in keywords:
        keyword_dict[keyword] = word_search(doc_list, keyword)
    return keyword_dict

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_string = ''
    l = '/'
    r = '\\'
    #draw upper half
    for i in range(1, int(height/2) + 1):
        temp_string = l * i + r * i
        temp_string = temp_string.center(height)
        diamond_string += temp_string + '\n'
    #draw lower half  
    l, r = r, l
    for i in range(int(height/2), 0, -1):
        temp_string = l * i + r * i
        temp_string = temp_string.center(height)
        diamond_string += temp_string + '\n'
    return diamond_string[:-1]

print(diamond(12))
q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
help(dict)
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    roulette_dict = {}
    #count
    for i in range(len(history) - 1):
        number = history[i]
        next_number = history[i + 1]
        if number not in roulette_dict.keys():
            roulette_dict[number] = {}
        if next_number not in roulette_dict[number].keys():     
            roulette_dict[number][next_number] = 0
        roulette_dict[number][next_number] += 1
        
    #calculate probs
    for key in roulette_dict.keys():
        total_counts = sum(roulette_dict[key].values())
        for inner_key in roulette_dict[key].keys():
            roulette_dict[key][inner_key] /= total_counts
        
    
    return roulette_dict

#print(conditional_roulette_probs([1, 3, 1, 5, 1]))
q5.check()
#q5.solution()