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
        return zip_code.isdigit()

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
    words_no_puncs = []
    docs_with_keywords = []
    for doc_index, document in enumerate(doc_list):
        for word in document.split(' '):
            word_no_puncs = word.lower().rstrip('?,.')
            if word_no_puncs == keyword:
                docs_with_keywords.append(doc_index)
                
    return docs_with_keywords

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
    keywords_with_matches = {}
    matches = []
    for keyword in keywords:
        matches = word_search(doc_list, keyword)
        keywords_with_matches[keyword] = matches
            
    return keywords_with_matches

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    midpoint = (height//2)+1
    line = []
    spaces = []
    left_diamond = []
    right_diamond = []
    diamond = ""
    
    for row in range(1,midpoint):
        line = []
        spaces = [" " for i in range(1,midpoint-row)]
        left_diamond = ["/" for i in range (1, row+1)]
        right_diamond = ["\\" for i in range (1, row+1)]
        line.append(''.join(spaces))
        line.append(''.join(left_diamond))
        line.append(''.join(right_diamond))
        line.append(''.join(spaces))
        #         print("".join(line))
        diamond += "".join(line) + "\n"

    line = []
    spaces = []
    left_diamond = []
    right_diamond = []    
    
    for row in range(midpoint, 1, -1):
         line = []
         spaces = [" " for i in range(midpoint-row, 0, -1)]
         left_diamond = ["\\" for i in range (row, 1, -1)]
         right_diamond = ["/" for i in range (row, 1, -1)]
         line.append(''.join(spaces))
         line.append(''.join(left_diamond))
         line.append(''.join(right_diamond))
         line.append(''.join(spaces))
#          print("".join(line))
         diamond += "".join(line) + "\n"
    
    return diamond

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
    pass


q5.check()
q5.solution()