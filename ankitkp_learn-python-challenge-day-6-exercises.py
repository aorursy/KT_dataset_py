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
    return zip_code.isnumeric() and len(zip_code) == 5

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
    indices = []
    for text in doc_list:
        words = text.lower().split()
        for word in words:
            if keyword.lower() == word.rstrip('.,'):
                indices.append(doc_list.index(text))
        
    return indices
            


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
    search_dict= {}
    for keyword in keywords:
       ind_list = word_search(doc_list, keyword)
       search_dict[keyword] = ind_list
        
    return search_dict
        

q3.check()
#q3.solution()
height = 10
fs = '/'
bs = '\\'
i = int(height / 2)
j = 0
for _ in range(int(height / 2)):  
  print(fs.rjust(i), bs.ljust(2*i - 1))
  i -= 1
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    pass


q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
#q4.solution()
history = [5, 1, 3, 1, 2, 1, 3, 3, 5, 1, 2]
main_dict ={}
#count_dict = {}
for i in range(len(history) - 1):
    if not history[i] in main_dict:
        main_dict[history[i]] ={}
        main_dict[history[i]][history[i + 1]] = ""
    else:
        main_dict[history[i]][history[i + 1]] = ""
            

for a in main_dict:
    la = len(main_dict[a])
    for b in main_dict[a]:
        print(main_dict[a][b])
        main_dict[a][b] = float(1 / la)
    
print(main_dict)
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    main_dict ={}
    
    for i in range(len(history) - 1):
        if not history[i] in main_dict:
            main_dict[history[i]] ={}
            main_dict[history[i]][history[i + 1]] = ""
        else:
            main_dict[history[i]][history[i + 1]] = ""
            
        
    for a in main_dict:
        la = len(main_dict[a])
        for b in main_dict[a]:
            main_dict[a][b] = float(1 / la)
        
    
    return main_dict
        


q5.check()
q5.solution()