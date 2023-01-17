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
    indices = []
    #for document_index in range(len(doc_list)):
    #    if ( keyword.lower()+" " in doc_list[document_index].lower() ) or ( " " + keyword.lower() in doc_list[document_index].lower() ):
    #        indices += [document_index]
        
        ## better .. use string functions
    for i , doc in enumerate(doc_list):
        words_list = doc.split()
        words_list = [word.rstrip(',.').lower() for word in words_list]
        if keyword in words_list:
            indices += [i]

    return indices

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
    keywords_indices = {}
    for i , word in enumerate(keywords):
        keywords_indices[word] = word_search(doc_list, word)
        
    return keywords_indices

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_str = "";
    
    for olc in range(1, height // 2+1):
        for ilc in range (height//2 - olc):
            diamond_str += " "
        for ilc in range (height//2 - olc, height//2):
            diamond_str += "/"
        for ilc in range(height//2 - olc, height//2):
            diamond_str += "\\"
        diamond_str += "\n"
    
    for olc in range(1, height // 2+1):
        for ilc in range (olc-1):
            diamond_str += " "
        for ilc in range (height//2 - olc+1):
            diamond_str += "\\"
        for ilc in range(height//2 - olc+1):
            diamond_str += "/"
        diamond_str += "\n"

    return diamond_str
            
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
    prob_dict = {}
    for n1 in history:
        next_items = {}
        for i , n1_ in enumerate(history):
            if n1_ == n1 and i+1 < len(history):
                if history[i+1] not in next_items:
                    next_items[history[i+1]] = 0
                next_items[history[i+1]] += 1
            
            ## Correct answer : if [1,2,3,1,2,5,1,4] is the list, then prob of occurence of 2 after 1 is 2/3 (2 : 2, 1 : 4)
            prob = {}
            sum_all_possibilities = sum(next_items.values())
            
            for item, count in next_items.items():
                prob[item] = count/sum_all_possibilities
            prob_dict[n1] = prob 
    
    return prob_dict
    

q5.check()
#q5.solution()