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
    return len(zip_code) == 5 and zip_code.isnumeric()

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
    found = []
    for doc_number, doc in enumerate(doc_list,start=0):
        filtered_doc = doc.lower()
        for character in [',', '.']: #filter commas and dots
            filtered_doc = filtered_doc.split(character)
            filtered_doc = "".join(filtered_doc)
        if keyword in filtered_doc.split():
            found.append(doc_number)
    
    return found


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
    
    kdict = {}
    for k in keywords:
        kdict[k] = word_search(doc_list,k)
    
    return kdict

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    half_height = height // 2
    dia = ""  
    mond = ""
    
    def diamond_builder(half_str, spc_n, slash_n, upper_half=True):
        if upper_half:
            while spc_n >= 0:
                half_str = half_str + (" " * spc_n) + ('/' * slash_n) + ('\\' * slash_n) + (" " * spc_n) + '\n'
                spc_n = spc_n - 1 
                slash_n = slash_n +1
        else:
            while slash_n >= 0:
                half_str = half_str + (" " * spc_n) + ('\\' * slash_n) + ('/' * slash_n) + (" " * spc_n)
                spc_n = spc_n + 1 
                slash_n = slash_n - 1 
                half_str = half_str + ('\n' if slash_n != 0 else "") 
        return half_str 
    
    dia = diamond_builder("",half_height-1,1)
    mond = diamond_builder("",0,half_height,False)

    return dia+mond


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
    pass


q5.check()
#q5.solution()