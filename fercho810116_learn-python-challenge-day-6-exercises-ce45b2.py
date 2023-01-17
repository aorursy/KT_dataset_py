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
    return True if (zip_code.isdigit() == True) and (len(zip_code) == 5) else False

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
    indexes = []
    prepString = ""
    
    for doc in doc_list:
        #Prepares the strings for review, by changing from upper to all lower. Also, splits 
        prepString = doc.replace(".","")
        prepString = prepString.replace(",","")
        prepString = prepString.lower()
        prepString = prepString.split()
        for word in prepString:
            if word == keyword:
                indexes.append(doc_list.index(doc))
                break
    return indexes
word_search(['The Learn Python Challenge Casino', 'crazy They bought a car, and a horse', 'Casinoville?', "He bought a casino. That's crazy."], "crazy")
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
    keyword_to_indices = {}
    for keyword in keywords:
        keyword_to_indices[keyword] = word_search(documents, keyword)
    return keyword_to_indices
q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    Udiamond = ""
    Ldiamond = ""
    Tdiamond = ""
    #H - 4 
    for line in range(1,int(height/2)+1):
        Udiamond = Udiamond +  int((height/2-line))*" " + line*"/" + line*"\\" + "\n"
        
    for line in range(int(height/2+1), height+1):
        Ldiamond = Ldiamond + int((height-2*(height-line+1))/2)*" " + int(height-line+1)*"\\"+ int(height-line+1)*"/"  + "\n"

    Tdiamond = Udiamond + Ldiamond
    return Tdiamond
print(diamond(10))
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
q5.solution()