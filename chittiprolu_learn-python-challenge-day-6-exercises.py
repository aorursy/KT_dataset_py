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
    pass
    type_of_zipcode=[]
    if(len(zip_code)==5):
        return zip_code.isnumeric()
    else:
        return False
                
    
q1.check()
#print(is_valid_zip('12345'))
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
    pass
    ind=[]
    doc_list_lower=[]
    doc_list_split=[]
    keyword_lower=[keyword.lower(),keyword.lower()+'.',keyword.lower()+',']
    for i in range(len(doc_list)):
        doc_list_lower.append(doc_list[i].lower())
    for i in range(len(doc_list_lower)):
        doc_list_split.append(doc_list_lower[i].split())
        
    for i in range(len(doc_list_split)):
        for j in range(len(doc_list_split[i])):
            if(doc_list_split[i][j] in keyword_lower):
                ind.append(i)
    return ind
#doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]            
#print(word_search(doc_list,'casino'))

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
    pass
    multi_search={}
    for keyword in keywords:
        multi_search[keyword]=word_search(doc_list,keyword)
    return multi_search
#doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]
#keywords = ['casino', 'they']
#multi_word_search(doc_list, keywords)
        
    
q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    pass
    string1=[]
    hh=height//2
    for i in range(hh):
        string1.append(' '*(hh-i-1))
        string1.append('/'*(i+1))
        if(i==0):
            string1=string1
        else:
            string1.append('\\'*(i)) 
        string1.append('\\\n')
    for i in range(hh):
        string1.append(' '*(i))
        string1.append('\\'*(hh-i))
        if(i==hh):
            string1=string1
        else:
            string1.append('/'*(hh-i-1))
        string1.append('/\n')
    string2=''.join(string1)
    return string2
#print(diamond(4))


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