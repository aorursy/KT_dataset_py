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
    return len(zip_code) == 5 and zip_code.isdigit()

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
    
    clean_doc_list = []
    for doc in doc_list:
        clean_doc_list.append(doc.replace(","," ").replace("."," ").lower()+" ")

    doc_list_index = []    
    for string in clean_doc_list:
        if string.find(" "+keyword.lower()+" ") >= 0:
            doc_list_index.append(clean_doc_list.index(string))
                
    return doc_list_index

q2.check()
#try on mah ownnnn

def word_search(doc_list, keyword):
    index = []
    for i, doc in enumerate(doc_list):
        words = doc.split()
        clean_words = [word.rstrip(".,").lower() for word in words]
        if keyword.lower() in clean_words:
            index.append(i)
    return index
    
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
    word_search_dict = {}
    for keyword in keywords:
        word_search_dict[keyword] = word_search(doc_list, keyword)
    return word_search_dict

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_list = []
    for i in range(1,int(height/2+1)):
        diamond_list.append(str(" " * int(height/2-i)) + str("/" * i) + str("\\" * i))
    for j in range(int(height/2)):
        diamond_list.append(str(" " * j) + str("\\" * int(height/2 - j)) + str("/" * int(height/2 - j)))
    return "\n".join("{}".format(line) for line in diamond_list)

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
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
    counts = {}
    for i in range(1,len(history)):
        prev, roll = history[i-1], history[i]
        if prev not in counts:
            counts[prev] = {}
        if roll not in counts[prev]:
            counts[prev][roll] = 0
        counts[prev][roll] += 1
    
    probs = {}
    for prev, nexts in counts.items():
        total = sum(nexts.values())
        sub_prob = {next_spin: next_count/total for next_spin, next_count in nexts.items()}
        probs[prev] = sub_prob
        
    return probs
    
q5.check()
q5.solution()