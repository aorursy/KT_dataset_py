# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex6 import *
print('Setup complete.')
a = ""
length = -1
q0.a.check()
b = "it's ok"
length = -1
q0.b.check()
c = 'it\'s ok'
length = -1
q0.c.check()
d = """hey"""
length = -1
q0.d.check()
e = '\n'
length = -1
q0.e.check()
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return (zip_code.isdigit() and len(zip_code)==5)

q1.check()
#q1.hint()
#q1.solution()
def word_search(doc_list, keyword):
    required_articles = []
    """
    Takes a list of documents (each document is a string) and a keyword. 
    Returns list of the index values into the original list for all documents 
    containing the keyword.

    Example:
    doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
    >>> word_search(doc_list, 'casino')
    >>> [0]
    """
    for i in range(len(doc_list)):
        l = doc_list[i].lower().replace(',', '').replace('.', '').split(' ')
        print(l)
        if keyword.lower() in l:
            required_articles.append(i)
    return required_articles


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
    output = {}
    
    for keyword in keywords:
        output[keyword] = word_search(doc_list, keyword)
    return output

q3.check()
#q3.solution()
def diamond(height):
    h = int(height/2)
    upper_half = ''
    lower_half = ''
    for i in range(h):
        upper_half += ' '*(h -i-1) + ('/'*(i+1)) + ('\\' * (i+1)) + ' '*(h -i-1) + '\n'
    for j in range(h, 0, -1):
        lower_half += ' '*(h -j) + ('\\'*(j)) + ('/' * (j)) + ' '*(h -j) + '\n'
    daimond =  upper_half + lower_half
    return daimond
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
def conditional_roulette_probs(history):
    occurences_after = {}
    for i in range(len(history)-1):
        n1 = history[i]
        n2 = history[i+1]
        if n1 not in occurences_after:
            occurences_after[n1] = [n2]
        else:
            occurences_after[n1].append(n2)
    print(occurences_after)
    probabilities = {}
    for i in range(len(history)-1):
        n1 = history[i]
        probabilities[n1] = {}
        for j in range(len(occurences_after[n1])):
            n2 = occurences_after[n1][j]
            probabilities[n1][n2] = occurences_after[n1].count(n2)/(len(occurences_after[n1])+0.0)
    print(probabilities)
    return probabilities
    """
    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
q5.check()
q5.solution()