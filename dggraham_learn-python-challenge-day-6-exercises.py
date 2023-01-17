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
    # A little overkill for the challenge... By importing string this code can strip
    # away other punctuation not just '.,'
    import string
    punc = set(string.punctuation)  # punctuation to be ignored
    dlnp = [] # doc_list, no punctuation
    for s in doc_list:
        new_s = ''
        for c in s.lower():
            if c not in punc:
                new_s += c
        dlnp.append(new_s)
    
    returnList = [] # indicies of strings containing the keyword
    for i, s in enumerate(dlnp):
        w = s.split()
        if keyword.lower() in w:
            returnList.append(i)

    return(returnList)


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
    found = {}
    for keyword in keywords:
        found[keyword] = word_search(doc_list, keyword)
    return found

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    myString = ''
    half = height // 2 # integer division
    l, r = '/', '\\'
    
    for row in range(height):
        if row == half:
            l, r = r, l  # swap the characters for the bottom half
        if row < half:
            nchars = row + 1 # zero index so add 1 for first row
        else:
            nchars = height - row
        left = (l * nchars).rjust(half) # this code from solution
        right = (r * nchars).ljust(half)
        myString += left + right + '\n'
    return myString[:-1] # do not return last \n


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
def conditional_roulette_probs(history):
    counts = {} # get the counts of the next roll
    for i in range(1,len(history)):
        roll, prev = history[i], history[i-1]
        if prev not in counts:
            counts[prev] = {} # new dictionary for the first roll
        if roll not in counts[prev]:
            counts[prev][roll] = 0
        counts[prev][roll] += 1
        
    # convert counts to probabilites
    probs = {}
    for prev, nexts in counts.items():
        total = sum(nexts.values())
        sub_probs = {next_spin: next_count/total for next_spin, next_count in nexts.items()}
        probs[prev] = sub_probs
    return(probs)


q5.check()
q5.solution()