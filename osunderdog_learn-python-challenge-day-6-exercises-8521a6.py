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
    return(len(zip_code) == 5 and zip_code.isdigit())

q1.check()
#q1.hint()
q1.solution()
def wordmatch(ukword, word):
    """
    determine if ukword (unknown word) is:
        word not a subcompoent
        ignoring punctuation (.)
        ignoring case
    """
    dotcount = len([c for c in ukword if c in ".,"])  ##c.punctuation anybody?
    return(word.upper() in ukword.upper() and len(word) == len(ukword)-dotcount )


print(wordmatch('casino', 'casino') == True)
print(wordmatch('Casino', 'casino') == True)
print(wordmatch('casino.', 'casino') == True)
print(wordmatch('cas.ino', 'casino') == False)
print(wordmatch('casino,', 'casino') == True)
print(wordmatch('casinoville', 'casino') == False)

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
    return([i for i,s in enumerate(doc_list) 
            if any([wordmatch(w,keyword) for w in s.split(" ")])
           ])
 

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
    ##initialize bags (better ways to do this in python, but going with what we know.)
    return({k:word_search(doc_list,k) for k in keywords})
    
q3.check()
#q3.solution()
def qd(height, orient='l', air = ' ', db = 'o'):
    """
    build a quadrant .. there are four in a diamond.
    """
    rslt = []
    for r in range(height):
        airst = (air * (height-r-1))
        dbst = (db * (r+1))
        rslt.append(airst + dbst if orient == 'l' else dbst + airst)
    return(rslt)
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    hh = height//2
    rstr = ""
    ##left side of diamond
    dl = qd(hh,orient='l',db='/') + list(reversed(qd(hh,orient='l',db='\\')))
    ##right side of diamond
    dr = qd(hh,orient='r',db='\\') + list(reversed(qd(hh,orient='r',db='/')))
    for i in range(height):
        rstr += dl[i] + dr[i] + "\n"
    return(rstr)

q4.check()
print(diamond(10))
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
    tpldict = {}
    hlen = len(history)
    for i,n in enumerate(history):
        if n not in tpldict:
            tpldict[n]={'runningtotal':0,'tally':{}}
        ni = i+1
        if ni < hlen: ##can do this for all but last one.
            tpldict[n]['runningtotal'] += 1
            nn = history[ni]
            if nn not in tpldict[n]['tally']:
                tpldict[n]['tally'][nn] = 0
            tpldict[n]['tally'][nn] += 1
    ##summarize
    rslt = {}
    for n,smry in tpldict.items():
        rt = smry['runningtotal']
        rslt[n] = {k:v/rt for k,v in smry['tally'].items()}
    return(rslt)


q5.check()
conditional_roulette_probs([1, 3, 1, 5, 1])
q5.solution()