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
length = -1
q0.c.check()
d = """hey"""
length = 3
q0.d.check()
e = '\n'
length = 1
q0.e.check()
import re

def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    if re.match('^(\d{5})$',zip_code):
        return True
    else:
        return False

q1.check()
q1.hint()
#q1.solution()
import re

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
    return [index for index, doc in enumerate(doc_list) if keyword.lower() in re.findall("\w+", doc.lower())]


q2.check()
q2.hint()
#q2.solution()
import re

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
    return {keyword: [index for index, doc in enumerate(doc_list) if keyword.lower() in re.findall("\w+", doc.lower())] for keyword in keywords}

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    seq=[n for n in range(1,int(height/2)+1)]
    rseq=seq.copy()
    rseq.reverse()
    seq.extend(rseq)
    result=''
    for index,row in enumerate(seq):
        if row>index:
            result=result+('/'*row+'\\'*row).center(height)
        else:
            result=result+('\\'*row+'/'*row).center(height)
        result=result+'\n'
    return result


q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
q4.solution()
import numpy
import pandas

def tabulate(l:list)->dict:
    """Returns frequencies of unique counts in a list.
    
    :param l: list to count frequencies in
    :return: counts as a dict
    """
    result={}
    for index, value in pandas.Series(l).value_counts().iteritems():
        result[index]=value/len(l)
    return result


def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    follows={}
    for index,n in enumerate(history):
        #skipping last item - no history ahead
        if index==len(history)-1:
            break
            
        if n not in follows:
            follows[n]=[]
        follows[n].append(history[index+1])
        
    result={}
    for n in numpy.unique(history):
        result[n]=tabulate(follows[n])
    return result


q5.check()
q5.solution()