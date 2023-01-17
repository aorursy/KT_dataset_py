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
    return (zip_code.isdigit() and len(str(zip_code)) ==5)

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
    doc=[]
    index=[]
    keyword=keyword.lower()
    for i in range(len(doc_list)):
        doc.append(doc_list[i])
        doc[i]= doc[i].lower()
        doc[i]=doc[i].replace('.','')
        doc[i]=doc[i].replace(',','')
        doc[i]=doc[i].strip('?')
        words=doc[i].split()
        for j in range(len(words)):
             if words[j]==keyword:
                    index.append(i)
    return index

q2.check()
q2.hint()
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
    dict1={}
    for i in range(len(keywords)):
        dict1[keywords[i]]=word_search(doc_list,keywords[i])
    return dict1

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    s=''
    i=0
    half=int(height/2)
    while i <half:
        j=0
        while j <(half-i-1):
            s+=' '
            j+=1
        k=0
        while k <=i:
            s+='/'
            k+=1
        l=0
        while l<=i:
            s+='\\'
            l+=1
        s+='\n'
        i+=1
    i=0
    while i <half:
        j=0
        while j <i:
            s+=' '
            j+=1
        k=0
        while k <=(half-i-1):
            s+='\\'
            k+=1
        l=0
        while l<=(half-i-1):
            s+='/'
            l+=1
        s+='\n'
        i+=1
    return s

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
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
    for i in range(1, len(history)):
        roll, prev = history[i], history[i-1]
        if prev not in counts:
            counts[prev] = {}
        if roll not in counts[prev]:
            counts[prev][roll] = 0
        counts[prev][roll] += 1

    # We have the counts, but still need to turn them into probabilities
    probs = {}
    for prev, nexts in counts.items():
        # The total spins that landed on prev (not counting the very last spin)
        total = sum(nexts.values())
        sub_probs = {next_spin: next_count/total
                for next_spin, next_count in nexts.items()}
        probs[prev] = sub_probs
    return probs


q5.check()
#q5.solution()