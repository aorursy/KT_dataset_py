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
help(str)
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    if len(zip_code)!=5:
        return False
    else:
        return zip_code.isdigit()
        

q1.check()
q1.hint()
q1.solution()
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
    res=[]
    for index, document in enumerate(doc_list):
        words = document.split()
        for word in words:
            if word.strip(',.').lower() == keyword.lower():
                res.append(index)
    return res
            


q2.check()
q2.hint()
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
    d={}
    
    for keyword in keywords:
        d[keyword]=word_search(doc_list, keyword)
    return d

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    res=''
    for i in range(height):
        if i<height//2:
            res=res+' '*((height-2*i-2)//2)+'/'*(i+1)+'\\'*(i+1)+' '*((height-2*i-2)//2)+'\n'
        else:
            res=res+' '*((2*i-height)//2)+'\\'*(height-i)+'/'*(height-i)+' '*((2*i-height)//2)+'\n'
    return res[:-1]
#print(diamond(4))
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
    d={}
    
    for i in range(len(history)-1):
        d[history[i]]=d.get(history[i],[])+[history[i+1]] # alternative d.setdefault(history[i+1],[]).append(history[i])
    d1={}
    for i in d:
        temp_d={}
        for j in d[i]:
            temp_d[j]=temp_d.get(j,0)+1
        for m in temp_d:
            temp_d[m]=temp_d[m]/len(d[i])
        d1[i]=temp_d
    return d1

#conditional_roulette_probs([1, 3, 1, 5, 1])
q5.check()
q5.solution()