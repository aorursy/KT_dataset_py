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
    if (zip_code.isnumeric()==True)&(len(zip_code)==5):
        return True
    else:
        return False
    pass

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
    r=[]
    for l in doc_list:
        a=l.lower()
        a=''.join(c for c in a if c not in ("!","@","#","$","%","^","&","*",":",";","/","?","+",",","."))
        a=a.split()
        if keyword in a:
             r.append(doc_list.index(l))
            #return doc_list.index(l)
    return r
    
    pass


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
    index=[]
    #word=[]
    a={}
    for words in keywords:
        index.append(word_search(doc_list,words))
        #word.append(words)
        #y={words:word_search(doc_list,words)}
    for i in range(len(keywords)): 
        a[keywords[i]]=index[i]
    return a
    pass

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    #a=''.join[('/'*i + '\\'*i) for i in range()]
    s=''
    for i in range(1,(height//2)+1):
        if i != 1: s += '\n'
        #s += ('/' * i + '\\' * i).center(height)
        s+=('/'*i + '\\'*i).center(height)
    for i in range((height//2),0,-1):
        s+="\n"+('\\'*i+'/'*i).center(height)
    return s
    pass


q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
history=[1,23,45,1,2,34,1]
s=set([1,23,45,1,2,34,1])
print(s)
dict_list = {item: [index for index, value in enumerate(history) if value == item]  for item in s}
print(dict_list)
dict_values = {key: [history[index + 1] for index in value if (index+ 1) < len(history)] for key, value in dict_list.items()}
print(dict_values)
prob={key: {element: value.count(element)/len(value) for element in value} for key,value in dict_values.items()}
print(prob)
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    
    s=set(history)
    dict_list = {item: [index for index, value in enumerate(history) if value == item]  for item in s}
    #print(dict_list)
    dict_values = {key: [history[index + 1] for index in value if (index+ 1) < len(history)] for key, value in dict_list.items()}
    #print(dict_values)
    prob={key: {element: value.count(element)/len(value) for element in value} for key,value in dict_values.items()}
    #print(prob)
    return(prob)
    pass


q5.check()
q5.solution()