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
    if len(zip_code)==5 and zip_code.isdigit(): return True
    else: return False
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
    lower_keyword = keyword.lower()
    copy_list_on_lower= [dc.lower() for dc in doc_list]
    finds= []
    for doc in copy_list_on_lower:
        words= doc.split()
        for word in words:
            if not word.isalpha():
                not_chars =[]
                for char in word:
                    if not char.isalpha():
                        not_chars.append(char)
                for char_strip in not_chars:
                    word = word.strip(char_strip)
            if word== keyword:
                finds.append(copy_list_on_lower.index(doc))
    return finds

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
    lower_keywords = [keyword.lower() for keyword in keywords]
    copy_list_on_lower= [dc.lower() for dc in doc_list]
    finds = {}
    for key in lower_keywords:
        finds [key] = []
    
    for doc in copy_list_on_lower:
        words= doc.split()
        for word in words:
            if not word.isalpha():
                not_chars =[]
                for char in word:
                    if not char.isalpha():
                        not_chars.append(char)
                for char_strip in not_chars:
                    word = word.strip(char_strip)
            for kw in lower_keywords:
                if word== kw:
                    finds[kw].append(copy_list_on_lower.index(doc))
    return finds
#doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]
#keywords = ['casino', 'they']
#multi_word_search(doc_list, keywords)
q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    a='/'
    b='\\'
    c='\n'
    diamond=""
    for i in range(height//2):
        diamond += ((i+1)*a +(i+1)*b).center(height) + c
    for i in range(height//2):
        diamond += ((height//2-i)*b +(height//2-i)*a).center(height) + c
    return diamond
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
    print(history)
    result={}
    for index,value in enumerate (history[:-1]):
        numOfRep=history[:-1].count(value)
        if(numOfRep>1 and not (value in result)):
            subDict={}
            counter=index
            nextNum= counter+1
            if (not history[nextNum] in subDict):
                subDict[history[nextNum]]= 1/numOfRep
            else:
                subDict[history[nextNum]] += 1/numOfRep
            for x in range(numOfRep-1):
                counter += 1
                nextNum= counter+1
                location=history[counter:].index(value)
                counter += location
                nextNum= counter+1
                if (not history[nextNum] in subDict):
                    subDict[history[nextNum]]= 1/numOfRep
                else:
                    subDict[history[nextNum]] += 1/numOfRep
            result[value]=subDict    
        if(numOfRep==1):
            result[value]={history[index+1]:1.0}
    print(result)
    return (result)

q5.check()
q5.solution()