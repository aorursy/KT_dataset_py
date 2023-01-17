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
    if zip_code.isdigit() and len(zip_code) == 5: return True
    return False
q1.check()
#q1.hint()
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
    pass
    list_index = []
    for i,sentence in enumerate(doc_list):
        word_list = sentence.split()# split the sentence into strings
    #    word_list = [string.strip('.,').lower() for string in word_list]
        word_list = [string.strip(',.') for string in word_list]#strip the strings 
        word_list = [string.lower() for string in word_list]#to lower case the stripped strings
        for j in range(len(word_list)):
            if word_list[j] == keyword: 
                list_index.append(i)
    return list_index
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
    pass
    keywords_dict = {keyword: [] for keyword in keywords}
    for i,sentence in enumerate(doc_list):
        word_list = sentence.split()# split the sentence into strings
        word_list = [string.strip(',.') for string in word_list]#strip the strings 
        word_list = [string.lower() for string in word_list]#to lower case the stripped strings
        for keyword in keywords:
            for j in range(len(word_list)):
                if word_list[j] == keyword: 
                    keywords_dict[keyword].append(i)
    return keywords_dict
q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    pass
    if (height%2) !=0: #to check whether height is an even integer or not
        height += 1
    left = "/"
    right = "\\"
    string = ""
    mid = height//2
    for i in range(mid-1,-1,-1): #upper diamond
        string += (left.ljust(mid-i,left)+right.rjust(mid-i,right)).center(height)+"\n"
    for i in range(mid): #lower diamond
        string += (right.ljust(mid-i,right)+left.rjust(mid-i,left)).center(height)+"\n"
    return string
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
    pass
    history_unique = list(set(history)) #getting unique elements in the history list
    count = {} #to count number of unique elements in the history list
    count_n1n2 = {} # to count [n1][n2] and to store the conditional probability
    for i in range(len(history_unique)):
        count[history_unique[i]] = 0 #initialising with 0
        count_n1n2[history_unique[i]] = {} # declaring 2nd dimensional dictionary for n2 give n1
    for i in range(len(history)-1):
        count[history[i]] += 1 # counting number of unique elements in the history list i.e sum[n1]
        count_n1n2[history[i]][history[i+1]] = 0 # initialising with 0
    for i in range(len(history)-1): #count the unique number of sum[[n1][n2]]
        count_n1n2[history[i]][history[i+1]] += 1
    for key in count_n1n2: # calculating the conditional probability
        for values in count_n1n2[key]:
            count_n1n2[key][values] /= count[key] # conditional probability = sum[[n1][n2]]/sum[n1]
    return count_n1n2
q5.check()
q5.solution()