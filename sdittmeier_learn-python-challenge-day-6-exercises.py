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
    return zip_code.isdigit() and len(zip_code)==5

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
    index_list = []
    for index, doc in enumerate(doc_list):
        words = doc.split()
        norm_words = [word.rstrip('.,').lower() for word in words]
        if keyword.lower() in norm_words:
            index_list.append(index)
    return index_list

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
    dictionary = {}
    for word in keywords:
        dictionary[word] = word_search(doc_list, word)
    return dictionary

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    """has height/2 layers, start with height/2-1 blanks, then /\\,escape"""
    dia = ""
    for i in range(height//2):
        dia += (height//2-i-1)*" " + (i+1)*"/"+(i+1)*"\\"+"\n"
    for i in range(height//2):
        dia += (i)*" " + (height//2-i)*"\\"+(height//2-i)*"/"+"\n"
    print(dia)
    return dia

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
    last_spin = -1
    dictionary = {}
    dictionary_inside = {}
    for spin in history:
        if last_spin != -1:
            if(last_spin in dictionary):
                dictionary_inside = dictionary[last_spin]
                if(spin in dictionary_inside):
                    dictionary_inside[spin] += 1.0
                else:
                    dictionary_inside[spin]= 1.0
            else:
                dictionary_inside = {spin: 1.0}
            dictionary[last_spin] = dictionary_inside
            
        last_spin = spin
  #  print(dictionary)
    
    for key in dictionary:
        sum = 0
        for key_inside in dictionary[key]:
            sum += dictionary[key][key_inside]
        for key_inside in dictionary[key]:
            dictionary[key][key_inside] = dictionary[key][key_inside]/sum
            
    print(history)
    print(dictionary)
    return dictionary

q5.check()
#q5.solution()