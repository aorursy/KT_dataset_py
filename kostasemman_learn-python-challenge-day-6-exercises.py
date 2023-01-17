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
a = 'and you are: too. me! me'
'me' in a
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return zip_code.isnumeric() and len(zip_code) == 5

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
    keyword = keyword.lower()
    result_list = []
    for doc in doc_list:
        doc_to_list = doc.lower().split()
        for word in doc_to_list:
            if keyword == word.strip(',.'):
                result_list.append(doc_list.index(doc))

    return result_list


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
    return {keyword: word_search(doc_list, keyword) for keyword in keywords}
    

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    for i in range(0, height // 2, 1):
        print(('/' * (i + 1) + '\\' * (i + 1)).center(height))
    for i in range(height // 2, 0, -1):
        print(('\\' * i + '/' * i).center(height))

diamond(8)
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    dmnd_str = ''
    for i in range(height // 2):
        dmnd_str += ('/' * (i + 1) + '\\' * (i + 1)).center(height) + '\n'
    for i in range(height // 2, 0, -1):
        dmnd_str += ('\\' * i + '/' * i).center(height) + '\n'
    return dmnd_str[:-1]

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
q4.solution()
a = {2: 2, 4: 3 ,3: 4}
b = [1, 2, 3]
help(list)

def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    probs_dict = {}
    for i in range(len(history[:-1])):
        num = history[i]
        num_count = history[:-1].count(num)
        if num not in probs_dict:
            probs_dict[num] = {}
            for j in range(1, len(history)):
                next_num = history[j]
                next_num_count = 0
                if next_num not in probs_dict[num] and history[j - 1] == num:
                    for k in range(len(history[:-1])):
                        if history[k] == num and history[k + 1] == next_num:
                            next_num_count += 1
                    probs_dict[num][next_num] = next_num_count / num_count
    return probs_dict

q5.check()
q5.solution()
help(range)