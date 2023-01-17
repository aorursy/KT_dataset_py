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
    z = zip_code
    if not z.isnumeric(): return False
    return (len(z) == 5 and z >= '00000' and z <= '99999')

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
    found = list()
    table = str.maketrans(dict.fromkeys(',.?'))
    
    for doc_index, doc in enumerate(doc_list):
        words = doc.lower().translate(table).split(' ')
        if keyword.lower() in words:
            found.append(doc_index)
    return found

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
    docs_keyword_indices = {keyword:[] for keyword in keywords}
    for keyword in keywords:
        docs_keyword_indices[keyword] = word_search(doc_list, keyword)

    return docs_keyword_indices
q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    def get_row(row, height):
        top_half = row < height / 2
        if top_half:
            first, second = "/", "\\"
            pad = ' ' * (height//2 - row -1)
            marks = row + 1
        else:
            first, second = "\\", "/"
            pad = ' ' * (row - height//2)
            marks = height - row
        answer = pad

        for c in range(marks):
            answer += first
        for c in range(marks):
            answer += second
        
        return answer

    diamond_str = ""
    for i in range(height):
        if i > 0:
            diamond_str += '\n'
        diamond_str += get_row(i,height) 
    return diamond_str

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
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    numbers = {number : {} for number in history}
    for ndx, number in enumerate(history[:-1]):
        next_num = history[ndx+1]
        if not next_num in numbers[number]:
            numbers[number][next_num] = 1
        else:
            numbers[number][next_num] += 1
    
    for k,v in numbers.items():
        trials = sum(v.values())
        for subsequent_number, frequency in v.items():
            numbers[k][subsequent_number] = frequency/trials

    return numbers
q5.check()
q5.solution()
