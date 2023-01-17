# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex6 import *
print('Setup complete.')
a = ""
length = 0
len(a)==length
q0.a.check()
b = "it's ok"
length = 7
len(b)==length
q0.b.check()
c = 'it\'s ok'
length = 7
len(c)==length
q0.c.check()
d = """hey"""
length = 3
len(d)==length
q0.d.check()
e = '\n'
length = 1
print(len(e))
len(e)==length
q0.e.check()
help(str)
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
#     if len(zip_code) == 5 and zip_code.isdigit():
#         return True
#     else:
#         return False
    return len(zip_code) == 5 and zip_code.isdigit()

print(is_valid_zip("20874"))
print(is_valid_zip("11a11"))
print(is_valid_zip("11111"))
print(is_valid_zip("118111"))
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
    indexList = []
    for i, checkstr in enumerate(doc_list):
        # split on white space
        strSplit = checkstr.split()
        # remove periods and commas, change to lower case
        cleanSplit = [strs.rstrip('.,').lower() for strs in strSplit]
        if keyword.lower() in cleanSplit:
            indexList.append(i)
    
    return indexList
doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
print(word_search(doc_list, 'casino'))
doc_list = ["The man in the yellow hat found George", "George is a curious monkey", "He was playing on the monkeybars"]
print(word_search(doc_list, 'monkey'))
doc_list = ["The man in the yellow hat found George", "George is a curious monkey", "George is a playful monkey", "He was playing on the monkeybars"]
print(word_search(doc_list, 'monkey'))

q2.check()
q2.hint()
q2.solution()
numbers = {'one':1, 'two':2, 'three':3}
print(numbers)
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
    indexDict = {}
    for keyword in keywords:
        indexDict[keyword.lower()] = word_search(doc_list,keyword.lower())
#         indexList = []
#         for i, checkstr in enumerate(doc_list):
#             # split on white space
#             strSplit = checkstr.split()
#             # remove periods and commas, change to lower case
#             cleanSplit = [strs.rstrip('.,').lower() for strs in strSplit]
#             if keyword.lower() in cleanSplit:
#                     indexList.append(i)
#         indexDict[keyword.lower()] = indexList
    
    return indexDict


doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
keyword_list = ["Casino", "bought"]
print(multi_word_search(doc_list, keyword_list))
doc_list = ["The man in the yellow hat found George", "George is a curious monkey", "He was playing on the monkeybars"]
print(multi_word_search(doc_list, ["MONKEY", "hat"]))
doc_list = ["The man in the yellow hat found George", "George is a curious monkey", "George is a playful monkey", "He was playing on the monkeybars"]
print(multi_word_search(doc_list, ["Monkey", "playful", "is"]))

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    s = ''
    # The characters currently being used to build the left and right half of 
    # the diamond, respectively. (We need to escape the backslash with another
    # backslash so Python knows we mean a literal "\" character.)
    l, r = '/', '\\'
    # The "radius" of the diamond (used in lots of calculations)
    rad = height // 2
    for row in range(height):
        # The first time we pass the halfway mark, swap the left and right characters
        if row == rad:
            l, r = r, l
        if row < rad:
            # For the first row, use one left character and one right. For
            # the second row, use two of each, and so on...
            nchars = row+1
        else:
            # Until we go past the midpoint. Then we start counting back down to 1.
            nchars = height - row
        left = (l * nchars).rjust(rad)
        right = (r * nchars).ljust(rad)
        s += left + right + '\n'
    # Trim the last newline - we want every line to end with a newline character
    # *except* the last
    return s[:-1]

print(diamond(6))
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

print(conditional_roulette_probs([1, 3, 1, 5, 1]))


q5.check()
q5.solution()