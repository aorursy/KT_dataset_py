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
    length_of_zip = len(zip_code)
    if length_of_zip == 5:
        return zip_code.isdecimal()
    else:
        return False

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
    length_of_doc_list = len(doc_list)
    length_of_keyword = len(keyword)
    
    list_of_index = []
    
    for m in range(length_of_doc_list):
        lowered_doc_list = doc_list[m].lower()
        
        splitted_doc_list = lowered_doc_list.split()
        
        length_of_splitted_string = len(splitted_doc_list)
        
        for k in range(length_of_splitted_string):
            selected_word = splitted_doc_list[k]
            length_of_selected_word = len(selected_word)
            if selected_word.startswith(keyword):
                if length_of_selected_word == length_of_keyword:
                    list_of_index.append(m)
                elif length_of_selected_word == length_of_keyword + 1:
                    if selected_word[length_of_selected_word-1] == "." or selected_word[length_of_selected_word-1] == ",":
                        list_of_index.append(m)
        
    return list_of_index
        


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
    number_of_keywords = len(keywords)
    dictionary = {}
    for i in range(number_of_keywords):
        selected_keyword = keywords[i]
        list_of_index = word_search(doc_list, selected_keyword)
        dictionary[selected_keyword] = list_of_index
    
    return dictionary

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    
    number_of_line = int(height/2)

    desired_string = ""

    for i in range(number_of_line):
        left_part = ""
        rigth_part = ""
        for k in range(i+1):
            left_part += "/"
            rigth_part += "\\"
        combined_line = (number_of_line-1-i)*" " + left_part + rigth_part + "\n"
        desired_string += combined_line
    
    reversed_left_part = number_of_line*"\\"
    reversed_rigth_part = number_of_line*"/"
    
    for m in reversed(range(number_of_line+1)):
        reversed_left_part = reversed_left_part[0:m]
        reversed_rigth_part = reversed_rigth_part[0:m]
        combined_reversed_line = (number_of_line-m)*" " + reversed_left_part + reversed_rigth_part + "\n"
        desired_string += combined_reversed_line
    
    #remove last added new line 
    desired_string = desired_string.rstrip()
    print(desired_string)
    return desired_string

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
q5.solution()