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
    return (len(zip_code) == 5) and (zip_code.isdecimal())

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
    lowered_keyword = keyword.lower()
    processed_doc_list = [element.lower().replace('.', ' ').replace(',', ' ').replace(';', ' ').replace('\'',' ').replace('"', ' ').replace('!', ' ').replace('?', ' ').replace('  ', ' ') for element in doc_list ]   
    indexes_list = []
    # print("input_list=", doc_list, "keyword=", keyword)
    # print("processed_list=", processed_doc_list)
    for index in range(len(processed_doc_list)):
        # print(processed_doc_list[index])
        if(lowered_keyword in processed_doc_list[index]) :
            find_start = processed_doc_list[index].index(lowered_keyword) 
            # print(processed_doc_list[index][find_start:])
            splitted_string = processed_doc_list[index][find_start:].split(" ")
            # print(processed_doc_list[index], "->" , splitted_string)
            if(splitted_string[0].replace(lowered_keyword, "") == "") :
                indexes_list.append(index)
    return indexes_list

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
    return { keyword : word_search(doc_list, keyword)   for keyword in keywords}

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_shape = ""
    for  line in range(int(height / 2), 0, -1) :
        diamond_top = "".join([" " for index in range (int((height / 2)) - line)]) + "".join(["/" for index in range(line)]) + "".join(["\\" for index in range(line)]) + "\n"
        diamond_bottom =  "".join([" " for index in range (int((height / 2)) - line)]) + "".join(["\\" for index in range(line)]) + "".join(["/" for index in range(line)]) + "\n"
        diamond_shape = diamond_top + diamond_shape + diamond_bottom
    return diamond_shape


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
    init_dictionary = {}
    for index in range(len(history)) :
        current = history[index]
        # print(current)
        if(index > 0) :
            init_dictionary[history[index - 1]].append(current)
        
        if (not (history[index] in init_dictionary))  :
            init_dictionary[current] = []
            
    print(init_dictionary)
    result_dict = {}
    for key, values_list in init_dictionary.items() :
        if(len(values_list) > 0) :
            result_dict[key] = {}
            for element in values_list :
                if(element in result_dict[key]) :
                    result_dict[key][element] += 1
                else :
                    result_dict[key][element] = 1

            for sub_key, sub_value in result_dict[key].items() :
                result_dict[key][sub_key] = result_dict[key][sub_key]  / len(values_list) 
    print(result_dict)
    return result_dict


q5.check()
q5.solution()