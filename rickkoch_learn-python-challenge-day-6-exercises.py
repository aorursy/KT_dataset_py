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
    return (zip_code.isdigit() and (len(zip_code) == 5))

q1.check()
#q1.hint()
#q1.solution()
doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
print("doc_list", doc_list)
print("doc_list length:", len(doc_list))
enspaced = list(map(lambda x: ' ' + x + ' ', doc_list))
print("enspaced:", enspaced)
lowered = list(map(lambda x: x.lower(), enspaced))
print("lowered:", lowered)
deperioded = list(map(lambda x: x.replace('.', ' '), lowered))
print("deperioded:", deperioded)
decommad = list(map(lambda x: x.replace(',', ' '), deperioded))
print("decommad:", decommad)

i = 0
while i < len(doc_list):
    print("i:", i)
    print("doc_list[i]:", doc_list[i])
    print("A")
    i += 1
print("all done")
    

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
    print("doc_list", doc_list)
    print("doc_list length:", len(doc_list))
    print("keyword:", keyword)
    lc_keyword = ' ' + keyword.lower() + ' ' # enable searching for space||keyword||space
    print("lc_keyword:", lc_keyword)
    enspaced = list(map(lambda x: ' ' + x + ' ', doc_list))
    print("enspaced:", enspaced)
    lowered = list(map(lambda x: x.lower(), enspaced))
    print("lowered:", lowered)
    deperioded = list(map(lambda x: x.replace('.', ' '), lowered))
    print("deperioded:", deperioded)
    decommad = list(map(lambda x: x.replace(',', ' '), deperioded))
    print("decommad:", decommad)
    ret_list = [] # create the list to return
    print("ret_list:", ret_list)
    i = 0
    while i < len(doc_list):
        print("i:", i)
        print("decommad[i]", decommad[i])
        print("decommad[i].find(lc_keyword)", decommad[i].find(lc_keyword))
        if decommad[i].find(lc_keyword) > -1:
            ret_list.append(i)
        i += 1
    return ret_list

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
    ret_dict = {} #create directory to return
    for keyword in keywords:
        ret_dict[keyword] =  word_search(doc_list, keyword)
    print("ret_dict:", ret_dict)
    return ret_dict

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    r_list = []
    for r_num in range(1, height + 1):
        #print('r_num: ', r_num)
        distance_from_widest_row = int(abs(r_num - (1+ height) / 2) - 0.5)
        #print('distance_from_widest_row: ', distance_from_widest_row )
        facets_in_current_row = height - distance_from_widest_row * 2
        #print('facets_in_current_row: ', facets_in_current_row)
        #
        # indent initial spaces
        for j in range(0, distance_from_widest_row):
            r_list.append(' ')
        # process left side facets
        for j in range(0, facets_in_current_row // 2):
            if r_num <= height // 2:
                r_list.append('/')
            else:
                r_list.append('\\')
        # process right side facets
        for j in range(0, facets_in_current_row // 2):
            if r_num <= height // 2:
                r_list.append('\\')
            else:
                r_list.append('/')
        # add newline if needed
        if r_num != height:
            r_list.append('\n')
        #print('r_num: ', r_num, 'r_list: ', r_list)
    return ''.join(r_list)


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
    pass
    # create dictionary of initial_spin with nested dictionary of next_spin
    d_current_spin = {}
    d_current_spin_count = {}
    d_result = {}
    #d_next_spin = {}
    #num_next_spins = len(history)
    #print('num_next_spins: ', num_next_spins)
    #set initial next spin
    next_spin = history.pop(0)
    #print('initial_next_spin: ', next_spin)
    while (len(history) > 0):
        #load dictionaries with values and counts
        current_spin = next_spin
        next_spin = history.pop(0)
        #print('current_spin: ', current_spin, 'next_spin: ', next_spin)
        # if dict entry does not exist for current spin, create it and initialize next_spin
        if current_spin not in d_current_spin:
            #print('a')
            d_current_spin[current_spin] = {}
            #print('b')
            d_current_spin[current_spin][next_spin] = 1
        else:
            #print('c')
            #print('d_current_spin[current_spin]: ',d_current_spin[current_spin])
            if next_spin not in d_current_spin[current_spin]:
                #print('c_prime')
                d_current_spin[current_spin][next_spin] = 1
                #print('d_prime')
            else:
                #print('d')
                d_current_spin[current_spin][next_spin] += 1
        # track total number next_spins for each current_spin
        if current_spin not in d_current_spin_count:
            d_current_spin_count[current_spin] = 1
        else:
            d_current_spin_count[current_spin] +=1
        #print('d_current_spin: ', d_current_spin)
        #print('d_current_spin_count: ', d_current_spin_count)
    #print('done processing')
    for k in d_current_spin.keys():
        d_result[k] = {}
        #dict_variable = {key:value for (key,value) in dictonary.items()}
        for (m, n) in d_current_spin[k].items():
            d_result[k][m] = (n / d_current_spin_count[k])
    return(d_result)

q5.check()
q5.solution()