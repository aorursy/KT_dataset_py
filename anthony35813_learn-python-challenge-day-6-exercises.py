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
    corr_digits = len(zip_code)
    try:
        if corr_digits == 5 and int(zip_code) >= 0:
            return True
        else:
            return False
    except:
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
    index_list = []
    for index, txt in enumerate(doc_list):
        words = txt.split()
        targets = [word.rstrip('.,').lower() for word in words]
        if keyword.lower() in targets:
            index_list.append(index)
            #print(index_list)
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
    dict = {}
    for keyword in keywords:
        index_list = word_search(doc_list, keyword)
        #returns index_list
        dict[keyword] = index_list
    #print(dict)    
    return dict
q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond = ''
    d_top = "/"
    d_bottom = "\\"          #need to escape \
    m = int(height / 2)      # need to know half way point to switch direction
    height_plus = height + 1 #add one to height to make printing multiples work
    mx = 1                   # counter needed to subtract away on bottom half
    if height > 0 and height <= 2:
        diamond = (d_top  + d_bottom) + '\n' + (d_bottom + d_top)  #simply print it
    
    else:
        for h in range(height_plus):
            if h < 1:
                pass
            elif h <= m:        #print the top
                diamond += (' ' * (m - h)) + (d_top * h) + (d_bottom * h) + '\n'
            
            elif h > m:       #print the bottom
                diamond += (' ' * (h-(m+1))) + (d_bottom * (h-mx)) + (d_top * (h-mx)) + '\n'
                mx = mx + 2
    return diamond
q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
def conditional_roulette_probs(history):            #couldnt work this one out myself!! need more dictionary revision
    """
    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """

    counts = {}                                  # create an empty dictionary
    for i in range(1, len(history)):            #miss the first number as it dosent have a previous roll
        roll, prev = history[i], history[i-1]   #acccess numbers in roll and the previous numbeer in prev
        if prev not in counts:                  #check if the number is in the dictionary if not then add it
            counts[prev] = {}
        if roll not in counts[prev]:            #check if next is already collected for the prev number 
            counts[prev][roll] = 0              #as this is the first time set up the counter
        counts[prev][roll] += 1                 # add one to the counter

    # We have the counts, but still need to turn them into probabilities
    probs = {}
    for prev, nexts in counts.items():                          #get the counts for each previous number
        # The total spins that landed on prev (not counting the very last spin)
        total = sum(nexts.values())                             #how many different numbers did it land on?
        sub_probs = {next_spin: next_count/total                #can't see how this bit works
                for next_spin, next_count in nexts.items()}
        probs[prev] = sub_probs
    return probs
q5.check()
#q5.solution()