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
    return (len(zip_code) == 5) and zip_code.isdigit()

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
    
    # creating list for storing results
    results = []
        
    # looping over doc_list
    for i in range(len(doc_list)):
        
        # preparing text
        text = doc_list[i]
               
        # splitting text
        words = text.split()
              
        # setting control variable
        found = False
        
        # looping over splitted text
        for word in words:
        
            # adjusting for final '.' or ',' using .rstrip method
            # adjusting for uppercases using .lower method
            word = word.rstrip(',.').lower()
            keyword = keyword.lower()
            
            # checking if keyword found
            if word == keyword:
                found = True
                        
        # appending succes to results list
        if found:
            results.append(i)
    
    print(keyword, results)
    return results

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
    
    results = {}
    
    print("\n", keywords)
    
    for keyword in keywords:
        results[keyword] = word_search(doc_list, keyword)
    
    print(results)
    return results

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    
    # setting initial string
    shape = """"""
    
    # calculating half height to simplify calculations
    half = int(height / 2)
    
    # upper half loop - counter of lines
    for i in range (1, half + 1):        
        width = i
        
        # centering line with spaces
        for diff in range(half - width):
            shape += ' '
        
        # generating left side    
        for i_left in range(width):
            shape += '/'
        
        # generating right side
        for i_right in range(width):
            shape += '\\'    
        
        # finishing line
        shape += '\n'
        
    
    # lower half loop - counter of lines    
    for j in range (1, half + 1):
        width = half - j + 1
        
        # centering line with spaces
        for diff in range(half - width):
            shape += ' '
            
        # generating left side    
        for i_left in range(width):
            shape += '\\'
            
        # generating right side    
        for i_right in range(width):
            shape += '/'    
        
        # finishing line
        shape += '\n'
    
    print(shape)
    
    return shape
    

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
    
    print()
    print("History: ", history)
    
    # generating top-dictionary for probabilities
    probs = {}
    probs = {num : {} for num in history}
    print("Unique numbers in history:", probs)
    
    # generating sub-dictionaries for each number from top-dictionary
    # to be included in sub-dictionary, 
    # each verified number must be preceeded by a number from top dictionary
    # skipping first element as it cannot be preceeded
    for prob in probs.keys():
        probs[prob] = {num: 0 for i, num in enumerate(history) if (i > 0) and (prob == history[i - 1])}
    
    # calculating probabilities
    
    # loop for numbers from top-dictionary
    for prob in probs.keys():
        
        # calculating frequency of particular number
        # skipping last element as it cannot be followed
        observations = history[:-1].count(prob)
        print("Number", prob, "happened for", observations, "times")
        print("...with followers:")
        
        # loop for events (followers) from each sub-dictionary
        for event in probs[prob]:
            print("     number", event, end = "")
            
            # calculating frequencies of each possible event (follower)
            count = 0
            for i in range(len(history) - 1):
                if history[i] == prob and history[i +1] == event:
                    count += 1

            # calculating probability of each possible event (follower)        
            probability = count / observations
            print(" with probability of:", probability)
            
            probs[prob][event] = probability
            
    return probs

q5.check()
q5.solution()