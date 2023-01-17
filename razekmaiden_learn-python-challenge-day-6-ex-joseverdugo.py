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

    return True if len(str(zip_code)) == 5 and str(zip_code).isnumeric() else False



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

    results = []

    keyword = keyword.lower()

    possible_keyword = [keyword, keyword + ".",  keyword + ","]

    for i, doc in enumerate(doc_list):

        doc = doc.lower()

        words = doc.split()

        if any([word in possible_keyword for word in words]): results.append(i) 

    return results





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

   

    return {keyword:word_search(doc_list, keyword) for keyword in keywords}



q3.check()
#q3.solution()
def diamond(height):

    """Return a string resembling a diamond of specified height (measured in lines).

    height must be an even integer.

    """

    if height % 2 != 0:

        height += 1

    axis = int(height/2)

    result = [] 

    for row in range(height):

        column = [" " for i in range(height)]

        #print(column)

        if row < axis:

            for i in range(axis+(-1-row),axis+1+row):

                #print(row,axis-(1-row), axis+row, i)

                #print(row, i, axis + (-1-row), axis+row)

                if i < axis:

                    column[i] = "/"

                else:

                    column[i] = "\\"

        else:

            for i in range(row-axis, (height)-(row-axis)):

                #print(row,axis-(1-row), axis+row, i)

                

                if i < axis:

                    column[i] = "\\"

                    #print(row, i, row-axis, (height)-(row-axis), "\\")

                else:

                    column[i] = "/"

                    #print(row, i, row-axis, (height)-(row-axis), "/")

        print("".join(column))

        result.append("".join(column))

            

    return "\n".join(result)



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

    # unique not present in standart modules

    unique_values = []

    history_temp = history[:(len(history)-1)] #not consider the last value

    [unique_values.append(value) for value in history_temp if value not in unique_values]

    results = {value:{} for value in unique_values}

    

    for i in unique_values:

        counter = {}

        for j in range(1,len(history)):

            if history[j-1] == i:

                if history[j] in counter.keys():

                    counter[history[j]] = counter[history[j]] + 1

                else:

                    counter[history[j]] = 1

        suma = sum(counter.values())

        results[i] = {value:counter[value]/suma for value in counter.keys()}

        

    #print(results)

    return results



q5.check()
q5.solution()