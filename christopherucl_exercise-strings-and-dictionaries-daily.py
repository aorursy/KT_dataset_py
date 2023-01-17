from learntools.core import binder; binder.bind(globals())

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

    if (len(zip_code) == 5 and (zip_code.isdecimal())):

        return True

    else:

        return False

    

    pass



q1.check()
#help(str)
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

    

    desiredIndex = []

    for i,name in enumerate(doc_list):

        toks1= name.split() # returns a list of splitted strings

        toks2 = [i.lower().strip(".,") for i in toks1] # normalises for comparison

        if keyword.lower().strip(" ") in toks2:

            desiredIndex.append(i)

    return desiredIndex

    

    pass



q2.check()
#q2.hint()

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

    desiredIndex1 = {}

    for i in keywords:

        desiredIndex1[i] = word_search(doc_list,i)

    return desiredIndex1

        

    pass



q3.check()
def multi_word_search(doc_list, keywords):

    """

    Takes list of documents (each document is a string) and a list of keywords.  

    Returns a dictionary where each key is a keyword, and the value is a list of indices

    (from doc_list) of the documents containing that keyword

    {'casino': [0, 1], 'they': [1]}

    """

    desiredIndex1 = {}



    for i,name in enumerate(doc_list):

        toks1= name.split() # returns a list of splitted strings

        toks2 = [i.lower().strip(".,") for i in toks1] # normalises for comparison

        a = len(keywords)

        for j in range(a):

            

            kw = keywords[j].lower().strip(" ")

            print(j, kw)

            if kw in toks2:

                desiredIndex1[kw] = toks2.index(kw)

                print(desiredIndex1[kw])

                    

    return desiredIndex1

    



doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]

keywords = ['casino', 'they']

multi_word_search(doc_list, keywords)
a =1,2

b = 2

print(list(a))

type(a)
#q3.solution()