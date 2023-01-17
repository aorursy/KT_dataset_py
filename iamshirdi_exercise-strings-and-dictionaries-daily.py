from learntools.core import binder; binder.bind(globals())

from learntools.python.ex6 import *

print('Setup complete.')
a = ""

length = len(a)

q0.a.check()
len(a)
b = "it's ok"

length = len(b)

print(length)

q0.b.check()
c = 'it\'s ok'

length = len(c)

print(length)

q0.c.check()
d = """hey"""

length = len(d)

print(length)

q0.d.check()
e = '\n'

length = len(e)

print(length)

q0.e.check()
def is_valid_zip(zip_code):

    """Returns whether the input string is a valid (5 digit) zip code

    """

    try:

        num=int(zip_code)

        if len(zip_code)==5:

            return True # no need when checking equality direct return

        else:

            return False

    except:

        return False

    #    return len(zip_code) == 5 and zip_code.isdigit()



q1.check()
#q1.hint()

#q1.solution()
d = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]

keyword='a'

a=[]

b=0

for i,c in enumerate(d):

    print(i)



    for j in c.split():

        j=j.lower().strip(',.')

        print(j)

        if j==keyword:

            a.append(i)

    

    

a
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

    a=[]



    

    for i,c in enumerate(doc_list):



        for j in c.split():

            j=j.lower().strip(',.')

            if j==keyword:

                a.append(i)

                break #prevent duplicate documents in string

    



    return a                

    

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

    keyword_to_indices = {}

    for keyword in keywords:

        keyword_to_indices[keyword] = word_search(doc_list, keyword) #creates keyword and stores its value in that

    return keyword_to_indices



q3.check()
#q3.solution()