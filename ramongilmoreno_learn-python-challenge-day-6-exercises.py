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
print(len(b))
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
    return (len(zip_code) == 5) and zip_code.isnumeric()

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
    r = []
    for doc in range(len(doc_list)):
        txt = " " + doc_list[doc].replace(",", " ").replace(".", " ").lower() + " "
        if (txt.find(" " + keyword.lower() + " ") != -1):
            r.append(doc)
    return r

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
    return { keyword: word_search(doc_list, keyword) for keyword in keywords }

q3.check()
#q3.solution()
def line (c1, c2, index, total, acc):
    """Displays the line at index, of total number of characters, with c1 and
    c2 open and closing characters, and the given starting char (for recursiveness).
    
    Example:
    >>> "[" + line("a", "b", 2, 4, "c") + "]"
    [ aaacbbb ]
    """
    if (index >= 0):
        return line(c1, c2, index - 1, total - 1, c1 + acc + c2)
    else:
        if (total > 0):
            return line(c1, c2, index - 1, total - 1, " " + acc + " ")
        else:
            return acc
        
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    acc = ""
    half = int(height / 2)
    # Draw the rectangle from inside out: from middle lines to top and bottom lines by concatenating lines.
    for l in reversed(range(half)):
        acc = line("/", "\\", l, half, "") + "\n" + (acc + "\n" if len(acc) > 0 else acc) + line("\\", "/", l, half, "")
    return acc

print(line("a", "b", 0, 4, "-"))
print(line("a", "b", 1, 4, "-"))
print(line("a", "b", 2, 4, "c"))
print(line("a", "b", 3, 4, "-"))
print(diamond(4))
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
    previous = None
    r = {}
    for current in history:
        if not current in r:
            r[current] = {}
        # Detect first one to not insert into inner dictionary
        if (previous == None):
            previous = current
        else:
            d = r[previous]
            if not current in d:
                d[current] = 1
            else:
                d[current] = d[current] + 1
            previous = current
    for current in history:
        values = r[current]
        total = sum(values.values())
        r[current] = { key: value / total for key, value in values.items() }
    return r

q5.check()
q5.solution()