# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex6 import *
print('Setup complete.')
a = ""
length = 0
q0.a.check()
bool("."), bool(a)
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
    return zip_code.isdigit() and len(zip_code)==5
    pass

q1.check()
help(str)
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
    contain = []
    for idx in range(len(doc_list)):
        if keyword.lower() in doc_list[idx].lower().replace(',','').replace('.','').split():
            contain.append(idx)
    return contain
    pass


q2.check()
doc_list = ["The Learn, Python, Challenge Casino.", "The bought a car", "Casinoville"]
keyword = 'the'
contain = []
for idx in range(len(doc_list)):
    temp = doc_list[idx]
    if keyword.lower() in temp.lower().split():
        contain.append(idx)
contain
doc_list[0].replace(',','').replace('.','')
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
    return {keyword: word_search(doc_list,keyword) for keyword in keywords}
    pass

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    hhh = int(height/2)
    n_slash = [hhh - abs(k - hhh +1) for k in range(2*hhh - 1)]
    n_slash
    raw_diamond1 = [' '*(hhh - n) + '/'*n + '\\'*n for n in n_slash[0:hhh]]
    raw_diamond2 = [' '*(hhh - n) + '\\'*n + '/'*n for n in n_slash[hhh-1:]]
    return('\n'.join(raw_diamond1) + '\n' + '\n'.join(raw_diamond2))
    pass


q4.check()
hhh = 4
#'\n'.join([' '*n_whites + '/''\\' +  ' '*n_whites for n_whites in range(hhh) ]) 
n_slash = [hhh - abs(k - hhh +1) for k in range(2*hhh - 1)]
n_slash
raw_diamond1 = [' '*(hhh - n) + '/'*n + '\\'*n for n in n_slash[0:hhh]]
raw_diamond2 = [' '*(hhh - n) + '\\'*n + '/'*n for n in n_slash[hhh-1:]]
#print('\n'.join(raw_diamond1) + '\n' + '\n'.join(raw_diamond2))
diamond(4)

d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
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
    l_hist = len(history)
    my_dict = {}
    cond_probs = {}
    consec_pairs = [ (history[n], history[n+1]) for n in range(l_hist - 1) ]
    keys = []
    # dictionary of absolute frequencies = { number thrown : { a possible next throw: its frequency,...}  }
    for a,b in consec_pairs:
        if not(a in keys):
            my_dict[a] = {b:1}
            keys.append(a)
        else:
            if b in list(my_dict[a]):
                my_dict[a][b] += 1
            else:
                my_dict[a][b] = 1
                
    for key in list(my_dict.keys()):
        temp_dict =  my_dict[key] # frequencies
        temp_total = sum(list(temp_dict.values()))
        cond_probs[key] = { temp_key:temp_dict[temp_key]/temp_total for temp_key in temp_dict.keys()}
    return(cond_probs)
    pass


q5.check()
history = [1, 3, 1, 5, 1, 3]
l_hist = len(history)
x = enumerate(history)
my_dict = {}
#my_dict = {1:2,1:3,3:1,5:3}
#for n in range(len(history)):
#my_dict.items()
#my_dict
consec_pairs = [ (history[n], history[n+1]) for n in range(l_hist - 1) ]
print(consec_pairs)
keys = []
for a,b in consec_pairs:
    if not(a in keys):
        #my_dict[a] = [b]
        my_dict[a] = {b:1}
        keys.append(a)
    else:
        if b in list(my_dict[a]):
            my_dict[a][b] = my_dict[a][b] + 1
        else:
            my_dict[a][b] = 1

print(my_dict)
my_dict.keys()
list(my_dict[1].keys())
my_dict[1][3]
cond_probs = {}
for key in list(my_dict.keys()):
    temp_dict =  my_dict[key] # frequencies
    temp_total = sum(list(temp_dict.values()))
    cond_probs[key] = { temp_key:temp_dict[temp_key]/temp_total for temp_key in temp_dict.keys()}
    
cond_probs

#temp_dict =  my_dict[1] # frequencies
#temp_total = sum(list(temp_dict.values()))
#{ temp_key:temp_dict[temp_key]/temp_total for temp_key in temp_dict.keys()}
#my_dict[1].append(4)
#print(my_dict)
q5.solution()