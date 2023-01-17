import os

dataset_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.split('.')[-1]=="txt": #only take the tect files

            dataset_files.append(os.path.join(dirname, filename))

print(dataset_files)
#load all data into a string



all_string = ''



for file in dataset_files:

    data = open(file).read()

    print(len(data))

    all_string += data



print("length of final string:",len(all_string))



# Extracting unique characters from the text feels simplier

unique_letters = list(set(all_string))

print("unique laters:",len(unique_letters))
# List all the Bengali characters

# Bengali unicode range is 0980-09FF

# Source: https://unicode.org/charts/PDF/U0980.pdf

print("All unique character from the text:",unique_letters)

filtered_letters = [' '] # space character is not in the range



for l in unique_letters:

    #dicarding numbers

    if (l >= '\u0980' and l <= '\u09E5') or (l >= '\u09F0' and l <= '\u09FF'):

        filtered_letters.append(l)

        

print("extracted valid charaters:",filtered_letters)

print("valid charaters:",len(filtered_letters))
unique_alpha = filtered_letters

unique_num = [chr(n) for n in range(0x09E6,0x09F0)]
#sort the unique letters according to their frequency

unique_alpha.sort(reverse=True, key=lambda e: all_string.count(e))

unique_num.sort(reverse=True, key=lambda e: all_string.count(e))



unique_alphanum = unique_alpha + unique_num

print('Total characters to fit:',len(unique_alphanum))
class Node:

    def __init__(self, val):

        self.val = val

        self.left = None

        self.right = None



letters = unique_alphanum



root = Node('start')



current = root

nexts = []



for char in letters:

    if current.left == None:

        current.left = Node(char)

    else:

        if current.right == None:

            current.right = Node(char)

        else:

            nexts.append(current.left)

            nexts.append(current.right)

            current = nexts.pop(0)

            current.left = Node(char)





print(root)
def traverse(rootnode, width):

  thislevel = [rootnode]

  a =('{:^'+str(width)+'}').format("")

  while thislevel:

    nextlevel = list()

    a = a[:int(len(a)/2)]

    for n in thislevel:

      print (a+str(n.val), end=a)

      if n.left: nextlevel.append(n.left)

      if n.right: nextlevel.append(n.right)

      

    print("\n")

    thislevel = nextlevel
traverse(root,200)
# applied number system ->

# 1 2 4 8 are the positional weights

# positoiinal values are 1 and 2

# binary number system but it skips the value 0

# decimal value = sum(positional_weight * positional value)

#

# This is placed to mimic the effect of binary tree



import numpy as np

def index_to_mc(higher_limit):

    n_digits = 6

    pos_weights = [1, 2, 4, 8, 16, 32]

    binary_index = [0, 0, 0, 0, 0, 0]

    mc = []

    for i in range(1, higher_limit+1):

        carry = 1

        #print("inter",i)

        for n in range(n_digits):

            _n = binary_index[n] + carry

            #print(_n)

            if _n > 2:

                binary_index[n] = 1

                carry = 1

                #print("greater")

            else:

                binary_index[n] = _n

                carry = 0

                #print("less")

        mc.append(

            ''.join(

                [str(x) for x in np.copy(binary_index)]

            ).replace('0','')

            .replace('1','-')

            .replace('2','.')

        )

    return mc
#create the translate dictionaries

from collections import OrderedDict 



mc_iter = index_to_mc(len(unique_alphanum))

conversion_dict = {unique_alphanum[i] : mc_iter[i] for i in range(len(unique_alphanum))}

reverse_dict = {v:k for k,v in conversion_dict.items()}



# order acourding to unicode/alphabetically

conversion_dict = OrderedDict(sorted(conversion_dict.items())) 
def transform_to_mc(value, inverse=False):

    r_value = ''

    if inverse:

        values = value.split('/')#user may use space

        for ch in values:

            if ch: # skipping empty elements in the list

                r_value += reverse_dict[ch]

        

    else:

        for ch in value:

            r_value += conversion_dict[ch] + '/'

                

    return r_value
print(transform_to_mc('মহাবিশ্ব সীমাহীন'))
print(transform_to_mc('--./..--/./..-/-./-.-./../..-/-/-../..../--././..--/..../---/', inverse=True))