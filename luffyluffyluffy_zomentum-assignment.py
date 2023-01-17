# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

path = "/kaggle/input/json-data/file.JSON"

f = open(path, "r")

data_dict= json.loads(f.read())

data_dict
# data_dict = {

# "op": "equal",

# "lhs": {

# "op": "multiply",

# "lhs": 3,

# "rhs": {

# "op": "add",

# "lhs": "x",

# "rhs": 2

# }

# },

# "rhs": 21

# }



# usefull functions and dicts

op_dict = {

    "multiply" : "*",

    "divide" : "/",

    "add" : "+",

    "subtract" : "-",

    "equal": "=",

}



op_neg_dict = {

    "multiply" : "divide",

    "divide" : "multiply",

    "add" : "subtract",

    "subtract" : "add",

    "+" : "-", 

    "-" : "+",

    "*" : "/",

    "/" : "*",

    

}



expressions = {"*", "/", "+", "-", "="}







def prefix_expression(data_dict, array=[]):

    for k in data_dict:

        if (type(data_dict[k]) == dict and data_dict[k] != np.nan):

            array = prefix_expression(data_dict[k], array)

#             print("hi")

        else:

            if data_dict[k] in op_dict:

                array.append(op_dict[data_dict[k]])

            else:

                array.append(data_dict[k])

            

    return array



def prerfix_to_infix(array):

    exp_string=[]

    length = len(array)

    for i in range(length-1, -1, -1):

        if array[i] in expressions:

            oprant_1 = exp_string[len(exp_string)-1]

            exp_string.pop()

            oprant_2 = exp_string[len(exp_string)-1]

            exp_string.pop()

            temp_str = ''

            if(i <= 1):

                temp_str =  oprant_1 + " " + array[i]+" "+oprant_2

            else:

                temp_str = "("+oprant_1+" "+array[i]+" "+oprant_2+")"

            exp_string.append(temp_str)

        else:

            exp_string.append(str(array[i]))

    

    return exp_string



def empy(arr):

    ar=[]

    for i in arr:

        if len(i) != 0:

            ar.append(i)

    return ar



def simplification(rhs, exp):

    exp_string=[]

    exp = ",".join(exp.split(' ')).replace("(", "").replace(")", "").replace("x", "").split(",")

    exp = empy(exp)

    length = len(exp)

    if(length>3):

        i=1

        while(i<length):

            if ((exp[i] in expressions) and (exp[i+1] in expressions)):

                exp_string.append(op_neg_dict[exp[i]]+exp[i-1]+")"+op_neg_dict[exp[i+1]])

                i+=1

            elif ((exp[i]) in expressions):

                exp_string.append(op_neg_dict[exp[i]]+exp[i-1])

            else:

                exp_string.append(exp[i])

            i+=1

        return "("+rhs+"".join(exp_string)

    return rhs+exp[0,len(exp)-1]
print("method 1: \n")

array = prefix_expression(data_dict, [])

exp_string = prerfix_to_infix(array)

print("infix notation: \n", exp_string[0])

expression = exp_string[0]

lhs = expression.split('=')[0]

rhs = expression.split('=')[1]

simple_exp = simplification(rhs, lhs)

print("simple exp : \n", "x =", simple_exp)

eval(simple_exp)

print("solution is :\n", "x = ", eval(simple_exp))



class node:

    def __init__(self, ):

        self.op = ''

        self.neg_op = ''

        self.lhs = ''

        self.rhs = ''





def process(data, parent):

    if type(data['lhs']) == dict:

        parent.lhs=node()

        process(data['lhs'], parent.lhs)

    else:

#         print(data['lhs'])

        parent.lhs = data['lhs']

    

#     print(data['op'])

    parent.op = data['op']

    

    if type(data['rhs']) == dict :

        parent.rhs = node()

        process(data['rhs'], parent.rhs)

    else:

#         print(data['rhs'])

        parent.rhs = data['rhs']



    return



def print_tree(tree, string ="", verbose=1 ):

    if type(tree.lhs) == int or tree.lhs == 'x':

        string = string + (str(tree.lhs))

#         print(string)



    else:

        string = string + str(" ( ")

        string = print_tree(tree.lhs, string, 0)

        string = string + str(" ) ")



    string = string + str(" ")

    string = string + op_dict[str(tree.op)]

    string = string + str(" ")

#     print(tree.op)

    

    if type(tree.rhs) == int or tree.rhs == 'x':

        string = string + (str(tree.rhs))

#         print(string)

    else:

        string = string + str(" ( ")

        string = print_tree(tree.rhs, string, 0)

        string = string + str(" ) ")



    if(verbose):

        print(string)

    return string

        

def isin(tree):

    if type(tree) == int:

        return 0

    if (tree) == 'x':

        return 1

    if tree.lhs == 'x':

        return 1

    if tree.rhs == 'x':

        return 1

    

    type_tree_lhs = type(tree.lhs)

    type_tree_rhs = type(tree.rhs)

    

    if type_tree_lhs != int:

        if isin(tree.lhs):

            return 1

    

    if type_tree_rhs != int:

        if isin(tree.rhs):

            return 1

    

    return 0






print("method 2: \n")



from sympy import symbols, solve

tree = node()

process(data_dict, tree)

print("expression :")

string = print_tree(tree)

lhs = tree.lhs

rhs = tree.rhs

tree.lhs = node()

tree.lhs.lhs = lhs

tree.lhs.rhs = rhs

tree.lhs.op = "subtract"

tree.rhs = 0

string = print_tree(tree)

x = symbols('x')

expr = string.split('=')[0]

sol = solve(expr)

print("ans :\nx = ", sol[0])
print("method 1:")

array = prefix_expression(data_dict, [])

exp_string = prerfix_to_infix(array)

print("infix notation: \n", exp_string[0])

expression = exp_string[0]

lhs = expression.split('=')[0]

rhs = expression.split('=')[1]

simple_exp = simplification(rhs, lhs)

print("simple exp : \n", "x =", simple_exp)

eval(simple_exp)

print("solution is :\n", "x = ", eval(simple_exp))







print("\n\nmethod 2:")



from sympy import symbols, solve

tree = node()

process(data_dict, tree)

print("expression :")

string = print_tree(tree)

lhs = tree.lhs

rhs = tree.rhs

tree.lhs = node()

tree.lhs.lhs = lhs

tree.lhs.rhs = rhs

tree.lhs.op = "subtract"

tree.rhs = 0

string = print_tree(tree)

x = symbols('x')

expr = string.split('=')[0]

sol = solve(expr)

print("ans :\nx = ", sol[0])