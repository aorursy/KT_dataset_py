1+2
a = 1.0
b = 2.0
print(a)
b
for i in range(10):
    print(i)
import requests
def square(x):
    """Square the input x"""
    n= x * x * x
    return n
square(4)
%time sum(x**2 for x in range(10000))
%%bash
ls
%load_ext line_profiler
def function_to_profile():
    print(sum(x**2 for x in range(1000)))
    print(sum(x**3 for x in range(10000)))
    
%lprun -f function_to_profile function_to_profile()
list1 = ["a", "b", "c", "d"]
list2 = [1, 2, 3, "4", [1,2]]
print(list1[0])
print(list1[0:4:2])
for a in list1:
    print(a)
print(list1[-1])
print(list1[1:2])
list1
list3 = []
for x in list1:
    list3.append(x + "_")
list3
list3 = [x + "_" for x in list1]
list3
with open("../input/sharepoint.txt") as f: ## Note: you will get errors if you dont reference an existing file on your directory
    lines = f.readlines()
print(lines)
sum([len(line.split(" ")) for line in lines])
%%bash
ls
(len(line.split(" ")) for line in lines)
%%timeit
all_words = []
for line in lines[:10000]:
    for word in line.split():
        all_words.insert(0, word)
%%timeit
all_words = []
for line in lines[:10000]:
    for word in line.split():
        all_words.append(word)
dict1 = {"a":1, "b":2, "c":3}
dict1["a"]
dict1["d"] = 4
"a" in dict1
{i:i**2 for i in range(10)}
class MyClass:
    def __init__(self, n):
        self.n = n
    
    def get_n(self):
        return self.n

a = MyClass(1)
a.get_n()
a.n
import matplotlib.pyplot as plt
%matplotlib notebook
import numpy as np
x = np.linspace(0,2*np.pi,100)
plt.plot(x, np.sin(x))
%matplotlib inline
plt.plot(x, np.sin(x))
