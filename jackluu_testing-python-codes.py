print('My name is {num} and my number is {name}'.format(num=12, name="Jack"))
my_list = ['a', 'b', 'c', 'd','e','f']
my_list.append('d')
print(my_list)
d = {'key1': 'value', 'key2':123}
d['key1']  
d['key2']
d = {'k1':[1,2,3,4,5,6,7]}
d['k1']
num = 12
name = 'Sam'
print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
list = [1,2,3,4,5]
for element in list:
    print(element)
list = [1,2,3,4,5,6]
for element in list:
    print('You are awesome!')
i = 1
while i < 5:
    print('i is: {}'.format(i))
    i = i+1
import math
def square(x):
    return math.sqrt(x)
print(square(10))
{1,2,3,1,2,1,2,3,3,3,3,2,2,2,1,1,2}