1 + 5
4 * 4

16 / 4
2 ** 4
4 % 2
5 % 2
(2 + 3) * (5 + 5)
# Can not start with number or special characters

name_of_var = 2
x = 2

y = 3
z = x + y
z
'single quotes'
"double quotes"
" wrap lot's of other quotes"
x = 'hello'
print(x)
num = 12

name = 'Sam'
print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
print('My number is: {}, and my name is: {}'.format(num,name))
[1,2,3]
['hi',1,[1,2]]
my_list = ['a','b','c']
my_list.append('d')
my_list
my_list[0]
my_list[1]
my_list[1:]
my_list[:1]
my_list[0] = 'NEW'
my_list
nest = [1,2,3,[4,5,['target']]]
nest[3]
nest[3][2]
nest[3][2][0]
d = {'key1':'item1','key2':'item2'}
d
d['key1']
True and False
True or False
t = (1,2,3)
t[0]
t[0] = 'NEW'
{1,2,3}
{1,2,3,1,2,1,2,3,3,3,3,2,2,2,1,1,2}
1 > 2
1 < 2
1 >= 1
1 <= 4
1 == 1
'hi' == 'bye'
(1 > 2) and (2 < 3)
(1 > 2) or (2 < 3)
(1 == 2) or (2 == 3) or (4 == 4)
if 1 < 2:

    print('Yep!')
if 1 < 2:

    print('first')

else:

    print('last')
if 1 > 2:

    print('first')

else:

    print('last')
if 1 == 2:

    print('first')

elif 3 == 3:

    print('middle')

else:

    print('Last')
seq = [1,2,3,4,5]
for item in seq:

    print(item)
for item in seq:

    print('Yep')
for jelly in seq:

    print(jelly+jelly)
i = 1

while i < 5:

    print('i is: {}'.format(i))

    i = i+1
range(5)
for i in range(5):

    print(i)
list(range(5))
x = [1,2,3,4]
out = []

for item in x:

    out.append(item**2)

print(out)
[item**2 for item in x]
def my_func(param1='default'):

    """

    Docstring goes here.

    """

    print(param1)
my_func
my_func()
my_func('new param')
my_func(param1='new param')
def square(x):

    return x**2
out = square(2)
print(out)
def times2(var):

    return var*2
times2(2)
lambda var: var*2
seq = [1,2,3,4,5]
map(times2,seq)
list(map(times2,seq))
list(map(lambda var: var*2,seq))
filter(lambda item: item%2 == 0,seq)
list(filter(lambda item: item%2 == 0,seq))
st = 'hello my name is Sam'
st.lower()
st.upper()
st.split()
tweet = 'Go Sports! #Sports'
tweet.split('#')
tweet.split('#')[1]
d
d.keys()
d.items()
lst = [1,2,3]
lst.pop()
lst
'x' in [1,2,3]
'x' in ['x','y','z']