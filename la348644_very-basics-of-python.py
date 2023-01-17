1 # integwer number
1.0 #floting point number
1+1 # basic arthmatic
1 * 3 # multiplication
1 / 2 #divisions
2 ** 4 #exponent
4 % 2 # reminder
(2 + 3) * (5 + 5)
name_of_var = 2 
x = 2
y = 3
z = x + y #Now Z=2+3 which are x and y defined in above steps
'Hello World' # single quote
"Hello World" # double quotes
" single quote's inside double quote"
x='hello'
x
print(x)
num = 12
name = 'Sam'
print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
[1,2,3]
['hi',1,[1,2]]
my_list = ['a','b','c']
my_list.append('d')
my_list
my_list[0]
my_list[1]
my_list[1:] #FROM 1 TILL END
my_list[:1] #ONLY 0 AS 1 IS NOT INCLUDED
my_list[0] = 'NEW'
nest = [1,2,3,[4,5,['target']]]
nest[3]
nest[3][2]
nest[3][2][0]
d = {'key1':'item1','key2':'item2'}
d['key1']
True
False
t = (1,2,3) # tuples are immutable and so is different from list
t[0]
t[0] = 'NEW'#should throw error
{1,2,3} #No duplicate values can exist in sets
{1,2,3,1,2,1,2,3,3,3,3,2,2,2,1,1,2}
1 > 2
1 < 2
1 >= 1
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
seq = [1,2,3,4,5]
for item in seq:
    print(item)
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
def square(x):
    return x**2
out = square(2)
print(out)
def times2(var):
    return var*2

times2(2)
lambda var: var*2
seq = [1,2,3,4,5]
map(times2,seq) #times is function defined in above cell
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
d #dictionary we defined earlier
d.keys()
d.items()
lst = [1,2,3]
lst.pop()
lst
'x' in [1,2,3]
'x' in ['x','y','z']
