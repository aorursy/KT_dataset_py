import numpy as np
x=np.array([["data", "science"],[10,20]])
x
#to change the string in upper case using numpy
np.char.upper(x)

#to capitalize the string in array using numpy
np.char.capitalize(x)
#to keep string in center and total length as 10 rest filled by *
z=np.char.center(x,10,' ')
z
y= np.array(["Python","Pocker"])
x,y
#join function concatenate each characters of y string with x whole string values 
np.char.join(x,y)
np.char.join(y,x)
np.char.join(x,x)
#to check strings are in lower case or not 
np.char.islower(x)
np.char.isupper(y)
#to replace character in strings
np.char.replace(y,"P","T")
z
np.char.strip(z)
t=np.char.swapcase(x)
u=np.char.swapcase(t)
t,u
a=np.char.encode(x,encoding='utf8')
a
np.char.decode(a,encoding='utf8')
np.char.splitlines(x)
np.char.title(x)
#string add will work as concatenate
np.char.add(x,y)
#to expant or split the string by tabs
g=np.array([['data\tPython', 'science\tPocker'],
       ['10\tPython', '20\tPocker']])
np.char.expandtabs(g)
np.char.partition(x,'t')
z
np.char.lstrip(z)
np.char.rstrip(z)
np.zeros(5)*2
s=np.ones((2,3))*0.7
s
np.ceil(s)
np.floor(s)
#nearest integer
np.rint(s)
t=np.sqrt(s)
u=np.negative(s)
t,u
np.sign(t), np.sign(u)
np.hypot(t,u)
np.exp(t)
