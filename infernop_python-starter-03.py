String0 = 'Taj Mahal is beautiful'
String1 = "Taj Mahal is beautiful"
String2 = '''Taj Mahal
is
beautiful'''
print(String0 , type(String0))
print(String1, type(String1))
print(String2, type(String2))
print(String0[4])
print(String0[4:])
print(String0.find('al'))
print(String0.find('am'))
print(String0[7])
print(String0.find('j',1))
print(String0.find('j',1,3))
String3 = 'observe the first letter in this sentence.'
print(String3.capitalize())
String0.center(70)
String0.center(70,'-')
String0.zfill(30)
s = 'h\te\tl\tl\to'
print(s)
print(s.expandtabs(1))
print(s.expandtabs())
print(String0.index('Taj'))
print(String0.index('Mahal',0))
print(String0.index('Mahal',10,20))
print(String0.endswith('y'))
print(String0.endswith('l',0))
print(String0.endswith('M',0,5))
print(String0.count('a',0))
print(String0.count('a',5,10))
'a'.join('*_-')
a = list(String0)
print(a)
b = ''.join(a)
print(b)
c = '/'.join(a)[18:]
print(c)
d = c.split('/')
print(d)
e = c.split('/',3)
print(e)
print(len(e))
print(String0)
print(String0.lower())
String0.upper()
String0.replace('Taj Mahal','Bengaluru')
f = '    hello      '
f.strip()
f = '   ***----hello---*******     '
f.strip('*')
print(f.strip(' *'))
print(f.strip(' *-'))
print(f.lstrip(' *'))
print(f.rstrip(' *'))
d0 = {}
d1 = dict()
print(type(d0), type(d1))
d0['One'] = 1
d0['OneTwo'] = 12 
print(d0)
print(d0['One'])
names = ['One', 'Two', 'Three', 'Four', 'Five']
numbers = [1, 2, 3, 4, 5]
d2 = zip(names,numbers)
print(d2)
a1 = dict(d2)
print(a1)
a1.clear()
print(a1)
for i in range(len(names)):
    a1[names[i]] = numbers[i]
print(a1)
a1.values()
a1.keys()
a1.items()
a2 = a1.pop('Four')
print(a1)
print(a2)
