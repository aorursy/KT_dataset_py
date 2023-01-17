import re
for i in dir(re):

    print(i, end=" , ")
print(len(dir(re)))
# help(re)
help(re.compile)
help(re.purge)
phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

match_obj = phoneNumRegex.search('My number is 415-555-4242.')

print('Phone number found: ' + match_obj.group())
pattern = r"Mango"

sequence = "Mango"

if re.match(pattern, sequence):

    print("Match!")

else: 

    print("Not a match!")
pattern = r"Mango"

sequence = "Orange"

print(re.match(pattern, sequence))
re.search(r'M.n.o', 'Mango').group()
re.search(r'M\wng\w', 'Mango').group()
re.search(r'S\Wgmail', 'S@gmail').group()
re.search(r'Eat\scake', 'Eat cake').group()
re.search(r'M\Sngo', 'Mango').group()
re.search(r'Mang\d', 'Mang0').group()
re.search(r'^Eat', 'Eat Rice').group()
re.search(r'Rice$', 'Eat Rice').group()
re.search(r'Number: [0-9]', 'Number: 8').group()
# Matches any character except 7

re.search(r'Number: [^7]', 'Number: 9').group()
re.search(r'\A[A-R]ice', 'Rice').group()
re.search(r'\b[A-R]ice', 'Rice').group()
superRegex = re.compile(r'Super(wo)*man')

match_obj1 = superRegex.search('The Adventures of Superman')

match_obj1.group()

mo2 = superRegex.search('The Adventures of Superwoman')

mo2.group()
mo3 = superRegex.search('The Adventures of Superwowowowoman')

mo3.group()
superRegex = re.compile(r'Super(wo)+man')

match_obj1 = superRegex.search('The Adventures of Superwoman')

match_obj1.group()

match_obj2 = superRegex.search('The Adventures of Superwowowowoman')

match_obj2.group()
match_obj3 = superRegex.search('The Adventures of Superman')

# match_obj3.group()

match_obj3==None
re.search(r'Colou?r', 'Color').group()
re.search(r'\d{9,10}', '0987654321').group()
result = re.match(r'Rua', 'Rua Analytics')

print(result)
print(result.group(0))
result = re.search(r'Analytics', 'Rua Analytics')

print(result.group(0))
result = re.findall(r'Rua', 'Rua Analytics Rua')

print(result)
result=re.split(r'e','occurrences')

print(result)
result=re.split(r'\s','It helps to get a list of all matching patterns')

print(result)
result=re.sub(r'notes','projects','Kaggle is the place to do data science notes')

print(result)