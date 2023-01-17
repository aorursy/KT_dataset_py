x = 'a string'

y = "a string"

x == y
multiline = """

one

two

three

"""
fox = "tHe qUICk bROWn fOx."
fox.upper()
fox.lower()
fox.title()
fox.capitalize()
fox.swapcase()
line = '         this is the content         '

line.strip()
line.rstrip()
line.lstrip()
num = "000000000000435"

num.strip('0')
line = "this is the content"

line.center(30)
line.ljust(30)
line.rjust(30)
'435'.rjust(10, '0')
'435'.zfill(10)
line = 'the quick brown fox jumped over a lazy dog'

line.find('fox')
line.index('fox')
line.find('bear')
line.index('bear')
line.rfind('a')
line.endswith('dog')
line.startswith('fox')
line.replace('brown', 'red')
line.replace('o', '--')
line.partition('fox')
line.split()
haiku = """matsushima-ya

aah matsushima-ya

matsushima-ya"""



haiku.splitlines()
'--'.join(['1', '2', '3'])
print("\n".join(['matsushima-ya', 'aah matsushima-ya', 'matsushima-ya']))
pi = 3.14159

str(pi)
"The value of pi is " + str(pi)
"The value of pi is {}".format(pi)
"""First letter: {0}. Last letter: {1}.""".format('A', 'Z')
"""First letter: {first}. Last letter: {last}.""".format(last='Z', first='A')
"pi = {0:.3f}".format(pi)
!ls *Python*.ipynb
import re

regex = re.compile('\s+')

regex.split(line)
for s in ["     ", "abc  ", "  abc"]:

    if regex.match(s):

        print(repr(s), "matches")

    else:

        print(repr(s), "does not match")
line = 'the quick brown fox jumped over a lazy dog'
line.index('fox')
regex = re.compile('fox')

match = regex.search(line)

match.start()
line.replace('fox', 'BEAR')
regex.sub('BEAR', line)
email = re.compile('\w+@\w+\.[a-z]{3}')
text = "To email Guido, try guido@python.org or the older address guido@google.com."

email.findall(text)
email.sub('--@--.--', text)
email.findall('barack.obama@whitehouse.gov')
regex = re.compile('ion')

regex.findall('Great Expectations')
regex = re.compile(r'\$')

regex.findall("the cost is $20")
print('a\tb\tc')
print(r'a\tb\tc')
regex = re.compile(r'\w\s\w')

regex.findall('the fox is 9 years old')
regex = re.compile('[aeiou]')

regex.split('consequential')
regex = re.compile('[A-Z][0-9]')

regex.findall('1043879, G2, H6')
regex = re.compile(r'\w{3}')

regex.findall('The quick brown fox')
regex = re.compile(r'\w+')

regex.findall('The quick brown fox')
email = re.compile(r'\w+@\w+\.[a-z]{3}')
email2 = re.compile(r'[\w.]+@\w+\.[a-z]{3}')

email2.findall('barack.obama@whitehouse.gov')
email3 = re.compile(r'([\w.]+)@(\w+)\.([a-z]{3})')
text = "To email Guido, try guido@python.org or the older address guido@google.com."

email3.findall(text)
email4 = re.compile(r'(?P<user>[\w.]+)@(?P<domain>\w+)\.(?P<suffix>[a-z]{3})')

match = email4.match('guido@python.org')

match.groupdict()