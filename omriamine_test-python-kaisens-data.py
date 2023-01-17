#We are using regular expression operations module

import re

#define a fuction that will matches a string with a regular expression pattern

def text_match(text):

    #the pattern of the regular expression 

    patterns = 'ab*?'

    #Scan through text looking for the first location where the regular expression pattern produces a match

    if re.search(patterns,  text):

        return 'Found a match!'

    else:

        return('Not matched!')

print(text_match("ssfsfsc"))

print(text_match("assfsfsc"))
def find_match(text):

    #text should start with one upper case letter followed by lower case letters

    patterns = '^[A-Z][a-z]+'

    if re.search(patterns,  text):

        return 'Found a match!'

    else:

        return('Not matched!')



print(find_match('Kaisens data'))

print(find_match("kaisens data"))

print(find_match("KAISENS DATA"))
def find_match(text):

    #text should contain an 'a'followed by anything and ending with 'b'

    patterns = 'a.*b$'

    if re.search(patterns,  text):

        return 'Found a match!'

    else:

        return('Not matched!')



print(find_match('a Kaisens data b'))

print(find_match("kaisens data b"))

print(find_match("a KAISENS DATA"))

print(find_match("ab"))

#Find the first occurrence of "z" where it is NOT at the beginning or the end of a word:, \B is the opposit of \b

def find_match(text):

        #text should ontain 'z',but not start or end with 'z' 

        patterns = '\Bz\B'

        if re.search(patterns,  text):

                return 'Found a match!'

        else:

                return('Not matched!')



print(find_match("kaisens data zip code"))

print(find_match("size paris"))

text = 'Kaisens Data : Experts dans la conception/réalisation de projets * Big data, data science'

print(re.split(': |, |\/|\*',text))
text = "it will certainly be the best start of my career as a data scientist if I do an internship at Kaisens Dataa"

#Return an iterator yielding MatchObject instances over all non-overlapping matches for the RE pattern in string

for m in re.finditer(r"\w+ly", text):

    print('%d-%d: %s' % (m.start(), m.end(), m.group(0)))
text = "i will be avilable for the intership starting from 01/03/2020 in english dormat it will be 2020-03-01 we can right is also like that 01/03/20"

match = re.findall(r'(\d{4}-\d{2}-\d{2})|(\d{2}\/\d{2}\/\d{4})|(\d{2}\/\d{2}\/\d{2})', text)

print(match)