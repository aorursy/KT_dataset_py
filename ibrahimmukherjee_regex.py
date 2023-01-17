# Source link :- https://www.programiz.com/python-programming/regex



# A Regular Expression (RegEx) is a sequence of characters that defines a search pattern. For example,



# ^a...s$



# The above code defines a RegEx pattern. The pattern is: any five letter string starting with a and ending with s.
import re

pattern = '^a...s$'

test_string = 'abyss'

result = re.match(pattern, test_string)

if result:

  print("Search successful.")

else:

  print("Search unsuccessful.")	



# Here, we used re.match() function to search pattern within the test_string. The method returns a match object if the search is successful. If not, it returns None.
# Square brackets specifies a set of characters you wish to match.



# [abc]	--> this would be the string to be matched. 



# a	1 match

# ac	2 matches

# Hey Jude	No match

# abc de ca	5 matches





# Here, [abc] will match if the string you are trying to match contains any of the a, b or c.



# You can also specify a range of characters using - inside square brackets.



# [a-e] is the same as [abcde].

# [1-4] is the same as [1234].

# [0-39] is the same as [01239].
# You can invert the character set by using caret ^ symbol at the start of a square-bracket.



# [^abc] means any character except a or b or c.

# [^0-9] means any non-digit character.
#$ - Dollar



# The dollar symbol $ is used to check if a string ends with a certain character.



# For the regex expression # a$



#	a	1 match

# formula	1 match

# cab	No match
# * - Star



# The star symbol * matches zero or more occurrences of the pattern left to it.



#Expression	ma*n

#mn	No match (no a character)

#man	1 match

#maaan	1 match

#main	No match (a is not followed by n)

#woman	1 match
# Source :- https://www.tutorialspoint.com/python/python_reg_expressions.htm

        

#!/usr/bin/python

import re



line = "Cats are smarter than dogs"



matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)



if matchObj:

   print ("matchObj.group() : ", matchObj.group())

   print ("matchObj.group(1) : ", matchObj.group(1))

   print ("matchObj.group(2) : ", matchObj.group(2))

else:

   print ("No match!!")
#!/usr/bin/python

import re



line = "Cats are smarter than dogs";



matchObj = re.match( r'dogs', line, re.M|re.I)

if matchObj:

   print ("match --> matchObj.group() : ", matchObj.group())

else:

   print ("No match!!")



searchObj = re.search( r'dogs', line, re.M|re.I)

if searchObj:

   print ("search --> searchObj.group() : ", searchObj.group())

else:

   print ("Nothing found!!")
#!/usr/bin/python

import re



phone = "2004-959-559 # This is Phone Number"



# Delete Python-style comments

num = re.sub(r'#.*$', "", phone)

print ("Phone Num : ", num)



# Remove anything other than digits

num = re.sub(r'\D', "", phone)    

print ("Phone Num : ", num)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# A Regular Expression (RegEx) is a sequence of characters that defines a search pattern. For example, ^a...s$ means any 5 letter

# string starting with an a and ending with an s. 



import re

pattern = '^a...s$'

test_string = 'abyss'

result = re.match(pattern, test_string)

if result:

  print("Search successful.")

else:

  print("Search unsuccessful.")	
# MetaCharacters

# Metacharacters are characters that are interpreted in a special way by a RegEx engine. Here's a list of metacharacters:



# [] . ^ $ * + ? {} () \ |



# [] - Square brackets



# Square brackets specifies a set of characters you wish to match.



# You can complement (invert) the character set by using caret ^ symbol at the start of a square-bracket.



# [^abc] means any character except a or b or c.

# [^0-9] means any non-digit character.

# You can also specify a range of characters using - inside square brackets.



# [a-e] is the same as [abcde].

# [1-4] is the same as [1234].

# [0-39] is the same as [01239].



# You can complement (invert) the character set by using caret ^ symbol at the start of a square-bracket.



# [^abc] means any character except a or b or c.

# [^0-9] means any non-digit character.



# he re.findall() method returns a list of strings containing all matches.



# Program to extract numbers from a string

import re

string = 'hello 12 hi 89. Howdy 34'

pattern = '\d+' #\d - Matches any decimal digit. Equivalent to [0-9]

result = re.findall(pattern, string) 

print(result)

# Output: ['12', '89', '34']
# re.split()

# The re.split method splits the string where there is a match and returns a list of strings where the splits have occurred.



import re

string = 'Twelve:12 Eighty nine:89.'

pattern = '\d+' #\d - Matches any decimal digit. Equivalent to [0-9]

result = re.split(pattern, string) 

print(result)

# Output: ['Twelve:', ' Eighty nine:', '.']



# You can pass maxsplit argument to the re.split() method. It's the maximum number of splits that will occur.



import re

string = 'Twelve:12 Eighty nine:89 Nine:9.'

pattern = '\d+'

maxsplit = 1

# split only at the first occurrence

result = re.split(pattern, string, maxsplit) 

print(result)

# Output: ['Twelve:', ' Eighty nine:89 Nine:9.']
# The method returns a string where matched occurrences are replaced with the content of replace variable.

# re.sub(pattern, replace, string)



# Program to remove all whitespaces

import re

# multiline string

string = 'abc 12\de 23 \n f45 6'

pattern = '\s+' # \s - Matches where a string contains any whitespace character. Equivalent to [ \t\n\r\f\v]

# empty string

replace = ''

new_string = re.sub(pattern, replace, string) 

print(new_string)

# Output: abc12de23f456
# The re.subn() is similar to re.sub() expect it returns a tuple of 2 items containing the new string 

# and the number of substitutions made



# Program to remove all whitespaces

import re

# multiline string

string = 'abc 12\de 23 \n f45 6'

# matches all whitespace characters

pattern = '\s+'

# empty string

replace = ''

new_string = re.subn(pattern, replace, string) 

print(new_string)

# Output: ('abc12de23f456', 4)
# The re.search() method takes two arguments: a pattern and a string. 

# The method looks for the first location where the RegEx pattern produces a match with the string.

# If the search is successful, re.search() returns a match object; if not, it returns None.

# match = re.search(pattern, str)



import re

string = "Python is fun"

# check if 'Python' is at the beginning

match = re.search('\APython', string) # \A - Matches if the specified characters are at the start of a string.

if match:

  print("pattern found inside the string")

else:

  print("pattern not found")  

# Output: pattern found inside the string
# Match object

# You can get methods and attributes of a match object using dir() function.

# Some of the commonly used methods and attributes of match objects are:



# match.group()

# The group() method returns the part of the string where there is a match.



import re

string = '39801 356, 2102 1111'

# Three digit number followed by space followed by two digit number

pattern = '(\d{3}) (\d{2})' # \d - Matches any decimal digit. Equivalent to [0-9]

# match variable contains a Match object.

match = re.search(pattern, string) 

if match:

  print(match.group())

else:

  print("pattern not found")

# Output: 801 35
# The start() function returns the index of the start of the matched substring. 

# Similarly, end() returns the end index of the matched substring.

match.start()

match.end()
# The span() function returns a tuple containing start and end index of the matched part.

match.span()
# The re attribute of a matched object returns a regular expression object. Similarly, string attribute returns the passed string.



match.re

re.compile('(\\d{3}) (\\d{2})')

match.string
# When r or R prefix is used before a regular expression, it means raw string. 

# For example, '\n' is a new line whereas r'\n' means two characters: a backslash \ followed by n.

# Backlash \ is used to escape various characters including all metacharacters. However, using r prefix makes \ treat as a normal character.



import re

string = '\n and \r are escape sequences.'

result = re.findall(r'[\n\r]', string) 

print(result)

# Output: ['\n', '\r']