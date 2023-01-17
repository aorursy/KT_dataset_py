# This Notebook will help the beginners to learn various python methods available on strings.
#1. capitalize() - This converts the first character of the string into upper case.

# syntax = string.capitalize()

inp = "ravi"

y = inp.capitalize()

print(y)
#2. casefold() - This method returns a string where all characters are lower case.

# syntax = string.casefold()

inp = "Hello World"

y = inp.casefold()

print(y)
#3. center() - This method will align the string to the center and fill remaining spaces with the given character.

# syntax = string.center(length,character)

inp = "Ravi"

y = inp.center(20,"*")

print(y)
#4. count() - This method will return the number of times a specified value appears in the string.

# syntax = string.count(value,start,end)

inp = "Ravi is learning python"

y = inp.count("a")

print(y)
#5. encode() - This method encodes the string using the specified encoding. If no encoding is specified, UTF-8 will be used.

# syntax = string.encode(encoding = encoding, errors = errors)

inp = "My name is Råvi"

y = inp.encode()

print(y)
#6. endswith() - This method will return true if the string ends with specified value, else false.

# syntax = string.endswith(value,start,end)

inp = "I am learning python"

y = inp.endswith("python")

print(y)
#7. expandtabs() - This method sets the tab size to the specified number of white spaces.

# syntax  = string.expandtabs(tabsize)

inp = "R\ta\tv\ti"

y = inp.expandtabs(5)

print(y)
#8. find() - This method finds the occurance of specified value and it returns -1 if value is not found.

# syntax = string.find(value,start,end)

inp = "My name is Raviteja"

y = inp.find("is")

print(y)
#9. format() - This method formats the specific values in the string.

# syntax = string.format(value1,value2,...)

inp = "My name is {myname}. I am {myage}"

y = inp.format(myname = "Raviteja",myage = 25)

print(y)
#10. format_map() - This method is used to return a dictionary's key value.

# syntax = string.format_map(mapping)

a = {'x':'John', 'y':'Wick'} 

print("{x}'s last name is {y}".format_map(a))
#11. index() - This method finds the first occurance of a specified value.

# syntax = string.index(value,start,end)

x = "Ravi is using Jupyter Notebook for python"

y = x.index("Notebook")

print(y)
#12. isalnum() - This method returns true if all the characters are alphanumeric.

# syntax = string.isalnum()

x = "Ravi is learning python from an e-learning course"

y = x.isalnum()

print(y)
#13. isalpha() - This method returns true if all the characters in the string are alphabetic.

# syntax = string.isalpha()

x = "The current time is 17:23 IST"

y = x.isalpha()

print(y)
#14. isdecimal() - This method returns true if all the characters in the string are decimals.

# syntax = string.isdecimal()

x = "\u0030" #\u0030 is the unicode for 0

y = x.isdecimal()

print(y)
#15. isdigit() - This method returns true if all the characters in the string are digits.

# syntax = string.isdigit()

x = "10000"

y = x.isdigit()

print(y)
#16. isidentifier() - This method returns true if the string is an identifier.

# syntax = string.isidentifier()

# A string is considered a valid identifier if it only contains alphanumeric letters (a-z) and (0-9), or underscores (_). 

# A valid identifier cannot start with a number, or contain any spaces.

x = "Ravi Teja"

y = x.isidentifier()

print(y)
#17. islower() - This method returns true if all the characters in the string are in lower case.

# syntax = string.islower()

x = "raviteja"

y = x.islower()

print(y)
#18. isnumeric() - This method returns true if all the characters in the string are numeric.

# syntax = string.isnumeric()

x = "50000"

y = x.isnumeric()

print(y)
#19. isprintable() - This method returns true if all the characters are printable.

# syntax = string.isprintable()

x = "Hello \t Are you Ravi? "

y = x.isprintable()

print(y)
#20. isspace() - This method returns true if all the characters in the string are white spaces.

# syntax = string.isspace()

x = "   Ravi   "

y = x.isspace()

print(y)
#21. istitle() - This method returns true if the string follows the rules of the title.

# syntax = string.istitle()

x = "HELLO, WELCOME TO MY WORLD"

y = x.istitle()

print(y)
#22. isupper() - This method returns true if all the characters in the string are in upper case.

# syntax = string.isupper()

x = "HELLO, WELCOME TO MY WORLD"

y = x.isupper()

print(y)
#23. join() - This method takes all items in an iterable and join them into a string.

# syntax = string.join(iterable)

mylist = ["Ravi","teja","Raviteja"]

y = "_".join(mylist)

print(y)
#24. ljust() - This method will left align the string using a specified character(default is space) as the fill character.

# syntax = string.ljust(length,character)

x = "Mango"

y = x.ljust(20)

print(y,"is my favourite fruit")
#25. maketrans() - This method returns a mapping table for translation usable for translate() method.

# syntax = string.maketrans()

# maketrans() creates a mapping of the character's Unicode ordinal to its corresponding translation.

dict = {"a": "123", "b": "456", "c": "789"}

string = "abc"

print(string.maketrans(dict))
#26. partition() - This method searches for a specified string, splits the string into three parts.

# First part is the elements which are before the specified string. Second part is the string and third part are the elements after the string.

# syntax = string.partition(value)

x = "Ravi is learning Python"

y = x.partition("is")

print(y)
#27. replace() - This method will replace a specified phrase with another specified phrase.

# syntax = string.replace(oldvalue,newvalue)

x = "My favourite fruit is Apple"

y = x.replace("Apple","Mango")

print(y)
#28. rfind() - This method finds the last occurance of the specified value. It returns -1 if the value is not found.

# syntax = string.rfind(value,start,end)

x = "Data science is a concept to unify statistics, data analysis, machine learning and their related methods in order to understand and analyze actual phenomena with data.[3] It uses techniques and theories drawn from many fields within the context of mathematics, statistics, computer science, and information science. Turing award winner Jim Gray imagined data science as a fourth paradigm of science (empirical, theoretical, computational and now data-driven) and asserted that everything about science is changing because of the impact of information technology and the data deluge"

y = x.rfind("is")

print(y)
#29. rindex() - This method finds the last occurance of a specified value.

# syntax = string.rindex(value,start,end)

x = "Data science is a concept to unify statistics, data analysis, machine learning and their related methods in order to understand and analyze actual phenomena with data.[3] It uses techniques and theories drawn from many fields within the context of mathematics, statistics, computer science, and information science. Turing award winner Jim Gray imagined data science as a fourth paradigm of science (empirical, theoretical, computational and now data-driven) and asserted that everything about science is changing because of the impact of information technology and the data deluge"

y = x.rindex("is")

print(y)
#30. rjust() - This method will right align the string using a specified character as a fill character.

# syntax = string.rjust(length,charcter)

x = "mango"

y = x.rjust(20)

print(y,"is my favourite fruit")
#31. rpartition() - This method searches for last occurance of specified string. Splits the string into 3 elements.

# First element is part before specified string, Second element is the specified string and third part is the part after the specified string.

# syntax = string.rpartition(value)

x = "Ravi is working on python. He loves coding"

y = x.rpartition("python")

print(y)
#32. rsplit() - This method will splits a string into a list starting from right.

# syntax = string.rsplit(separator)

x = "Data science is a concept to unify statistics, data analysis, machine learning and their related methods in order to understand and analyze actual phenomena with data.[3] It uses techniques and theories drawn from many fields within the context of mathematics, statistics, computer science, and information science. Turing award winner Jim Gray imagined data science as a fourth paradigm of science (empirical, theoretical, computational and now data-driven) and asserted that everything about science is changing because of the impact of information technology and the data deluge"

y = x.rsplit(" ")

print(y)
#33. rstrip() - This method removes characters at the end of the string.

# syntax = string.rstrip(characters)

x = "         Ravi        "

y = x.rstrip()

print(y,"is my name")
#34. split() - This method splits the string at specified separator and returns a list

# syntax = strip.split(separator,max split)

x = "Data science is a concept to unify statistics, data analysis, machine learning and their related methods in order to understand and analyze actual phenomena with data."

y = x.split(",")

print(y)
#35. splitlines() - This method splits a string into list. The splitting is done at line breaks.

# syntax = string.splitlines()

x = "Thank you for the music\nWelcome to the jungle"

y = x.splitlines()

print(y)
#36. startswith() - This method returns true if the string starts with specified value, else false.

# syntax = string.startswith(value,start,end)

x = "Ravi is learning python"

y = x.startswith("Ravi")

print(y)
#37. strip() - This method will remove leading and trailing characters of a string.

# syntax = string.strip(characters)

x = "*****Raviteja***"

y = x.strip("*")

print(y)
#38. swapcase() - This method returns string where all upper case letters becomes lower case and vice versa.

# syntax = string.swapcase()

x = "Hello World"

y = x.swapcase()

print(y)
#39. title() - This method converts the first letter of each word into upper case.

# syntax = string.title()

x = "Python methods on strings"

y = x.title()

print(y)
#40. translate() - This method returns a string where each character is mapped to its corresponding character in the translation table.

# syntax = string.translate(table)

firstString = "abc"

secondString = "ghi"

thirdString = "ab"

string = "abcdef"

print("Original string:", string)

translation = string.maketrans(firstString, secondString, thirdString)

# translate string

print("Translated string:", string.translate(translation))
#41. upper() - This method converts a string into upper case.

# syntax = string.upper()

x = "hello world"

y = x.upper()

print(y)
#42. zfill() - This method adds zeroes at the beginning of the string until it reaches a specified length.

# If the value of length parameter is less than the length of the string then no filling will be done.

# syntax = string.zfill(len)

x = "Ravi"

y = x.zfill(6)

print(y)