# Lets create a text object to work upon
txt = "A word is meaningful combination of more than one character."
print(txt)
# Find the length of a String (no. of charater).
len(txt)
## Split a sentence into words
# The split() function below breaks our sentence into words based on the split character provided as paraneter.
# The default splitting character is space. split() returns a list of tokens/words.
words = txt.split(' ')
print(words)
# How many words/splits we have?
len(words)
# len() works in same way for strings & lists both.
## Filterinag words in a list
# Suppose we want to subset only for those words that are 4 or more character long
[word for word in words if len(word) >= 4]
# Find words with first letter as CAPITAL
[word for word in words if word.istitle()]
# Find words ends with 'n'
[word for word in words if word.endswith('n')]
# Edit: MM_20180518
# A SHORTCUT to use key character to find both upper and lower case strings
# Use lower() or upper() in conjunction.

[word for word in 'To be or not to be'.split(' ') if word.upper().startswith('T')]
### Finding Unique words among the list of words
# When you are working with text, you wouldn't want to keep dulicate words in your bag.
# This will cause redundant works for the program, and will consume extra memory as well

# Let's create a string/sentence with duplicate words.
txt1 = 'To be or not to be'           # Here 'to' & 'be' are duplicates.
words1 = txt1.split(' ')                # Split the words

# Print number of elements in the list
len(words1)
# Print unique number of elements in the list
len(set(words1))
# Here, we had 4 unique words but the instances of 'to' have different cases.
# We can convert these in lower cases to remove complete redundancy.
len(set([word.lower() for word in words1]))
# We can subset the string just like a list.
# index in python starts with 0, and is exclusive of upper range
txt2 = 'Hello World!'

txt2[0:4]
# We can omit lower bound in case we want to filter from start
txt2[:4]     # This will return same result as above
# We can omit upper bound in case we want to filter till the end
txt2[4:]
# We can also put subsiquent filters to put a series of conditions
txt2[6:][:1]      # To filter only the letter 'W' from the string
### Fetch characters from a String
# We can break our string into a character list in two ways
# Using the list() function:
print(list(txt2))

# By looping in:
print([c for c in txt2])

# Both would give same output.
txt3 = "Hello"
# Starts With
txt2.startswith('H')    # Whether the string starts with 'H'
# Ends With
txt3.endswith('H')     # Whether the string ends with 'H'
# Find character in string
'e' in txt3            # Whether we have character 'e' in string
# Whether it is in upper case
txt3.isupper()
# Whether it is in lower case
txt3.islower()
# Whether it is in camel case
txt3.istitle()
# Whether the string has all alphabets only
txt3.isalpha()
# Whether the string is all numerics/digits
txt3.isdigit()
# Whether the string is alpha-numeric
'Hello123'.isalnum() 
# Change cases of strings
print(txt3.lower())   # To lower case
print(txt3.upper())   # To upper case
print(txt3.title())   # to camel case or title case
# SPLIT AND JOIN strings

# We have seen how to split a string at the beginning, let's see how to join them back.
txt4 = "This is a sentence."
words4 = txt4.split()
print(words4)

# We can join the tokens back to a sentence using join() function
' '.join(words4)
# SPLIT String into sentences.

# The function splitlines() can be used to split a string into sentences.
# It splits by the new line character. Useful when working with text files (will see in later sections).
txt5 = "Hi, how are you?\nHope everything is fine at your end."
txt5.splitlines()
# Stripping/Trimming white spaces from string
txt6 = "  This is a string with white spaces on both sides..  "

# Length of the text before stripping
len(txt6)
# Stripping white spcae from both side
print(txt6.strip())
print(len(txt6.strip()))
# Stripping white spcae from right side only
print(txt6.rstrip())
print(len(txt6.rstrip()))
# Stripping white spcae from left side only
print(txt6.lstrip())
print(len(txt6.lstrip()))
## Find a character in a string from left
print(txt2)
txt2.find('o')   # Returns location/index of first occurance (from left) 'o' in String.
## Find a character in a string from right
txt2.rfind('o')   # Returns location/index of first occurance (from right) 'o' in String.