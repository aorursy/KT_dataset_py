import re # regex library in python
mystring = "preposterous"



# Gives you the index number of the start of the match

mystring.find("rous")
# If it doesn't find the matching string, it returns None

re.search("rous", mystring)
f = open("../input/alice-in-wonderland.txt") 

alice_lines = f.readlines() # Read all the lines

alice_lines = [l.rstrip() for l in alice_lines] # strip whitespace in each line and then add it in list (list comprehension)

f.close()



for line in alice_lines:

     if re.search("Hatter", line): print( line ) # Look through each line and if a line has word "Hatter" in it, print that line
for line in alice_lines:

    if re.search("riddle", line): print(line)
# Find word Hatter but also match not just "H"atter but also "h"atter



for line in alice_lines:

    if re.search("[Hh]atter", line): print(line)
for line in alice_lines:



     if re.search("[0-9][0-9][0-9]", line): print(line)
for line in alice_lines:

    if re.search("b\w\w\wed", line): print( line )
# ... : Three consecutive sequences of any single character

for line in alice_lines:

    if re.search("b...ed", line): print(line)
# Lines with words with at least 7 characters

for line in alice_lines:

    if re.search(".......+", line): print(line)
# look for lines in Alice in Wonderland that start with "The"

for line in alice_lines:

    if re.search("^The", line): print(line)
#  Look for lines that have the word "Alice" occurring in the end of a line

for line in alice_lines:

    if re.search("Alice$", line): print(line)
for line in alice_lines:

    if re.search(r"\bsing\b", line): print(line)