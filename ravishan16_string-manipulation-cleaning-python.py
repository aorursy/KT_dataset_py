!ls ../
print('double quoted "" string')

print("single quoted '' string")

print("""Multi line String/Text 

        Use Triple double/single quotes""")

import string

print ("String Punctuations ",string.punctuation)

print ("String Printable ",string.printable)
print("lower() copy of the string converted to uppercase- ","Lorem ipsum".lower())

print("upper() copy of the string converted to uppercase- ","Lorem ipsum".upper())

print("capitalize() first character capitalized - ","lorem ipsum".capitalize())

print("swapcase() copy of the string with uppercase characters converted to lowercase and vice versa - ","Lorem ipsum".swapcase())

print("title() copy of the string with title case- ","Lorem ipsum".title())
print("Length of String - ",len("Lorem ipsum"))

print("Concatenate multiple Strings - ","Lorem"+" "+"ipsum"+" "+str(1))

print("Split String (default is space) to list - ","Lorem ipsum".split())

print("Split comma seperated String to list - ","Lorem,ipsum".split(','))

print("First letter -","Lorem ipsum"[0])

print("Last letter -","Lorem ipsum"[-1])

print("Extract first 3 letters -","Lorem ipsum"[:3])

print("Extract last 3 letters -","Lorem ipsum"[-3:])

print("Remove first 3 letter -","Lorem ipsum"[3:])

print("Remove last 3 letter -","Lorem ipsum"[:-3])

print("Reverse a String -","Lorem ipsum"[::-1])

print("Substring range of letters 3 to 8 -","Lorem ipsum"[2:8])
print("Remove Leading/Trailing Whitespaces -","  Lorem ipsum    ".strip())

print("Remove Leading Whitespaces -","  Lorem ipsum    ".lstrip())

print("Remove Trailing Whitespaces -","  Lorem ipsum    ".rstrip())
import re

re.sub(r'\W', '','01.Lorem_ipsum!?')
import re

re.sub(r'[^a-zA-Z0-9]', '','01.Lorem ipsum!?')
import re

re.sub(r'[^0-9.]', '','01.50 Lorem ipsum!?')
import re

re.sub(r'[^a-zA-Z]', '','01.Lorem ipsum!?')
print("Replace all occurance of , with ; -","Lorem, ipsum, Lorem, ipsum ".replace(',',';'))

print("Replace first occurance of , with ; -","Lorem, ipsum, Lorem, ipsum ".replace(',',';',1))

print("Replace first two occurance of , with ; -","Lorem, ipsum, Lorem, ipsum ".replace(',',';',2))
print("Find first occurance of , in the text from left:","Lorem, ipsum, Lorem, ipsum ".find(','))

print("Find first occurance of , in the text from right:","Lorem, ipsum, Lorem, ipsum ".rfind(','))

print("Count number of occurance of , in the text:","Lorem, ipsum, Lorem, ipsum ".count(','))
