#The library "re" supports regular expression

import re



from IPython.display import Image

import os
sampletext_to_search = '''abcdefghijklmnopqurtuvwxyz

ABCDEFGHIJKLMNOPQRSTUVWXYZ

1234567890

123abc



Hello HelloHello



MetaCharacters (Need to be escaped):

. ^ $ * + ? { } [ ] \ | ( )



utexas.edu

 

821-545-4271 

823.559.1938



daniel-mitchell@utexas.edu



Mr. Johnson

Mr Smith

Ms Davis

Mrs. Robinson

Mr. T '''
re.split('\n',sampletext_to_search)
#2.use "finditer" function to search the matching character just like the  pattern created in previous step.

matching_results = pattern_literal.finditer(sampletext_to_search)



#3.print the results

for char in matching_results:

    print(char)

# cross verify the result by searching the index and see the results 

print(sampletext_to_search[68:71])
#create a pattern to find dot(.) character

pattern_specialchar = re.compile(r'.')



matching_results = pattern_specialchar.finditer(sampletext_to_search)



#print the results

for char in matching_results:

    print(char)

pattern_specialchar = re.compile(r'\.')



matching_results = pattern_specialchar.finditer(sampletext_to_search)



#print the results

for char in matching_results:

    print(char)
#lets find any number character.



#set the pattern

pattern_anynum = re.compile(r'\d')
#pass the entire text to pattern to find matching result

matching_results = pattern_anynum.finditer(sampletext_to_search)



#print the results

for num in matching_results:

    print(num)
pattern_anyNumChar = re.compile(r'\d\w')

matching_results =  pattern_anyNumChar.finditer(sampletext_to_search)



#print the results

for char in matching_results:

    print(char)




# Search_text: Hello HelloHello

pattern_wordboundry = re.compile(r'Hello\b')

matching_results =  pattern_wordboundry.finditer(sampletext_to_search)



#print the results

for char in matching_results:

    print(char)
# Search_text: Hello HelloHello

pattern_wordboundry = re.compile(r'\bHello\b')



matching_results = pattern_wordboundry.finditer(sampletext_to_search)



#print the results

for char in matching_results:

    print(char)









pattern_charset = re.compile(r'[13]')



matching_results = pattern_charset.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_charset = re.compile(r'[34]\w')



matching_results = pattern_charset.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_charset = re.compile(r'[a-z][a-z]')



matching_results = pattern_charset.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_charset = re.compile(r'[^a-z][^a-z]')



matching_results = pattern_charset.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_charsetgrp = re.compile(r'(bcd|efg|ijkl)')



matching_results = pattern_charsetgrp.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_charsergrp = re.compile(r'([A-Z]|io)[a-z]')



matching_results = pattern_charsetgrp.finditer(sampletext_to_search)



for char in matching_results:

    print(char)

                                



                                

                                
pattern_quantify = re.compile(r'Mr\.?\s[A-Z]')



matching_results = pattern_quantify.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_quantify = re.compile(r'M(s|rs)')



matching_results = pattern_quantify.finditer(sampletext_to_search)



for char in matching_results:

    print(char)



                              
pattern_quantify = re.compile(r'\d{3}[.-]\d{4}')



matching_results = pattern_quantify.finditer(sampletext_to_search)



for char in matching_results:

    print(char)
pattern_email = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')



matching_results = pattern_email.finditer(sampletext_to_search)



for char in matching_results:

    print(char)