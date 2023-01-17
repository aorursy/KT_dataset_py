text1 = 'It does not matter how slowly you go so long as you do not stop. Confucius'
len(text1)
# splitting the sentence into words
text2 = text1.split(' ')
text2
len(text2)
# words in text2 that are of length more than 3
text3 = [w for w in text2 if len(w)>3]
text3
# Words with initial letter as capital
text4 = [w for w in text2 if w.istitle() ]
text4
# words that end withs
text5 = [w for w in text2 if w.endswith('s') ]
text5
text6 = 'To be or not to be'
# the set() function finds out the unique elements from a list
len(text6)
text7 = text6.split(' ')
text7
text8 = set(text7)
text8 # be is not repeated here but to is repeated 
text9 = (set([w.lower() for w in text7 ]))
text9 #lower() is used to convert all the letters in a word to lowercase
text10 = 'Delhi'
text10.startswith('D')
text10.endswith('h')
text10.islower()
text10.isupper()
text10.istitle()
text11= 'Capital'
text11.isalpha()
text11.isdigit()
text11.isalnum() #checks for both alphabets and digits
text12='Del0011'
text12.isalnum()
text13 = 'Numbers have an important story to tell. They rely on you to give them a voice'
text14 = text13.lower()
text14
text15 = text14.upper()
text15
text16 = text14.capitalize()
text16
text17 = text13.split('e')
text17
'e'.join(text17) #using join() on split() to retrieve original text
text18 = '    Numbers have an important story to tell. \n They rely on you to give them a voice'
print(text18)
text19 = text18.split(' ')
text19 # we see empty spaces and newline character which are unnecessary
text20 = text18.splitlines()
text20
# strip() is used for removing all whitespaces
text21 = text18.strip()
text21
text22 = 'Stats is the future \n'
text22.rstrip() #removes \n
text23 = 'For every two degrees the temperature goes up, check-ins at ice cream shops go up by 2%'
text23.find('e') # finds the first e from the left of the string
text23.rfind('e') # finds the first e from the right of the string
text23.replace('e','E')