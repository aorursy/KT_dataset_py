#Some Ways to create Strings.

my_string = 'Hello'

print(my_string)



my_string = "Hello"

print(my_string)



my_string = '''Hello'''

print(my_string)
#String with multiple Line

my_string = """

Hello, welcome to

the world of Python"""

print(my_string)
my_string="Jovian"

print(my_string,type(my_string))
str = 'Jovian'

print('str = ', str)



#first character

print('str[0] = ', str[0])



#last character

print('str[-1] = ', str[-1])



#slicing 2nd to 5th character

print('str[1:5] = ', str[1:5])



#slicing 2nd to 2nd last character

print('str[1:-1] = ', str[1:-1])
str1 = 'Hello'

str2 ='World!'



# using +

print('str1 + str2 = ', str1 + str2)



# using *

print('str1 * 3 =', str1 * 3)
count = 0

for letter in 'Hello World':

    if(letter == 'l'):

        count += 1

print(count,'letters found')
'a' in 'Jovian'

'at' not in 'battle'
str = 'jovian'



# enumerate()

list_enumerate = list(enumerate(str))

print('list[(enumerate(str)] = ', list_enumerate)



#character count

print('len(str) = ', len(str))
# using triple quotes

print('''He said, "What's there?"''')



# escaping single quotes

print('He said, "What\'s there?"')



# escaping double quotes

print("He said, \"What's there?\"")
# default(implicit) order

default_order = "{}, {} and {}".format('Amit','Bhavesh','Nilesh')

print('\n-   Using Default Order   -')

print(default_order)



# order using positional argument

positional_order = "{1}, {0} and {2}".format('Amit','Bhavesh','Nilesh')

print('\n-  Using Positional Order   -')

print(positional_order)



# order using keyword argument

keyword_order = "{a}, {b} and {n}".format(a='Amit',b='Bhavesh',n='Nilesh')

print('\n-  Using Keyword Order   -')

print(keyword_order)
name="John"

age=25

print(f"My name is {name},and I am {age} years old.")
str.lower()
str.upper()
str="Hello World"

str.split()
l=['Happy','Birthday']

''.join(l)
str.find('o')
str.replace('Hello','Happy')