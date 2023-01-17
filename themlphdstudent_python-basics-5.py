with open('../input/demo-file-for-file-handling/python-practice.txt') as content:

    programming_language = content.read()

    print(programming_language)

    

# output

# Python

# Java

# R

# HTML
with open('../input/demo-file-for-file-handling/python-practice.txt') as filecontent:

    for line in filecontent:

        print(line)



# output

# Python



# Java



# R



# HTML


with open('../input/demo-file-for-file-handling/python-practice.txt') as filecontent:

    lines = filecontent.readlines()

    

for line in lines:

    print(line)



# output

# Python



# Java



# R



# HTML
# this operation is not possible in kaggle 

with open('../input/demo-file-for-file-handling/cars.txt', 'w') as file_content:

    file_content.write('Audi\n')

    file_content.write('BMW\n')

    file_content.write('Toyota')
# this is also not possible in kaggle as kaggle is only read only system

with open('../input/demo-file-for-file-handling/python-practice.txt', 'a') as file_content:

    file_content.write('CSS\n')

    file_content.write('Ruby\n')



with open('programming_language.txt') as content:

    programming_language = content.read()

    print(programming_language)

    

# output

# Python

# Java

# R

# HTML

# CSS

# Ruby
a = 10

b = 0



print(10/0)



# Traceback (most recent call last):

#  File "demo.py", line 1, in <module>

#    print(10/0)

# ZeroDivisionError: division by zero
a = 10

b = 0



try:

  print(a/b)

except ZeroDivisionError:

  print("You can't divide number by 0.")

  

# output

# You can't divide number by 0.