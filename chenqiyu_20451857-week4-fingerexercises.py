#Exercise 1: Write a program that accept an user input in miles and 

#convert it km as well as meters. 

#The formulas for conversion are km = miles /0.62137 

#and meters = 1000 * (value in km)

#For example, if the user enters 5

#the program should print the result in the following form:

                    #5 miles is equivalent to

                    #8.0467 km / 8046.7 meters
print('Please enter the miles')



miles = input('The entered miles is : ')

miles = float(miles)

km = miles/0.62137

meters = 1000*km

print('***',miles, 'miles is equilvalent to','***')

print('***',km,'km','/',meters,'meters','***')
#Exercise 2 :Write a program that asks the user’s name and then age. 

#Use appropriate variable names to store these variables. 

#Calculate how old the user will be in 2047 years. 

#For example, if the user enters Bob and 20, the program should print Hi Bob! In 2047 you will be 47!
print('Please enter your name')

name = input('Your entered name is :')

name = str(name)

print('Please enter your age')

age = (input('Your entered age is :'))

age = int(age)

age_in2047 = age + 27

print('***','Hi',name, '!', 'In 2047 you will be',age_in2047,'***')