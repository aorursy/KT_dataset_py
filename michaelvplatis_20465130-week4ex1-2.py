
## Finger Exercise 1 
## Program to Convert Miles into Kilometers and then into Meters
# Figuring out how many Miles we need to convert
Miles = input('How many Miles do we need to convert into Kilometers:')
x=float(Miles)
## Kilometers equals Miles / Conversion Rate, so we need to define the conversion rate
convert = 0.621371
#Kilometers found by dividing the x (miles) by convert (conversion factor)
kilometers = x/convert
# Convert Kilometers into Meters
meters = 1000 * kilometers
#print out a message explaining how many kilometers the input of miles is
print(x, "miles is converted to",meters, 'meters')
## Inputs needed are 'Name' and 'Age'
Name = input("Hello, what is your name?")
Age = input("How old are you?")
x = int(Age)

## Output needed is 'Age_in_2047' as an integer, adding 27 to your age will work since 2047 is 27 years from now.
Age_in_2047 = x + 27

## Return a friendly string of text and int
print ('Hi',Name,', you will be', Age_in_2047,'in 2047!')