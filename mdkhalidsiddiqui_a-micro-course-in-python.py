# hello.py
# By Md Khalid
# My first Python program

# hello.py
# By Md Khalid
# My first Python program
print("Hello World")
a = 1
b = 2
c = a + b
print(c)

# A program to demonstrate the use of print function
# Print out a shape of a diamond
print("     *")
print("    * *")
print("   *   *")
print("  *     *")
print(" *       *")
print("*         *")
print(" *       *")
print("  *     *")
print("   *   *")
print("    * *")
print("     *")

# A program to demonstrate the use of print function
# Print out Unimovies program
print("Welcome to Unimovies!");
print() #this code prints a newline
print("Thursday July 30 at 7.15pm: Inside Out")
print()
print("Director: Pete Docter, Ronaldo Del Carmen")
print("Writer: Pete Docter, Meg LeFauve")
print("Starring: Diane Lane, Amy Poehler, Mindy Kaling")
print("Released: June 18, 2015")
print("Rating: PG")
print("Runtime: 102 minutes")
print("Websites: http://movieweb.com/movie/inside-out-2015")
# Print Full Name using string concatenation
first_name = "Md Khalid"
last_name = "Siddiqui"
full_name = first_name + " " + last_name
print("My name is " + full_name + ".")

#Print Subjects list enrolled in using concatenation
subject1 = "ISIT111"
subject2 = "MATH101"
subject3 = "ACCY113"
print("My enrolled subjects are: " + subject1 + ", " + subject2
+ ", " + subject3 + ".")


#Whats wrong with this code

#fav_number = 7
#print("My favorite number is " + fav_number)

#Corrected code
fav_number = 7
print("My favorite number is " + str(fav_number))

#str function application example 1
number1 = 10
number2 = 20
sum = number1 + number2
print("The sum of " + str(number1) + " and " + str(number2)
+ " is " + str(sum) + ".")

#str function application example 2
SECOND_PER_MINUTE = 60
minute = 5
second = minute * SECOND_PER_MINUTE
print(str(minute) + " minutes has " + str(second) + " seconds")

#Convert Number to string example
number1 = 10
number2 = 20
sum = number1 + number2
print("The sum of " + str(number1) + " and " + str(number2)
+ " is " + str(sum) + ".")

subject1_code = "CSCI111"
subject1_mark = 80
subject2_code = "MATH103"
subject2_mark = 75
subject3_code = "PHYS101"
subject3_mark = 85
print("Exam result. " + subject1_code + ": " + str(subject1_mark) + ", " + subject2_code + ": " + str(subject2_mark) + ", " + subject3_code + ": " + str(subject3_mark) + ".")

subject1_code = "CSCI111"
subject1_mark = 80
subject2_code = "MATH103"
subject2_mark = 75
subject3_code = "PHYS101"
subject3_mark = 85
print("Exam result. " \
 + subject1_code + ": " + str(subject1_mark) \
 + ", " \
 + subject2_code + ": " + str(subject2_mark) \
 + ", " \
 + subject3_code + ": " + str(subject3_mark) \
 + ".")

#How to include quotes/tab/single quote/back slash/backspace in your output 

#print("Welcome to Unimovies!")
#print("Thursday July 30 at 7.15pm: "Inside Out"") #Invalid Syntax
#use escape sequence[\"..string inside quotes..\"] for displaying quotes as output

print("Welcome to Unimovies!")
print("Thursday July 30 at 7.15pm: \"Inside Out\"")

######Escape Sequence Meaning
######\\ Backslash (\)
######\' Single quote (')
######\" Double quote (")
######\b Backspace
######\n New line
######\t Tab

#Example : Escape seqence 

print("Your details:\n")
print("\tName: \"Khalid Siddiqui\"")
print("\tSN: \"2012345\"")
print("\nEnrolment record:\n")
print("\tMATH101")
print("\tCSCI201")
#Escape sequence examples

print("Escape sequence:")
print("\\n : Insert a newline.")
print("\\t : Insert a tab.")
print("\\\" : Insert a double quote character.")   #Why 3 times ///
print("\\\' : Insert a single quote character.")
print("\\\\ : Insert a backslash character.")
fname = "John"
lname = "Smith"
print("Hi {0} {1}!".format(fname, lname))
print("{1} {2} is {0} years old".format(fname, lname, 20)) #WTF
print("And his favorite number is {0}".format(7))
#left , right and center alignment
#< left
#> right
#^ center
print("Exam result:")
print("{0:<10}{1:<15}{2:>5}{3:>5}".format("COMM104", "Commerce I", "75", "D"))
print("{0:<10}{1:<15}{2:>5}{3:>5}".format("FIN201", "Accounting", "85", "HD"))
print("{0:<10}{1:<15}{2:>5}{3:>5}".format("MTH202", "Analysis", "100", "HD"))
print("{0:<10}{1:<15}{2:>5}{3:>5}".format("ECTE110", "Circuits", "90", "HD"))
print("1234567890123456789012345678901234567890")

print("Exam result:")
print("{0:<10}{1:^15}{2:>5}{3:>5}".format("COMM104", "Commerce I", "75", "D"))
print("{0:<10}{1:^15}{2:>5}{3:>5}".format("FIN201", "Accounting", "85", "HD"))
print("{0:<10}{1:^15}{2:>5}{3:>5}".format("MTH202", "Analysis", "100", "HD"))
print("{0:<10}{1:^15}{2:>5}{3:>5}".format("ECTE110", "Circuits", "90", "HD"))
print("1234567890123456789012345678901234567890")

#Table of 2 w/o alignment
print("{0} x {1} = {2}".format(1, 5, 1*5))
print("{0} x {1} = {2}".format(2, 5, 2*5))
print("{0} x {1} = {2}".format(3, 5, 3*5))
print("{0} x {1} = {2}".format(4, 5, 4*5))
print("{0} x {1} = {2}".format(5, 5, 5*5))
print("{0} x {1} = {2}".format(6, 5, 6*5))
print("{0} x {1} = {2}".format(7, 5, 7*5))
print("{0} x {1} = {2}".format(8, 5, 8*5))
print("{0} x {1} = {2}".format(9, 5, 9*5))
print("{0} x {1} = {2}".format(10, 5, 10*5))

#Table of 2 with right alignment
print("{0:>2} x {1:>1} = {2:>2}".format(1, 5, 1*5))
print("{0:>2} x {1:>1} = {2:>2}".format(2, 5, 2*5))
print("{0:>2} x {1:>1} = {2:>2}".format(3, 5, 3*5))
print("{0:>2} x {1:>1} = {2:>2}".format(4, 5, 4*5))
print("{0:>2} x {1:>1} = {2:>2}".format(5, 5, 5*5))
print("{0:>2} x {1:>1} = {2:>2}".format(6, 5, 6*5))
print("{0:>2} x {1:>1} = {2:>2}".format(7, 5, 7*5))
print("{0:>2} x {1:>1} = {2:>2}".format(8, 5, 8*5))
print("{0:>2} x {1:>1} = {2:>2}".format(9, 5, 9*5))
print("{0:>2} x {1:>1} = {2:>2}".format(10, 5, 10*5))
#Examples Naming Convention
first_name = "John"                       #lower_case_with_underscores for normal variables
last_name = "Smith"                       #lower_case_with_underscores for normal variables
full_name = first_name + " " + last_name  #lower_case_with_underscores for normal variables
fav_number = 7                            #lower_case_with_underscores for normal variables
subject1 = "ISIT111"
subject2 = "MATH101"
subject3 = "ACCY113"
SECOND_PER_MINUTE = 60                    #UPPER_CASE_WITH_UNDERSCORES for constant
minute = 5
second = minute * SECOND_PER_MINUTE

#3.5 Miscellaneous
#Candy Box: $4/each or $10/for 3 boxes
box_count = 50
group_of_3_count = box_count // 3
left_over_count = box_count - 3 * group_of_3_count
cost = group_of_3_count * 10 + left_over_count * 4
print("{0} candy boxes cost: ${1}".format(box_count, cost))
#Full name program with input command
first_name = input("Please enter your first name: ")
last_name = input("Please enter your last name: ")
full_name = first_name + " " + last_name
print("Your name is " + full_name + ".")

#using int() function

input1 = input("Enter the first integer: ")
num1 = int(input1)
input2 = input("Enter the second integer: ")
num2 = int(input2)
sum = num1 + num2
print("The sum of {0} and {1} is {2}".format(num1, num2, sum))
#Using float() function

input1 = input("Enter the first number: ")
num1 = float(input1)
input2 = input("Enter the second number: ")
num2 = float(input2)
sum = num1 + num2
print("The sum of {0} and {1} is {2}".format(num1, num2, sum))
#Code without rounding
input_a = input("Enter number of students with grade A: ")
grade_a = int(input_a)
input_b = input("Enter number of students with grade B: ")
grade_b = int(input_b)
input_c = input("Enter number of students with grade C: ")
grade_c = int(input_c)
total_student = grade_a + grade_b + grade_c
# calculate percentage
pct_a = grade_a * 100 / total_student
pct_b = grade_b * 100 / total_student
pct_c = grade_c * 100 / total_student
print("Total number of students: {0}".format(total_student))
print("Grade statistics: A {0}%, B {1}%, C {2}%".format(pct_a,
pct_b, pct_c))

#Code with rounding
input_a = input("Enter number of students with grade A: ")
grade_a = int(input_a)
input_b = input("Enter number of students with grade B: ")
grade_b = int(input_b)
input_c = input("Enter number of students with grade C: ")
grade_c = int(input_c)
total_student = grade_a + grade_b + grade_c
# calculate percentage
pct_a = round(grade_a * 100 / total_student)
pct_b = round(grade_b * 100 / total_student)
pct_c = round(grade_c * 100 / total_student)
print("Total number of students: {0}".format(total_student))
print("Grade statistics: A {0}%, B {1}%, C {2}%".format(pct_a,
pct_b, pct_c))

#rounding previous result upto 2 decimal places
input1 = input("Enter assignment 1 mark: ")
a1 = float(input1)
input2 = input("Enter assignment 2 mark: ")
a2 = float(input2)
input3 = input("Enter assignment 3 mark: ")
a3 = float(input3)
# calculate average mark
average = round((a1 + a2 + a3)/3, 2)    #rounding to 2 decimals
print("Average mark: {0}".format(average))
box_input = input("How many boxes would you like? ")
box_count = int(box_input)
UNIT_PRICE = 4
cost = UNIT_PRICE * box_count
if (box_count == 1):
 print("{0} box: ${1}".format(box_count, cost))      #Note the indention to specify the statement is under if block
else:
 print("{0} boxes: ${1}".format(box_count, cost))    #Note the indention to specify the statement is under else block
#Example 1
#grade A: 100-80, B: 79-60, C: 59-40, D: 39-0
mark_input = input("Please enter mark: ")
mark = int(mark_input)
if (mark >= 80):
 grade = "A"
elif (mark >= 60):
 grade = "B"
elif (mark >= 40):
 grade = "C"
else:
 grade = "D"
print("Mark {0}, Grade {1}".format(mark, grade))
#Exmple 2
#Even/Odd Integer
num_input = input("Enter an integer: ")
num = int(num_input)
if (num == 0):
 print("This number is zero")
elif (num > 0):                 #same indention as 'if'
 if (num % 2 == 0):
  print("This number is positive and even")
 else:
  print("This number is positive and odd")
else:                           #same indention as 'if' and 'elif'
 if (num % 2 == 0):
  print("This number is negative and even")
 else:
  print("This number is negative and odd")
#print maximum of given numbers
input1 = input("Enter the 1st integer: ")
n1 = int(input1)
input2 = input("Enter the 2nd integer: ")
n2 = int(input2)
input3 = input("Enter the 3rd integer: ")
n3 = int(input3)
max_n = n1
if (n2 > max_n):
 max_n = n2

if (n3 > max_n):
 max_n = n3
print("Max of {0}, {1}, {2} is {3}".format(n1, n2, n3, max_n))


# show menu
print("------------------------------------------------")
print(" Welcome to Science Park! ")
print()
print("Admission Charges: Adult $35, Child $20 ")
print("Stargazing Show: $10/person ")
print()
print("Free Science Park Hats if you spend $150 or more")
print("10% discount if you spend $200 or more ")
print("------------------------------------------------")
print()

# take order from user
print("Please make your order.")
print()

# ask number of adults
adult_input = input("Enter number of adults: ")
adult = int(adult_input)
# ask number of children
child_input = input("Enter number of children: ")
child = int(child_input)
# ask the additional star show
star_show_input = input("Add Stargazing Show: (Y/N) ")

ADULT_PRICE = 35
CHILD_PRICE = 20
SHOW_PRICE = 10
# calculate the total charge, no discount calculation yet
adult_cost = ADULT_PRICE * adult
child_cost = CHILD_PRICE * child
if ((star_show_input == "Y") or (star_show_input == "y")):
 show_cost = SHOW_PRICE * (adult + child)
else:
 show_cost = 0
total_cost = adult_cost + child_cost + show_cost

DISCOUNT_MIN = 200 # the minimum amount to have discount
DISCOUNT_PCT = 10 # the discount percentage
# calculate the final charge, take discount into consideration
if (total_cost >= DISCOUNT_MIN):
 # eligible for discount
 final_cost = total_cost * (100 - DISCOUNT_PCT) / 100
 print("Total cost: ${0}".format(total_cost))
 print("Discount {0}%".format(DISCOUNT_PCT))
 print("Final charge: ${0}".format(final_cost))
else:
 # not eligible for discount
 final_cost = total_cost
 print("Final charge: ${0}".format(final_cost))

FREE_HAT_MIN = 150 # the minimum amount to have free hat
# check Free Hat
if (total_cost >= FREE_HAT_MIN):
 print("Please collect your free Science Park Hats at the counter.")
print()
print("Enjoy your day!!!")
#Numbers in each row, and in each column,
#and in each diagonals, all add up to the same number!
print("Magic square")
print("m11 m12 m13")
print("m21 m22 m23")
print("m31 m32 m33")
#get user input
input11 = input("Enter m11: ")
m11 = int(input11)
input12 = input("Enter m12: ")
m12 = int(input12)
input13 = input("Enter m13: ")
m13 = int(input13)
input21 = input("Enter m21: ")
m21 = int(input21)
input22 = input("Enter m22: ")
m22 = int(input22)
input23 = input("Enter m23: ")
m23 = int(input23)
input31 = input("Enter m31: ")
m31 = int(input31)
input32 = input("Enter m32: ")
m32 = int(input32)
input33 = input("Enter m33: ")
m33 = int(input33)
# display the numbers
print("{0:>10}{1:>10}{2:>10}".format(m11, m12, m13))
print("{0:>10}{1:>10}{2:>10}".format(m21, m22, m23))
print("{0:>10}{1:>10}{2:>10}".format(m31, m32, m33))
# calculate the sums
r1 = m11 + m12 + m13
r2 = m21 + m22 + m23
r3 = m31 + m32 + m33
c1 = m11 + m21 + m31
c2 = m12 + m22 + m32
c3 = m13 + m23 + m33
d1 = m11 + m22 + m33
d2 = m13 + m22 + m31
print("Row sums: {0}, {1}, {2}".format(r1, r2, r3))
print("Column sums: {0}, {1}, {2}".format(c1, c2, c3))
print("Diagonal sums: {0}, {1}".format(d1, d2))
# checking the magic square condition
if ((r2 == r1) and (r3 == r1) and (c1 == r1) and (c2 == r1) and
(c3 == r1) and (d1 == r1) and (d2 == r1)):
 print("This is a magic square")
else:
 print("This is not a magic square")
