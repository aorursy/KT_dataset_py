#Finger Exercise 1

miles = input('How many miles need to be conerted in km:')
x = float(miles)
#Convert miles to km

km = x/0.62137
#Convert km to meters

meters = km * 1000

#Final product
print(x,"miles is equivalent to",km,"km/",meters,"meters")
#Finger Exercise 2
name = input("Enter Name:")
age = input("Enter your age:")
x = int(age)
age_in_2047 = x + 2047-2020
print('Hi',name,'! In 2047 you will be',age_in_2047)
