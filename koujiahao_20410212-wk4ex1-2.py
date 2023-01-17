# Exercise 1

miles = float(input("Enter miles: "))              
km = round(miles/0.62137,4)
meters = round(1000 * km,2)

print(str(miles),"miles is equivalent to")
print(str(km),"km","/",str(meters),"meters")
# Exercise 2

name = input("What is your name: ")
age =  input("What is your age: ")

age_in_2047 = int(age) + (2047-2020)

print("Hi",name +"!","In 2047 you will be",str(age_in_2047) +"!")