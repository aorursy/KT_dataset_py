
# Write a program that accept an user input in miles and convert it km as well as meters.

miles= input("Enter the Number of Miles:") #funtion for creating input interface


miles=float(miles)                        #converting to float

km= miles/0.62137                         #inputting conversion formula

meters= 1000*km                           #formula for converting fom meters to kilometers

mile_km=miles/0.62137

mile_meters=1000*mile_km

print("{:.2f} miles is equivalent to" .format(miles)) #function for printng user feedback

print("{:6.4f}km/{:6.1f} meters" .format(mile_km, mile_meters)) # funtion for printing output

#Write a program that asks the user’s name and then age

name=input("Please input your name:")

age= input("How old are you?:")

#name and age are string variables

age= int(age)
age_in_year_2047=age+2047-2020

#fuction to print feedback

print("Hi,",name, "in 2047 you will be",age_in_year_2047,"years!")

