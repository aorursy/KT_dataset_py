#handle inputs
distance = input ('Enter the distance in miles: ')                              # input the distance
alpha = float(distance)                                                         # change the string to float

#handle calcualtaions
km = alpha /0.62137                                                             #calcualte km
meter = 1000 * km                                                               #calculate meter

#handle outputs
####print ( distance,"mile(s) is equivalent to","\n",km,"km/",meter, "meters")  # (optional)print out the answer in 2 lines
print ( distance,"mile(s) is equivalent to")                                    # print out the answer
print ( round(km,4),"km/",round(meter,1), "meters")                             # print out the answer
#handle input
name = input ("What is your name? ")                                            #ask for name
age = input ("What is your age now? ")                                          #ask for age now
age_2020 = float (age)                                                          #make sure the age is truned from string to float

#handle calcualtions
age_2047 = age_2020 + 27                                                        #2047 is 27 years from now

#handle outputs
print("Hi",name,"! In 2047 you will be",age_2047,"!")                           #printing the needed outputs