#file name: Arithmetic Mean and Mean
#This Calculator for Positive Value, otherwise it may be occured false result!
#How many number are there? Number
Counter = float(input("Please enter number of numbers: "))
Number_of_Numbers = Counter
Total = 0
Loop = 1 #it starting with first loop
#Total for mean value.
#Loops for enter numbers.
while(Counter!=0):
    Enter = float(input("Please enter %s. number = " % Loop))
    Loop+=1
    Counter-=1
    Total = Total + Enter
#print("%d" %Total)
#print is for control
#avarage operation
Mean = Total / Number_of_Numbers
print("Mean value is equal to = %.3f" % Mean)

