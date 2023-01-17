#File name: Mode Calculator
#Warning! This Program supports only one Mode Value!!!
import statistics as stat
#How many number are there? Number
Counter = float(input("Please enter number of numbers: "))
Coef_Counter = Counter #Coefficient Counter(Static Counter)
OurList = []
Loop = 1 #it starting with first loop
#Loops for enter numbers.
while(Counter!=0):
    Enter = float(input("Please enter %s. number = " % Loop))
    OurList.append(Enter)
    Loop += 1
    Counter -= 1
OurList.sort()
print("Your Data Set is:\n")
print(OurList)
try:
    print("Your Mode Value = %.2f" % stat.mode(OurList))
except:
    print("There are a lot of Mode Value, Our Program obtains only one mode.")
