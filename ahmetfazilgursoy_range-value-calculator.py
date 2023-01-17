#file name: Range Calculator
#How many number are there? Number
Counter = float(input("Please enter number of numbers: "))
Coef_Counter = Counter #Coefficient Counter(Static Counter)
SortingList = []
Loop = 1 #it starting with first loop
#Loops for enter numbers.
while(Counter!=0):
    Enter = float(input("Please enter %s. number = " % Loop))
    SortingList.append(Enter)
    Loop += 1
    Counter -= 1
SortingList.sort()
print("\n***\nYour Data Set ==> ",SortingList)
print("\n*******\nYour Maximum Value = ",max(SortingList))
print("\n**********\nYour Minimum Value = ",min(SortingList))
Range = max(SortingList)-min(SortingList)
print("\n********************\nYour Range Value = ",Range)
#print is for control
