#file name: Median Calculator
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
#print(SortingList)
#print is for control

if (Coef_Counter%2 == 0): #If quantity of Number is even.
    middle = Coef_Counter/2
    second_middle = middle-1
    Even_Median = (SortingList[int(middle)]+SortingList[int(second_middle)])/2
    print("Median is equal to %.3f" % Even_Median)
    
else:#If quantity of Number is odd.
    middle = (Coef_Counter/2)+0.5
    middle = middle-1
    print("Median is equal to %.3f" % SortingList[int(middle)])
    
