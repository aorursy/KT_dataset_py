#file name: Quartiles Calculator
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

if (Coef_Counter%2 == 0):   #Do not forget, 2nd quartile means that median!!!!
    middle = Coef_Counter/2
    second_middle = middle-1
    Second_Quartile = (SortingList[int(middle)]+SortingList[int(second_middle)])/2
    print("\n***\n2nd Quartile is equal to %.3f" % Second_Quartile)
else:
    middle = (Coef_Counter/2)+0.5
    middle = middle-1
    print("\n***\n2nd Quartile is equal to %.3f" % SortingList[int(middle)])
if (Coef_Counter%2 == 0):
    middle = Coef_Counter/4
    second_middle = middle-1
    First_Quartile = (SortingList[int(middle)]+SortingList[int(second_middle)])/2
    print("\n***\n1st Quartile is equal to %.3f" % First_Quartile)
else:
    middle = (Coef_Counter/4)+0.75
    middle = middle-1
    print("\n***\n1st Quartile is equal to %.3f" % SortingList[int(middle)])
if (Coef_Counter%2 == 0):
    middle = Coef_Counter*3/4
    second_middle = middle-1
    Third_Quartile = (SortingList[int(middle)]+SortingList[int(second_middle)])/2
    print("\n***\n3rd Quartile is equal to %.3f" % Third_Quartile)
else:
    middle = (Coef_Counter*3/4)+0.5
    middle = middle-1
    print("\n***\n3rd Quartile is equal to %.3f" % SortingList[int(middle)])