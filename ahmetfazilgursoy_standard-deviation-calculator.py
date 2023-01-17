#file name: Standard Deviation and Its Step
from math import sqrt
#How many number are there? Number
Counter = float(input("Please enter number of numbers: "))
Static_Counter = Counter #Coefficient Value of Counter
Total = 0
Loop = 1 #it starting with first loop
#Total for mean value.
#Loops for enter numbers.
Our_Data_Set = []
while(Counter!=0):
    Enter = float(input("Please enter %s. number = " % Loop))
    Our_Data_Set.append(Enter)
    Loop+=1
    Counter-=1
    Total = Total + Enter
Our_Data_Set.sort()
print("Your Data Set ==>",Our_Data_Set)
print("\n***\nAdditions of your Data Set = %.3f" %Total)
#print is for control
#Mean Value Operation
Avarage = Total/Static_Counter
print("\n******\n Mean Value = %.3f" % Avarage)
#Mean Value -(minus) Values Operation,After absolute value and square operation, then add this values...
Addition=0
for i in range(int(Static_Counter)):
    Subs = abs(Avarage-Our_Data_Set[i])
    Square = Subs**2
    Addition = Addition+Square
Standard_Deviation = sqrt(Addition)
print("\n*********\n Standard Deviation = %.3f" % Standard_Deviation)
