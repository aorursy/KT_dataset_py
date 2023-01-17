#Exercise 3

InSet=set(map(float,input('Please enter the numbers splitted by commas",": ').split(',')))
OutSet=set()

for A in InSet:
    if A%2==1:
        OutSet.add(A)
        
if OutSet:
    print('The largest odd number is ',int(max(OutSet)),'.',sep='')
else:
    print('None of the numbers entered is an odd.')
#Exercise 4

numXs = int(input('How many times should I print the letter X? '))
toPrint = 0

#concatenate X to toPrint numXs times

while numXs != 0:
    numXs -= 1
    toPrint += 1

print("X"*toPrint)