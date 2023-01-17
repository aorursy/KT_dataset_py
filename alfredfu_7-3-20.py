#1
import random
number = i
number = random.randint(1,100)
mylist = []
while i != number:
    num = int(input('choose your answer'))
    mylist.append(num)
    if num == number:
        print('you are correct')
        break
    elif num < number:
        print('you guessed too little of a number')
    else:
        print('you guessed too large of a number')
print(mylist)
#2
rows = 6
for i in range(rows):
    for x in range(i):
        print('*',end = '')
    print('')
#3
for x in [1,2,3,4,5,6,7,8,9,10]:
    y = x+1
    print(x,'*',y,'=',x*y)