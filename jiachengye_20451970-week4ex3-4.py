# Aim of the program: Examines three variables x, y, and z 
# And check which one is the lagest odd or none of them are odd 
x = input('Enter an Integer: ')            # ask the user to input an integer
y = input('Enter an Integer: ')
z = input('Enter an Integer: ')
x = int(x)                          # convert the string input to an integer
y = int(y)
z = int(z)
if x%2 ==0 and y%2 == 0 and z%2 == 0:  
# perform the modulo operation and # check the remainder is equal to 0 or not 
  print('None of them are Odd')                      # print the number is even 
else:
  print('Odd')                                       # print the number is odd 
  print('Done with conditional')                       # print the work done
  if x%2!= 0 and y%2 == 0 and z%2 == 0:
    print('x is largest')
  if y%2!= 0 and x%2 == 0 and z%2 == 0:
    print('y is largest')
  if z%2!= 0 and x%2 == 0 and y%2 == 0:
    print('z is largest')
  if x > y and x > z and x%2!= 0: 
    print('x is largest') 
  elif y > z and y%2!= 0 : 
    print('y is largest') 
  elif z%2!= 0: 
    print('z is largest')
  
numXs = int(input('How many times should I print the letter X? ')) 
toPrint = '' 
toPrint = []
while (numXs!= 0 ):
    numXs-=1
    toPrint.append('X')
print(toPrint)   

#Find out maxium even number from inputed numbers
num_list = []
for i in range(10):
    num = int(input('Number' + str(i+1) + ':'))
    num_list.append(num)    
    #the append function is used to add the num input by the user to the list
    
even = []
for numb in num_list:
    if numb%2 == 0:
        even.append(numb) #add the even number to the list
        
if even:
    print (sorted(even)[-1]) # sort the list and output the maxium even
else:
    print ("No even")
    
X = int(input('Enter an integer:'))
ans = 0
i = 0
for pwr in range(2,7):
    while ans**pwr < X:
        ans+=1
        if ans**pwr!= X:
            ans = 0
            i+=1
    print('root is:', ans ,'pwr is:', pwr)
if i == 5:
    print ('no such pair of integers exists')
        