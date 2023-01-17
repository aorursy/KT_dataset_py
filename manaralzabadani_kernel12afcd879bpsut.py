#Manar Al-zabadani

#Rotate an array 90 degree clockwise without extra space and with low complexity



#1. define the array and print it

L=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

print("Array Before Rotate 90 degree:")

for i in range(4):

    for j in range(4): 

         print(str(L[i][j])+' ', end = '')

    print()



#2. rotate the array using this structure which created by me

size=4

numberOfRotates = int((size*size) / 4)

dec = 0

row = 0

col = size - 1

rowC = row

colC = col

for i in range(numberOfRotates):

   temp = L[row][rowC]

   L[row][rowC]=L[rowC][col]

   L[rowC][col] = L[col][colC]

   L[col][colC] = L[colC][row]

   L[colC][row] = temp

   rowC+=1;

   colC-=1;

   if i >= (size-2)+dec:

      row+=1

      col-=1

      rowC = row

      colC = col

      dec += 2

    

#3. print the array after rotate it

print("Array Aftear Rotate 90 degree:")

for i in range(4):

    for j in range(4): 

         print(str(L[i][j])+' ', end = '')

    print()

               