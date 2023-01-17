x = input("enter x dog long x=:")
if x.isdigit():
   y = input("enter x dog long y=:")
   if y.isdigit():
       z = input("enter z dog long z=:")
       if z.isdigit():
           x = int(x)
           y = int(y)
           z = int(z)
           if x > y:
               x, y = y, x
           if y > z:
               y, z = z, y
           if z % 2 == 1:
               print(z)
           elif y % 2 == 1:
               print(y)
           elif x % 2 == 1:
               print(x)
           else:
               print("三个数中没有奇数")
       else:
           print("enter number")
   else:
       print("enter number")
else:
   print("enter number")
numXs = int( input("How many times should I print the letter X? ") )
toPrint = ''
if numXs < 1:
    print ("Value of numXs is less than or equal to zero")
else:
    while numXs > 0:
      toPrint = toPrint + "x"
      numXs = numXs -1
    print(toPrint)
