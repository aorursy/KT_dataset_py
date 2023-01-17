sumabc=1000

a = 1

while (a**2) % (sumabc-a) != 0:

  a=a+1



cnb=a**2/(sumabc-a)

c=int((sumabc-a+cnb)/2)

b=int(sumabc-a-c)

print('a=',a,'b=',b,'c=',c)

print('abc=',a*b*c)