num=int(input("Enter a number whose table must be printed: "))

i=1

while i<=10:

    print("%d x %d = %d" %(num,i,(num*i)))

    i+=1
char=input("Enter an alphabet: ")

ascii=ord(char)

if(((ascii>=97 and ascii<=122) or (ascii>=65 and ascii<=90)) and 

   (char=='a' or char=='e' or char=='i' or char=='o' or char=='u' or char=='A' or char=='E' or char=='I' 

   or char=='O' or char=='U')):

    print(char," is a vowel")

elif(((ascii>=97 and ascii<=122) or (ascii>=65 and ascii<=90)) and 

   (char!='a' and char!='e' and char!='i' and char!='o' and char!='u' and char!='A' and char!='E' and char!='I' 

   and char!='O' and char!='U')):

    print(char," is a consonant")

else:

    print(char," is not an alphabet")
month=input("Enter month name (Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec): ")

if(month=='Jan'or month=='Mar'or month=='May'or month=='Jul'or month=='Aug'or month=='Oct'or month=='Dec'):

    print("Number of days = 31")

elif(month=='Feb'):

    print("Number of days = 28/29")

elif(month=='Apr'or month=='Jun'or month=='Sep'or month=='Nov'):

    print("Number of days = 30")

else:

    print("Wrong input.")
s1=int(input("Enter 1st side: "))

s2=int(input("Enter 2nd side: "))

s3=int(input("Enter 3rd side: "))

if(s1==s2 and s2==s3):

    print("Triangle is Equilateral")

elif(s1==s2 or s2==s3 or s3==s1):

    print("Triangle is Isosceles")

else:

    print("Triangle is Scalene")
n = [10, 70, 30]

n.sort()

length = len(n)

if length%2==0:

    m1=n[length//2]

    m2=n[length//2-1]

    m=(m1+m2)/2

else:

    m=n[length//2]

print("Median is: " + str(m)) 
x1=int(input("Enter x co=ordinate of 1st point: "))

y1=int(input("Enter y co=ordinate of 1st point: "))

x2=int(input("Enter x co=ordinate of 2nd point: "))

y2=int(input("Enter y co=ordinate of 2nd point: "))

dist= ((((x2-x1 )**2) + ((y2-y1)**2) )**0.5)

print("Distance between (%d,%d) and (%d,%d) = %f" %(x1,y1,x2,y2,dist))
b=int(input("Enter the Base: "))

p=int(input("Enter the Perpendicular: "))

h=(p**2 + b**2)**0.5

print("Hypotenuse = ",h)
d = int(input("Input days: ")) 

d = d * 3600 * 24

h = int(input("Input hours: ")) 

h = h * 3600

mins = int(input("Input minutes: "))

mins = mins * 60

secs = int(input("Input seconds: "))

time = d + h + mins + secs

print("Total seconds = ", time)
s = 0

a = int(input('Enter a number: '))

a1=a

while a!=0:

    rem = a%10

    s = s + rem

    a = a//10

print("Sum of all digits in ",a1,"= ",s)
char=input("Enter a character: ")

ascii=ord(char)

print("ASCII value of ",char,"is: ", ascii)
x=30

y=20

print("%d+%d=%d" %(x,y,(x+y)))
base = int(input('Enter base: '))

height = int(input('Enter height: '))

area = base * height

print("Area is: ", area)
name = 'sample'

print(name[10]+name[12])
import calendar

print(calendar.calendar(2025))