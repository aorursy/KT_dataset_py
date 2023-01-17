print("WELCOME PYTHON BASİC TUTORİAL")
#print() =>> This function very important because we print everything in programming languages with this function.

print("Omer Faruk Elikara")
#VARİABLES

#var = 20        =>  #int

#day = "monday"  =>  #string

#flo = 2,15      =>  #float
day0 = "sun"

day1 = "day"

day2 = day0 + day1

print(day2)
num0 = 10

num1 = 11

num2 = num0 + num1

print(num2)



num3 = "10"

num4 = "11"

num5 = num3 + num4

print(num5)
a = "Omer"

b = 1.6

c = 1.4

d = "111"

variable_type = type(a)             #Which data type

print(variable_type )           



print(len(a))  #length of data type



print(round(b))

print(round(c))    #rounding



type(int(d))  #converting a data type to a number

#FUNCTİONS



#user defined function

var1 = 5

var2 = 10

def first_function(x,y):

    output = x+y

    

    return output

sonuc = first_function(var1,var2)

print(sonuc)



#default flexible function

def circumference(r):

    pi =3

    output = 2*pi*r

    return output 

circumference(2)
def calculate(boy,kilo,*temp):

    print(temp)

    output = (boy+kilo)*temp[1]

    return output



calculate(10,20,5,6,1)
output = lambda x: x*x

print(output(5))
#LIST



firstlist = ["sunday","monday","tuesday","wednesday","thursday","friday","saturday"] 

print(firstlist[0:3])

print(firstlist[-1])

firstlist.append(9)

print(firstlist)

firstlist.remove(9)

print(firstlist)
lists = [1,7,9,5,6,4,3,2,8]

lists.sort()

print(lists)
#DİCTİONARY



dictin = {"Ömer":20, "Ali":19,"Mehmet":25}   #keys = string     values = int

print(dictin["Ömer"])