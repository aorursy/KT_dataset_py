from random import randint

a= randint(100,999)

print("Random 3 digit number generated = ",a)

a1=a

s=0

while a!=0:

    r=a%10

    s=s+r

    a=a//10

print("Sum of digits in %d = %d" %(a1,s))
st = 'India has the world 3rd largest population, and india is among the world 5th strongest Army.'.lower()

print("The total occurances of 'India' in the given string = ",st.count("india"))
var = 'I am Indian'

print('Before replacing: ',var)

print('After replacing: ',var.replace('Indian','INDIAN'))
w = input("Enter a string: ")

l = len(w)

print("Length of %s = %d " %(w,l))
string = """ This is a String """



#to print the string without stripping 

print(string)  

  

# to print the string by removing leading and trailing whitespaces 

print(string.strip())    

  

#to print the string by removing This 

print(string.strip(' This is')) 
string =  "www.theax.in" 

print(string)  

n_s = string.strip('www.')

print(n_s.strip('.in'))
string = "www.theax.in".upper()

print(string)
vowels = 'a e i o u'

print(vowels.split())
if 'e' in vowels:

    print("Present")

else:

    print("Not present")
f_n = input("Enter first name: ")

l_n = input("Enter last name: ")

print("%s %s" %(f_n,l_n))
v1 = "AI"

v2 = "was"

v3 = "coined"

v4 = "in"

v5 = "1956"

v="{} {} {} {} {}"

print(v.format(v1,v2,v3,v4,v5))
print("artificial intelligence".capitalize())
print("artificial intelligence".title())