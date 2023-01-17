num1=int(input('enter 1st number'))
num2=int(input('enter 2nd number'))
res=num1+num2
print(res)
import math
list_sq=[]
list_sqrt=[]
list1 =[1,3,6,9,11,15]
for x in list1:
    list_sq.append(x**2)

for x in list1:
    list_sqrt.append(x**(1/2))
    
print (list_sq)
print (list_sqrt)
lst1 = ['aaa','bbb','ccc','ddd','eee']
join_str=[]                                   
index = 0                                         
totalloops= int((len(lst1)/2)) 
length=len(lst1)

if length%2 == 0:
    for i in range (totalloops):
        join_str.append("".join([lst1[index],lst1[index+1]]))
        index = index+2
      
    
else:
    for i in range(totalloops):
        join_str.append("".join([lst1[index],lst1[index+1]]))
        index = index+2
    join_str.append(lst1[length-1])  

print(join_str)




def printValues():
 l = list()
 for i in range(1,21):
  l.append(sqrt.i)
 print(l)

printValues()



