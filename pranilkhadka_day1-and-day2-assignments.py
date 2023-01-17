num1 = 5
num2 =7
res = num1 +num2
print(f"The sum of {num1} and {num2} is {res}")
print("The sum of {} and {} is {}".format(num1 ,num2, res))
list_square=[]
list_square_root=[]
list1 =[1,3,6,9,11,15]
for i in list1:
    list_square.append(i**2)

for i in list1:
    list_square_root.append(i**(1/2))
    
print (list_square)
print (list_square_root)
lst_input = ['abc','def','ghi','jkl','mno']
join_str=[]                                   
index = 0                                         
total_no_of_loops= int((len(lst_input)/2)) # Since we need to add 2 consecutive string, the no of time need to loop is halfed
length=len(lst_input)

if length%2 == 0:
    for i in range (total_no_of_loops):
        join_str.append("".join([lst_input[index],lst_input[index+1]]))
        index = index+2
      
    
else:
    for i in range(total_no_of_loops):
        join_str.append("".join([lst_input[index],lst_input[index+1]]))
        index = index+2
#LAST STRING IN THE LIST( eg: int(9/2) =  4, so it will loop 4 times which means last item is yet to be added in the join_str list)
    join_str.append(lst_input[length-1])  

print(join_str)
a = [10,20,30,40]
b=[]
for i in(a):
    b.append(i)
print(b)
a = ['aaa','bbb','ccc']
b = []
for i in a: # Here i is not an index, it refers to a specific element
    print(''.join([i,i]))
z = []
for i in range(3,31,3):
    z.append(i)
print(z)
# Methods for printing

# Method 1 ( String COncatenation)

print('Sum of '+ str(num1) + ' and '+ str(num2) +' is '+ str(res))
# method 2 ( COmma seperator)
print('Sum of',num1,'and',num2,'is',res)
# Method 3 : Formatted String
print(f'Sum of {num1} and {num2} is {res}')