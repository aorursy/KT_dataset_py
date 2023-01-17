# We'll learn how to automate this sort of list in the next lecture
list1 = [1,2,3,4,5,6,7,8,9,10]
for num in list1:
    print(num)
17 % 5
# 3 Remainder 1
10 % 3
# 2 Remainder 4
18 % 7
# 2 no remainder
4 % 2
for num in list1:
    if num % 2 == 0:
        print(num)
for num in list1:
    if num % 2 == 0:
        print(num)
    else:
        print('Odd number')
# Start sum at zero
list_sum = 0 

for num in list1:
    list_sum = list_sum + num

print(list_sum)
# Start sum at zero
list_sum = 0 

for num in list1:
    list_sum += num

print(list_sum)
for letter in 'This is a string.':
    print(letter)
tup = (1,2,3,4,5)

for t in tup:
    print(t)
list2 = [(2,4),(6,8),(10,12)]
for tup in list2:
    print(tup)
# Now with unpacking!
for (t1,t2) in list2:
    print(t1)
d = {'k1':1,'k2':2,'k3':3}
for item in d:
    print(item)
# Create a dictionary view object
d.items()
# Dictionary unpacking
for k,v in d.items():
    print(k)
    print(v) 
list(d.keys())
sorted(d.values())