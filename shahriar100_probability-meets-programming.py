import random
odd = [1,3,5,7,9]
even = [0,2,4,6,8]

lst =  []
acc = 0
for i in range(100000):
    num = ''
    num += f"{random.choice(odd)}"
    num += f"{random.choice(even)}"
    num += f"{random.choice(odd)}"
    num += f"{random.choice(even)}"
    num += f"{random.choice(odd)}"

    num = int(num)
    lst.append(num)


acc = 0
for num in set(lst):
    if num <= 78000:
        acc += 1
acc
len(set(lst))
lst = []
def sample_space(flips, n):
    if n == 0:
        print (flips)
        lst.append(flips)
        return
    sample_space(flips + "H", n-1)
    sample_space(flips + "T", n-1)
sample_space('',3)
lst
acc = 0
for sample in lst:
    if 'H' in sample:
        acc  += 1
        
print(acc/len(lst))
