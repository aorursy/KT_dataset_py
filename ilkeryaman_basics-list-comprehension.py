list1 = [1, 2, 3, 4, 5, 6]
list2 = []

for x in list1:
    list2.append(x)
    
print(list2)
list3 = [i for i in list1]

print(list3)
list4 = [[1, 2], [3, 4, 5], [6, 7, 8, 9, 10]]
list5 = list()

for i in list4:
    for y in i:
        list5.append(y)

print(list5)
list6 = [y for i in list4 for y in i]

print(list6)