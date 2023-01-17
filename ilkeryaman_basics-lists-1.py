myList = ["apple", 35, 5.15, True]
print(type(myList))
print(myList)
myList = list("apple")
print(myList)
print(myList[3:])
print(myList[2:4])
print(myList[2::2])
list1 = [1, 2, 3]
list2 = [3, 4, 5]
list3 = list1 + list2
print(list3)
print(list3 * 2)
list3[3:] = [7, 10, 9, 8]
print(list3)
list3.append(22)       # add an element to list
list3.append("ilker")
print(list3)
list3.pop()     # remove last element
print(list3)
list3.pop(1)    # remove element at index 1
print(list3)
list3.sort()
print(list3)
list3.sort(reverse=True)  # reverse sorting
print(list3)