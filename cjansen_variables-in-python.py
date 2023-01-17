1+1
2-1
2*4
2**30
2*2*2*2
8/2
9/2
9//2
9%2

type(8)
type(9/2)
a=55
a
a/2
A/2
b = a/2
b
a
b
type(a)
type(b)
a + b
type(a+b)
h = 1
h=2
h
del h
h
print(a)
print(b)
print(a,b)
print("a")
type("a")
print(1)
print("1")
type(1)
type("1")
print("1"+b)
print("1" + str(b))
print(int("1") + b)
name = "Charles"
type(name)
len(name)
name[0]
name[3]
myList = [90, 11, 23, 89897, 675]
type(myList)
myList[0]
len(myList)
myList[5]
myList2 = ["hello", 3, 6.5]
myList2
myList2[-1]
myList2[-2]
myList2[-3]
l1 = myList2.append(myList)
print(l1)
myList2
myList.append("test")
myList
myList2
l1 = ["a","b","a","c","2","d","r","g","dd","2"]
l2 = l1
l2.remove("a")
l2
del l2[3]
l2
l2[5]
l2[5] ="change"
l2
l1  #!!
l3 = l1.copy()
l3.append("will it work?")
l1
l3
l3.insert(2,33)
l3
l3.extend([5,6])
l3
l3.append([5,6])
print(l3)
bigList = list(range(100))
print(bigList) #show without print too, one number per line
len(bigList)
bigList[-1]
bigList[90:] #90 inclusive
bigList[:10] #10 not inclusive
bigList[15:22]
bigList[15:22:2] #2 is the step
bigList[::7]
print(bigList[::-1])
print(bigList[90:70:-2])#from, to! 90 first 
print(bigList[70:90:-2])
test = bigList[5:10]
test[2]
test[2] = 88
test
print(bigList[5:10])


letterList = ["A","B","C","D","E","F","G","H","I","J","K"]

myTuple = (9898, 7)
myTuple[0]
myTuple[0]=1
emptyTuple=()
type(emptyTuple)
print(a, b)
a, b = b, a
print(a,b)
tuple1 = 1,
type(tuple1)
tuple1
tuple2 = (1,)
tuple2
dico = {"key1" : "value1"}
dico
dico2 = {"name" : "Billy", "age": 43, "job" : "researcher"}
dico2
dico2["age"] 
dico2["age"]=5
dico2
dico2["adress"] = "55 water street"
dico2
dico2.pop("age")
dico2
del dico2["adress"]
dico2
boardgame = {}
boardgame["a",1] = "white tower"
boardgame

a = "abracadabra"
b = "alacazam"
a=set(a)
b=set(b)
print(a)
print(b)
print(a - b)
l4=[1,2,7,2,1,9,2,4,9]
l4
set(l4)
print(list(set(l4)))
True
False
type(True)
a = True
a
spglobalToBe = ['o', '&', 'l', 'a', 'b', 'o', 'l', 'G', ' ', 'P', '&', 'S', 'u', 'K']


spglobalToBe = ['o', '&', 'l', 'a', 'b', 'o', 'l', 'G', ' ', 'P', '&', 'S', 'u', 'K']
start = len(spglobalToBe)-3
spglobalToBe = spglobalToBe[start:1:-1]
spglobalToBe