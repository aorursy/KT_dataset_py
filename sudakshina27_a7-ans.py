colors_list = ['Red','Green','White','Black']

colors_list.pop(1)

colors_list.pop(-2)

print(colors_list)
list1 = ['M','na','i','Su']

list2 = ['y','me','s','mit']

list = []

i=0

while i<=3:

    x = list1[i]+list2[i]

    list.append(x)

    i+=1

print(list)
list1 = [100,200,300,400,500]

list1.reverse()

print(list1)
list1 = ["Hello ", "Welcome "]

list2 = ["Dear", "Sir"]

l=[]

for i in list1:

    for j in list2:

        x = i + j

        l.append(x)

print(l)
def fun(v): 

    a = [""] 

    if (v in a): 

        return False

    else: 

        return True

   

list1 = ["AI", "", "Emma", "DL", "", "NLP"]

l = []

filtered = filter(fun, list1) 

  

for s in filtered: 

    l.append(s)

print(l)
list1 = ['Sumit', 'Keerat',['Gunit', 'Ankur', ['Akshat', 'Tanmay'], 'Tanishka'], 'Preeti','Nishi','Shivani']

list1[2][2].append('Purvi')

print(list1)
tup = (10, 20, 30, 40, 50)

new_tup = tup[::-1] 

print(new_tup)
Tup = ("Orange", ['Black', 'Grey', 30], (5, 15, 25))

print(Tup[1][1])
tup = (50)

type(tup)
tuple1 = (11, 22)

tuple2 = (99, 88)

tuple_new = tuple2 + tuple1 

print('tuple1 = (',tuple2,')','(',tuple1,')')

print(tuple_new)
tuple1 = ('cherry', 'banana', 'guava', 'apple', 'mango', 'lichi')

new_tuple = tuple1[3:5]

print(new_tuple)
Set1 = {"NumPy", "Pandas", "OpenCv"}

Set2 = {"Scikit", "Matplotlib", "Keras"}

Set1.union(Set2)
Set1 = {"NumPy", "OpenCv","Scikit-Learn"}

Set2 = {"Scikit", "Matplotlib", "Keras","Scikit-Learn"}

Set3 = Set1 & Set2

print(Set3)
Set1 = {"NumPy", "OpenCv","Scikit-Learn"}

Set2 = {"Scikit", "Matplotlib", "Keras","Scikit-Learn"}

Set3 = Set1.union(Set2)

print(Set3)
Set1 = {"NumPy", "OpenCv","Scikit-Learn"}

Set2 = {"Scikit", "Matplotlib", "Keras","Scikit-Learn"}

Set1.difference_update(Set2)

print(Set1)
Set1 = {"NumPy", "OpenCv","Scikit-Learn"}

Set2 = {"Scikit", "Matplotlib", "Keras","Scikit-Learn"}

Set3 = Set1.symmetric_difference(Set2)

print(Set3)
Set1 = {"NumPy", "OpenCv","Scikit-Learn"}

Set2 = {"Scikit", "Matplotlib", "Keras","Scikit-Learn"}

Set1.symmetric_difference_update(Set2)

print(Set1)
Set1 = {"NumPy", "OpenCv","Scikit-Learn"}

Set2 = {"Scikit", "Matplotlib", "Keras","Scikit-Learn"}

Set3 = Set1 & Set2

print(Set3)