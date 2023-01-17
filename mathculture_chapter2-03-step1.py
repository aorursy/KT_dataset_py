a=[4, 8, 3, 4, 1]



print(len(a))

print(max(a))

print(min(a))

print(sum(a))

print(sorted(a))
1.2 + 3.8

10 // 100

1 >= 0

'Hello World' == 'Hello World'

not 'Chainer' != 'Tutorial'

all([True, True, False]) 

any([True, True, False])

abs(-3)

2 // 0
a=[4, 8, 3, 4, 1]

print(a[1:])

print(a[:-1])

a.append(100)

print(a)
a=[4, 8, 3, 4, 1] 

b = [i % 2 for i in a]

print(b)

print(sum(b))

c = [i for i in b if i == 1]

print(c)
a = [str(i) for i in range(100)]

" ".join(a)
f"{1.0 / 7.0:.9f}"
class DataManager:

    def __init__(self, x, y, z):

        self.x = x

        self.y = y

        self.z = z

        

    def add_x(self, delta):

        self.x += delta

    

    def add_y(self, delta):

        self.y += delta

    

    def add_z(self, delta):

        self.z += delta

    

    def sum(self):

        return self.x + self.y + self.z
data_manager = DataManager(2, 3, 5)

print(data_manager.sum())  # => 10

data_manager.add_x(4)      # => data_manager.x の値が 2 から 6 に更新される

print(data_manager.sum())  # => 14

data_manager.add_y(0)      # => data_manager.y の値が 3 から 3 に更新される

print(data_manager.sum())  # => 14

data_manager.add_z(-9)     # => data_manager.z の値が 5 から -4 に更新される

print(data_manager.sum())  # => 5

def f(a):

    a = [6, 7, 8]



def g(a):

    a.append(1)
def somefunction():

    a0 = [1, 2, 3]

    f(a0)

    print(a0)



    a1 = [1, 2, 3]

    g(a1)

    print(a1)



somefunction()
a = [i for i in range(100)]

flag = [True for i in range(100)]



for i in range(2, 100):

    if flag[i]:

        prime = a[i]

        print(prime)

        flag[prime] = False

        for j in range(prime, 99 // prime + 1):

            flag[prime*j] = False