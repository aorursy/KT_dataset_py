import numpy as np
from imageio import imread #ฟังก์ชันที่เอาไว้สำหรับอ่านรูปภาพและเก็บไว้ใน Numpy array
import matplotlib.pyplot as plt

img = imread('../input/thaikeras.jpeg') #ใน workshop นี้เราได้ใส่รูปไว้ใน directory นี้
plt.imshow(img)
plt.show()
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# ผลลัพธ์ที่ได้ "[1, 1, 2, 3, 6, 8, 10]"
!python --version
x = 3
print(type(x)) # ตัวแปรนี้เป็นจำนวนเต็ม พิมพ์ "<class 'int'>"
print(x)       # พิมพ์ "3"
print(x + 1)   # พิมพ์ "4"
print(x - 1)   # พิมพ์ "2"
print(x * 2)   # นี่คือการคูณ พิมพ์ "6"
print(x ** 2)  # นี่คือการยกกำลัง พิมพ์ "9"
print(x / 2)   # นี่คือการหารแบบจำนวนจริง พิมพ์ "1.5"
print(x // 2)  # นี่คือการหารแบบจำนวนเต็ม (ปัดเศษทิ้ง) พิมพ์ "1"

x += 1
print(x)  # พิมพ์ "4"
x *= 2
print(x)  # พิมพ์ "8"

y = 3.0   # ใส่จุดทศนิยมเพื่อให้เป็นตัวแปรของจำนวนจริง 
print(type(y)) # ตัวแปรนี้เป็นจำนวนจริง พิมพ์ "<class 'float'>"
print(y, y + 1, y * 2, y ** 2, y / 2, y // 2)

z = 2. # ใส่จุดทศนิยมโดยไม่มีเลขศูนย์ก็ได้และนิยมใช้กัน
print(type(z), z) # ตัวแปรนี้เป็นจำนวนจริง พิมพ์ "<class 'float'>"
t = True
f = False
print(type(t)) # พิมพ์ "<class 'bool'>"
print(t and f) # Logical AND; พิมพ์ "False"
print(t or f)  # Logical OR; พิมพ์ "True"
print(not t)   # Logical NOT; พิมพ์ "False"
print(t != f)  # Logical XOR; พิมพ์ "True"
hello = 'hello'    # เราสามารถนิยามตัวแปรสตริงด้วย ' '
world = "world"    # หรือ " " ได้โดยไม่มีความแตกต่างกัน
print(hello)       # พิมพ์ "hello"
print(len(hello))  # ฟังก์ชันที่รีเทิร์นความยาวสตริง; พิมพ์ "5"
hw = hello + ' ' + world  # การ concat (เชื่อมต่อ) สตริง ทำได้อย่างง่ายดายเมื่อเทียบกับภาษาอื่น
print(hw)  # พิมพ์ "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # วิธีการใช้ฟังก์ชั่นเสมือน sprintf บน python ก็ทำได้ง่ายๆ
print(hw12)  # พิมพ์ "hello world 12"

# ตัวอย่างของการซ้อน ' ' หรือ " "
print('hello "world"')
print("hello 'world'")
s = "hello world"
print(s.capitalize())  # เปลี่ยนเฉพาะตัวอักษรแรกเป็นตัวพิมพ์ใหญ่
print(s.upper())       # เปลี่ยนทั้งหมดเป็นตัวพิมพ์ใหญ่

s = 'hello'
print(s.rjust(7))      # เปลี่ยนสตริงให้มีความยาวตามที่กำหนด โดยแปะชองว่างไว้ข้างหน้า; พิมพ์ "  hello"
print(s.center(7))     # เปลี่ยนสตริงให้มีความยาวตามที่กำหนด โดยจัดสตริงเดิมไว้ตรงกลาง; พิมพ์ " hello "
print(s.replace('l', '(ell)'))  # Replace ซับสตริง (substring) ที่กำหนด;
                                # พิมพ์ "he(ell)(ell)o"
print('  world '.strip())  # ลบ space ด้านหน้าและด้านหลังของสตริง; พิมพ์ "world"
xs = [3, 1, 2]    # สร้างตัวแปร container ประเภท list
print(xs, xs[2])  # พิมพ์ "[3, 1, 2] 2"
print(xs[0])     # index เริ่มต้นที่ 0; พิมพ์ "3"
print(xs[-1])     # index ที่เป็นเลขลบ จะไล่ข้อมูลมาจากด้านหลังของ list; พิมพ์ "2"
print(xs[-2])     # index ที่เป็นเลขลบ จะไล่ข้อมูลมาจากด้านหลังของ list; พิมพ์ "1"
xs[2] = 'foo'     # สามารถบรรจุข้อมูลได้หลายประเภทใน list เดียว
print(xs)         # พิมพ์ "[3, 1, 'foo']"
xs.append('bar')  # แปะข้อมูลให่ไว้ที่ท้ายสุดของ list
print(xs)         # พิมพ์ "[3, 1, 'foo', 'bar']"
x = xs.pop()      # ลบข้อมูลตัวสุดท้ายออกจาก list และรีเทิร์นไว้ในตัวแปร x
print(x, xs)      # พิมพ์ "bar [3, 1, 'foo']"
nums = list(range(5))     
# range เป็น built-in function ที่เอาไว้สร้าง list ของตัวเลขจำนวนนับเริ่มจาก 0 ให้มีความยาวตามที่กำหนด

print(nums)               # พิมพ์ "[0, 1, 2, 3, 4]"

print(nums[2:4])          
# รีเทิร์น sublist ตั้งแต่ index 2 ถึง index 4-1 = 3; พิมพ์ "[2, 3]"
# หมายเหตุ: ใน syntax ของ Python slicing, index ตัวสุดท้ายจะถูกลบด้วย 1

print(nums[2:])           # รีเทิร์น sublist ตั้งแต่ index 2 ถึงตัวสุดท้าย; พิมพ์ "[2, 3, 4]"
print(nums[:2])           # รีเทิร์น sublist ตั้งแต่ตัวแรก ถึง index 2-1 = 1; พิมพ์ "[0, 1]"
print(nums[:])            # รีเทิร์น sublist ทั้งหมด (มีประโยชน์ใน list หลายมิติ); พิมพ์ "[0, 1, 2, 3, 4]"
print(nums[:-1])          # รีเทิร์น sublist ตั้งแต่ตัวแรก ถึง index -1-1 = -2; พิมพ์ "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # เปลีย่น sublist ใน slice ที่กำหนด
print(nums)               # พิมพ์ "[0, 1, 8, 9, 4]"
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# พิมพ์ "#1: cat", "#2: dog", "#3: monkey", แบบเว้นบรรทัด
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums] # list comprehension
print(squares)   # Prints [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"
d = {'cat': 'cute', 'dog': 'furry'}  # สร้าง dictionary ที่บรรจุ (key,value) ไว้สองคู่
print(d['cat'])       # หา value ใน dictionary ที่คู่กับ key 'cat'; พิมพ์ "cute"
print('cat' in d)     # วิธีเช็คว่ามี key ชื่อ 'cat' ใน dictionary หรือไม่; พิมพ์ "True"
d['fish'] = 'wet'     # กำหนด (key,value) เพิ่มเติม; key = 'fish', value = 'wet'
print(d['fish'])      # พิมพ์ "wet"
# print(d['monkey'])  # ตัวอย่าง KeyError: 'monkey' ไม่ได้เป็น key ใน d
print(d.get('monkey', 'N/A'))  # วิธีการค้นข้อมูลใน dict โดยให้รีเทิร์นค่า default ="N/A" ถ้าไม่มี key นั้นๆ
print(d.get('fish', 'N/A'))    # พิมพ์ "wet"
del d['fish']         # ลบข้อมูลใน dictionary
print(d.get('fish', 'N/A')) # "fish" ไม่อยู่ใน dict d อีกต่อไป; พิมพ์ "N/A"
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# พิมพ์ "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # พิมพ์ "{0: 0, 2: 4, 4: 16}"

animals = {'cat', 'dog'}
print('cat' in animals)   # การเช็คสมาชิกใน set; พิมพ์ "True"
print('fish' in animals)  # พิมพ์ "False"
animals.add('fish')       # เพิ่มสมาชิกใน set
print('fish' in animals)  # พิมพ์ "True"
print(len(animals))       # ดูจำนวนสมาชิก set; พิมพ์ "3"
animals.add('cat')        # เพิ่มสมาชิกที่มีอยู่เดิมแล้ว จะไม่เกิดผลอะไร
print(len(animals))       # พิมพ์ "3"
animals.remove('cat')     # ลบสมาชิกออกจาก set
print(len(animals))       # พิมพ์ "2"
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# จะพิมพ์ลำดับไหนออกมา เช่น "#1: fish", "#2: dog", "#3: cat" หรือลำดับอื่นๆ ก็เป็นไปได้ 

from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # พิมพ์ "{0, 1, 2, 3, 4, 5}"

t = (5, 6)        # สร้าง tuple ที่มีสมาชิก 2 ตัว (หรือ "คู่ลำดับ" นั่นเอง)
print(type(t))    # พิมพ์ "<class 'tuple'>"
print(t[0])       # พิมพ์ "5"
t[0] = 1          # ตัวอย่างที่จะแจ้ง error เนื่องจาก tuple ไม่สามารถเปลี่ยนค่าได้ด้วยเครื่องหมาย '='
d = {(x, x + 1): x for x in range(10)}  # dictionary comprehension เพื่อสร้าง dict ที่มี tuple เป็น keys
print(d[t])       # พิมพ์ "5"
print(d[(1, 2)])  # พิมพ์ "1"

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# พิมพ์ "negative", "zero", "positive"
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # พิมพ์ "Hello, Bob"
hello('Fred', loud=True)  # พิมพ์ "HELLO, FRED!"
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # สร้าง instance ของ Greeter class
g.greet()            # เรียกใช้งาน instance method; พิมพ์ "Hello, Fred"
g.greet(loud=True)   # รียกใช้งาน instance method; พิมพ์ "HELLO, FRED!"
import numpy as np        # วิธีการที่จะเรียกใช้ numpy library โดยกำหนดชื่อย่อว่า np

a = np.array([1, 2, 3])   # สร้าง array 1 มิติ (เวกเตอร์) จาก Python list [1, 2, 3]
print(type(a))            # พิมพ์ "<class 'numpy.ndarray'>"
print(a.shape)            
# shape เป็น method ที่ไว้เรียกดูขนาดของ array 
# พิมพ์ "(3,)" แปลว่าเป็น array 1 มิติ ที่มีสมาชิกเท่ากับ 3 (เวกเตอร์ที่ความยาวเท่ากับ 3)

print(a[0], a[1], a[2])   # พิมพ์ "1 2 3"
a[0] = 5                  # เปลี่ยนค่าใน array
print(a)                  # พิมพ์ "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # สร้าง array 2 มิติ (เมตริกซ์)
print(b.shape)                     # พิมพ์ "(2, 3)" แปลว่าเป็น array 2 มิติ ที่มีความยาวเท่ากับ 2 และ 3 ตามลำดับ
                                   # พูดง่ายๆ ว่าเป็นเมตริกซ์ขนาด 2x3 นั่นเอง 
print(b[0, 0], b[0, 1], b[1, 0])   # พิมพ์ "1 2 4"
import numpy as np

a = np.zeros((2,2))   # สร้าง array 2 มิติขนาด 2x2 ที่มีสมาชิกเป็น 0 ทั้งหมด
print(a)              # พิมพ์ "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # สร้าง array 2 มิติขนาด 1x2 ที่มีสมาชิกเป็น 1 ทั้งหมด
print(b)              # พิมพ์ "[[ 1.  1.]]"

c = np.full((2,2), 7)  # สร้าง array 2 มิติขนาด 2x2 ที่มีสมาชิกเป็นค่าคงที่เท่ากับ 7 ทั้งหมด
print(c)               # พิมพ์ "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # สร้าง "เมตริกซ์เอกลักษณ์" ขนาด 2x2 นั่นคือมีสมาชิกในเส้นแทยงมุมเป็น 1 นอกนั้นเป็น 0 ทั้งหมด
print(d)              # พิมพ์ "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # สร้าง array 2 มิติขนาด 2x2 ที่มีสมาชิกเป็นค่าสุ่ม (random numbers) ในช่วง (0,1)
print(e)                     # พิมพ์ค่าสุ่มอาทิเช่น "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

# [[ 1  2] + [[ 5  6] =  [[ 6  8]
#   [3 4]]    [ 7  8]]    [10 12]]
print(x + y)

# [[-4 -4]
#  [-4 -4]]
print(x - y)

# [[ 5 12]
#  [21 32]]
print(x * y)

# ตัวอย่างการหาร เมื่อเลขไม่ลงตัว สมาชิกใน array จะเปลี่ยนเป็นจำนวนจริงโดยอัตโนมัติเหมือนกรณีการหารตัวเลข (ดูหัวข้อบน)
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
# ยกกำลัง
print(x ** 2)

# ถอดรากที่สองมีสองวิธีคือเรียกฟังก์ชั่นของ numpy หรือ ยกกำลังด้วย 1/2
print(np.sqrt(x))
print(x ** (1/2))
import numpy as np

A = np.random.random((2,3))
B = np.random.random((3,4))
print(A.shape, B.shape)

C = np.dot(A,B)
print(C.shape) # ผลลัพธ์ "(2,4)"

C = np.dot(B,A) # จะแจ้ง Error ตามที่อธิบายด้านบน
print(C.shape)
import numpy as np
A = np.ones((3,3))
x = np.array([1, 1, 1]) # นี่คือ array 1 มิติที่ควรหลีกเลี่ยง

print(np.dot(A,x))
print(np.dot(A,x).shape,'\n')
print(np.dot(x,A))
print(np.dot(x,A).shape)
# ทั้งสองกรณีให้ผลลัพธ์เท่ากันทั้งที่ควรจะ error ในกรณีที่สอง 
# เราใช้ตัวแปรเดียวกับเซลล์ก่อนหน้า ดังนั้นถ้ารันแล้ว error เนื่องจากไม่ได้นิยามตัวแปร ให้รันเซลล์ก่อนหน้าก่อน
x = x.reshape(3,1) # จาก x ในเซลล์ที่แล้วเราสร้างคอลัมภ์เวกเตอร์โดยการ reshape จาก array 1 มิติ ให้เป็น 2 มิติ
print('Now x is 2D array with shape: ', x.shape,'\n')

print('Now np.dot(A,x) is :')
print(np.dot(A,x))
print(np.dot(A,x).shape,'\n')
print('And np.dot(x,A) will cause error :')
print(np.dot(x,A))
print(np.dot(x,A).shape)
# error ในกรณีที่สองตามที่ควรจะเป็น
import numpy as np

A = np.array([[1,2],[3,4]]) # สร้างเมตริกซ์ขนาด 2x2
print(A)

print(np.sum(A))  # รีเทิร์นผลบวกของทุกสมาชิกในเมตริกซ์; พิมพ์ "10"
print(np.sum(A, axis=0))  # รีเทิร์นผลบวกในมิติแรก (axis = 0) หรือผลบวกตามแถวของทุกสมาชิกในเมตริกซ์; พิมพ์ "[4 6]"
print(np.sum(A, axis=1))  # รีเทิร์นผลบวกในมิติที่สอง (axis = 1) หรือผลบวกตามคอลัมภ์ของทุกสมาชิกในเมตริกซ์; พิมพ์ "[3 7]"

import numpy as np

A = np.array([[1,2], [3,4]])
print(A)    # พิมพ์ "[[1 2]
            #          [3 4]]"
print(A.T)  # พิมพ์ "[[1 3]
            #          [2 4]]"
# การสั่ง Transpose กับ array 1 มิติไม่มีผลอะไร
v = np.array([1,2,3])
print('This is 1D array')
print(v)    # พิมพ์ "[1 2 3]"
print(v.T)  # พิมพ์ "[1 2 3]"
print(v.shape, v.T.shape,'\n')

# ที่ถูกต้องคือการสร้าง array 2 มิติ สร้างได้ 2 วิธีดังนี้
print('This is 2D array')
v = v.reshape(np.size(v),1) #วิธีที่หนึ่ง reshape จาก array 1 มิติ
print(v)
print(v.shape,'\n')

v = np.array([[1,2,3]]) # วิธีที่สองสร้าง array ใหม่ 
# สังเกตว่ามี [[]] ซ้อนกันเป็น list สองชั้น เพื่อแจ้ง np.array ว่าเราต้องการสร้าง array เป็นสองมิติ
# อย่างไรก็ดี วิธีนี้จะได้เป็น row vector ไม่ใช่ column vector 
print(v.shape)
print(v,'\n')
# ดังนั้นถ้าอยากได้คอลัมภ์เวกเตอร์เราต้อง transpose 
print(v.T.shape)
print(v.T)
import numpy as np
W = np.array([[1,2,3],[4,5,6],[7,8,9]]) #สมมติเรามีเมตริกซ์ขนาด 3x3
print('We have a 3x3 matrix : ')
print(W,'\n')

x = np.array([1,0,1])
y = x.reshape(3,1) # สร้างเวกเตอร์แนวตั้ง หรือคอลัมภ์เวกเตอร์
z = x.reshape(1,3) # สร้างเวกเตอร์แนวนอน
print('We also have two vectors with shapes: ',y.shape, z.shape)
print(y)
print(z,'\n')

print('Broadcasting addition with respect to the column vector : ')
print(W+y,'\n') # บวกทุกคอลัมภ์ของเมตริกซ์ด้วยเวกเตอร์แนวตั้ง
print('Broadcasting addition with respect to the row vector : ')
print(W+z,'\n') # บวกทุกแถวของเมตริกซ์ด้วยเวกเตอร์แนวนอน

# จริงๆ แล้วการคูณเมตริกซ์ด้วยค่าคงที่ก็คือ Broadcasting ชนิดหนึ่ง 
print('Broadcasting constant multiplication example : ')
print(W*2)
import numpy as np
W = np.array([[1,2,3],[4,5,6],[7,8,9]]) #สมมติเรามีเมตริกซ์ขนาด 3x3
print('If we have a 3x3 matrix : ')
print(W,'\n')

print('And the following 2x1 vector : ')
x = np.array([1,10])
x = x.reshape(2,1) # สร้างเวกเตอร์แนวตั้ง 2x1
print(x,'\n')

print('We are sorry that we cannot add them as broadcasting mismatched arrays is not possible : ')
print(W+x) # error
import numpy as np

# สร้าง array 2มิติ ที่มีขนาด (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# ใช้ slicing เพื่อดึงสองแถวแรก และคอลัมภ์ที่สองและสาม 
# สังเกตว่า index เริ่มจาก 0 ดังนั้นคอลัมภ์ที่สองและสามจึงมี index 1 และ 2
# และอย่าลืมว่าใน syntax ของ Python จะนำ index ตัวสุดท้ายไปลบด้วยหนึ่ง
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)

# นอกจากนี้ b ยังเสมือนเป็น pointer ที่ชี้ไปยังข้อมูลชุดเดียวกับ a
# ดังนั้นถ้าเปลี่ยนค่าใน b ซึ่งเป็น slice ของ a ข้อมูลใน a ก็จะเปลี่ยนด้วย
print(a[0, 1])   # พิมพ์ "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # พิมพ์ "77"


import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# สังเกตว่าการ index แถวที่สองทำได้สองวิธีคือ ใช้ index 1 หรือใช้ slicing 1:2
# ซึ่งใน syntax ของ python จะให้ผลลัพธ์ที่ __แตกต่างกัน__
# การใช้ index เป็นเพียงตัวเลข เช่น 1 จะลดมิติของ array ลง 1 มิติ (ดูตัวอย่างข้างล่าง)
# ในขณะที่การใช้ slicing index เช่น 1:2 เราจะได้ผลลัพธ์เป็น array ที่มีมิติเท่าเดิม
row_r1 = a[1, :]    # ผลลัพธ์คือ array 1 มิติ
row_r2 = a[1:2, :]  # ผลลัพธ์คือ array 2 มิติ
print(row_r1, row_r1.shape)  # พิมพ์ "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  #พิมพ์ "[[5 6 7 8]] (1, 4)"

import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # รีเทิร์นเป็น array ขนาดเดียวกับ a 
                     # ค่า bool_idx[i,j] มีค่า เป็น True ถ้า a[i,j] > 2
                     # นอกนั้นจะมีค่าเป็น False

print(bool_idx.shape)
print(bool_idx)      # พิมพ์     "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# เราใช้บูลเลียน index นี้สร้าง array 1 มิติ จาก a ที่จะรีเทริน์เฉพาะ index ที่ bool_idx = True
print(a[bool_idx])  # พิมพ์ "[3 4 5 6]"

# ถ้าคล่องแล้ว เราสามารถลัดขั้นตอนทั้งหมดเหลือเพียงบรรทัดเดียว
print(a[a > 2])     # พิมพ์ "[3 4 5 6]"
import numpy as np
import matplotlib.pyplot as plt # import library ที่ใช้สำหรับพล็อต

# สร้าง x ให้เป็น array ของตัวเลขตั้งแต่ 0 ไปจนถึง 3pi โดยแต่ละจุดห่างกันเท่ากับ 0.1
x = np.arange(0, 3 * np.pi, 0.1)
# กำหนด y ให้เป็น array มีค่าเท่ากับไซน์ของ x ในแต่ละจุดซึ่งเรียกโดยใช้ฟังก์ชัน np.sin
y = np.sin(x)

# ระบุว่าเราต้องการพล็อดกราฟของ x และ y
plt.plot(x, y)
plt.show()  # เมื่อระบุสิ่งที่เราต้องการพล็อตครบแล้ว สามารถแสดงผลการพล็อตได้ด้วยคำสั่ง plt.show()
import numpy as np
import matplotlib.pyplot as plt

# สร้างกราฟทั้งไซน์และโคไซน์ของ x
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# สั่งให้พล็อตทั้งสองกราฟพร้อมกัน
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label') # กำหนดชื่อบนแกน x
plt.ylabel('y axis label') # กำหนดชื่อบนแกน y
plt.title('Sine and Cosine') # กำหนดชื่อไตเติ้ล
plt.legend(['Sine', 'Cosine']) # กำหนดลาเบลของกราฟแต่ละเส้น (ตามลำดับคำสั่ง plt.plot)
plt.show() # เมื่อระบุสิ่งที่เราต้องการพล็อตครบแล้ว สามารถแสดงผลการพล็อตได้ด้วยคำสั่ง plt.show()
import numpy as np
import matplotlib.pyplot as plt

# เตรียมข้อมูลเหมือนในตัวอย่างที่แล้ว
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# ใช้คำสั่ง subplot 
# argument 2 ตัวแรกระบุว่าเราต้องการแบ่งกราฟเป็นสวนย่อยที่มี 2 แถว และ 1 คอลัมภ์ (นั่นคือแบ่งเป็นสองส่วน)
# argument ตัวที่สามระบุว่าเราจะเริ่มพล็อตจากส่วนที่หนึ่งก่อน
plt.subplot(2, 1, 1)

# ระบุกราฟที่ต้องการพล็อตใน subplot ที่ระบุ
plt.plot(x, y_sin)
plt.title('Sine')

# argument ตัวที่สามระบุว่าเราจะเริ่มพล็อตจากส่วนที่สอง
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# โชว์กราฟทั้งหมด
plt.show()
import numpy as np
from imageio import imread #ฟังก์ชันที่เอาไว้สำหรับอ่านรูปภาพและเก็บไว้ใน Numpy array
import matplotlib.pyplot as plt

img = imread('../input/cat.jpg') #ใน workshop นี้เราได้ใส่รูปไว้ใน directory นี้
print(img.shape) 
# พิมพ์ (400, 248, 3)

img_tinted = img * [1, 0.9, 0.8] 
#ตัวอย่างการใช้ Broadcasting เพื่อปรับสีสำหรับทุกๆ pixel (ดูคำอธิบายเรื่อง Broadcasting ด้านบน)
#ในมิติที่ 3 นั้นเรียงลำดับด้วย แดง เขียว น้ำเงิน ดังนั้นการคูณด้วย [1, 0.9, 0.8]
#ก็คือการปรับสีเขียวลง 10% และปรับสีน้ำเงินลง 20% นั่นเอง ดังนั้นภาพ output จะมืดลงเล็กน้อย

# โชว์ภาพต้นฉบับ
plt.subplot(1, 2, 1)
plt.imshow(img)

# โชว์ภาพที่เราปรับแสง
plt.subplot(1, 2, 2)
# หมายเหตุ แสงสีน้ำเงิน เป็นแสงที่ทำร้ายสายตา การที่เราปรับลดแสงสีน้ำเงินลง อาจมองเป็นโหมดถนอมสายตาได้ :)

# คำสั่ง imshow ต้องการข้อมูลที่เป็นชนิด uint8 ดังนั้นเราจึง cast ก่อน
plt.imshow(np.uint8(img_tinted))
plt.show()
from imageio import imread, imwrite
from skimage.transform import resize

# อ่านภาพ JPEG ให้อยู่ในรูป numpy array
img = imread('../input/cat.jpg')
print(img.dtype, img.shape)  # พิมพ์ "uint8 (400, 248, 3)"

img_tinted = img * [1, 0.95, 0.9]

# เปลี่ยนขนาดของรูปภาพให้เป็น 300x300
img_tinted = resize(img_tinted, (300, 300))

# เซพไฟล์ที่เปลีย่นขนาดลงฮาร์ดไดร์พ
imwrite('cat_tinted.jpg', img_tinted)

# พล็อตเช่นเดียวกับตัวอย่างด้านบน
plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)

plt.imshow(np.uint8(img_tinted))
plt.show()

import numpy as np
from scipy.spatial.distance import pdist, squareform

# สมมติเรามีเซ็ตที่แทนจุด 3 จุดในปริภูมิ 2 มิติ
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# คำนวนระยะของจุดในทุกคู่
# โดยระยะระหว่างจุด i และ j แทนด้วย d[i, j] 
# คำนวนได้จากระยะแบบยูคลิดสำหรับจุด x[i, :] and x[j, :],
# d จะเป็นเมตริกซ์สมมาตรดังนี้
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
