%matplotlib inline
import torch
import PIL
import PIL.Image
import numpy as np 
import matplotlib.pyplot as plt
!ls -al /kaggle/input/flower_data/flower_data/train/20
# find folder
path = '/kaggle/input/flower_data/flower_data/train/20/'
filename = 'image_04897.jpg'
pf = path+filename

img = PIL.Image.open(pf)
img_arr = np.array(img)
h,w,c = img_arr.shape
# make variabel vector to use tourch.rand
vc = torch.rand(3)
arr = np.array(vc)
print(arr)
# make manual vector
vector = [
    255,
    255,
    0
]

vc = torch.tensor(vector)
print(vc)
arr = np.array(vector)
print(arr)
uk = arr.shape
print(uk)
# make manual matrix
matrix = [
    [11,12,13],
    [21,22,23],
    [31,32,33],
]

mx = torch.tensor(matrix)
print(mx)
arr = np.array(matrix)
print(arr)
uk = arr.shape
print(uk)
plt.imshow(matrix, cmap='gray')
#taking all row on column-3
# pengambilan semua baris pada kolom ke-3
a = mx[:,2]
print(a)

#taking all column on row-1
# pengambilan semua kolom pada baris ke-1
b = mx[0,:]
print(b)

#taking row on column-2
# pengambilan baris ke-3 pada kolom ke-2
c = mx[2,1]
print(c)
# make manual tensor 
tensor = [
    [
        [225,255,0],
        [0,0,255],
        [0,0,255],
    ],
    
    [
        [225,0,0],
        [0,0,255],
        [255,0,255], 
    ],
    
    [
        [0,0,0],
        [255,0,0],
        [255,0,0]
    ]
]

tc = torch.tensor(tensor)
print(tc)
ar = np.array(tensor)
print(ar)
uk = ar.shape
print(uk)
plt.imshow(ar, cmap='gray')
