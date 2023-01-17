import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd   #import thư viện pandas (dùng pd thay cho pandas thuận tiện hơn trong việc code) 
data = pd.read_csv('/kaggle/input/data-linear/data_linear.csv', header=0).values  # Đọc file cvs
data
import matplotlib.pyplot as plt
plt.scatter(data[:,0],data[:,1])  
plt.show()
# Đọc ảnh từ input
import cv2 # import thư viện openCV
img = cv2.imread('/kaggle/input/imagetest/image.png') #đọc ảnh
print(img.shape)
plt.imshow(img)
plt.show()
# Cắt góc phần tư trái trên của ảnh.
height, width, N = img.shape
img1 = img[:height//2,width//2:]

plt.imshow(img1)
plt.show()
# Resize ảnh, dài rộng còn một nửa.
_shape = (width//2,height//2)
print(_shape)

img2 = cv2.resize(img,_shape)
print(img2.shape)
plt.imshow(img2)
plt.show()
# Thực hiện Gaussian blur ảnh.
img3 = cv2.GaussianBlur(img,(15,15),0)   # áp dụng với kernel có kích thước 15 * 15

plt.imshow(img3)
plt.show()
# Phát hiện edge trong ảnh.
img4 = cv2.Canny(img,200,300)

plt.imshow(img4)
plt.show()