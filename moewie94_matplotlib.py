import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax3 = fig.add_subplot(2, 2, 3)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
from numpy.random import randn



x = randn(50).cumsum()

print(x)

plt.plot(x, "k--")
import numpy as np



x = [1, 1, 3, 2, 4, 3]

print(x)

_ = plt.hist(x, bins=20, color="g", alpha=0.5)
import numpy as np



# x = [0, 1, ..., 29]

# y = x + 3 * noise

plt.scatter(x=np.arange(30), y=np.arange(30) + 3 * np.random.randn(30), color="g", alpha=0.3)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))

for i in range(2):

    for j in range(2):

        axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.plot(np.random.rand(100).cumsum(), "g--")
plt.plot(np.random.rand(100).cumsum(), linestyle="-", color="black")
plt.plot(np.random.randn(30).cumsum(), 'k*--')
plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='--', marker='*')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])

labels = ax.set_xticklabels(['Nghi', 'Thien', 'Nam', 'Tung', 'Duc'], 

                            rotation=30, fontsize='small')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])

labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], 

                            rotation=30, fontsize='small')









ax.set_title('My first matplotlib plot')

ax.set_xlabel('Stages')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum(), 'r', label='Nghi')

ax.plot(randn(1000).cumsum(), 'g--', label='Thien')

ax.plot(randn(1000).cumsum(), 'b.', label='Duc')

ax.legend(loc="best")
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])

labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], 

                            rotation=30, fontsize='small')

ax.set_title('My first matplotlib plot')

ax.set_xlabel('Stages')

ax.text(300, 0, 'Hello world!',

        family='monospace', fontsize=10)
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])

labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], 

                            rotation=30, fontsize='small')

ax.set_title('My first matplotlib plot')

ax.set_xlabel('Stages')

ax.annotate("Giang", xy=(10, 10), xytext=(10, 20), arrowprops={"facecolor": "black"},

            horizontalalignment='left', verticalalignment='top')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)

ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])

labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], 

                            rotation=30, fontsize='small')

ax.set_title('My first matplotlib plot')

ax.set_xlabel('Stages')

ax.annotate("Giang", xy=(10, 10), xytext=(10, 20), arrowprops={"facecolor": "black"},

            horizontalalignment='left', verticalalignment='top')

plt.savefig("ben.svg")
import cv2 # OpenCV-version-2



path = "190805181053-barack-obama-190406-exlarge-169.jpg"



image = cv2.imread(path)
type(image), image.shape, image.dtype
import matplotlib.pyplot as plt



# `plt` show R-G-B image. cv2 read image in B-G-R mode

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

plt.show()
# Output: new window

cv2.imshow(image)



# Output: console

plt.imshow(image)
!wget https://upload.wikimedia.org/wikipedia/commons/8/85/Ho_Chi_Minh_City_%28Saigon%29_1980s_government_license_plate.jpg
# Thư viện để đọc ảnh từ máy tính và lưu trong chương trình dưới dạng NumPy array

import cv2



# Thư viện để visualize ảnh và dữ liệu

import matplotlib.pyplot as plt
image = cv2.imread("Ho_Chi_Minh_City_(Saigon)_1980s_government_license_plate.jpg")
type(image)
image.shape, image.dtype
# Vẽ ảnh `image` vào subplot

# Khi mọi người dùng CV2 để đọc ảnh thì ảnh đọc vào sẽ có chế độ màu BGR (Blue-Green-Red)

# Trong khi ảnh ta nhìn thấy là ảnh RGB (Red-Green-Blue)

# Ta cần đảo lại kênh màu cho ảnh trước khi hiển thị 

# Khi nào nên đảo kênh màu sang RGB: khi cần hiển thị ảnh

revised_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(revised_image)



# Hiển thị figure

plt.show()