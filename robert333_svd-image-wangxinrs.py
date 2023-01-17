# 导入第三方库

import numpy as np 

from scipy.linalg import svd

from PIL import Image

import matplotlib.pyplot as plt
# 原图像加载

image = Image.open('../input/cartoon.jpg');

A = np.array(image);

# 显示原图像

plt.imshow(A);

plt.axis('off');

plt.show();
### 将图像进行灰度化处理



# 原图像加载并转化为灰度图

image = Image.open('../input/cartoon.jpg').convert('L');

# 图像矩阵

A = np.array(image);

# 图像显示

plt.imshow(A);

plt.axis('off');

plt.show();
# 对图像矩阵A进行奇异值分解，得到p, s, q

p, s, q = svd(A, full_matrices=False);

print('图像矩阵A有%d个特征值！' % len(s));

print('前五个特征值为');

print(s[:5]);
# 取前k个特征，并对图像进行还原

# 对矩阵A进行奇异值分解，得到A=psq'

# 输入：p, s, q 和 特征数量k

# 输出：返回取了k个特征之后的图像

def get_image_features(p, s, q, k):

    # 对于s，取前k个特征

    temp_k = np.zeros(s.shape[0]);

    temp_k[:k] = s[:k];

    s_k = temp_k * np.identity(s.shape[0]);

    # 用新的s_k，以及p, q重构A

    temp = np.dot(p, s_k);

    temp = np.dot(temp, q);

    plt.imshow(temp);

    plt.axis('off');

    plt.show();
# 取前k个特征，对图像进行还原

get_image_features(p, s, q, 5);

get_image_features(p, s, q, 50);

get_image_features(p, s, q, 500);