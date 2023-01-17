#codes from Endi Niu @niuddd

import cv2

import matplotlib.pyplot as plt

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_01.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

import cv2

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Wai_Ming/Wai_Ming_11.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_05.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_11.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_10.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_09.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Zhang_Xiao_Gang_张晓刚/Zhang_Xiao_Gang_09.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
from IPython.display import Image

import os

!ls ../input/
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Luo_Zhong_Li_羅中立/Luo_Zhongli_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
Image('../input/chinese-fine-art/Dataset/Qin_Feng_秦风/Qin_Feng_08.jpg')
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Giuseppe_Castiglione_郎世宁/Giuseppe_Castiglione_15.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#Import numpy for numerical calculations



import numpy as np
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Li_Zijian_李自健/Li_Zijian_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Lu_Xin_Jian_陆新建/Lu_Xin_Jian_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#Randomly Initialize Weights

input_weights = np.around(np.random.uniform(-5,5,size=6), decimals=2)

bias_weights = np.around(np.random.uniform(size=3), decimals=2)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/He_Jilan_何紀嵐/He_Jilan_14.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
print(input_weights)

print(bias_weights)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Yu_Zhi_Ding_禹之鼎/Yu_Zhi_Ding_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
# Assign values to inputs.

x_1 = 0.5 #input 1

x_2 = 0.82 #input 2



print('Input x1 is {} and Input x2 is {}'.format(x_1,x_2))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Gu_Wenda_谷文达/Gu_Wenda_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#Calculate linear combination of inputs

z_11 = x_1 * input_weights[0] + x_2 * input_weights[1] + bias_weights[0]



print('The linear combination of inputs at the first node of the hidden layer is {}'.format(z_11))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Chu_Chu_儲楚/Chu_Chu_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
z_12 = x_1 * input_weights[2] + x_2 * input_weights[3] + bias_weights[1]



print('The linear combination of inputs at the second node of the hidden layer is {}'.format(z_12))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Luo_Jian_Wu_罗建武/Luo_Jian_Wu_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#Calculate Output of Activation Function. We don't have Relu.

a_11 = max(0.0, z_11)



print('The output of the activation function at the first node of the hidden layer is {}'.format(np.around(a_11, decimals=4)))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Jin_Ting_Biao_金廷标/Jin_Ting_Biao_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
a_12 = max(0.0, z_12)



print('The output of the activation function at the second node of the hidden layer is {}'.format(np.around(a_12, decimals=4)))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input//chinese-fine-art/Dataset/Ai_Xuan_艾軒/Ai_Xuan_02.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#Repeat the steps until you get your final output.

z_2 = a_11 * input_weights[4] + a_12 * input_weights[5] + bias_weights[2]



print('The linear combination of inputs at the output layer is {}'.format(z_2))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Chang_Dai_Chien_張大千/Chang_Dai_Chien_17.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#Feeding this summation into a non-linear activation function known as sigmoid function which is best suited for Output Layer.

#We don't have Sigmoid.

y = 1.0 / (1.0 + np.exp(-z_2))



print('The output of the network for the given inputs is {}'.format(np.around(y, decimals=6)))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/He_Baili_何百里/He_Baili_11.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Yao_Wen_Han_姚文瀚/Yao_Wen_Han_01.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Wang_Hui_王翚/Wang_Hui_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Kuo_Sung_刘国松/Liu_Kuo_Sung_13.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Li_Fang_Ying_李方膺/Li_Fang_Ying_01.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Zou_Chuan_An_邹传安/Zou_Chuan_An_03.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Qiu_Ying_仇英/Qiu_Ying_04.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Zhao_Kailin_赵开霖/Zhao_Kailin_13.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_13.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Ling_Jian_凌健/Ling_Jian_06.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_08.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Ling_Jian_凌健/Ling_Jian_01.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)