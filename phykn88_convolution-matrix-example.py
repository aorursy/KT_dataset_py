from PIL import Image

from io import BytesIO

import requests

import numpy as np

import matplotlib.pyplot as plt
class kernel:

    def identity():

        k = np.array([

            [0, 0, 0],

            [0, 1, 0],

            [0, 0, 0],

        ])

        return k

    

    def edge_1():

        k = np.array([

            [1, 0, -1],

            [0, 0, 0],

            [-1, 0, 1],

        ])

        return k

    

    def edge_2():

        k = np.array([

            [0, 1, 0],

            [1, -4, 1],

            [0, 1, 0],

        ])

        return k

        

    def edge_3():

        k = np.array([

            [-1, -1, -1],

            [-1, 8, -1],

            [-1, -1, -1],

        ])

        return k

    

    def sharpen():

        k = np.array([

            [0, -1, 0],

            [-1, 5, -1],

            [0, -1, 0],

        ])

        return k

    

    def box_blur():

        k = np.array([

            [1, 1, 1],

            [1, 1, 1],

            [1, 1, 1],

        ])/9

        return k

    

    def gaussian_blur_3x3():

        k = np.array([

            [1, 2, 1],

            [2, 4, 2],

            [1, 2, 1],

        ])/16

        return k

    

    def gaussian_blur_5x5():

        k = np.array([

            [1, 4, 6, 4, 1],

            [4, 16, 24, 16, 4],

            [6, 24, 36, 24, 6],

            [4, 16, 24, 16, 4],

            [1, 4, 6, 4, 1],

        ])/256

        return k

    

    def unsharp_masking_5x5():

        k = np.array([

            [1, 4, 6, 4, 1],

            [4, 16, 24, 16, 4],

            [6, 24, -476, 24, 6],

            [4, 16, 24, 16, 4],

            [1, 4, 6, 4, 1],

        ])*(-1)/256

        return k 

    

def convolution(img, kernel):   

    hight, width, chennel = img.shape

    k_size = kernel.shape[0]

    k_flat = kernel.flatten().T

    

    result = []

    result_h = hight - k_size + 1

    result_w = width - k_size + 1

    

    for c in range(chennel):

        img_c = img[:, :, c]

        tmp = []

        tmp_append = tmp.append

        for i in range(result_h):

            for j in range(result_w):

                img_select = img_c[i:i+k_size, j:j+k_size]

                img_select = img_select.flatten()

                tmp_append(img_select)

        tmp = np.array(tmp)

        tmp = np.dot(tmp, k_flat)

        tmp = np.resize(tmp, [hight-k_size+1, width-k_size+1])

        result.append(tmp)

    result = np.stack(result, axis=2)

    

    # Bound Min : 0

    result = (result + abs(result)) / 2

    # Bound Max : 255

    result = ((result - 255) - abs(result - 255)) / 2 + 255    

    return result.astype('int')
def get_img(url):

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    img = np.array(img)

    return img



def plot(location, title, img):

    plt.subplot(location)

    plt.title(title)

    plt.imshow(img)
img = get_img('https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png')

plt.figure(figsize=(10, 12))

plot(331, 'Identity', convolution(img, kernel.identity()))

plot(332, 'Edge Detection 1', convolution(img, kernel.edge_1()))

plot(333, 'Edge Detection 2', convolution(img, kernel.edge_2()))

plot(334, 'Edge Detection 3', convolution(img, kernel.edge_3()))

plot(335, 'Sharpen', convolution(img, kernel.sharpen()))

plot(336, 'Box blur', convolution(img, kernel.box_blur()))

plot(337, 'Gaussian blur 3 x 3', convolution(img, kernel.gaussian_blur_3x3()))

plot(338, 'Gaussian blur 5 x 5', convolution(img, kernel.gaussian_blur_5x5()))

plot(339, 'Unsharp masking 5 x 5', convolution(img, kernel.unsharp_masking_5x5()))

plt.show()
img = get_img('http://www.tv-asahi.co.jp/shinchan/character/img/01.png')

plt.figure(figsize=(10, 12))

plot(331, 'Identity', convolution(img, kernel.identity()))

plot(332, 'Edge Detection 1', convolution(img, kernel.edge_1()))

plot(333, 'Edge Detection 2', convolution(img, kernel.edge_2()))

plot(334, 'Edge Detection 3', convolution(img, kernel.edge_3()))

plot(335, 'Sharpen', convolution(img, kernel.sharpen()))

plot(336, 'Box blur', convolution(img, kernel.box_blur()))

plot(337, 'Gaussian blur 3 x 3', convolution(img, kernel.gaussian_blur_3x3()))

plot(338, 'Gaussian blur 5 x 5', convolution(img, kernel.gaussian_blur_5x5()))

plot(339, 'Unsharp masking 5 x 5', convolution(img, kernel.unsharp_masking_5x5()))

plt.show()
img = get_img('https://media.licdn.com/dms/image/C5103AQF7oIQOVSIhTw/profile-displayphoto-shrink_200_200/0?e=1575504000&v=beta&t=-f1Z9GQSIdp4zU2-zXRJXSWnHf7Sk_81znbQJ7mQdOk')

plt.figure(figsize=(10, 12))

plot(331, 'Identity', convolution(img, kernel.identity()))

plot(332, 'Edge Detection 1', convolution(img, kernel.edge_1()))

plot(333, 'Edge Detection 2', convolution(img, kernel.edge_2()))

plot(334, 'Edge Detection 3', convolution(img, kernel.edge_3()))

plot(335, 'Sharpen', convolution(img, kernel.sharpen()))

plot(336, 'Box blur', convolution(img, kernel.box_blur()))

plot(337, 'Gaussian blur 3 x 3', convolution(img, kernel.gaussian_blur_3x3()))

plot(338, 'Gaussian blur 5 x 5', convolution(img, kernel.gaussian_blur_5x5()))

plot(339, 'Unsharp masking 5 x 5', convolution(img, kernel.unsharp_masking_5x5()))

plt.show()