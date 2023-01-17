import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageOps



img = np.array(Image.open('../input/-emma/emma_stone.jpg'))

plt.figure(figsize=(8,8))

plt.imshow(img)
print('# of dims: ',img.ndim)

print('img shape: ',img.shape)

print('dtype: ',img.dtype)
# pixel value at [R, G, B]

print(img[20, 20])



# min pixel value at channel B

print(img[:, :, 2].min())
path = 'emma.jpg'

pil_img = Image.fromarray(img)

pil_img.save(path)
degrees = 90

img = np.array(Image.open('../input/-emma/emma_stone.jpg'))

# img = img.sum(2) / (255*3) # converting to grayscale



fig = plt.figure(figsize=(10, 10))



fig.add_subplot(1, 2, 1)

plt.imshow(img)

plt.title("original")

img0 = img.copy()



# Algo: image(ndarray) -> transpose -> mirror image across y axis (middle column)



for _ in range(degrees // 90):

    img0 = img0.transpose(1, 0, 2)

    

    for j in range(0, img0.shape[1] // 2):



        c = img0[:, j, :].copy()

        img0[:, j, :] = img0[: , img0.shape[1]-j-1, :]

        img0[: , img0.shape[1]-j-1, :] = c

        

fig.add_subplot(1, 2, 2)

plt.imshow(img0)

plt.title("rotated")

# or you could have used np.rot90(img)

plt.imshow(np.rot90(img))

# This does the other way around
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))

img_grey = img.sum(2) / (255*3) # summing over axis=2 (channel axis) to get grey scaled image

fig = plt.figure(figsize=(10, 10))

img_grey = 255*3 - img_grey        # 255 * 3 because we added along channel axis previously

fig.add_subplot(1, 2, 1)

plt.imshow(img_grey)

plt.title('Negative of Grey image')



img = 255 - img

fig.add_subplot(1, 2, 2)

plt.imshow(img)

plt.title('Negative of RGB image')
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))

img_grey = img.sum(2) / (255*3)

img0 = img_grey.copy()

img0 = np.pad(img0, ((100,100),(100,100)), mode='constant')

plt.imshow(img0)
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



img_R, img_G, img_B = img.copy(), img.copy(), img.copy()



img_R[:, :, (1, 2)] = 0

img_G[:, :, (0, 2)] = 0

img_B[:, :, (0, 1)] = 0







img_rgb = np.concatenate((img_R,img_G,img_B), axis=1)

plt.figure(figsize=(15, 15))

plt.imshow(img_rgb)
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



# Making Pixel values discrete by first division by // which gives int and then multiply by same factor



img_0 = (img // 64) * 64    

img_1 = (img // 128) * 128



img_all = np.concatenate((img, img_0, img_1), axis=1)



plt.figure(figsize=(15, 15))

plt.imshow(img_all)
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



fig = plt.figure(figsize=(10, 10))



fig.add_subplot(1, 2, 1)

plt.imshow(img)

plt.title('Original')



img0 = img[128:-128, 128:-128, :]



fig.add_subplot(1, 2, 2)

plt.imshow(img0)

plt.title('Trimmed')



src = np.array(Image.open('../input/-emma/emma_stone.jpg').resize((128, 128)))

dst = np.array(Image.open('../input/-emma/emma_stone.jpg').resize((256, 256))) // 4



dst_copy = dst.copy()

dst_copy[64:128, 128:192] = src[32:96, 32:96]



fig = plt.figure(figsize=(10, 10))



fig.add_subplot(1, 2, 1)

plt.imshow(src)

plt.title('Original')



fig.add_subplot(1, 2, 2)

plt.imshow(dst_copy)

plt.title('Pasted with slice')
fig = plt.figure(figsize=(10, 10))



img = np.array(Image.open('../input/-emma/emma_stone.jpg'))

fig.add_subplot(1, 2, 1)

plt.imshow(img)

plt.title('Original')



img0 = img.transpose(1, 0, 2)



fig.add_subplot(1, 2, 2)

plt.imshow(img0)

plt.title('Flip_rotated')
# np.linspace:     Gives grdually increasing or decreasing 1-D array



x = np.linspace(0, 10, 3)  # start, stop, num of samples

# array([ 0.,  5., 10.])









# np.tile:         Repeats our given 1-D/ 2-D array in either/both axes

x= np.arange(5).reshape(1,-1)

np.tile(x, (3, 1))

# array([[0, 1, 2, 3, 4],

#        [0, 1, 2, 3, 4],

#        [0, 1, 2, 3, 4]])



np.tile(x, (2, 2))

# array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],

#        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
# Lets get into Shading now



Height = 256

Width = 512

start_list = (0, 0, 192)

stop_list = (255, 255, 64)

is_horizontal_list = (True, False, False)





def gradation_2d(start, stop, width, height, is_horizontal):

    if is_horizontal:

        return np.tile(np.linspace(start, stop, width), (height, 1))

    else:

        return np.tile(np.linspace(start, stop, height), (width, 1)).T   #imagine it in your head, you'll get it

    

    

def gradation_3d(width, height, start_list, stop_list, is_horizontal_list):

    result = np.zeros((height, width, len(start_list)),dtype=np.float)

    

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):

        result[:, :, i] = gradation_2d(start, stop, width, height, is_horizontal)

    return result
# Width, Height, start_list, stop_list, is_horizontal_list

img0 = gradation_3d(256, 256, (0,0,0),(255,255,255),(True,True,True))

img0 = np.uint8(img0)



plt.figure(figsize=(10, 5))

plt.imshow(img0)
img0 = gradation_3d(Width, Height, start_list, stop_list, is_horizontal_list)

img0 = np.uint8(img0)



plt.figure(figsize=(10, 5))

plt.imshow(img0)
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



img_64 = (img > 64) * 255

img_128 = (img > 128) * 255



fig = plt.figure(figsize=(15, 15))



img_all = np.concatenate((img, img_64, img_128), axis=1)

plt.imshow(img_all)



img = np.array(Image.open('../input/-emma/emma_stone.jpg'))

img0 = np.array(Image.open('../input/mountains/mountains.jpg').resize(img.shape[1::-1])) # resize takes 2 arguments (WIDTH, HEIGHT)



print(img.dtype)

# uint8



dst = (img * 0.6 + img0 * 0.4).astype(np.uint8)   # Blending them in



plt.figure(figsize=(10, 10))

plt.imshow(dst)
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



ones = np.ones((img.shape[0] // 2, img.shape[1] // 2, 3))

zeros = np.zeros(((img.shape[0] // 4, img.shape[1] // 4, 3)))





zeros_mid = np.zeros(((img.shape[0] // 2, img.shape[1] // 4, 3)))

up = np.concatenate((zeros, zeros, zeros, zeros), axis=1)

middle = np.concatenate((zeros_mid, ones, zeros_mid), axis=1)

down = np.concatenate((zeros, zeros, zeros, zeros), axis=1)





mask = np.concatenate((up, middle, down), axis=0)

mask = mask / 255



img0 = mask * img



fig = plt.figure(figsize=(10, 10))

fig.add_subplot(1, 2, 1)

plt.imshow(img)



fig.add_subplot(1, 2, 2)

plt.imshow(img0)



img0 = img.copy()



for i in range(img0.shape[0] // 2):

    c = img0[i, :, :].copy()

    img0[i, :, :] = img0[img0.shape[0] - i - 1, :, :]

    img0[img0.shape[0] - i - 1, :, :] = c

        

plt.imshow(img0)
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



fig = plt.figure(figsize=(10, 10))

fig.add_subplot(1, 2, 1)

plt.imshow(np.flipud(img))



fig.add_subplot(1, 2, 2)

plt.imshow(np.fliplr(img))
img = np.array(Image.open('../input/-emma/emma_stone.jpg'))



img_flat = img.flatten()



plt.hist(img_flat, bins=200, range=[0, 256])

plt.title("Number of pixels in each intensity value")

plt.xlabel("Intensity")

plt.ylabel("Number of pixels")

plt.show()