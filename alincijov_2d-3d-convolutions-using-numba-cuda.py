import numpy as np

import matplotlib.pyplot as plt

from numba import cuda, float32

import PIL

import math
image = np.asarray(PIL.Image.open('../input/fruit-recognition/train/train/Pineapple/Pineapple_108.jpg'))
img2d = image[:,:,0]

img3d = image
f, axarr = plt.subplots(1,2,figsize=(15,15))

axarr[0].set_title('2D')

axarr[0].imshow(img2d)

axarr[1].set_title('3D')

axarr[1].imshow(img3d)
MASK_WIDTH = 3

MASK_RADIUS = MASK_WIDTH // 2

TILE_WIDTH = 8

W = TILE_WIDTH + MASK_WIDTH - 1
@cuda.jit

def conv2d(I, O, M, height, width):

    s = cuda.shared.array(shape=(W, W), dtype=float32)

    

    dest = cuda.threadIdx.x + cuda.threadIdx.y * TILE_WIDTH

    destY = dest // W

    destX = dest % W



    srcY = destY + cuda.blockIdx.y * TILE_WIDTH - MASK_RADIUS

    srcX = destX + cuda.blockIdx.x * TILE_WIDTH - MASK_RADIUS

    src = srcX, srcY



    if destY < W:

        if (srcY >= 0) and (srcY < height) and (srcX >= 0) and (srcX < width):

            s[destY][destX] = I[src]

        else:

            s[destY][destX] = 0



    cuda.syncthreads()



    dest = cuda.threadIdx.x + (cuda.threadIdx.y * TILE_WIDTH) + TILE_WIDTH * TILE_WIDTH

    destY = dest // W

    destX = dest % W



    srcY = destY + (cuda.blockIdx.y * TILE_WIDTH) - MASK_RADIUS

    srcX = destX + (cuda.blockIdx.x * TILE_WIDTH) - MASK_RADIUS

    src = srcX, srcY



    if(destY < W):

        if(srcY >= 0 and srcY < height and srcX >= 0 and srcX < width):

            s[destY][destX] = I[src]

        else:

            s[destY][destX] = 0;



    cuda.syncthreads()



    sum = 0

    for y in range(MASK_WIDTH):

        for x in range(MASK_WIDTH):

            sum = sum + s[cuda.threadIdx.y + y][cuda.threadIdx.x + x] * M[x, y * MASK_WIDTH]



    y = cuda.threadIdx.y + (cuda.blockIdx.y * TILE_WIDTH)

    x = cuda.threadIdx.x + (cuda.blockIdx.x * TILE_WIDTH)



    if(y < height and x < width):

        O[x, y] = sum
np.random.seed(23432423)

I = img2d / 255.

O = np.zeros((100,100))

M = np.random.randn(3,3)
dimBlock = (TILE_WIDTH, TILE_WIDTH, 1)

dimGrid = ((I.shape[0] + TILE_WIDTH - 1) // TILE_WIDTH, (I.shape[1] + TILE_WIDTH - 1) // TILE_WIDTH)
# Return a contiguous array (ndim >= 1) in memory (C order)

I = np.ascontiguousarray(I, dtype=np.float32)

O = np.ascontiguousarray(O, dtype=np.float32)
conv2d[dimGrid, dimBlock](I, O, M, 100, 100)
plt.figure(figsize=(8,8))

plt.imshow(O * 255.)
@cuda.jit

def conv3d(I, O, M, height, width, depth):

    s = cuda.shared.array(shape=(W, W, W), dtype=float32)



    dest = cuda.threadIdx.x + (cuda.threadIdx.y * TILE_WIDTH) + (cuda.threadIdx.z * TILE_WIDTH * TILE_WIDTH)

    destTmp = dest

    destX = destTmp % W

    destTmp = destTmp // W

    destY = destTmp % W

    destTmp = destTmp // W

    destZ = destTmp



    srcZ = destZ + (cuda.blockIdx.z * TILE_WIDTH) - MASK_RADIUS

    srcY = destY + (cuda.blockIdx.y * TILE_WIDTH) - MASK_RADIUS

    srcX = destX + (cuda.blockIdx.x * TILE_WIDTH) - MASK_RADIUS

    src = srcX, srcY, srcZ



    if(srcZ >= 0 and srcZ < depth and srcY >= 0 and srcY < height and srcX >= 0 and srcX < width):

        s[destZ][destY][destX] = I[src]

    else:

        s[destZ][destY][destX] = 0



    dest = cuda.threadIdx.x + (cuda.threadIdx.y * TILE_WIDTH) + (cuda.threadIdx.z * TILE_WIDTH * TILE_WIDTH) + TILE_WIDTH * TILE_WIDTH * TILE_WIDTH

    destTmp = dest

    destX = destTmp % W

    destTmp = destTmp // W

    destY = destTmp % W

    destTmp = destTmp // W

    destZ = destTmp



    srcZ = destZ + (cuda.blockIdx.z * TILE_WIDTH) - MASK_RADIUS

    srcY = destY + (cuda.blockIdx.y * TILE_WIDTH) - MASK_RADIUS

    srcX = destX + (cuda.blockIdx.x * TILE_WIDTH) - MASK_RADIUS

    src = srcX, srcY * width, srcZ * width * height



    if(destZ < W):

        if(srcZ >= 0 and srcZ < depth and srcY >= 0 and srcY < height and srcX >= 0 and srcX < width):

            s[destZ][destY][destX] = I[src]

        else:

            s[destZ][destY][destX] = 0



    cuda.syncthreads()



    sum = 0.



    for z in range(MASK_WIDTH):

        for y in range(MASK_WIDTH):

            for x in range(MASK_WIDTH):

                sum = sum + s[cuda.threadIdx.z + z][cuda.threadIdx.y + y][cuda.threadIdx.x + x] * M[x, y * MASK_WIDTH,  z * MASK_WIDTH * MASK_WIDTH]



    z = cuda.threadIdx.z + (cuda.blockIdx.z * TILE_WIDTH)

    y = cuda.threadIdx.y + (cuda.blockIdx.y * TILE_WIDTH)

    x = cuda.threadIdx.x + (cuda.blockIdx.x * TILE_WIDTH)



    if(z < depth and y < height and x < width):

        O[x, y, z] = sum



    cuda.syncthreads()
np.random.seed(23432423)

I = img3d / 255.

O = np.zeros((100,100,3))

M = np.random.randn(3,3,3)
dimBlock = (TILE_WIDTH, TILE_WIDTH, TILE_WIDTH)

dimGrid = ((I.shape[0] + TILE_WIDTH - 1) // TILE_WIDTH, (I.shape[1] + TILE_WIDTH - 1) // TILE_WIDTH, (I.shape[2] + TILE_WIDTH - 1) // TILE_WIDTH)
# Return a contiguous array (ndim >= 1) in memory (C order)

I = np.ascontiguousarray(I, dtype=np.float32)

O = np.ascontiguousarray(O, dtype=np.float32)
conv3d[dimGrid, dimBlock](I, O, M, 100, 100, 3)
plt.figure(figsize=(8,8))

plt.imshow(O * 255.)