!pip install git+https://github.com/dwgoon/jpegio

import jpegio as jio
import numpy as np

from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt



INPUT_PATH = Path('/kaggle/input/alaska2-image-steganalysis/')
files_cover = [x for x in (INPUT_PATH / 'Cover').glob('**/*') if x.is_file()]

files_cover.sort()



len(files_cover)
def JPEGdecompressYCbCr(path):

    jpegStruct = jio.read(str(path))



    nb_colors = len(jpegStruct.coef_arrays)



    [Col, Row] = np.meshgrid(range(8), range(8))

    T = 0.5 * np.cos(np.pi * (2 * Col + 1) * Row / (2 * 8))

    T[0, :] = T[0, :] / np.sqrt(2)



    sz = np.array(jpegStruct.coef_arrays[0].shape)



    imDecompressYCbCr = np.zeros([sz[0], sz[1], nb_colors])

    szDct = (sz / 8).astype("int")



    for ColorChannel in range(nb_colors):

        tmpPixels = np.zeros(sz)



        DCTcoefs = jpegStruct.coef_arrays[ColorChannel]

        if ColorChannel == 0:

            QM = jpegStruct.quant_tables[ColorChannel]

        else:

            QM = jpegStruct.quant_tables[1]



        for idxRow in range(szDct[0]):

            for idxCol in range(szDct[1]):

                D = DCTcoefs[

                    idxRow * 8 : (idxRow + 1) * 8, idxCol * 8 : (idxCol + 1) * 8

                ]

                tmpPixels[

                    idxRow * 8 : (idxRow + 1) * 8, idxCol * 8 : (idxCol + 1) * 8

                ] = np.dot(np.transpose(T), np.dot(QM * D, T))

        imDecompressYCbCr[:, :, ColorChannel] = tmpPixels

        

    return imDecompressYCbCr.astype(np.float32)
img_v1 = JPEGdecompressYCbCr(files_cover[0])

img_v1.shape
def JPEGdecompressYCbCr_v2(path):

    jpegStruct = jio.read(str(path))



    [col, row] = np.meshgrid(range(8), range(8))

    T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))

    T[0, :] = T[0, :] / np.sqrt(2)



    img_dims = np.array(jpegStruct.coef_arrays[0].shape)

    n_blocks = img_dims // 8

    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)

    

    YCbCr = []

    for i, dct_coeffs, in enumerate(jpegStruct.coef_arrays):



        if i == 0:

            QM = jpegStruct.quant_tables[i]

        else:

            QM = jpegStruct.quant_tables[1]

        

        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)

        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)

        dct_coeffs = dct_coeffs.reshape(broadcast_dims)

        

        a = np.einsum('ijkl,ilkm->ijkm', qm * dct_coeffs, t)

        b = np.einsum('ijkl,ilkm->ijkm', np.transpose(t, axes=(0, 3, 2, 1)), a)

        YCbCr.append(b.reshape(img_dims))

                    

    return np.stack(YCbCr, -1).astype(np.float32)





def JPEGdecompressYCbCr_v3(path):

    jpegStruct = jio.read(str(path))



    [col, row] = np.meshgrid(range(8), range(8))

    T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))

    T[0, :] = T[0, :] / np.sqrt(2)



    img_dims = np.array(jpegStruct.coef_arrays[0].shape)

    n_blocks = img_dims // 8

    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)

    

    YCbCr = []

    for i, dct_coeffs, in enumerate(jpegStruct.coef_arrays):



        if i == 0:

            QM = jpegStruct.quant_tables[i]

        else:

            QM = jpegStruct.quant_tables[1]

        

        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)

        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)

        dct_coeffs = dct_coeffs.reshape(broadcast_dims)

        

        a = np.transpose(t, axes=(0, 2, 3, 1))

        b = (qm * dct_coeffs).transpose(0,2,1,3)

        c = t.transpose(0,2,1,3)

                

        z = a @ b @ c

        z = z.transpose(0,2,1,3)

        YCbCr.append(z.reshape(img_dims))

                    

    return np.stack(YCbCr, -1).astype(np.float32)
img_v2 = JPEGdecompressYCbCr_v2(files_cover[0])  # np.einsum version

img_v3 = JPEGdecompressYCbCr_v3(files_cover[0])  # matrix multiplication version

img_v2.shape, img_v3.shape
# Check if they are the same

np.allclose(img_v1, img_v2), np.allclose(img_v1, img_v3)
%%timeit -n 100

JPEGdecompressYCbCr(files_cover[0])
%%timeit -n 100

JPEGdecompressYCbCr_v2(files_cover[0])
%%timeit -n 100

JPEGdecompressYCbCr_v3(files_cover[0])
%%timeit -n 100

Image.open(files_cover[0]).convert('YCbCr')
img_pil = np.array(Image.open(files_cover[0]).convert('YCbCr'), dtype=np.int32) - 128

img_pil[0]
img_v3[0]
difference = img_pil - img_v3

print(f"Mean difference: {np.mean(difference):0.5f}, Std Dev: {np.std(difference):0.5f}")

plt.hist(difference.flatten(), bins=50);