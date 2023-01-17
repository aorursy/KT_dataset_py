import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import cv2

import matplotlib.pyplot as plt



import ipywidgets as widgets

from ipywidgets import interact



import glob

import zipfile
def plot_image(number):

    """

    Визуализирует 3 вида одной фотографии: [BGR, Grayscale, RGB] с выбором названия

    """

    fig = plt.figure(figsize=(16, 10))

    image = cv2.imread(f"{dirname}/0{number}.png")

    

    fig.add_subplot(1, 3, 1)

    plt.imshow(image)

    plt.title('BGR', fontsize=15)

    

    fig.add_subplot(1, 3, 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')

    plt.title('Grayscale', fontsize=15)

    

    fig.add_subplot(1, 3, 3)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.title('RGB', fontsize=15)

    



# Validation image: 0 + [801; 900]

ip = widgets.interactive(plot_image,

                         number=widgets.IntSlider(min=801, max=900, step=1, value=866, 

                                                  description=r'$Number\;of\;image$',

                                                  style={'description_width': 'initial'}, 

                                                  layout=dict(width='80%')))



display(widgets.HBox(ip.children[:1]))

display(ip.children[-1])

ip.update()
def images_save(cv_format, name_directory, save_directory):

    """

    Сохраняет "png" файлы из name_directory в save_directory, используя формат cv_format

    """

    images = [(os.path.basename(file), 

               cv2.cvtColor(cv2.imread(file), cv_format)) for file in glob.glob(f"{name_directory}/*.png")]

    

    save_list = [cv2.imwrite(filename=f"{save_directory}/{name_file}", img=image) for name_file, image in images]

    return images, all(save_list)
open_dirr = '/kaggle/input/movavi-dataset/DIV2K_valid_HR'

save_dirr = 'original'



!mkdir -p {save_dirr}



list_images, flag = images_save(cv2.COLOR_BGR2GRAY, open_dirr, save_dirr)

flag
def plot_directory(name_directory, count=None, gray=False):

    """

    Отрисовывает count "png" файлов из name_directory и возвращает список файлов из изображений, 

    открытых в grayscale, если gray=True

    """

    images = []

    

    list_png = glob.glob(f"{name_directory}/*.png")

    count = len(list_png) if count == None else count

    

    fig = plt.figure(figsize=(20, count * 2))

    

    for i, png in enumerate(sorted(list_png)[:count]):

        # пусть 3 столбца

        fig.add_subplot(count // 3 + 1, 3, i + 1)

        image = cv2.imread(png, 0 if gray else 1)

        plt.imshow(image, cmap='gray' if gray else None)

        plt.title(os.path.basename(png), fontsize=15)



        images.append((os.path.basename(png), image))

        

    return images





count_images = 10

list_images = plot_directory(name_directory=save_dirr, count=count_images, gray=True)
negativ = lambda image: cv2.bitwise_not(image)
def list_save(l, save_directory, fun_format, **other):

    """

    Записывает список фотографий l в save_directory, 

    применив функцию fun_format c именованными аргументами other

    """

    for elem in l:

        pixels = fun_format(elem[1], **other)

        cv2.imwrite(filename=f"{save_directory}/{elem[0]}", img=pixels)
save_dirr = 'negative'

!mkdir -p {save_dirr}



list_save(list_images, save_dirr, negativ)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
logaryth = lambda image: cv2.normalize(np.uint8(np.log1p(image)), None, alpha=0, beta=255, 

                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)



# точно в [0; 255]

logaryth2 = lambda image: np.uint8(255 / np.log1p(np.max(image)) * np.log1p(image))



# (np.clip(> 255) -> 255) -> uint8

logaryth3 = lambda image, c: np.uint8(np.clip(c * np.log1p(np.int64(image)), 0, 255))
fig = plt.figure(figsize=(18, 14))



image = cv2.imread('./original/0801.png')

fig.add_subplot(1, 4, 1)

plt.imshow(image)

plt.title('Original', fontsize=15)



fig.add_subplot(1, 4, 2)

plt.imshow(logaryth(image))

plt.title('logaryth', fontsize=15)



fig.add_subplot(1, 4, 3)

plt.imshow(logaryth2(image))

plt.title('logaryth2', fontsize=15)



c = 35

fig.add_subplot(1, 4, 4)

plt.imshow(logaryth3(image, c))

plt.title('logaryth3', fontsize=15)



plt.show()
save_dirr = 'logarythm'

!mkdir -p {save_dirr}



# можно logaryth / logaryth3

list_save(list_images, save_dirr, logaryth2)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
gamm = lambda image, c, gamma: cv2.LUT(image, np.array([np.uint8(np.clip(c * ((i / 255.0) ** gamma), 0, 255)) \

                                              for i in np.arange(0, 256)]).astype("uint8"))
save_dirr = 'gamma'

!mkdir -p {save_dirr}



c = 255

gamma = 2.5

list_save(list_images, save_dirr, gamm, c=c, gamma=gamma)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
equalization = lambda image: cv2.equalizeHist(image)
plt.subplots(figsize=(16, 10))



image = cv2.imread('./original/0801.png', 0)



equ = equalization(image)



# -> hist ~ Rav() по блокам

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

cl1 = clahe.apply(image)



res = np.hstack((image, equ, cl1)) 



plt.imshow(res, cmap='gray')

plt.title('Original / Histogram Equalization / CLANE', fontsize=15)

plt.show()
save_dirr = 'histo'

!mkdir -p {save_dirr}



list_save(list_images, save_dirr, equalization)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
def linear_smooth(image, method, width, height):

    """

    На основании method вернуть нужное сглаживание image с параметрами окна [width; height] 

    """

    options = {

        'average': cv2.blur(image, (width, height)),

        'gaussian': cv2.GaussianBlur(image, (width, height), 0)

    }

    return options[method]
width, height = 17, 17

image2 = linear_smooth(image, 'gaussian', width, height)

image3 = linear_smooth(image, 'average', width, height)



plt.subplots(figsize=(16, 10))



plt.imshow(np.hstack((image, image2, image3)), cmap='gray')

plt.title(f'Original / Gaussian({width}; {height}) / Average({width}; {height})', fontsize=15)

plt.show()
save_dirr = 'smooth'

!mkdir -p {save_dirr}



list_save(list_images, save_dirr, linear_smooth, method='average', width=width, height=height)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
# сначала сохранить больше информации о интенсивности пикселя (64 бита)

# затем в uint8 (p.s. отрицательные -> положительные; положительные > 255 -> 255), т.е. абсолютный

laplacian1 = lambda image: cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F))



# числа в диапозоне [0; 255] с преобразованием: отрицательное -> 0

laplacian2 = lambda image: cv2.Laplacian(image, ddepth=-1, ksize=1)
plt.subplots(figsize=(14, 10))

plt.imshow(np.hstack([laplacian1(image), laplacian2(image)]), cmap='gray')

plt.title('Абсолютный Лаплассиан / С потерей информации', fontsize=15)

plt.show()
save_dirr = 'laplassian'

!mkdir -p {save_dirr}



list_save(list_images, save_dirr, laplacian1)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
save_dirr = 'laplassian2'

!mkdir -p {save_dirr}



# вторая вариация

list_save(list_images, save_dirr, laplacian2)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
def standart(image):

    """

    Возвращает стандартизованную форму (без абсолютных значений) image в uint8

    """

    image2 = image.copy()

    image2[image2 < 0.0] = 0.0

    image2[image2 > 255.0] = 255.0

    image2 = image2.astype(np.uint8)

    return image2





# abs_uint8(f + c * abs(lapl))

sharp_laplacian = lambda image, c: cv2.convertScaleAbs(np.int64(image) + c * cv2.Laplacian(image, cv2.CV_64F))
plt.subplots(figsize=(18, 14))

plt.imshow(np.hstack([standart(np.int64(image) -1 * cv2.Laplacian(image, cv2.CV_64F)), 

                      sharp_laplacian(image, -1)]), cmap='gray')

plt.title('Стандартизованный лаплассиан (с = -1) / Абсолютный (с = -1)', fontsize=15)

plt.show()
save_dirr = 'sharp'

!mkdir -p {save_dirr}



# c = -3 для лучших контуров

list_save(list_images, save_dirr, sharp_laplacian, c=-3)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
# выделение границ -> abs(intensity)

sobel_X = lambda image: np.uint8(np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0)))

sobel_Y = lambda image: np.uint8(np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1)))



# Sobel - объединение (X, Y)

sobel_XY = lambda image: cv2.bitwise_or(sobel_X(image), sobel_Y(image))
plt.subplots(figsize=(20, 16))

plt.imshow(np.hstack([sobel_X(image), sobel_Y(image), sobel_XY(image)]), cmap='gray')

plt.title(r'Sobel: X / Y / X  $\vee$ Y', fontsize=15)

plt.show()
save_dirr = 'sobel'

!mkdir -p {save_dirr}



list_save(list_images, save_dirr, sobel_XY)

_ = plot_directory(name_directory=save_dirr, count=count_images, gray=False)
def gradation_cor(image, c):

    """

    Вернуть градационную коррекцию image с параметром масштаба c

    """

    image2 = image.copy()

    image2 = image2 - image2.min()

    image2 = c * image2 / image2.max()

    return image2
image = cv2.imread('../input/moon-opencv/moon.jpg', 0)



# диагональная маска

kernel = np.ones((3, 3))

kernel[1,1] = -8



laplassian_64bit = cv2.filter2D(image, cv2.CV_64F, kernel)

# laplassian_64bit = cv2.Laplacian(image, cv2.CV_64F)



gradcor_64bit = gradation_cor(laplassian_64bit, 255)



plt.subplots(figsize=(16, 12))

plt.imshow(np.hstack([image, standart(laplassian_64bit), standart(gradcor_64bit)]), cmap='gray')

plt.title('Исходное (1) / Лаплассиан / Градационная коррекция на лаплассиане (2)', fontsize=15)

plt.show()



plt.subplots(figsize=(10, 10))

plt.imshow(np.hstack([standart(np.int64(image) - 1 * laplassian_64bit), 

                      standart(np.int64(image) - 1 * gradcor_64bit)]), cmap='gray')

plt.title(r'Исходное - 1 $\cdot$ Лаплассиан / Исходное - 1 $\cdot$ Градационная коррекция лаплассиана (3)')

plt.show()
sobel = sobel_XY(image)

sobel_smoothed = linear_smooth(sobel, method='average', width=5, height=5)
plt.subplots(figsize=(12, 10))

# standart == обычному

plt.imshow(np.hstack([standart(sobel), standart(sobel_smoothed)]), cmap='gray')

plt.title(r'Собель X $\vee$ Y (4) / Собель $\wedge$ Average (5)', fontsize=15)

plt.show()
laplacian_sobel = cv2.bitwise_and(standart(laplassian_64bit), sobel_smoothed)

log = linear_smooth(standart(laplassian_64bit), method='gaussian', width=5, height=5)
plt.subplots(figsize=(16, 12))

plt.imshow(np.hstack([laplacian_sobel, standart(laplassian_64bit), log]), cmap='gray')

plt.title(r'Лаплассиан $\cdot$ сглаженный Собель (6) / Лаплассиан / Лаплассиан $\wedge$ Гауссиан', fontsize=15)

plt.show()
plt.subplots(figsize=(16, 12))

plt.imshow(np.hstack([standart(np.int64(image) - 1 * np.int64(laplacian_sobel)), 

                      gamm(image, 255, 0.5)]), cmap='gray')

plt.title(r'Исходное - 1 $\cdot$ Лаплассиан $\cdot$ сглаженный Собель (7) / + Гамма преобразование $(\gamma = 0.5)$ (8)',

         fontsize=15)

plt.show()
def combination(image, kernel):

    """

    Возвращает закреплённое по горизонтали изображение:

    [Исходное, Исходное - Лаплассиан, Собель, Исходное - Лаплассиан * Собель, Гамма коррекция последнего]

    kernel - окно со значениями для лаплассиана

    """

    laplassian_64bit = cv2.filter2D(image, cv2.CV_64F, kernel)

    gradcor_64bit = gradation_cor(laplassian_64bit, 255)

    

    sobel = sobel_XY(image)

    sobel_smoothed = linear_smooth(sobel, method='average', width=5, height=5)

    laplacian_sobel = cv2.bitwise_and(standart(laplassian_64bit), sobel_smoothed)

    log = linear_smooth(standart(laplassian_64bit), method='gaussian', width=5, height=5)



    return np.hstack([image, standart(np.int64(image) - 1 * laplassian_64bit), standart(sobel), 

                          standart(np.int64(image) - 1 * np.int64(laplacian_sobel)),

                                  gamm(image, 255, 0.5)])
save_dirr = 'enhancement'

!mkdir -p {save_dirr}



list_save(list_images, save_dirr, combination, kernel=kernel)



for image in sorted(glob.glob(f"{save_dirr}/*.png")):

    plt.subplots(figsize=(20, 10))

    plt.imshow(cv2.imread(image), cmap='gray')

    plt.title(fr"{image} / - Лаплассиан / Собель / - Лаплассиан $\cdot$ Собель / - Лаплассиан $\cdot$ Собель с гамма преобразованием",

             fontsize=15)

    plt.show()
def zip_and_remove(path):

    """

    Преобразует все папки в zip-файлы, удаляя файлы из исходных

    [чтобы при просмотре kernel-а не было сотен фотографий]

    """

    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)

    

    for root, dirs, files in os.walk(path):

        for file in files:

            file_path = os.path.join(root, file)

            ziph.write(file_path)

            os.remove(file_path)

    

    ziph.close()



    

for dirname, dirs, filenames in os.walk('/kaggle/working'):

    for dirname in dirs:

        zip_and_remove(dirname)