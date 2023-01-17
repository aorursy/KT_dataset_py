import numpy as np # linear algebra

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)
import cv2

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.patches as patches
img1=cv2.imread('/kaggle/input/gerek-resimler/img_real1.jpg')

img1 = cv2.resize(img1, (64,64), interpolation = cv2.INTER_AREA) 

img1rgb=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2=cv2.imread('/kaggle/input/gerek-resimler/img_real2.jpg')

img2 = cv2.resize(img2, (64,64), interpolation = cv2.INTER_AREA) 

img2rgb=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3=cv2.imread('/kaggle/input/gerek-resimler/img_real3.jpg')

img3 = cv2.resize(img3, (64,64), interpolation = cv2.INTER_AREA) 

img3rgb=cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

img4=cv2.imread('/kaggle/input/gerek-resimler/img_real4.jpg')

img4 = cv2.resize(img4, (64,64), interpolation = cv2.INTER_AREA) 

img4rgb=cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

img5=cv2.imread('/kaggle/input/gerek-resimler/img_real5.jpg')

img5 = cv2.resize(img5, (64,64), interpolation = cv2.INTER_AREA) 

img5rgb=cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)

img6=cv2.imread('/kaggle/input/gerek-resimler/img_real6.jpg')

img6 = cv2.resize(img6, (64,64), interpolation = cv2.INTER_AREA) 

img6rgb=cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)

img7=cv2.imread('/kaggle/input/gerek-resimler/img_real7.jpg')

img7 = cv2.resize(img7, (64,64), interpolation = cv2.INTER_AREA) 

img7rgb=cv2.cvtColor(img7, cv2.COLOR_BGR2RGB)

img8=cv2.imread('/kaggle/input/gerek-resimler/img_real8.jpg')

img8 = cv2.resize(img8, (64,64), interpolation = cv2.INTER_AREA) 

img8rgb=cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)

fig = plt.figure()

fig.set_figwidth(18)

a=fig.add_subplot(1, 8, 1)

a.set_title('1.img '+str(img1rgb.shape))

plt.imshow(img1rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 2)

a.set_title('2.img '+str(img2rgb.shape))

plt.imshow(img2rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 3)

a.set_title('3.img '+str(img3rgb.shape))

plt.imshow(img3rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 4)

a.set_title('4.img '+str(img4rgb.shape))

plt.imshow(img4rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 5)

a.set_title('5.img '+str(img5rgb.shape))

plt.imshow(img5rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 6)

a.set_title('6.img '+str(img6rgb.shape))

plt.imshow(img6rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 7)

a.set_title('7.img '+str(img7rgb.shape))

plt.imshow(img7rgb)

plt.axis('off')

a=fig.add_subplot(1, 8, 8)

a.set_title('8.img '+str(img8rgb.shape))

plt.imshow(img8rgb)

plt.axis('off')

kart1=cv2.imread('/kaggle/input/gerek-kartlar/kart1.jpg')

kart1 = cv2.resize(kart1, (256,256), interpolation = cv2.INTER_AREA) 

kart1rgb=cv2.cvtColor(kart1, cv2.COLOR_BGR2RGB)

kart2=cv2.imread('/kaggle/input/gerek-kartlar/kart2.jpg')

kart2 = cv2.resize(kart2, (256,256), interpolation = cv2.INTER_AREA)

kart2rgb=cv2.cvtColor(kart2, cv2.COLOR_BGR2RGB)

kart3=cv2.imread('/kaggle/input/gerek-kartlar/kart3.jpg')

kart3 = cv2.resize(kart3, (256,256), interpolation = cv2.INTER_AREA)

kart3rgb=cv2.cvtColor(kart3, cv2.COLOR_BGR2RGB)

kart4=cv2.imread('/kaggle/input/gerek-kartlar/kart4.jpg')

kart4 = cv2.resize(kart4, (256,256), interpolation = cv2.INTER_AREA)

kart4rgb=cv2.cvtColor(kart4, cv2.COLOR_BGR2RGB)

kart5=cv2.imread('/kaggle/input/gerek-kartlar/kart5.jpg')

kart5 = cv2.resize(kart5, (256,256), interpolation = cv2.INTER_AREA)

kart5rgb=cv2.cvtColor(kart5, cv2.COLOR_BGR2RGB)

kart6=cv2.imread('/kaggle/input/gerek-kartlar/kart6.jpg')

kart6 = cv2.resize(kart6, (256,256), interpolation = cv2.INTER_AREA)

kart6rgb=cv2.cvtColor(kart6, cv2.COLOR_BGR2RGB)

kart7=cv2.imread('/kaggle/input/gerek-kartlar/kart7.jpg')

kart7 = cv2.resize(kart7, (256,256), interpolation = cv2.INTER_AREA)

kart7rgb=cv2.cvtColor(kart7, cv2.COLOR_BGR2RGB)

fig = plt.figure()

fig.set_figwidth(18)

a=fig.add_subplot(1, 7, 1)

a.set_title('1.kart '+str(kart1rgb.shape))

plt.imshow(kart1rgb)

plt.axis('off')

a=fig.add_subplot(1, 7, 2)

a.set_title('2.kart '+str(kart2rgb.shape))

plt.imshow(kart2rgb)

plt.axis('off')

a=fig.add_subplot(1, 7, 3)

a.set_title('3.kart '+str(kart3rgb.shape))

plt.imshow(kart3rgb)

plt.axis('off')

a=fig.add_subplot(1, 7, 4)

a.set_title('4.kart '+str(kart4rgb.shape))

plt.imshow(kart4rgb)

plt.axis('off')

a=fig.add_subplot(1, 7, 5)

a.set_title('5.kart '+str(kart5rgb.shape))

plt.imshow(kart5rgb)

plt.axis('off')

a=fig.add_subplot(1, 7, 6)

a.set_title('6.kart '+str(kart6rgb.shape))

plt.imshow(kart6rgb)

plt.axis('off')

a=fig.add_subplot(1, 7, 7)

a.set_title('7.kart '+str(kart7rgb.shape))

plt.imshow(kart7rgb)

plt.axis('off')
def concat_tile(im_list_2d):

    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

im_tile = concat_tile([[img5rgb, img4rgb, img5rgb, img4rgb],

                       [img4rgb, img5rgb, img4rgb, img5rgb],

                       [img5rgb, img4rgb, img5rgb, img4rgb],

                       [img4rgb, img5rgb, img4rgb, img5rgb]])

plt.imshow(im_tile)

plt.axis('off')
def kernel_size(image):

    size=[int((image.shape[0])/4),int((image.shape[1])/4)]

    return size
def solust_ortalama(img_binary):

    count=0

    sum=0

    for i in range (0,kernel_size(img_binary)[0]):

        for j in range(0,kernel_size(img_binary)[1]):

            sum=sum+img_binary[i,j]

            count=count+1

    result=sum/count

    return (result)
def sagalt_ortalama(img_binary):

    count=0

    sum=0

    for i in range (img_binary.shape[0]-(kernel_size(img_binary)[0]),img_binary.shape[0]):

        for j in range(img_binary.shape[1]-(kernel_size(img_binary)[1]),img_binary.shape[1]):

            sum=sum+img_binary[i,j]

            count=count+1

    result=sum/count

    return (result)
def sagust_ortalama(img_binary):

    count=0

    sum=0

    for i in range (0,kernel_size(img_binary)[0]):

        for j in range(img_binary.shape[1]-(kernel_size(img_binary)[1]),img_binary.shape[1]):

            sum=sum+img_binary[i,j]

            count=count+1

    result=sum/count

    return (result)
def solalt_ortalama(img_binary):  

    count=0

    sum=0

    for i in range (img_binary.shape[0]-(kernel_size(img_binary)[0]),img_binary.shape[0]):

        for j in range(0,kernel_size(img_binary)[1]):

            sum=sum+img_binary[i,j]

            count=count+1

    result=sum/count

    return (result)
def orta_ortalama(img_binary):

    count=0

    sum=0

    for i in range (int(img_binary.shape[0]/2)-int(kernel_size(img_binary)[0]/2),int(img_binary.shape[0]/2)+int(kernel_size(img_binary)[0]/2)):

        for j in range(int(img_binary.shape[1]/2)-int(kernel_size(img_binary)[1]/2),int(img_binary.shape[1]/2)+int(kernel_size(img_binary)[1]/2)):

            sum=sum+img_binary[i,j]

            count=count+1

    result=sum/count

    return (result)
def tahmin(image):

    #resized = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA) 

    img_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _,img_binary=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)

    imgrgb=cv2.cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

    #fig,ax = plt.subplots(1)

    #ax.add_patch(rect)

    #ax.imshow(imgrgb,cmap=cm.gray, vmin=0, vmax=255)

    #plt.axis('off')

    tahmin="Bilemedim"

    if solalt_ortalama(img_binary)-sagust_ortalama(img_binary)>127:

        tahmin="Sağ üst renkli"

    elif solust_ortalama(img_binary)-sagalt_ortalama(img_binary)>127:

        tahmin="Sağ alt renkli"

    elif sagust_ortalama(img_binary)-solalt_ortalama(img_binary)>127:

        tahmin="Sol alt renkli"

    elif sagalt_ortalama(img_binary)-solust_ortalama(img_binary)>127:

        tahmin="Sol üst renkli"

    elif (solust_ortalama(img_binary)+sagalt_ortalama(img_binary)+orta_ortalama(img_binary))/3>200.0:

        tahmin="Tamamen beyaz"

    elif (solust_ortalama(img_binary)+sagalt_ortalama(img_binary)+orta_ortalama(img_binary))/3<50.0:

        tahmin="Tamamen renkli"

    elif orta_ortalama(img_binary)<127:

        tahmin="Orta renkli"

    elif orta_ortalama(img_binary)>127:

        tahmin="Orta beyaz"

    return tahmin,imgrgb
sonuc,image=tahmin(img1)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img2)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img3)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img4)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img5)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img6)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img7)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
sonuc,image=tahmin(img8)

print(sonuc)

rect = patches.Rectangle((0,0),63,63,linewidth=2,edgecolor='black',facecolor='none')

fig,ax = plt.subplots(1)

ax.add_patch(rect)

ax.imshow(image,cmap=cm.gray, vmin=0, vmax=255)

plt.axis('off')
def kart_oku(kart_resmi):

    kup_liste=[img1,img2,img3,img4,img5,img6,img7,img8]

    thm1of16=img = np.zeros((64,64,3), np.int8)

    thm2of16=img = np.zeros((64,64,3), np.int8)

    thm3of16=img = np.zeros((64,64,3), np.int8)

    thm4of16=img = np.zeros((64,64,3), np.int8)

    thm5of16=img = np.zeros((64,64,3), np.int8)

    thm6of16=img = np.zeros((64,64,3), np.int8)

    thm7of16=img = np.zeros((64,64,3), np.int8)

    thm8of16=img = np.zeros((64,64,3), np.int8)

    thm9of16=img = np.zeros((64,64,3), np.int8)

    thm10of16=img = np.zeros((64,64,3), np.int8)

    thm11of16=img = np.zeros((64,64,3), np.int8)

    thm12of16=img = np.zeros((64,64,3), np.int8)

    thm13of16=img = np.zeros((64,64,3), np.int8)

    thm14of16=img = np.zeros((64,64,3), np.int8)

    thm15of16=img = np.zeros((64,64,3), np.int8)

    thm15of16=img = np.zeros((64,64,3), np.int8)

    thm16of16=img = np.zeros((64,64,3), np.int8)

    img1of16=kart_resmi[0:64, 0:64]

    sonuc,temp_cube=tahmin(img1of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm1of16=temp_img

    img2of16=kart_resmi[0:64, 65:128]

    sonuc,temp_cube=tahmin(img2of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm2of16=temp_img

    img3of16=kart_resmi[0:64, 129:192]

    sonuc,temp_cube=tahmin(img3of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm3of16=temp_img

    img4of16=kart_resmi[0:64, 193:256]

    sonuc,temp_cube=tahmin(img4of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm4of16=temp_img

    img5of16=kart_resmi[65:128, 0:64]

    sonuc,temp_cube=tahmin(img5of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm5of16=temp_img

    img6of16=kart_resmi[65:128, 65:128]

    sonuc,temp_cube=tahmin(img6of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm6of16=temp_img

    img7of16=kart_resmi[65:128, 129:192]

    sonuc,temp_cube=tahmin(img7of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm7of16=temp_img

    img8of16=kart_resmi[65:128, 193:256]

    sonuc,temp_cube=tahmin(img8of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm8of16=temp_img

    img9of16=kart_resmi[129:192, 0:64]

    sonuc,temp_cube=tahmin(img9of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm9of16=temp_img

    img10of16=kart_resmi[129:192, 65:128]

    sonuc,temp_cube=tahmin(img10of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm10of16=temp_img

    img11of16=kart_resmi[129:192, 129:192]

    sonuc,temp_cube=tahmin(img11of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm11of16=temp_img

    img12of16=kart_resmi[129:192, 193:256]

    sonuc,temp_cube=tahmin(img12of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm12of16=temp_img

    img13of16=kart_resmi[193:256, 0:64]

    sonuc,temp_cube=tahmin(img13of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm13of16=temp_img

    img14of16=kart_resmi[193:256, 65:128]

    sonuc,temp_cube=tahmin(img14of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm14of16=temp_img

    img15of16=kart_resmi[193:256, 129:192]

    sonuc,temp_cube=tahmin(img15of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm15of16=temp_img

    img16of16=kart_resmi[193:256, 193:256]

    sonuc,temp_cube=tahmin(img16of16)

    for i in (kup_liste):

        temp_sonuc,temp_img=tahmin(i)

        if temp_sonuc==sonuc:

            thm16of16=temp_img

    im_tile = concat_tile([[thm1of16, thm2of16, thm3of16, thm4of16],

                           [thm5of16, thm6of16, thm7of16, thm8of16],

                           [thm9of16, thm10of16, thm11of16, thm12of16],

                           [thm13of16, thm14of16, thm15of16, thm16of16]])

    fig = plt.figure()

    fig.set_figwidth(10)

    a=fig.add_subplot(1, 2, 1)

    a.set_title('Kart ')

    plt.imshow(kart_resmi)

    plt.axis('off')

    a=fig.add_subplot(1, 2, 2)

    a.set_title('Dizgi ')

    plt.imshow(im_tile)

    plt.axis('off')

kart_oku(kart1)
kart_oku(kart2)
kart_oku(kart3)
kart_oku(kart4)
kart_oku(kart5)
kart_oku(kart6)
kart_oku(kart7)