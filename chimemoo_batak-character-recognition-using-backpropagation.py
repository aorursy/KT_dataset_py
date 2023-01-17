%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import cv2

from skimage.morphology import skeletonize
Karo = ['A','Ba','Ca','Da','Ga','I','Ja','Ka','La','Ma','Na','Nga','Pa','Ra','Sa','Ta','U','Wa','Ya']

Mandailing = ['A','Ba','Ca','Da','Ga','Ha','I','Ja','Ka','La','Ma','Na','Nga','Nya','Pa','Ra','Sa','Ta','U','Wa','Ya']

Pakpak = ['A','Ba','Da','Ga','I','Ja','Ka','La','Ma','Na','Nga','Pa','Ra','Sa','Ta','U','Wa','Ya']

Simalungun = ['A','Ba','Da','Ga','Ha','I','Ja','La','Ma','Mba','Na','Nda','Nga','Nya','Pa','Ra','Sa','Ta','U','Wa','Ya']

Toba = ['A','Ba','Da','Ga','Ha','I','Ja','Ka','La','Ma','Na','Nga','Nya','Pa','Ra','Sa','Ta','U','Wa','Ya']

print("Total Huruf")

print("Huruf Karo : "+str(str(Karo).count(",")+1))

print("Huruf Mandailing : "+str(str(Mandailing).count(",")+1))

print("Huruf Pakpak : "+str(str(Pakpak).count(",")+1))

print("Huruf Simalungun : "+str(str(Simalungun).count(",")+1))

print("Huruf Toba : "+str(str(Toba).count(",")+1))
#FUNGSI UNTUK MELAKUKAN PREPROCESSING

def preprocessing(image):

    image = np.uint8(image) #Convert to unsigned integer 8

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grayscalling

    ret,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY) #To Binary

    P = image.shape[0]

    Q = image.shape[1]

    for p in range(0,P):

        for q in range(0,Q):

            if image[p,q] == 255:

                image[p,q] = 0

            elif image[p,q] == 0:

                image[p,q] = 1

    skeleton = skeletonize(image)

    return 1*skeleton
hurufkaro = {'A':{},'Ba':{},'Ca':{} ,'Da':{} ,'Ga':{} ,'I':{} ,'Ja':{} ,'Ka':{} ,'La':{} ,'Ma':{} ,'Na':{} ,'Nga':{} ,'Pa':{} ,'Ra':{} ,'Sa':{} ,'Ta':{} ,'U':{} ,'Wa':{} ,'Ya':{} }

hurufmandailing = {'A':{},'Ba':{} ,'Ca':{} ,'Da':{} ,'Ga':{} ,'Ha':{} ,'I':{} ,'Ja':{} ,'Ka':{},'La':{},'Ma':{},'Na':{},'Nga':{},'Nya':{},'Pa':{},'Ra':{},'Sa':{},'Ta':{},'U':{} ,'Wa':{} ,'Ya':{}} 

hurufpakpak = {'A':{},'Ba':{} ,'Da':{} ,'Ga':{} ,'I':{} ,'Ja':{} ,'Ka':{} ,'La':{} ,'Ma':{} ,'Na':{} ,'Nga':{} ,'Pa':{} ,'Ra':{} ,'Sa':{} ,'Ta':{},'U':{},'Wa':{},'Ya':{}}

hurufsimalungun = {'A': {}, 'Ba': {}, 'Da': {}, 'Ga': {}, 'Ha': {}, 'I': {}, 'Ja': {}, 'La': {}, 'Ma': {}, 'Mba': {}, 'Na': {}, 'Nda':{}, 'Nga': {}, 'Nya': {} , 'Pa': {}, 'Ra': {}, 'Sa':{}, 'Ta': {}, 'U': {}, 'Wa': {}, 'Ya': {} }

huruftoba = {'A':{},'Ba':{},'Da':{},'Ga':{},'Ha':{},'I':{},'Ja':{},'Ka':{},'La':{},'Ma':{},'Na':{},'Nga':{},'Nya':{},'Pa':{},'Ra':{},'Sa':{},'Ta':{},'U':{},'Wa':{},'Ya':{} }

surat_batak = [

    'Karo',

    'Mandailing',

    'Pakpak',

    'Simalungun',

    'Toba'

]



for surat in surat_batak:

    list_surat = locals()[surat]

    for i in list_surat:

        for a in range(1,11):

            alamat = '../input/suratbatak/surat batak/'+str(surat)+' Final/'+str(i)+'/'+str(a)+'.png'

            img = cv2.imread(alamat)

            if surat == 'Karo':

                hurufkaro[i][str(a)] = preprocessing(img)

            elif surat == 'Mandailing':

                hurufmandailing[i][str(a)] = preprocessing(img)

            elif surat == 'Pakpak':

                hurufpakpak[i][str(a)] = preprocessing(img)

            elif surat == 'Simalungun':

                hurufsimalungun[i][str(a)] = preprocessing(img)

            elif surat == 'Toba':

                huruftoba[i][str(a)] = preprocessing(img)
f, axarr = plt.subplots(4, 5)

no = 0

for baris in range(0,4):

    for kolom in range(0,5):

        axarr[baris, kolom].imshow(huruftoba[Toba[no]]['1'],cmap='gray')

        axarr[baris, kolom].set_title(Toba[no])

        no = no+1

f.subplots_adjust(hspace=0.9)
print('Total Karakter Pada Dataset : '+str(99*10))

print('Masing-masing karakter memiliki 10 sample 7 untuk ditraining dan 2 untuk testing');



jml_karo = str(Karo).count(",")+1

jml_mand = str(Mandailing).count(",")+1

jml_pak  = str(Pakpak).count(",")+1

jml_sim  = str(Simalungun).count(",")+1

jml_toba = str(Toba).count(",")+1

total = jml_karo + jml_mand + jml_pak + jml_sim + jml_toba;

print("Jumlalh yang digunakan : "+str(total))

def make_a(image):

    P = image.shape[0]

    Q = image.shape[1]

    inp_new = 0*image

    for p in range(1,P-1):

        for q in range(1, Q-1):

            if image[p,q] == 1:

                if (image[p-1,q-1] != 0 or image[p+1,q+1] != 0):

                    image[p,q] = 5

                if (image[p-1, q] != 0 or image[p+1, q] != 0):

                    image[p,q] = 2

                if (image[p-1,q+1] != 0 or image[p+1,q-1] != 0):

                    image[p,q] = 3

                if (image[p, q-1] != 0 or image[p,q+1] != 0):

                    image[p,q] = 4

    return image
karo_arah = {'A':{},'Ba':{},'Ca':{} ,'Da':{} ,'Ga':{} ,'I':{} ,'Ja':{} ,'Ka':{} ,'La':{} ,'Ma':{} ,'Na':{} ,'Nga':{} ,'Pa':{} ,'Ra':{} ,'Sa':{} ,'Ta':{} ,'U':{} ,'Wa':{} ,'Ya':{} }

mandailing_arah = {'A':{},'Ba':{} ,'Ca':{} ,'Da':{} ,'Ga':{} ,'Ha':{} ,'I':{} ,'Ja':{} ,'Ka':{},'La':{},'Ma':{},'Na':{},'Nga':{},'Nya':{},'Pa':{},'Ra':{},'Sa':{},'Ta':{},'U':{} ,'Wa':{} ,'Ya':{}} 

pakpak_arah = {'A':{},'Ba':{} ,'Da':{} ,'Ga':{} ,'I':{} ,'Ja':{} ,'Ka':{} ,'La':{} ,'Ma':{} ,'Na':{} ,'Nga':{} ,'Pa':{} ,'Ra':{} ,'Sa':{} ,'Ta':{},'U':{},'Wa':{},'Ya':{}}

simalungun_arah = {'A': {}, 'Ba': {}, 'Da': {}, 'Ga': {}, 'Ha': {}, 'I': {}, 'Ja': {}, 'Ka': {}, 'La': {}, 'Ma': {}, 'Mba': {}, 'Na': {}, 'Nda':{}, 'Nga': {}, 'Nya': {} , 'Pa': {}, 'Ra': {}, 'Sa':{}, 'Ta': {}, 'U': {}, 'Wa': {}, 'Ya': {} }

toba_arah = {'A':{},'Ba':{},'Da':{},'Ga':{},'Ha':{},'I':{},'Ja':{},'Ka':{},'La':{},'Ma':{},'Na':{},'Nga':{},'Nya':{},'Pa':{},'Ra':{},'Sa':{},'Ta':{},'U':{},'Wa':{},'Ya':{} }



def dfeprocess(arah):

    tot = 11

    if arah == 'Karo':

        for i in Karo:

            for a in range(1,tot):

                karo_arah[i][str(a)] = make_a(hurufkaro[i][str(a)])

    elif arah == 'Mandailing':

        for i in Mandailing:

            for a in range(1,tot):

                mandailing_arah[i][str(a)] = make_a(hurufmandailing[i][str(a)])

    elif arah == 'Pakpak':

        for i in Pakpak:

            for a in range(1,tot):

                pakpak_arah[i][str(a)] = make_a(hurufpakpak[i][str(a)])

    elif arah == 'Simalungun':

        for i in Simalungun:

            for a in range(1,tot):

                simalungun_arah[i][str(a)] = make_a(hurufsimalungun[i][str(a)])

    elif arah == 'Toba':

        for i in Toba:

            for a in range(1,tot):

                toba_arah[i][str(a)] = make_a(huruftoba[i][str(a)])



for x in surat_batak:

    dfeprocess(x)
print(toba_arah['A']['10'])
dtset_karo = []

dtset_mandailing = []

dtset_pakpak = []

dtset_simalungun = []

dtset_toba = []

tot = 8

for i in Karo:

    for a in range(1,tot):

        dtset_karo.append(i)

for i in Mandailing:

    for a in range(1,tot):

        dtset_mandailing.append(i)

for i in Pakpak:

    for a in range(1,tot):

        dtset_pakpak.append(i)

for i in Simalungun:

    for a in range(1,tot):

        dtset_simalungun.append(i)

for i in Toba:

    for a in range(1,tot):

        dtset_toba.append(i)
def count_kode_arah(image_label):

    P = image_label.shape[0]

    Q = image_label.shape[1]

    inp_new = 0*image_label

    K = [0,0,0,0,0,0,0,0]

    for p in range(1,P-1):

        for q in range(1, Q-1):

            if image_label[p,q] == 2:

                K[0] = K[0] + 1

                K[4] = K[4] + image_label[p,q]

            if image_label[p,q] == 3:

                K[1] = K[1] + 1

                K[5] = K[5] + image_label[p,q]

            if image_label[p,q] == 4:

                K[2] = K[2] + 1

                K[6] = K[6] + image_label[p,q]

            if image_label[p,q] == 5:

                K[3] = K[3] + 1

                K[7] = K[7] + image_label[p,q]

    K[0] = K[0]/P

    K[1] = K[1]/P

    K[2] = K[2]/P

    K[3] = K[3]/P

    K[4] = K[4]/P

    K[5] = K[5]/P

    K[6] = K[6]/P

    K[7] = K[7]/P

    return K
c_kode_arah_karo = []

c_kode_arah_mandailing = []

c_kode_arah_pakpak = []

c_kode_arah_simalungun = []

c_kode_arah_toba = []

for i in Karo:

    for a in range(1,tot):

        c_kode_arah_karo.append(count_kode_arah(karo_arah[i][str(a)]))

for i in Mandailing:

    for a in range(1,tot):

        c_kode_arah_mandailing.append(count_kode_arah(mandailing_arah[i][str(a)]))

for i in Pakpak:

    for a in range(1,tot):

        c_kode_arah_pakpak.append(count_kode_arah(pakpak_arah[i][str(a)]))

for i in Simalungun:

    for a in range(1,tot):

        c_kode_arah_simalungun.append(count_kode_arah(simalungun_arah[i][str(a)]))

for i in Toba:

    for a in range(1,tot):

        c_kode_arah_toba.append(count_kode_arah(toba_arah[i][str(a)]))

import pandas as pd

lihat_karo = np.column_stack((c_kode_arah_karo, dtset_karo))

df2 = pd.DataFrame(lihat_karo,columns=['K[0]', 'K[1]', 'K[2]','K[3]','K[4]','K[5]','K[6]','K[7]','Label'])

df2
from sklearn.neural_network import MLPClassifier

clf_karo = MLPClassifier(alpha=0.000000001,hidden_layer_sizes=(100,100,100), max_iter=10000,random_state=0) 

clf_mandailing = MLPClassifier(alpha=0.000000001,hidden_layer_sizes=(100,100,100), max_iter=10000,random_state=0)

clf_pakpak = MLPClassifier(alpha=0.0000000001,hidden_layer_sizes=(100,100,100), max_iter=10000,random_state=0)

clf_simalungun = MLPClassifier(alpha=0.00000001,hidden_layer_sizes=(100,100,100), max_iter=10000,random_state=13423)

clf_toba = MLPClassifier(alpha=0.000000001,hidden_layer_sizes=(100,100,100), max_iter=10000,random_state=0)

clf_karo.fit(c_kode_arah_karo,dtset_karo)
clf_mandailing.fit(c_kode_arah_mandailing,dtset_mandailing)
clf_pakpak.fit(c_kode_arah_pakpak,dtset_pakpak)
clf_simalungun.fit(c_kode_arah_simalungun,dtset_simalungun)
clf_toba.fit(c_kode_arah_toba,dtset_toba)
dtest_karo = []

dtest_mandailing = []

dtest_pakpak = []

dtest_simalungun = []

dtest_toba = []

tot = 11

for i in Karo:

    for a in range(8,tot):

        dtest_karo.append(i)

for i in Mandailing:

    for a in range(8,tot):

        dtest_mandailing.append(i)

for i in Pakpak:

    for a in range(8,tot):

        dtest_pakpak.append(i)

for i in Simalungun:

    for a in range(8,tot):

        dtest_simalungun.append(i)

for i in Toba:

    for a in range(8,tot):

        dtest_toba.append(i)
c_kode_arah_karo_t = []

c_kode_arah_mandailing_t = []

c_kode_arah_pakpak_t = []

c_kode_arah_simalungun_t = []

c_kode_arah_toba_t = []

for i in Karo:

    for a in range(8,tot):

        c_kode_arah_karo_t.append(count_kode_arah(karo_arah[i][str(a)]))

for i in Mandailing:

    for a in range(8,tot):

        c_kode_arah_mandailing_t.append(count_kode_arah(mandailing_arah[i][str(a)]))

for i in Pakpak:

    for a in range(8,tot):

        c_kode_arah_pakpak_t.append(count_kode_arah(pakpak_arah[i][str(a)]))

for i in Simalungun:

    for a in range(8,tot):

        c_kode_arah_simalungun_t.append(count_kode_arah(simalungun_arah[i][str(a)]))

for i in Toba:

    for a in range(8,tot):

        c_kode_arah_toba_t.append(count_kode_arah(toba_arah[i][str(a)]))
print('Akurasi untuk huruf Karo')

print('ACCURACY USING DATA TRAINING : ' + str(clf_karo.score(c_kode_arah_karo,dtset_karo))) # ACCURACY USING DATA TRAINING

print('ACCURACY USING DATA TESTING  : ' + str(clf_karo.score(c_kode_arah_karo_t,dtest_karo))) #ACCURACY USING DATA TESTING

print()

print('Akurasi untuk huruf Mandailing')

print('ACCURACY USING DATA TRAINING : ' + str(clf_mandailing.score(c_kode_arah_mandailing,dtset_mandailing))) # ACCURACY USING DATA TRAINING

print('ACCURACY USING DATA TESTING  : ' + str(clf_mandailing.score(c_kode_arah_mandailing_t,dtest_mandailing))) #ACCURACY USING DATA TESTING

print()

print('Akurasi untuk huruf Pakpak')

print('ACCURACY USING DATA TRAINING : ' + str(clf_pakpak.score(c_kode_arah_pakpak,dtset_pakpak))) # ACCURACY USING DATA TRAINING

print('ACCURACY USING DATA TESTING  : ' + str(clf_pakpak.score(c_kode_arah_pakpak_t,dtest_pakpak))) #ACCURACY USING DATA TESTING

print()

print('Akurasi untuk huruf Simalungun')

print('ACCURACY USING DATA TRAINING : ' + str(clf_simalungun.score(c_kode_arah_simalungun,dtset_simalungun))) # ACCURACY USING DATA TRAINING

print('ACCURACY USING DATA TESTING  : ' + str(clf_simalungun.score(c_kode_arah_simalungun_t,dtest_simalungun))) #ACCURACY USING DATA TESTING

print()

print('Akurasi untuk huruf Toba')

print('ACCURACY USING DATA TRAINING : ' + str(clf_toba.score(c_kode_arah_toba,dtset_toba))) # ACCURACY USING DATA TRAINING

print('ACCURACY USING DATA TESTING  : ' + str(clf_toba.score(c_kode_arah_toba_t,dtest_toba))) #ACCURACY USING DATA TESTING
clf_karo.predict([[0.166666666666666,0.16666666666666666,1.1333333333333333,0.16666666666666666,0.166666666666666,0.16666666666666666,1.1333333333333333,0.16666666666666666]])
import os

print(os.listdir("../input/suratbatakwiki/surat batak/Karo"))
alamat = '../input/suratbatak/surat batak/Karo Final/A/7.png'

img = cv2.imread(alamat)

gambar = preprocessing(img)

dfe = make_a(gambar)

kode_arah = count_kode_arah(dfe)

clf_karo.predict([kode_arah])