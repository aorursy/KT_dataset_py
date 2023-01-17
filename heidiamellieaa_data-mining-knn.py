

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        



# Any results you write to the current directory are saved as output.
        print(os.listdir("../input")) #menampilkan direktori yang ada di direktori input
# show the first food in dataset

from skimage import io #import image, dengan library skimage

food0 = io.imread('../input/food5k-image-dataset/training/food/1486.jpg') #bikin objek baru namanya food0, imread buat membaca lokasi gmbarnya

io.imshow(food0) #menampilkan gambar

print(food0.shape) #print size dari gambar tsb
from skimage.transform import resize #ambil syntax resize dari library skimage.transform

food0_ = resize(food0, (200,200,3)) #resize si food0 jadi ukuran 200 x 200

from skimage.color import rgb2gray # ambil syntax rgb2gray dari library skimage.transform (krn lebih mudah diproses jika gambar berupa grayscale)

food0_gs = rgb2gray(food0_)



import matplotlib.pyplot as plt #matplottlib(matlab like ploting framework) untuk mngkombinasikan pyplot(python ploting) sm matplotlib dan diringkas namanya menjadi plt

fig=plt.figure() #untuk menunjukkan objek fig adalah objek yang berupa gambar

columns = 2

rows = 1

fig.add_subplot(rows, columns, 1) #menunjukkan posisi (baris, kolom)

plt.imshow(food0_)

fig.add_subplot(rows, columns, 2)

plt.imshow(food0_gs) #yg udh diproses sm matlab

plt.show()
# Get the first 100 pixels of food and nonfood to train

x = [] #bikin array kosong x dan y, buat ngeprint ukuran dari gambar nya

y = []

for i in range(1,101): #untuk looping ukuran 100 pixel di sumbu x dan y

    food = rgb2gray(resize(io.imread('../input/food5k-image-dataset/training/food/{}.jpg'.format(i)), (200,200)))

    x.append(food) #append adl syntax untuk menambahkan data kedalam array

    y.append(0)



    #{} = all

    

for i in range(1,101):

    non_food = rgb2gray(resize(io.imread('../input/food5k-image-dataset/training/non_food/{}.jpg'.format(i)), (200,200)))

    x.append(non_food)

    y.append(1)



x, y = np.asarray(x), np.asarray(y)

print('x shape: ', x.shape, 'y shape: ', y.shape)

#gambar itu array, krna komputer bacanya pixel

fig=plt.figure()

columns = 2

rows = 1

fig.add_subplot(rows, columns, 1)

plt.imshow(food)

fig.add_subplot(rows, columns, 2)

plt.imshow(non_food)
class NearestNeighbor(object): #petunjuk untuk prediksi menggunakan metode KNN

    def __init__(self): #inisialisasi bhw  kelas ini bs memakai nilainya sendiri

        pass



    def fit(self, X, y): #ngepasin nilai x dan y jadi nilainya neighbor, knn baca data gabungan food sm non food yg di x dan y

        """ X is N x D where each row is an example. Y is 1-dimension of size N """

        # the nearest neighbor classifier simply remembers all the training data

        self.Xtr = X #self.Xtr = buat ambil data x training, karena si x berasal dari kelas lain, maka self berfungsi untuk membuat nilai x jadi kelasnya neighbor biar bisa dipake sama neighbor

        self.ytr = y



    def predict(self, X, k):

        # find the nearest training image to the i'th test image

        # using the L1 distance (sum of absolute value differences)

        distances = [np.sum(np.abs(self.Xtr[i] - X)) for i in range(0, len(self.Xtr))] #np.abs (numpy absolute) untuk hitung nilai absolut berdasarkan elemen (elemen yng dipakai disini itu self.Xtr)

        #print(distances)

        min_indexs = np.argsort(distances)[:k] #mensortir/menyaring dari distance spy fokus ke objek (ex: food sj tnpa bckground)

        #print(min_indexs)

        y_ = self.ytr[min_indexs] #sumbu yg digunakan untuk mengkalkulasi self.ytr dg min_indexs

        #print(y_)

        counts = np.bincount(y_)

        #print(np.argmax(counts))

        return np.argmax(counts) #nilai yg didapat dr counts nnti dipakai oleh np.argmax, np.argmax adalah untuk memilih kemungkinan terbesar apakah food atau non food
knn = NearestNeighbor()

knn.fit(x,y)

results_100 = []

for k in (1,2,3,5,10):

    print('k = ',k)

    for i in range(1,101):

        results_100.append(knn.predict(rgb2gray(resize(io.imread('../input/food5k-image-dataset/training/non_food/{}.jpg'.format(i)), (200,200))),5))

    unique, counts = np.unique(results_100, return_counts=True)

    print(dict(zip(unique, counts)))