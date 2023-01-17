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


import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from matplotlib import style



import os

print(os.listdir('../input/'))
# reading the data

data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')



# getting the shape of the data

print(data.shape)
data.head()
data.describe()
# checking if the dataset contains any null values

data.isnull().sum().sum()
plt.rcParams['figure.figsize'] = (15, 5)

plt.style.use('_classic_test')

sns.countplot(data['gender'], palette = 'bone')

plt.title('Perbandingan Laki-laki dan Perempuan', fontweight = 30)

plt.xlabel('Gender')

plt.ylabel('Count')

plt.show()
plt.rcParams['figure.figsize'] = (15,9)

plt.style.use('ggplot')



sns.countplot(data['race/ethnicity'], palette = 'pink')

plt.title('Perbandingan Berbagai Kelompok', fontweight = 30, fontsize = 20)

plt.xlabel('Groub')

plt.ylabel('count')

plt.show()
plt.rcParams['figure.figsize'] = (15 ,9)

plt.style.use('fivethirtyeight')



sns.countplot(data['parental level of education'], palette= 'Blues')

plt.title('Perbandingan Pendidikan Orang Tua',fontweight = 30, fontsize = 20)

plt.xlabel('Degree')

plt.ylabel('count')

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

plt.style.use('seaborn-talk')



sns.countplot(data['lunch'], palette = 'PuBu')

plt.title('Perbandingan berbagai jenis makan siang ', fontweight = 30, fontsize = 20)

plt.xlabel('types of lunch')

plt.ylabel('count')

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

plt.style.use('tableau-colorblind10')



sns.countplot(data['math score'], palette = 'BuPu')

plt.title('Perbandingan nilai matematika', fontweight = 30, fontsize = 20)

plt.xlabel('score')

plt.ylabel('count')

plt.xticks(rotation = 90)

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

plt.style.use('tableau-colorblind10')



sns.countplot(data['reading score'], palette = 'RdPu')

plt.title('Perbandingan skor Membaca', fontweight = 30, fontsize = 20)

plt.xlabel('score')

plt.ylabel('count')

plt.xticks(rotation = 90)

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

plt.style.use('tableau-colorblind10')



sns.countplot(data['writing score'], palette = 'prism')

plt.title('Perbandingan skor menulis', fontweight = 30, fontsize = 20)

plt.xlabel('score')

plt.ylabel('count')

plt.xticks(rotation = 90)

plt.show()


# Fitur rekayasa pada data untuk memvisualisasikan dan menyelesaikan dataset lebih akurat



# Menetapkan tanda kelulusan bagi siswa untuk lulus pada tiga mata pelajaran secara individual

passmarks = 40



# membuat kolom pass_math baru, kolom ini akan memberi tahu kita apakah siswa lulus atau gagal

data['pass_math'] = np.where(data['math score']< passmarks, 'Gagal', 'Lulus')

data['pass_math'].value_counts().plot.pie(colors = ['orange', 'red'])



plt.title('Lulus / Gagal dalam Matematika', fontweight = 30, fontsize = 20)

plt.xlabel('status')

plt.ylabel('count')

plt.show()
# membuat kolom pass_reading baru, kolom ini akan memberi tahu kita apakah siswa lulus atau gagal

data['pass_reading'] = np.where(data['reading score']< passmarks, 'Gagal', 'Lulus')

data['pass_reading'].value_counts(dropna = False).plot.pie(colors = ['grey', 'black'])



plt.title('Gagal/Lulus dalam Membaca', fontweight = 30, fontsize = 20)

plt.xlabel('status')

plt.ylabel('count')

plt.show()
data['pass_writing'] = np.where(data['writing score']< passmarks, 'Gagal', 'Lusus')

data['pass_writing'].value_counts(dropna = False).plot.pie(colors = ['lightblue', 'lightgreen'])



plt.title('Luslus/Gagal dalam Menulis', fontweight = 30, fontsize = 20)

plt.xlabel('status')

plt.ylabel('count')

plt.show()