# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image

Image(filename='../input/mc-regular-ising-supplementary/Pseudocode.PNG')
# 1. Generating initial configurations, with the amount element and box length



def initgrid(box,ratio):

    matrix = np.random.randint(2, size=(box, box))      

    sum = np.sum(matrix)

    if sum == box**2*ratio:

        return matrix

    else:

        return initgrid(box,ratio)
from PIL import Image



def display_regular_field(field):

    return Image.fromarray(np.uint8(field * 255)) #
# swap: 1st debug on detecting element **location to swap**, initial **value, local energy, microstate**

def debug_1():

    print('Element 1:', (rdm1,rdm2),'Element 2:',(rdm3,rdm4))

    print('Initial Value 1:', matrix_1[rdm1,rdm2],'Initial Value 2:',matrix_1[rdm3,rdm4])

    print('Initial 1:', local_energy(matrix_1,rdm1,rdm2),'Element 2:',local_energy(matrix_1,rdm3,rdm4))

    print(matrix_1)

    print('-------------------------')

    

# swap: 2nd debug on detecting element final **microstate, value, local energy**

def debug_2():

    print(matrix_1)

    print('Final Value 1:', matrix_1[rdm1,rdm2],'Final Value 2:',matrix_1[rdm3,rdm4])            

    print('Final 1:', local_energy(matrix_1,rdm1,rdm2),'Final 2:',local_energy(matrix_1,rdm3,rdm4))

box,ratio = 24,0.5

def next_(i):

    if i < box-1:

        return i+1

    else:

        return 0



def prec(i):

    if i >= 1:

        return i-1

    else:

        return box-1
from ipywidgets import interact



def display_regular_sequence(images):

    def _show(frame=(0, len(images) - 1)):

        return display_regular_field(images[frame])

    return interact(_show)



def display_ising_sequence(images):

    def _show(frame=(0, len(images) - 1)):

        return display_spin_field(images[frame])

    return interact(_show)
import random

def swap_(matrix_1,beta=0.5):

    #Creating 2 random row and column for swapping

    rdm1 = random.randint(0,box)

    rdm2 = random.randint(0,box)

    rdm3 = random.randint(0,box)

    rdm4 = random.randint(0,box)

  

    #Creating 2 random row and column for swapping  

    if matrix_1[rdm1,rdm2] != matrix_1[rdm3,rdm4]:

        total = 0

        

        Ei_1 = abs(4*matrix_1[rdm1,rdm2] - (matrix_1[next_(rdm1),rdm2] + matrix_1[prec(rdm1),rdm2] + matrix_1[rdm1,prec(rdm2)] + matrix_1[rdm1,next_(rdm2)]))

        Ei_2 = abs(4*matrix_1[rdm3,rdm4] - (matrix_1[next_(rdm3),rdm4] + matrix_1[prec(rdm3),rdm4] + matrix_1[rdm3,prec(rdm4)] + matrix_1[rdm3,next_(rdm4)]))

        Ei = 2 * (Ei_1+Ei_2)

        

        Ef_1 = abs(4*matrix_1[rdm3,rdm4] - (matrix_1[next_(rdm1),rdm2] + matrix_1[prec(rdm1),rdm2] + matrix_1[rdm1,prec(rdm2)] + matrix_1[rdm1,next_(rdm2)]))

        Ef_2 = abs(4*matrix_1[rdm1,rdm2] - (matrix_1[next_(rdm3),rdm4] + matrix_1[prec(rdm3),rdm4] + matrix_1[rdm3,prec(rdm4)] + matrix_1[rdm3,next_(rdm4)]))

        Ef = 2 * (Ef_1+Ef_2)

        

        dE = Ef-Ei

        

        if dE <= 0 or np.exp(-dE * beta) > np.random.rand():

            #print(rdm1,rdm2)

            #print(rdm3,rdm4)

            

            store = matrix_1[rdm3,rdm4]

            matrix_1[rdm3,rdm4] = matrix_1[rdm1,rdm2]

            matrix_1[rdm1,rdm2] = store



            return matrix_1

        

        else:

            return swap_(matrix_1,beta=0.5)

    else:

        return swap_(matrix_1,beta=0.5)
def random_regular_field(N, M):

    return np.random.choice([0, 1], size=(N, M))



from tqdm import tqdm

regular_images = [random_regular_field(box+1, box+1)]

for i in tqdm(range(1000**2)):

    regular_images.append(swap_(regular_images[-1].copy()))
display_regular_sequence(regular_images);
display_regular_field(regular_images[0]).resize(size=(250, 250))
display_regular_field(regular_images[-1]).resize(size=(250, 250))
def swap_general(matrix_1,beta=0.5):

    #Creating 2 random row and column for swapping

    rdm1 = random.randint(0,box)

    rdm2 = random.randint(0,box)

    rdm3 = random.randint(0,box)

    rdm4 = random.randint(0,box)

  

    #Creating 2 random row and column for swapping  

    if matrix_1[rdm1,rdm2] != matrix_1[rdm3,rdm4]:

        total = 0

        

        Ei_1 = abs(4*matrix_1[rdm1,rdm2] - (matrix_1[next_(rdm1),rdm2] + matrix_1[prec(rdm1),rdm2] + matrix_1[rdm1,prec(rdm2)] + matrix_1[rdm1,next_(rdm2)]))

        Ei_2 = abs(4*matrix_1[rdm3,rdm4] - (matrix_1[next_(rdm3),rdm4] + matrix_1[prec(rdm3),rdm4] + matrix_1[rdm3,prec(rdm4)] + matrix_1[rdm3,next_(rdm4)]))

        Ei = 2 * (Ei_1+Ei_2)

        

        Ef_1 = abs(4*matrix_1[rdm3,rdm4] - (matrix_1[next_(rdm1),rdm2] + matrix_1[prec(rdm1),rdm2] + matrix_1[rdm1,prec(rdm2)] + matrix_1[rdm1,next_(rdm2)]))

        Ef_2 = abs(4*matrix_1[rdm1,rdm2] - (matrix_1[next_(rdm3),rdm4] + matrix_1[prec(rdm3),rdm4] + matrix_1[rdm3,prec(rdm4)] + matrix_1[rdm3,next_(rdm4)]))

        Ef = 2 * (Ef_1+Ef_2)

        

        dE = Ef-Ei

        

        if dE <= 0 or np.exp(-dE * beta) > np.random.rand():

            #print(rdm1,rdm2)

            #print(rdm3,rdm4)

            

            store = matrix_1[rdm3,rdm4]

            matrix_1[rdm3,rdm4] = matrix_1[rdm1,rdm2]

            matrix_1[rdm1,rdm2] = store



            return matrix_1

        

        else:

            return matrix_1

    else:

        return matrix_1
from tqdm import tqdm

regular_images_general = [random_regular_field(box+1, box+1)]

for i in tqdm(range(1000**2)):

    regular_images.append(swap_general(regular_images_general[-1].copy()))
display_regular_sequence(regular_images_general);
display_regular_field(regular_images_general[-1]).resize(size=(250, 250))
from IPython.display import Image

Image(filename='../input/mc-regular-ising-supplementary/Regular Part 1.PNG')
Image(filename='../input/mc-regular-ising-supplementary/Regular Part 2.PNG')
from IPython.display import YouTubeVideo

YouTubeVideo('Wy9YoEYv-fA', width=600, height=350)
# 1. Generating initial configurations

def random_spin_field(N, M):

    return np.random.choice([-1/2, 1/2], size=(N, M))



initial_ising = random_spin_field(10, 10)

initial_ising
# Supplementary 1

from PIL import Image



def display_spin_field(field):

    return Image.fromarray(np.uint8((field*2 + 1) * 0.5 * 255))  # 0 ... 255



display_spin_field(random_spin_field(10, 10))
# 2. Creating initial steps of the Ising Model

def ising_step(field, beta=2):

    N, M = field.shape

    for n_offset in range(2):

        for m_offset in range(2):

            for n in range(n_offset, N, 2):

                for m in range(m_offset, M, 2):

                    _ising_update(field, n, m, beta)

    return field



# 2. Creating initial steps of the Ising Model

def _ising_update(field, n, m, beta):

    total = 0

    N, M = field.shape

    for i in range(n-1, n+2):

        for j in range(m-1, m+2):

            if i == n and j == m:

                continue

            total += field[i % N, j % M]

    dE = 2 * field[n, m] * total

    if dE <= 0:

        field[n, m] *= -1

    elif np.exp(-dE * beta) > np.random.rand():

        field[n, m] *= -1
display_spin_field(ising_step(random_spin_field(200, 200)))
from tqdm import tqdm



images = [random_spin_field(200, 200)]

for i in tqdm(range(1000)):

    images.append(ising_step(images[-1].copy()))
from PIL import Image

display_ising_sequence(images);
from IPython.display import Image

Image(filename='../input/mc-regular-ising-supplementary/Ising.PNG')