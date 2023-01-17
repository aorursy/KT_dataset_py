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
def initgrid():

    matrix = np.random.randint(2, size=(10, 10))                                  #creating a 10 by 10 matrix abd filling it with 

    sum = np.sum(matrix)                                                          #sum up all values in matrix

    if sum == 50:                                                                 #making sure that the original matrix has 50 oil, 50 water

        return matrix

    else:

        return initgrid()                                                           #else rerun function

dummy_matrix = initgrid()

dummy_matrix
matrix_initial = np.zeros((12, 12))

matrix_initial
np.arange(100).reshape((10, 10))
L = 10

#next

for i in range(L):

    if i < L-1:

        print(i+1)

    else:

        print(0)

print('------boundary--------')

#prec

for i in range(L):

    if i >= 1:

        print(i-1)

    else:

        print(0)
#next

for i in range(L):

    if i < L-1:

        print(i+1)

    else:

        print(0)

print('------boundary--------')

#prec

for i in range(L):

    if i >= 1:

        print(i-1)

    else:

        print(0)
def initgrid(box,ratio):

    matrix = np.random.randint(2, size=(box, box))                                  #creating a 10 by 10 matrix abd filling it with 

    sum = np.sum(matrix)                                                          #sum up all values in matrix

    if sum == box**2*ratio:                                                                 #making sure that the original matrix has 50 oil, 50 water

        return matrix

    else:

        return initgrid(box,ratio)

    

box,ratio = 10,0.5

    

dummy_matrix = initgrid(box,ratio)

dummy_matrix
def next_(i):

    if i < box-1:

        return i+1

    else:

        return 0



def prec(i):

    if i >= 1:

        return i-1

    else:

        return 9
#Energy Interaction

for i in range(box):

    print('--------------------')

    for j in range(box):

        print("Start",(i,j),":Right:",(next_(i),j), ',Left:',(prec(i),j), ',Up:',(i,prec(j)), ',Down:', (i,next_(j)))
#Density: Diagonal Interaction

for i in range(box):

    print('--------------------')

    for j in range(box):

        print("Start",(i,j),":Left-Down:",(prec(i),next_(j)), ',Left-Up:',(prec(i),prec(j)), ',Right-Up:',(next_(i),prec(j)), ',Right-Down:', (next_(i),next_(j)))
import random



for i in range(100):

    counter = random.randint(0,1)

    dummy_list = [random.randint(0,1) for i in range(4)]

    

    print(counter, dummy_list, abs(4*counter - sum(dummy_list)))
import random

from collections import Counter



def state_to_density(dummy_matrix):

    density_state = []

    for i in range(box):

        for j in range(box):

            density = dummy_matrix[i,j] + dummy_matrix[next_(i),j] + dummy_matrix[prec(i),j] + dummy_matrix[i,prec(j)] + dummy_matrix[i,next_(j)] + dummy_matrix[prec(i),next_(j)] + dummy_matrix[prec(i),prec(j)] + dummy_matrix[next_(i),prec(j)] + dummy_matrix[next_(i),next_(j)]

            density_state.append(density)

    return pd.DataFrame.from_dict(Counter(density_state), orient='index')
def state_to_energy(dummy_matrix):

    energy_state = np.zeros((10,10))

    for i in range(box):

        for j in range(box):

            energy_state[i,j] = abs(4*dummy_matrix[i,j] - (dummy_matrix[next_(i),j] + dummy_matrix[prec(i),j] + dummy_matrix[i,prec(j)] + dummy_matrix[i,next_(j)])) 

    return energy_state
dummy_matrix
energy_state = state_to_energy(dummy_matrix)

energy_state
def local_energy(dummy_matrix,i,j):

    return abs(4*dummy_matrix[i,j] - (dummy_matrix[next_(i),j] + dummy_matrix[prec(i),j] + dummy_matrix[i,prec(j)] + dummy_matrix[i,next_(j)]))



#Put Boltzmann Criteria



def swap(matrix_1):

    #Creating 2 random row and column for swapping

    rdm1 = random.randint(0,9)

    rdm2 = random.randint(0,9)

    rdm3 = random.randint(0,9)

    rdm4 = random.randint(0,9)

  

    #Creating 2 random row and column for swapping  

    if matrix_1[rdm1,rdm2] != matrix_1[rdm3,rdm4]:

        

        def debug_1():

            print('Element 1:', (rdm1,rdm2),'Element 2:',(rdm3,rdm4))

            print('Initial Value 1:', matrix_1[rdm1,rdm2],'Initial Value 2:',matrix_1[rdm3,rdm4])

            print('Initial 1:', local_energy(matrix_1,rdm1,rdm2),'Element 2:',local_energy(matrix_1,rdm3,rdm4))

            print(matrix_1)

            print('-------------------------')

        #debug_1()

        start = local_energy(matrix_1,rdm1,rdm2) + local_energy(matrix_1,rdm3,rdm4)

        

        store = matrix_1[rdm3,rdm4]

        matrix_1[rdm3,rdm4] = matrix_1[rdm1,rdm2]

        matrix_1[rdm1,rdm2] = store

        

        def debug_2():

            print(matrix_1)

            print('Final Value 1:', matrix_1[rdm1,rdm2],'Final Value 2:',matrix_1[rdm3,rdm4])            

            print('Final 1:', local_energy(matrix_1,rdm1,rdm2),'Final 2:',local_energy(matrix_1,rdm3,rdm4))

        

        end = local_energy(matrix_1,rdm1,rdm2) + local_energy(matrix_1,rdm3,rdm4)

        #debug_2()    

        

        local_dif = 2*(end-start)

        return local_dif, matrix_1

  

    #Creating 2 random row and column for swapping

    else:

        return swap(matrix_1)

    

dummy_new_state = swap(dummy_matrix)
dummy_new_state
energy_new_state = state_to_energy(dummy_new_state[1])

energy_new_state
energy_state.sum(), energy_new_state.sum()
counter = 0

while counter < 1000:

    counter += 100    

    print(counter)
energy_diff = []

density_pandas = []



from tqdm import tqdm

for i in tqdm(range(1000)):

    dummy_new_state = swap(dummy_new_state[1])

    

    energy_diff.append(dummy_new_state[0]+energy_state.sum())

    density_pandas.append(state_to_density(dummy_new_state[1]))

density_table = pd.concat(density_pandas,axis=1)

density_table.columns = [i+1 for i in range(1000)]

density_table = density_table.transpose()

density_table
import matplotlib.pyplot as plt

nrow,ncol=2,5

size = 4



fig, axes = plt.subplots(nrow, ncol,figsize=(ncol*size,nrow*size))



count=0

for r in range(nrow):

    for c in range(ncol):

        density_table[count].plot(ax=axes[r,c])

        axes[r,c].title.set_text('Density: '+str(count))

        count += 1
df = pd.DataFrame(np.asarray(energy_diff), columns=['Col1'])

df.boxplot(column=['Col1'])
dummy_matrix = initgrid(box,ratio)-1/2

dummy_matrix
J, h = 1, 1

def local_hamiltonian(dummy_matrix,i,j):

    sum_spin = dummy_matrix[i,j] + dummy_matrix[next_(i),j] + dummy_matrix[prec(i),j] + dummy_matrix[i,prec(j)] + dummy_matrix[i,next_(j)]

    spin_interaction = dummy_matrix[i,j]*(dummy_matrix[next_(i),j] + dummy_matrix[prec(i),j] + dummy_matrix[i,prec(j)] + dummy_matrix[i,next_(j)])

    return -J*spin_interaction + h*sum_spin

from IPython.display import YouTubeVideo

YouTubeVideo('rN7g4gzO2sk', width=600, height=350)
#Creating random model of spin-up and spin-down
import numpy as np



def random_spin_field(N, M):

    return np.random.choice([-1/2, 1/2], size=(N, M))



random_spin_field(10, 10)
#Create image visualization by converting (-1/2,1/2) to (0,255), (-0.5+1.5=)
# pip install pillow

from PIL import Image



def display_spin_field(field):

    return Image.fromarray(np.uint8((field*2 + 1) * 0.5 * 255))  # 0 ... 255



display_spin_field(random_spin_field(10, 10))
def ising_step(field, beta=0.1):

    N, M = field.shape

    for n_offset in range(2):

        for m_offset in range(2):

            for n in range(n_offset, N, 2):

                for m in range(m_offset, M, 2):

                    _ising_update(field, n, m, beta)

    return field



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
from ipywidgets import interact

from tqdm import tqdm



def display_ising_sequence(images):

    def _show(frame=(0, len(images) - 1)):

        return display_spin_field(images[frame])

    return interact(_show)



images = [random_spin_field(200, 200)]

for i in tqdm(range(1000)):

    images.append(ising_step(images[-1].copy()))

display_ising_sequence(images);
def initgrid(box,ratio):

    matrix = np.random.randint(2, size=(box, box))                                  #creating a 10 by 10 matrix abd filling it with 

    sum = np.sum(matrix)                                                          #sum up all values in matrix

    if sum == box**2*ratio:                                                                 #making sure that the original matrix has 50 oil, 50 water

        return matrix

    else:

        return initgrid(box,ratio)

    

box,ratio = 200,0.5

    

dummy_matrix = initgrid(box,ratio)

dummy_matrix
def display_regular_field(field):

    return Image.fromarray(np.uint8(field * 255))  # 0 ... 255



display_regular_field(dummy_matrix)
def display_regular_sequence(images):

    def _show(frame=(0, len(images) - 1)):

        return display_regular_field(images[frame])

    return interact(_show)



images = [dummy_matrix]

for i in tqdm(range(1000)):

    images.append(ising_step(images[-1].copy()))

display_regular_sequence(images);
def ising_step(field, beta=0.1):

    N, M = field.shape

    for n_offset in range(2):

        for m_offset in range(2):

            for n in range(n_offset, N, 2):

                for m in range(m_offset, M, 2):

                    _ising_update(field, n, m, beta)

    return field



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
def random_spin_field(N, M):

    return np.random.choice([0, 1], size=(N, M))

initial_regular = random_spin_field(10, 10)

initial_regular
from PIL import Image



def display_spin_field(field):

    return Image.fromarray(np.uint8(field * 255))  # 0 ... 255



display_spin_field(random_spin_field(400, 400))
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
initial_regular
import random

def swap(matrix_1):

    #Creating 2 random row and column for swapping

    rdm1 = random.randint(0,9)

    rdm2 = random.randint(0,9)

    rdm3 = random.randint(0,9)

    rdm4 = random.randint(0,9)

  

    #Creating 2 random row and column for swapping  

    if matrix_1[rdm1,rdm2] != matrix_1[rdm3,rdm4]:

        

        print(rdm1,rdm2)

        print(rdm3,rdm4)

        # Getting initial local energy

        # start = local_energy(matrix_1,rdm1,rdm2) + local_energy(matrix_1,rdm3,rdm4)

        

        #debug_1()

        

        # Swapping positions

        store = matrix_1[rdm3,rdm4]

        matrix_1[rdm3,rdm4] = matrix_1[rdm1,rdm2]

        matrix_1[rdm1,rdm2] = store

        # Getting final local energy

        # end = local_energy(matrix_1,rdm1,rdm2) + local_energy(matrix_1,rdm3,rdm4)

        #debug_2()    

        # local_dif = 2*(end-start)

        

        

        return matrix_1 #, local_dif

  

    #Creating 2 random row and column for swapping

    else:

        return swap(matrix_1)
start = initial_regular

start
swap(initial_regular)
swap(initial_regular)
end = initial_regular

end
start == end
box = 200

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
def swap_(matrix_1,beta=0.5):

    #Creating 2 random row and column for swapping

    rdm1 = random.randint(0,9)

    rdm2 = random.randint(0,9)

    rdm3 = random.randint(0,9)

    rdm4 = random.randint(0,9)

  

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
start = initial_regular

start
swap_(initial_regular)
swap_(initial_regular)
end = initial_regular

end
start == end
from ipywidgets import interact



def display_ising_sequence(images):

    def _show(frame=(0, len(images) - 1)):

        return display_spin_field(images[frame])

    return interact(_show)
from tqdm import tqdm



images = [random_spin_field(200, 200)]

for i in tqdm(range(1000)):

    images.append(swap_(images[-1].copy()))

display_ising_sequence(images);
from tqdm import tqdm

for i in tqdm(range(1000)):

    dummy_new_state = swap(dummy_new_state[1])

    

    energy_diff.append(dummy_new_state[0]+energy_state.sum())

    density_pandas.append(state_to_density(dummy_new_state[1]))

density_table = pd.concat(density_pandas,axis=1)

density_table.columns = [i+1 for i in range(1000)]

density_table = density_table.transpose()

density_table