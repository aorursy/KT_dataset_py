import numpy as np
arr = np.zeros(12, dtype=int)
arr = np.insert(arr, range(0,13,3), range(1,6))
arr
arr = np.zeros(8*8, dtype=int).reshape(8,8)
arr[1::2,::2] = 1 #linha : start = 1, stop = fim, step 2 / coluna : start = inicio, stop = fim, step 2
arr[::2,1::2] = 1 #linha : start = 0, stop = fim, step 2 / coluna : start = 1, stop = fim, step 2
arr
arr = np.array(range(0,25)).reshape(5,5)
arr[[0,1]] = arr[[1,0]]
arr
arr = np.array(range(0,25)).reshape(5,5)
arr[[2,3]] = arr[[3,2]]
arr
arr = np.array(range(0,25)).reshape(5,5)
arr[:,[2,3]] = arr[:,[3,2]]
arr
ini = np.array([1,2,3])
arr = np.array([ini]*4)
arr
ini = np.array([1,2,3])
arr = np.array([ini]*3).transpose()
arr
arr = np.array([[4, 95, 37, 64, 42, 19, 55, 38, 46, 83, 48, 67, 98, 21, 10, 88]])
arr[np.where(np.logical_and(arr >= 35, arr <= 55))]

arr = np.arange(0,101)
arr
f = np.random.uniform(0,100)
f
np.abs(arr - f).argmin()

np.random.seed(100)
arr = np.random.uniform(1, 50, 30)
arr
arr[np.where(arr < 10,)] = 10
arr[np.where(arr > 30,)] = 30
arr
def calc_determinante(array):
    
    #adiciona as 2 primeiras colunas no final
    c = array[:,0:2]      
    array = np.append(array, c , axis=1) 

    #diagonais aei , bfg , cdh 
    d1 = array.diagonal()
    d2 = array.diagonal(1,0)
    d3 = array.diagonal(2,0)

    #antidiagonais afh , bdi, ceg
    ad1 = np.fliplr(array).diagonal()  
    ad2 = np.fliplr(array).diagonal(1,0)  
    ad3 = np.fliplr(array).diagonal(2,0)  

    #formula
    det = (np.prod(d1) + np.prod(d2) + np.prod(d3)) - (np.prod(ad1) + np.prod(ad2) + np.prod(ad3))
    
    return det   
arr = np.array(range(1,10)).reshape(3,3)
calc_determinante(arr)