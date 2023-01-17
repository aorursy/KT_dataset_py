# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:53:09 2020

@author: Tarro
"""
from scipy.stats import hypergeom
import numpy as np
import seaborn as sns; sns.set()
import time
start = time.time()
import matplotlib.pyplot as plt
from scipy import stats
#A = np.random.normal(32, 2.5, size=(200, 400))
np.random.seed(10)
from numpy.lib.stride_tricks import as_strided
from matplotlib.offsetbox import AnchoredText
import random
from scipy.sparse import coo_matrix, vstack, hstack
impreso1 = False
impreso2 = False

Metod_Importaco = np.zeros(4)
Metod_Suma3 = np.zeros(4)
Metod_Suma5 = np.zeros(4)
Metod_Suma10 = np.zeros(4)
Error_perc_3 = []
Error_perc_5 = []
Error_perc_10 = []


def read_confussion(matrix):
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    d = matrix[3]
    Sensibilidad= d/(d+c)
    True_Negative = a/(a+b)
    Exactitud = d/(c+d)
    Especifidad = a/(a+b)
    Precision = (a+d)/(a+b+c+d)
    
    print('Sensibilidad (True Positive): ', Sensibilidad)
    print('True Negative: ', True_Negative)
    print('Exactitud: ', Exactitud)
    print('Especifidad: ', Especifidad)
    print('Precisión: ', Precision)

def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 
    """
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1] 
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
    return arr


calidad_calibs_big =[]
calidad_satchets = []

num_containers = 1
num_tot = int(30000*num_containers)

results = np.zeros([num_tot,7])
num_sachet = 0
size_temp = (250, 50)

temp = np.zeros((size_temp[0],size_temp[1],2))
delta =0

AA = np.random.choice([31,32,33,34,35,36],3,p= [0.1,0.2,0.2,0.2,0.2,0.1])
AA2 = [i+np.random.uniform(low=0,high=999)/1000 for i in AA]
calidad_calibs =[]
while num_sachet<(num_tot-1):
#    Calibre = np.random.normal(35, 2.5, size=size_temp)
#    Calibre = np.random.triangular(31,34,36, size=size_temp)
    Calibre = np.random.triangular(30.67,34,36, size=size_temp)
    Peso = -12.345 + 0.705 * Calibre # Regresion Sacos Claudia
    #Peso = -22.1+.96*Calibre  # Proporcion Lineal entre Calibre y Peso
#    Peso = np.random.normal(12, 1.5, size=size_temp)
    cumplen_temp = np.count_nonzero(Calibre<32)
    tot_temp = np.count_nonzero(Calibre)
#    print('percent_temp=', 100*cumplen_temp/tot_temp)
    calidad_calibs.append(100*cumplen_temp/tot_temp)
    if num_sachet ==0:
        Calibre_Full =Calibre.flatten()
    else:
        B = coo_matrix(Calibre[temp[:,:,0]==0].flatten())
        Calibre_Full= hstack([B, Calibre_Full]).toarray()
        
    temp[temp[:,:,0]==0,0] = Calibre[temp[:,:,0]==0]
    temp[temp[:,:,1]==0,1] = Peso[temp[:,:,1]==0]
    #temp[:,:,0] = Calibre
    #temp[:,:,1] = Peso
#    if not impreso1:
#        plt.figure()
#        ax1 = sns.heatmap(temp[:,:,0], vmin=31, vmax=36,xticklabels=False, yticklabels=False)
##        plt.title('Tolva Inicial')
#    else:
#        plt.figure()
#        ax1 = sns.heatmap(temp[:,:,0], vmin=31, vmax=36,xticklabels=False, yticklabels=False)
##        plt.title('Tolva Después del relleno')
#        break
        
    for i in range(1000):
    
        box = np.zeros((300,2))
        box_mass = 0 
        box_index =0
        while box_mass <=605 + delta:
            size_clip = np.random.choice([1,2,3,4],2)
            if box_mass > 400 + delta and box_mass <= 500+delta :
                size_clip = np.random.choice([1,2,3],2)
            elif box_mass > 500 + delta and box_mass <= 550 + delta:
                size_clip = np.random.choice([1,2],2)
            elif box_mass > 550+delta:
                
                size_clip = [1,1]
            index = np.random.randint(0,size_temp[1]-size_clip[1]+1)
            
            A = temp[size_temp[0]-size_clip[0]:,index:index+size_clip[1]:,:].copy()
            sum_cand = A[:,:,1].sum()
            if box_mass + sum_cand >608 + delta:
#                print('hubiese sido muy grande')
                continue
            
            temp[size_temp[0]-size_clip[0]:,index:index+size_clip[1]:,:] = np.zeros((size_clip[0],size_clip[1],2))
            r = [1 if i >= index and i < index+ size_clip[1] else 0 for i in range(size_temp[1])]
            r = np.dot(r,size_clip[0])
            temp = indep_roll(temp, r,axis =0)
        #    print(A[:,:,1].sum(), size_clip, 'Tamaño Bloque entrante')
            
        
            box_mass = box_mass + sum_cand
        #    print(box_mass, 'peso sachet')
            b=A.reshape(-1,2)
            box[box_index:box_index+b.shape[0]]=b
            box_index +=b.shape[0]
            if box_mass > 595 + delta:
                break
        tot_nuts  = np.count_nonzero(box[:,0])
        good_nuts = np.count_nonzero(box[:,0]>32)
        bad_nuts = tot_nuts-good_nuts
    #    print(tot_nuts, good_nuts, 100*good_nuts/tot_nuts)
        percent = 100*bad_nuts/tot_nuts
        results[num_sachet,0] = 1
        results[num_sachet,3] = tot_nuts
        results[num_sachet,4] = bad_nuts 
        results[num_sachet,5] = good_nuts 
        results[num_sachet,6] = box_mass
        if percent >10:
            # no cumple con criterio
            results[num_sachet,1] = 0
            
        else:
            # si cumple con criterio
            results[num_sachet,1] = 1
        
        
        results[num_sachet,2] = percent
        
        num_sachet = num_sachet+1
        if num_sachet>=num_tot:
            break
#        if np.count_nonzero(temp[-220,:,0]==0) >0 and not impreso1:
#            print(i)
#            plt.figure()
#            ax2 = sns.heatmap(temp[:,:,0], vmin=31, vmax=36,xticklabels=False, yticklabels=False)
##            plt.title('Tolva Vaciándose')
#            impreso1 = True
#        if np.count_nonzero(temp[-100,:,0]==0) >0 and not impreso2:
#            print(i)
#            plt.figure()
#            ax2 = sns.heatmap(temp[:,:,0], vmin=31, vmax=36,xticklabels=False, yticklabels=False)
##            plt.title('Tolva Vaciándose')
#            impreso2 = True
        if np.count_nonzero(temp[-1,:,0]==0) >0:
#            print(i)
#            plt.figure()
#            ax2 = sns.heatmap(temp[:,:,0], vmin=31, vmax=36, xticklabels=False, yticklabels=False)
##            plt.title('Tolva Antes del Relleno')
            break

        if num_sachet%1500 ==0:
            print("---------------------")
            print("Muestreo: ")
            print(num_sachet)
            Test = results[num_sachet-1500:num_sachet,:]
            Buenos = np.count_nonzero(Test[:,1])
            Malos = np.count_nonzero(Test[:,1]==0)
            Total = np.count_nonzero(Test[:,1]>=0)
            prob_aceptar  = 100*hypergeom.pmf(0, Total, Malos, 3)
            print("Buenos: {}".format(Buenos) )
            print("Malos: {}".format(Malos) )
            print("Probabilidad Aceptar Container: ", prob_aceptar)
            
            promedio_Test = np.mean(Test[:,2])
            sample3 = random.choices(Test,k=3)
            sample3_tots = np.sum(sample3,axis=0)[3]
            sample3_malos = np.sum(sample3,axis=0)[4]
            sample3_percent = 100*sample3_malos/sample3_tots
            op3=np.array([i[2] for i in sample3])
            malos_sacados = np.count_nonzero(op3>10)

            sample5 = random.choices(Test,k=5)
            sample5_tots = np.sum(sample5,axis=0)[3]
            sample5_malos = np.sum(sample5,axis=0)[4]
            sample5_percent = 100*sample5_malos/sample5_tots
            
            sample10 = random.choices(Test,k=10)
            sample10_tots = np.sum(sample10,axis=0)[3]
            sample10_malos = np.sum(sample10,axis=0)[4]
            sample10_percent = 100*sample10_malos/sample10_tots
            
            
            print('Porcentaje Calibre últimos 1500 satchets: ', promedio_Test)
            print('Porcentaje Muestreo 3 satchets: ', sample3_percent)
            print('Porcentaje Muestreo 5 satchets: ', sample5_percent)
            print('Porcentaje Muestreo 10 satchets: ', sample10_percent)
            print('Satchets con descalibre en test ', malos_sacados)
            Error_perc_3.append(promedio_Test - sample3_percent)
            Error_perc_5.append(promedio_Test - sample5_percent)
            Error_perc_10.append(promedio_Test - sample10_percent)
            if promedio_Test < 10:
                if malos_sacados <1:
                    Metod_Importaco[0]+=1
                else:
                    Metod_Importaco[1]+=1
                
                if sample3_percent < 10:
                    Metod_Suma3[0] += 1
                else:
                    Metod_Suma3[1] += 1
                
                if sample5_percent < 10:
                    Metod_Suma5[0] += 1
                else:
                    Metod_Suma5[1] += 1
                
                if sample10_percent < 10:
                    Metod_Suma10[0] += 1
                else:
                    Metod_Suma10[1] += 1
            else:
                if malos_sacados <1:
                    Metod_Importaco[2]+=1
                else:
                    Metod_Importaco[3]+=1
                
                if sample3_percent < 10:
                    Metod_Suma3[2] += 1
                else:
                    Metod_Suma3[3] += 1
                
                if sample5_percent < 10:
                    Metod_Suma5[2] += 1
                else:
                    Metod_Suma5[3] += 1

                if sample10_percent < 10:
                    Metod_Suma10[2] += 1
                else:
                    Metod_Suma10[3] += 1                
                    
                    
#        plt.figure()
#        ax2 = sns.heatmap(temp[:,:,0])
#        plt.title('Saco Vaciándose')
#        break
    
    
    
plt.figure()      
plt.hist(Calibre.flatten())
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.axvline(x=32, color ='red')
tot = np.count_nonzero(Calibre.flatten())
malas = np.count_nonzero(Calibre.flatten()<32)
perc_desc = round(100 * malas/tot,2)
plt.text(31, plt.ylim()[1]-200, "Porcentaje\nDescalibre: {}".format(perc_desc),
         fontsize=14, verticalalignment='top', bbox=props)
plt.title('Distribución  % calibre en Tolva ')
plt.ylabel('Frecuencia')
plt.xlabel('% de Calibrado sobre 32 por Tolva')

plt.figure()      
plt.hist(results[:,2], int(max(results[:,2])-min(results[:,2])))
plt.axvline(x=10, color ='red')
desc_satchets = round(100*(1-np.mean(results[:,1])),2)
plt.text(plt.xlim()[1]-8, plt.ylim()[1]-200, "Porcentaje\nDescalibre Satchets: {}".format(desc_satchets),
         fontsize=14,verticalalignment='top', bbox=props)
plt.title('Distribución  % calibre en satchets ')
plt.ylabel('Frecuencia')
plt.xlabel('% de Calibrado sobre 32 por satchet')

plt.figure()      
plt.hist(results[:,3],int(max(results[:,3])-min(results[:,3])))
plt.title('Distribución  # nueces por satchet ')
plt.ylabel('Frecuencia')
plt.xlabel('# nueces por satchet')        
           
plt.figure()      
plt.hist(Calibre_Full.flatten())
plt.title('Distribución de Calibre Container')
plt.ylabel('Frecuencia')
plt.xlabel('Calibre')   
plt.axvline(x=32, color ='red')  

tot_full = np.count_nonzero(Calibre_Full.flatten())
malas_full = np.count_nonzero(Calibre_Full.flatten()<32)
perc_desc_full = round(100 * malas_full/tot_full,2)
plt.text(31, plt.ylim()[1]-200, "Porcentaje\nDescalibre: {}".format(perc_desc_full),
         fontsize=14, verticalalignment='top', bbox=props)       
        
satchets_fail = stats.percentileofscore(results[:,2],10)
    
    #plt.figure()
    #plt.hist(results[:,2][results[:,6]>605])
    #plt.title('Distribución  % calibre en satchets Sin considerar >605gr')
    #plt.ylabel('Frecuencia')
    #plt.xlabel('% de Calibrado sobre 32 por satchet')
#print('time Total:', time.time()-start)  
            
    #ax = sns.heatmap(A)
calidad_calibs_big.append(np.mean(calidad_calibs))
calidad_satchets.append(100*(1-np.mean(results[:,1])))
print('Porcentaje satchets que no cumplen:  ',1-np.mean(results[:,1]),' %')

print('Resultados Test Importaco:')
read_confussion(Metod_Importaco)
print('---------------')

print('Resultados Test 3:')
read_confussion(Metod_Suma3)
print('Error Percentual Test 3: ',np.mean(Error_perc_3))
print('---------------')

print('Resultados Test 5:')
read_confussion(Metod_Suma5)
print('Error Percentual Test 5: ',np.mean(Error_perc_5))
print('---------------')

print('Resultados Test 10:')
read_confussion(Metod_Suma10)
print('Error Percentual Test 10: ',np.mean(Error_perc_10))
print('---------------')


print('time:', time.time()-start)


#print('percentil que calza con container: ',np.percentile(results[:,2], 96.5))

#plt.figure()
#plt.plot(calidad_satchets, calidad_calibs_big)
#plt.title('Calidad Satchets vs Calidad Calibre')
#plt.ylabel('Calidad Calibs')
#plt.xlabel('Calidad Satchet')
#    
