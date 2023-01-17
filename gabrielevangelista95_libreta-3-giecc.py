# Asignar listas vacias o de ceros para inicializacion de variables 
# Usar abs y len y randi
data = [38,48,54,39,48,39,40,30,23,32,49,23,47,37,23,48,42,48,49,32,36,39,53]
N = 10        # numero de ciclos
M = 12        # numero de elementos por ciclo
L = len(data) # cantidad de elementos en la variable data 
import numpy as np 
from numpy import random 
random.randint(50) # Toma como argumento el numero maximo 
r_index = random.randint(len(data)) # La longitud de la lista sera el numero limite {0-length_of(data)}
empty_list = []
zeros_list = np.zeros(M)

print("  empty_list content : {} \n  zeros_list content : {}".format(empty_list,zeros_list))
for m in range(M): 
    r_index = random.randint(L)

    empty_list.append(data[r_index])
    zeros_list[m] = data[r_index]
        
print("Resultado : \n  empty_list : {} \n  zeros_list : {}".format(empty_list,zeros_list))
inicializado_como_lista = []
inicializado_como_zeros = np.zeros((N,M))

print("  empty_list content : \n\t\t\t{} \n  zeros_list content : \n{}".format(inicializado_como_lista,
                                                                               inicializado_como_zeros))
for n in range(N):
    this_cicle_list = []
    for m in range(M): 
        r_index = random.randint(L)

        this_cicle_list.append(data[r_index])
        inicializado_como_zeros[n][m] = data[r_index]
    
    inicializado_como_lista.append(this_cicle_list)
    
print("Resultado : \n  inicializado_como_lista : {} \n  zeros_list : {}".format(inicializado_como_lista,
                                                                                inicializado_como_zeros))
print("Resultado : ")

print("  inicializado_como_lista : ")

for a_list in inicializado_como_lista:
    print(a_list)
    
print("  inicializado_como_zeros : ")
# for a_row in inicializado_como_zeros: 
#     print(a_row)
print(inicializado_como_zeros)

from numpy.random import choice as rand_choice # usualmente usado como np.random.choice() 
from matplotlib import pyplot as graficar      # usualmente usado como plt()
rand_choice(data,5)
rand_choice(data,5)
rand_choice(data,5)
choices = [rand_choice(data,12) for n in range(10)]

for a_choice in choices:
    print(a_choice)
stds = [np.std(this_choice) for this_choice in choices]

for a_std in stds:
    print(a_std)
    
graficar.plot(stds); graficar.show()
means = [np.mean(this_choice) for this_choice in choices]

for a_mean in means: 
    print(a_mean)
    
graficar.plot(means); graficar.show()
medians = [np.median(this_choice) for this_choice in choices]

for a_median in medians: 
    print(a_median)
    
graficar.plot(medians); graficar.show()
[stds,means,medians]
# # Las listas se pueden convertir en arreglos vectoriales individualmente 

# stds_np = np.array(stds)
# means_np = np.array(means)
# medians_np = np.array(medians)

# print(type(stds_np)," : \n\t\t", stds_np)
# print(type(means_np)," : \n\t\t", means_np)
# print(type(medians_np)," : \n\t\t", medians_np)
# Las listas pueden ser convertidas a arreglos vectoriales mediante la implicacion de la libreria numpy
np.vstack((stds,means,medians)).T 
results = np.vstack((stds,means,medians)).T 
results[0,:]
results[:,0]
graficar.figure(figsize=(16,5))   # Indicar tamano de la figura 

graficar.subplot(1,3,1); graficar.plot(stds);    graficar.title("Desviaciones estandard")
graficar.subplot(1,3,2); graficar.plot(means);   graficar.title("Promedios")
graficar.subplot(1,3,3); graficar.plot(medians); graficar.title("Medianas")

graficar.show()
graficar.figure(figsize=(16,5))   # Indicar tamano de la figura 

graficar.subplot(1,3,1); graficar.plot(results[:,0]); graficar.title("Desviaciones estandard")
graficar.subplot(1,3,2); graficar.plot(results[:,1]); graficar.title("Promedios")
graficar.subplot(1,3,3); graficar.plot(results[:,2]); graficar.title("Medianas")

graficar.show()
fig, axs = graficar.subplots(1, 3, figsize=(16, 5))
axs[0].bar(list(range(N)),stds);      axs[0].set_title("Desviaciones estandard")
axs[1].scatter(list(range(N)),means); axs[1].set_title("Promedios")
axs[2].plot(list(range(N)),medians);  axs[2].set_title("Medianas")

fig.text(0.5, 0.04, 'ciclos', ha='center', va='center')
fig.text(0.1, 0.5,  'magnitud', ha='center', va='center', rotation='vertical')

graficar.show()
fig, axs = graficar.subplots(1, 3, figsize=(16, 5), sharey=True, sharex=True)
axs[0].bar(list(range(N)),stds);      axs[0].set_title("Desviaciones estandard")
axs[1].scatter(list(range(N)),means); axs[1].set_title("Promedios")
axs[2].plot(list(range(N)),medians);  axs[2].set_title("Medianas")

fig.text(0.5, 0.04, 'ciclos', ha='center', va='center')
fig.text(0.1, 0.5,  'magnitud', ha='center', va='center', rotation='vertical')

graficar.show()
q_25 = [np.quantile(this_choice, 0.25) for this_choice in choices]
q_75 = [np.quantile(this_choice, 0.75) for this_choice in choices]
fig, (ax0, ax1) = graficar.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 6))

ax0.set_title('Grafica de error')
ax0.errorbar(list(range(N)), means, yerr=stds, linestyle=':', fmt='o')

ax1.set_title('Grafica de caja')
ax1.boxplot(results.T)

fig.text(0.5, 0.04, 'ciclos', ha='center', va='center')
fig.text(0.1, 0.5,  'magnitud', ha='center', va='center', rotation='vertical')

graficar.show()
print("  ->  Promedio   : {}".format(np.mean(means)))
print("  ->  Media      : {}".format(np.mean(medians)))
print("  ->  Desv. est. : {}".format(np.mean(stds)))
graficar.figure(figsize=(16,5))

graficar.subplot(1,2,1);     graficar.imshow(choices);             graficar.colorbar(); 
graficar.xlabel("Elemento"); graficar.ylabel("Ciclo")

graficar.subplot(1,2,2);     graficar.imshow(choices, cmap='jet'); graficar.colorbar()
graficar.xlabel("Elemento"); graficar.ylabel("Ciclo")

graficar.show()
