import math

import scipy.constants as sc

from tqdm import tqdm

import numpy as np

from scipy.special import factorial as fac

import scipy

import cmath

from scipy.special import sph_harm

import pandas as pd

import matplotlib.pyplot as plt



# Loai phan tu

type_of_mol = 3

# print(type_of_mol)



if type_of_mol == 1:

    B = 0.39/219474.63068

    alpha_para = 4.05/0.14818474347690475

    alpha_pedi = 1.95/0.14818474347690475

elif type_of_mol == 2:

    B = 1.4377/219474.63068

    alpha_para = 2.35/0.14818474347690475

    alpha_pedi = 1.21/0.14818474347690475

else:

    B = 1.998/219474.63068

    alpha_para = 2.38/0.14818474347690475

    alpha_pedi = 1.45/0.14818474347690475





#parameter 

# 1 auT = 0.024189 fs

    auT = 0.024189

########################################################################################################################

####################### LASER ##########################################################################################

pi = np.pi

# print("Nhập vào cường độ laser:")

I = 2*(10**13)



# print("Nhập vào bước sóng của laser:")

lamda = 0.0000008



# angular freq in atomic units 

omega0 = 0.057



# init_phase = 0

t_fwhm = 60/auT



#

muy_zero = 4*pi*(10**(-7))

speed_of_light = sc.c

Emax = math.sqrt(I/(3.5*(10**16)))

f = speed_of_light/lamda



def E(t):

    return Emax*np.exp(-2*np.log(2)*(t**2/(t_fwhm**2)))*math.cos(omega0*t)



#    return Emax*np.exp(-2*np.log(2)*(t**2/(t_xung**2/4)))*math.cos(2*pi*f*t)



####################### TIME ##########################################################################################

time = []



ta = -2*(t_fwhm)

tb = 2*t_fwhm



t_step = 500

delta_t = (tb-ta)/t_step

e = []

for i in range(t_step):

    time.append(ta + i*delta_t)

    e.append(E(ta + i*delta_t + delta_t/2))



df = pd.DataFrame({

    'time': time,

    'E': e

})

df.head()



#delta_t = delta_t/(2.42*(10**-17))



plt.figure(figsize=(50, 10))



plt.subplot(132)

plt.plot(df['time'], df['E'])

plt.suptitle('Laser Plotting')

plt.show()

# plt.savefig('laser.png')
####################### TRANG THAI #####################################################################################

jMax = 11

index = []

index.append((0, 0))

index.append((1, -1))

index.append((1, 0))

index.append((1, 1))

index.append((2, -2))

index.append((2, -1))

index.append((2, 0))

index.append((2, 1))

index.append((2, 2))

for j in range(3, jMax):

    for m in range(7):

        index.append((j, m - 3))

# print(np.array(index).shape)

nMax = np.array(index).shape[0]

########################################################################################################################





####################### HE SO C THEO THOI GIAN #########################################################################

c = []

for k in range(nMax):

    temp = []

    for j in range(nMax):

        if (index[j][0] == index[k][0] and index[j][1] == index[k][1]):

            temp.append(1)

        else:

            temp.append(0)

    c.append(temp)



c = np.array(c)

print(c.shape)



pi = np.pi





def a(j, m):

    return np.sqrt(((2 * j + 1) / (4 * pi)) * (fac(j - m) / fac(j + m)))





def ffi(m, mm):

    if (m == mm):

        return 1

    else:

        return 0





########################################################################################################################



####################### MA TRAN TOAN TU COS^2 THETA ####################################################################

cos_theta = []



# N = 16

x, w = np.polynomial.legendre.leggauss(jMax)

phi_r = (2 * pi - 0) / 2

u_r = 1

phi_m = (2 * pi + 0) / 2

u_m = 0





def ftheta_t(j, m, jj, mm, u, t):

    if (j == jj or j == jj + 2 or j == jj - 2):

        return (scipy.special.lpmv(m, j, u) * scipy.special.lpmv(mm, jj, u) * (

            np.exp(-1j * ((-1) * (e[t] ** 2) * (alpha_para * u ** 2 + alpha_pedi * (1 - u ** 2)) / 2) * delta_t)))

    else:

        return 0





em = []

for i in range(nMax):

    em.append((cmath.exp(-1j * B * index[i][0] * (index[i][0] + 1) * delta_t / 2)))

em = np.array(em)
####################### SPLIT OPERATOR #################################################################################

c_temp = []

for q in tqdm(range(nMax)):

    temp = c[q]

    for i in range(t_step):

        t = ta + i * delta_t

        c_1 = em*temp

        c_1 = np.array(c_1)

        c_2 = []

        for m in range(nMax):

            sumt = 0j

            for n in range(nMax):

                if (index[m][1] == index[n][1]) and (index[m][0] == index[n][0] or index[m][0] == (index[n][0] + 2) or index[m][0] == (index[n][0] - 2)):

                    fi = 2*pi

                    thet = 0

                    for k in range(jMax):

#                          thet = thet + w[k] * ftheta_t(index[m][0], index[m][1], index[n][0],index[n][1], u_r * x[k] + u_m, i)

                         thet = thet + w[k] * ftheta_t(index[m][0], index[m][1], index[n][0],index[n][1], x[k], i)

#                     thet = thet * u_r

                    sumt = sumt + c_1[n] * a(index[m][0], index[m][1]) * a(index[n][0],index[n][1]) * thet * fi

                else:

                    sumt = sumt + 0

            c_2.append(sumt)

        c_2 = np.array(c_2)

        c_3 = em*c_2

        c_3 = np.array(c_3)

        temp = c_3

    c_temp.append(temp)

c = c_temp



c_init = c
indx = []

wb = []

B = B * 219474.63068

c0 = scipy.constants.c

T = 20  # room temperature

k = scipy.constants.k  # boltzmann constant

kB = 69.50348004  # in m^-1 K^-1

h = scipy.constants.h



sum_exp = 0

for j in range(nMax):

    if j % 2 == 0:

        sum_exp = sum_exp + np.exp(-1 * B * 100 * index[j][0] * (index[j][0] + 1) / (kB * T))

    else:

        sum_exp = sum_exp + 2* np.exp(-1 * B * 100 * index[j][0] * (index[j][0] + 1) / (kB * T))



print(sum_exp)



for j in range(nMax):

    indx.append(j)

    if j % 2 == 0:

        wb.append(2 * np.exp(-1 * B * 100 * index[j][0] * (index[j][0] + 1) / (kB * T)) / sum_exp)

    else:

        wb.append(np.exp(-1 * B * 100 * index[j][0] * (index[j][0] + 1) / (kB * T)) / sum_exp)



df = pd.DataFrame({

    'J': indx,

    'W_j': wb

})

df.shape



plt.plot(df['J'],df['W_j'])

plt.suptitle('boltzmann distribution')

plt.show()
B = B/219474.63068



t_start = 0.0

t_end = 16000/auT

t_step = 500



delta_t = (t_end - t_start)/t_step

time = []

for i in range(t_step):

    time.append(t_start + i * delta_t)

#delta_t = delta_t/(2.42*(10**-17))



theta_r = pi / 2

theta_m = pi / 2

theta = []

for i in range(jMax):

    theta.append(np.arccos(x[i]))



theta.reverse()

from scipy.special import sph_harm

ro = []

# time = []

c = c_init

for i in tqdm(range(t_step)):

#     t = t_start/(2.42*(10**-17)) + i * delta_t

    t = t_start + i * delta_t

#     time.append(ta + i*delta_t)

    ro_temp = []

    c_temp = []

    for m in range(nMax):

        c_1 = []

        for n in range(nMax):

#             print(m,n)

            c_1.append((cmath.exp(-1j*B*index[n][0]*(index[n][0]+1)*t)*c[m][n]))

        c_temp.append(c_1)

#     print(len(c_temp))

    for t in range(jMax):

        sumt = []

        for m in range(nMax):

            temp = 0j

            for n in range(nMax):

                temp = temp + c_temp[m][n]*sph_harm(index[n][1],index[n][0],0,theta[t])

            sumt.append(temp)

        temp = 0j

        for i in range(jMax):

            temp = temp + (np.abs(sumt[i])**2)*wb[i]

        ro_temp.append(temp)

    ro.append(ro_temp)





cos_theta = []

time = []



# delta_t = delta_t*(2.42*(10**-17))

for i in tqdm(range(t_step)):

    time.append(t_start + i * delta_t)

    thet = 0

    for k in range(jMax):

        thet = thet + w[k] * ro[i][k] * ((u_r * x[k] + u_m) ** 2)

    cos_theta.append(thet * 2 * pi)  # /2.070419103)





df = pd.DataFrame({

    'time': time,

    'cos^2theta': cos_theta

})



plt.plot(df['time'],df['cos^2theta'])

plt.suptitle('COS^2 THETA')

plt.show()

# plt.savefig('cos2theta.png')