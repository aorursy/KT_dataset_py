import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cmath
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def ABCD2S(ABCD,Z0):
    S = np.zeros(shape=(2,2),dtype=complex)
    tmp = ABCD[0,0] + ABCD[0,1]/Z0 + ABCD[1,0]*Z0 + ABCD[1,1]
    S[0,0] = (ABCD[0,0] + ABCD[0,1]/Z0 - ABCD[1,0]*Z0 - ABCD[1,1])/tmp
    S[0,1] = 2 * (ABCD[0,0] * ABCD[1,1] - ABCD[0,1] * ABCD[0,1])/tmp
    S[1,0] = 2 / tmp
    S[1,1] = (-ABCD[0,0] + ABCD[0,1]/Z0 - ABCD[1,0]*Z0 + ABCD[1,1])/tmp
#     print(S)
    return S
# ABCD = numpy.zeros(shape=(2,2),dtype=complex)
# ABCD = np.array([[(1+1j)/pow(2,0.5),(1+1j)/pow(2,0.5)],[(1+1j)/pow(2,0.5),(1+1j)/pow(2,0.5)]],dtype=complex)
Z1 = 100
llambda = 0.2
bl = 2*np.pi*llambda
ABCD = np.array([[np.cos(bl),1j*Z1*np.sin(bl)],[1j*(1/Z1)*np.sin(bl),np.cos(bl)]],dtype=complex)
S = ABCD2S(ABCD,50)
print('S11 = ',S[0,0])
print('S21 = ',S[1,0])
def zsc(zo,bl):
    out = 1j*zo*np.tan(bl)
    return out
def zoc(zo,bl):
    out = -1j*zo*1/(np.tan(bl))
    return out
Z0=50
Z1=100
S11e = (zoc(Z1,bl/2) - Z0)/(zoc(Z1,bl/2) + Z0)
S11o = (zsc(Z1,bl/2) - Z0)/(zsc(Z1,bl/2) + Z0)
print('S11 = ',(S11e+S11o)/2)
print('S21 = ',(S11e-S11o)/2)
print('Show that |S11|^2+|S21|^2 = 1 for a lossless transmission line.')
print('np.abs((S11e-S11o)/2)**2+np.abs((S11e+S11o)/2)**2 = ',np.abs((S11e-S11o)/2)**2+np.abs((S11e+S11o)/2)**2)
def zin(zo,zl,bl):
    out = zo*(zl+1j*zo*np.tan(bl))/(zo+1j*zl*np.tan(bl))
    return out
Zin = zin(Z1,Z0,bl)
S11 = (zin(Z1,Z0,bl)-Z0)/(zin(Z1,Z0,bl)+Z0)
print('S11 =', (zin(Z1,Z0,bl)-Z0)/(zin(Z1,Z0,bl)+Z0))
tmp = 1 - np.abs(S11)**2
print('Computing only phase gives a wrong result. S21 !=',pow(tmp,0.5)*(np.cos(bl)-1j*np.sin(bl)))
S21 = (S11e-S11o)/2
print('Phase of S21',cmath.phase(S21)*180/np.pi,' degrees')
print('Phase computed from the lenght of TL',-bl*180/np.pi,' degrees')
#Computing S12 using circuit theory
refl=(Z0-Z1)/(Z0+Z1)
V2n = 1+refl
# V1p = ((Z0+Zin)/(2*Zin))*((np.cos(bl)+1j*np.sin(bl))+refl*(np.cos(bl)-1j*np.sin(bl)))
V1p = ((Z0+Zin)/(2*Zin))*(np.exp(1j*bl)+refl*np.exp(-1j*bl))
print('S12 =',V2n/V1p)
#test np.exp(complex())
np.exp(1j*(np.pi/2))