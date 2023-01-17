# One feature constructing
import pandas as pd

import numpy as np

import pyarrow.parquet as pq
metatrain=pd.read_csv ('../input/metadata_train.csv')

m1i=metatrain[metatrain.target==1].index

m0i=metatrain[metatrain.target==0].index


suin0 = pq.read_pandas('../input/train.parquet', \

                        columns=[str(i) for i in m0i [:500]]).to_pandas()

suin0=suin0.transpose ()

suin0.index=suin0.index.astype ('int')

suin1 = pq.read_pandas('../input/train.parquet', \

                        columns=[str(i) for i in m1i [:500]]).to_pandas()

suin1=suin1.transpose ()

suin1.index=suin1.index.astype ('int')
import matplotlib.pyplot as plt

from scipy.fftpack import fft,dct,idct,dstn,idstn

from scipy.signal import find_peaks, peak_widths

from scipy.ndimage import maximum_filter1d

# This procedure shifts the maximum of the signal to zero

def shift1 (v):

  nn=len (v)

  s1=np.argmax (v); s4=s1

  s2=set(find_peaks (v)[0])

  s2.difference_update({s1})

  for i2 in s2:

    if v [i2]>0.85*v [s1]:

      if (s1-i2) % nn < nn/6:

        s3=((s1-i2) % nn)//2

        s4=(i2+s3) % nn

      elif (i2-s1) % nn < nn/6:

        s3=((i2-s1) % nn)//2

        s4=(s1+s3) % nn      

  s4=s4.astype ('int') 

  return np.roll (v,-s4),s4



# This procedure finds the difference between the signal and its approximation

# using scipy.fftpack.dct (also shift1 used)

def redim (vec):

  nidct=150

  vec_trunc=np.zeros ([nidct])

  vec_dct=dct (vec,n=800000,norm='ortho')

  vec_nidct= idct (vec_dct,n=nidct, norm='ortho')

  s1=np.argmax (vec_nidct)

  vec_nidct_shift,s5=shift1 (vec_nidct)

  vec_nidct_shift_dct =dct (vec_nidct_shift,norm='ortho')

  vec_trunc[0:15]=vec_nidct_shift_dct [0:15]

  vec_trunc_idct=idct (vec_trunc,n=800000, norm='ortho')

  vec_shift=np.roll(vec,-s5*800000//nidct)

  vec_dif=vec_shift-vec_trunc_idct

  return vec_dif

                
# This procedure uses some filter and constructs some characteristic

# for 15 rectangulars formed by the signal. Why so? Don't know. The best

# parameters I found are here

def shtuk2 (vecvdi):

  nv=5; nv1=3; nve=800//nv; ng=1000//nv1; nyc=nve*ng; nb=nyc//6

  mat44=np.empty ([nv,nv1]); 

  for i1 in range (nv1):

    for i0 in range(nv):

      ve2=vecvdi.reshape (800,-1)[i0*nve:(i0+1)*nve,i1*ng:(i1+1)*ng].reshape (-1)

      ve3=idct (ve2.reshape (-1),norm='ortho');

      ve3s=ve3 [nb:]

      ve4=maximum_filter1d(ve3s,58,mode='wrap')

      ff=find_peaks (ve4,height=1,distance=2,width=1)

      ff1=ff [1]['peak_heights'];

      fs=ff1.sum ()/ff1.mean ()

      mat44 [i0,i1]=fs

  vem=np.min(mat44)

  return vem
# We collect the values of shtuk2 procedure in two lists for 500 normal and 

# 500 fault signals

min0=[]; min1=[]; 

for i in range(500):

  no0=m0i[i]

  ve0=suin0.loc [no0,:]

  vdi0=redim (ve0)

  chi0=shtuk2 (vdi0)

  min0.append (chi0);

for i in range(500):  

  no1=m1i[i]

  ve1=suin1.loc [no1,:]

  vdi1=redim (ve1)

  chi1=shtuk2 (vdi1)

  min1.append (chi1)

  

  
# For this number ku we have that 153 of 500 faults are greater than ku, and 

# only 4 of 500 normal signals are greater than ku

ku=np.sort(min0)[-5]

aa=pd.DataFrame(min1); bb=pd.DataFrame (min0);

print ((bb>ku).sum(), (aa>ku).sum())
plt.title('500 normal vs 500 faults')

plt.hist((min1,min0))
