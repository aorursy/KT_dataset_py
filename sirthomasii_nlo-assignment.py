import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


theta = np.radians(29.2) #radians

lamba = np.linspace(.200,1,10000)

n_o = np.sqrt(2.7359+0.01878/(lamba**2-0.01822)-0.01354*lamba**2) #ordinary refractive index
n_e = np.sqrt(2.3753+0.01224/(lamba**2-0.01667)-0.01516*lamba**2) #extraordinary refractive index

df = pd.DataFrame(zip(*[lamba,n_o,n_e]),columns=["lamba", "n_o", "n_e"])
ax = df.plot(x="lamba",figsize=(10,8))

n_e_theta = np.sqrt(1/(((np.sin(theta)**2)/n_e**2)+((np.cos(theta)**2)/n_o**2)))
df_theta = pd.DataFrame(zip(*[lamba,n_e_theta]),columns=["lamba", "n_e_theta"])
df_theta.plot(x="lamba",ax=ax)
ax.set_ylabel("RFI")
ax.set_xlabel("Wavelength (um)")


L = np.linspace(0,25,100)

n_e_theta_400nm = float(df_theta[df_theta['lamba'].between(.3995, .4005)]['n_e_theta'])
n_o_800nmm = float(df[df['lamba'].between(.7995, .8005)]['n_o'])

delta_k = 4*np.pi*(n_e_theta_400nm-n_o_800nmm)/.8
print("crystal length", delta_k)

I_2 = np.sin(delta_k*L/2)**2/(delta_k**2)
delta_k = .1

I_2_lowerK = np.sin(delta_k*L/2)**2/(delta_k**2)
delta_k = .5

I_2_lowK = np.sin(delta_k*L/2)**2/(delta_k**2)
print("phase-matched doubled frequency n_e(θ,λ/2) = ",n_e_theta_400nm)

df_I2 = pd.DataFrame(zip(*[L,I_2]),columns=["L", "I_2 (k=0.0)"])
df_I2_lowerK = pd.DataFrame(zip(*[L,I_2_lowerK]),columns=["L", "I_2 (k=0.1)"])
df_I2_lowK = pd.DataFrame(zip(*[L,I_2_lowK]),columns=["L", "I_2 (k=0.5)"])

ax = df_I2.plot(x="L",figsize=(10,8))
df_I2_lowerK.plot(x="L",ax=ax)
df_I2_lowK.plot(x="L",ax=ax)
L=3e6

delta_k_array = np.zeros(len(lamba))
I_2 = np.zeros(len(lamba))
for i in range(len(lamba)):
    
    delta_k = (4*np.pi*(n_e_theta[i]-n_o[i])/lamba[i])
    delta_k_array[i] = delta_k
    
    I_2[i]= (np.sin(delta_k*(L/2))**2/(delta_k**2))


df_I2 = pd.DataFrame(zip(*[lamba,I_2]),columns=["wavelength", "I_2 (k=0.0)"])

ax = df_I2.plot(x="wavelength",figsize=(10,8),xlim=[.390,.410])
df_dk = pd.DataFrame(zip(*[lamba,delta_k_array]),columns=["wavelength", "dk"])
df_dk.plot(x="wavelength")

