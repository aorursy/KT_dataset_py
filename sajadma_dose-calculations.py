%pylab inline
import math
import numpy as np
import matplotlib.pyplot as plt
################################################ Dose #########################################################
############################################ Aquisition parameters ############################################
Energy       = 22*10**3                                #eV
ExposureTime = 2*10**(-3)                              #s/projection
Flux         = 7*10**4                                 #photons/s/um²
NumberOfProj = [900, 450, 300, 225, 180,150,128, 112]  #Number of projections
mu_en        = 0.4727/(10**4)                          #um⁻¹
rho          = 1                                       #g/cm³
FieldSize    = 2000*1000*3                             #um²
BeamSize     = 3*3                                     #um²  
SampleDepth  = 2.5*10**(4)                             #um
ConversionF  = 1.602*10**(-4)                          #joule/eV
###################################### Entrance Skin Dose [Gy] ################################################
ESD = ConversionF*np.array(NumberOfProj)*Flux*ExposureTime*Energy*mu_en/(rho) 
print(ESD)
###################################### Absorbed Dose [Gy] ######################################################
D = ConversionF*np.array(NumberOfProj)*Flux*ExposureTime*(1-math.exp(-mu_en*SampleDepth))*FieldSize*Energy/(FieldSize*SampleDepth*rho)
print(D)
#============================================== CNR ==========================================================#
#Gridrec                                             PR vs AC 
I_outGC   = [5.436, 4.848]
I_inGC    = [29.74, 29.90]
SD_outGC  = [.6563, 15.13]
SD_inGC   = [.3559, 16.4]
#SIRT
I_inS=[13.32, 13.69]
I_outSC=[-12.19, -11.72]
SD_inSC=[1.05, 16.95]
SD_outSC=[0.9278, 16.62]

CNRGC = (np.array(I_inGC)-np.array(I_outGC))/np.sqrt(np.power(SD_inGC,2)+np.power(SD_outGC,2))
CNRSC=(np.array(I_inSC)-np.array(I_outSC))/np.sqrt(np.power(SD_inSC,2)+np.power(SD_outSC,2))
print('PR            AC')
print(CNRGC)
print(CNRSC)
#                                         CNR vs N-angles 
#Gridrec
#Corner ROI (CNR)
I_outG   = [5.444, 5.550, 5.521, 5.639, 5.896, 5.720, 5.698, 5.439]
I_inG  = [29.68, 29.64, 29.65, 29.54, 30.32, 29.60, 29.52, 29.85]
SD_outG  = [.6690, .7155, .7679, .9931, 1.102, 1.84, 2.092, 1.593]
SD_inG = [.244, .4656, .5949, 1.064, .9461, 1.326, 1.556, 2.148]
#Center ROI (CNR1)
I_inG1   = [24.42, 24.58, 24.81, 24.98, 24.49, 25.02, 24.64, 24.95]
I_outG1  = [2.458, 2.508, 2.399, 2.459, 2.340, 2.491, 2.301, 2.224]
SD_inG1  = [0.4031, 0.5961, 0.7600, 1.066, 1.092, 1.550, 1.955, 2.146]
SD_outG1 = [0.2349, 0.4041, 0.5229, 0.6017, 1.489, 1.055, 1.738, 1.540]
#SIRT
I_inS   = [13.32, 13.49, 13.47, 13.90, 12.85, 13.78, 13.12, 13.74]
I_outS  = [-12.19, -12.03, -12.27, -11.96, -15.30, -12.12, -11.62, -12.05]
SD_inS  = [1.05, 1.509, 1.836, 2.792, 2.900 , 3.567, 4.178, 5.058]
SD_outS = [0.9278, 1.262, 2.065, 2.576, 3.375, 3.250, 5.372, 4.725]
CNRG  = (np.array(I_inG)-np.array(I_outG))/np.sqrt(np.power(SD_inG,2)+np.power(SD_outG,2))
CNRG1 = (np.array(I_inG1)-np.array(I_outG1))/np.sqrt(np.power(SD_inG1,2)+np.power(SD_outG1,2))
CNRS  = (np.array(I_inS)-np.array(I_outS))/np.sqrt(np.power(SD_inS,2)+np.power(SD_outS,2))
N_Angles = [900, 450, 300, 225, 180, 150, 128, 112]
print('Number of projection angles:', N_Angles)
print('CNR Gridrec Corner ROI:',CNRG)
print('CNR Gridrec Center ROI:',CNRG1)
print('CNR SIRT',CNRS)
####################################################################################
CNR_Dose_Gridrec  = np.array(CNRG)/np.array(ESD) 
CNR_Dose1_Gridrec = np.array(CNRG1)/np.array(ESD)
CNR_Dose_SIRT     = np.array(CNRS)/np.array(ESD)
print(CNR_Dose_Gridrec)
print(CNR_Dose1_Gridrec)
print(CNR_Dose_SIRT)
plt.figure(figsize=(15,6))
plt.plot(N_Angles,CNR_Dose,'ko')
plt.plot(N_Angles,CNR_Dose1,'bo')
plt.xlabel('Number of Projection angles',fontsize=16)
plt.ylabel('CNR/Dose',fontsize=16)
plt.axis([0, 1000, 1.5, 4.1])
plt.figure(figsize=(15,6))
plt.plot(N_Angles,CNR_Dose_SIRT,'ko')
plt.xlabel('Number of Projection angles',fontsize=16)
plt.ylabel('CNR/Dose',fontsize=16)
plt.axis([0, 1000, 0.8, 1.6])