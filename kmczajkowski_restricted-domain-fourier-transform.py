import numpy as np # linear algebra

import scipy.sparse as sp

import matplotlib.pyplot as plt



def FTax(N):

    if N%2:

        x=np.concatenate((np.arange((N-1/2)+1),np.arange(-(N-1)/2,0)))  # [0:(N-1)/2,-(N-1)/2:-1] 

    else:

         x=np.concatenate((np.arange(N/2),np.arange(-N/2,0))) # 0:N/2-1,-N/2:-1]

    return x



class RDFTprep:    

    def __init__(self,MaskFT):

        """

        rdft - restricted domain Fourier transform

        """        

         

        self.MaskFT= MaskFT['fk']

        self.Ny,self.Nx=MaskFT['shape']

        self.px=FTax(self.Nx)

        self.py=FTax(self.Ny)

        self.Px,self.Py=np.meshgrid(self.px,self.py)



        self.NN=np.lcm(self.Nx,self.Ny)

        self.selfx=self.NN/self.Nx

        self.selfy=self.NN/self.Ny

        self.mx=self.Px[self.MaskFT]*self.selfx

        self.my=self.Py[self.MaskFT]*self.selfy

        self.PHI=np.exp(-2j*np.pi/self.NN*np.arange(self.NN))        

        self.M=lambda SpVals: self.PHI[(self.mx*self.Px[np.where(SpVals)].T+self.my*self.Py[np.where(SpVals)].T)%self.NN]



    def M(self, SpVals):

        ir,ic,nonzeros=sp.find(SpVals)

        return self.PHI[np.mod((np.kron(self.mx[:,np.newaxis],self.Px[ir,ic].T)+np.kron(self.my[:,np.newaxis],self.Py[ir,ic].T)),self.NN).astype('int32')]        

        

    def SparseDFT2(self,SpVals):

        M=self.M(SpVals)

        FT=spalloc(nnz(self.MaskFT),self.Nx*self.Ny,numel(M))

        FT[:,np.where(SpVals)] =M

        return lambda X: FT*X.flatten

         

    def PartDFT2(self,SpVals):

        M=self.M(SpVals)

        DFT2=lambda X: M*X.flatten

        return DFT2   

    

    def rdft2(self,SpVals):

        ir,ic,nonzeros=sp.find(SpVals)

        return self.PHI[np.mod((np.kron(self.mx[:,np.newaxis],self.Px[ir,ic].T)+np.kron(self.my[:,np.newaxis],self.Py[ir,ic].T)),self.NN).astype('int32')]@nonzeros
import time



Nobj=50 # number of non-zero elements in the image domain

Nft=300 # number of FT coefficients to be calculated

Nrep=5 # number of repetitions





Px=np.round(2**np.arange(10,14.5,0.5)).astype('int32')# matrix sizes (10:.5:14)

Px[1:len(Px)-1:2]=Px[1:len(Px)-1:2]+1 # to test also the odd sizes

Px=Px[:4]



Py=Px+0

tfft=np.zeros((len(Px),Nrep))

trdft=np.zeros((len(Px),Nrep))

Err=np.zeros((len(Px),Nrep))

for i in range(len(Px)):

    # optimize the fftw algorithm

    #fftw('planner','measure')

    #fft2(np.random.randn(Py[i],Px[i]))

    for j in range(Nrep): 

        print((i,j,len(Px)))

        # Select the frequencies that will be needed and store their positions

        # in a spase matrix MaskFT

        MaskFT=dict()

        fkx=np.random.randint(0,Px[i]-1,size=Nft)

        fky=np.random.randint(0,Py[i]-1,size=Nft)         

        MaskFT['fk']=(fky,fkx)

        MaskFT['shape']=(Py[i],Px[i])

        #MaskFT=sp.csc_matrix((np.ones(Nft),(fkx,fky)),shape=(Py[i],Px[i])).astype('bool')

        

        # prepare for fft evaluation

        rdft2=RDFTprep(MaskFT)

        

        # Create Py[i] x Px[i] matrix with Nobj non-zero elements

        xix=np.random.randint(0,Px[i]-1,size=Nobj)

        xiy=np.random.randint(0,Py[i]-1,size=Nobj)

        Xsparse=sp.csc_matrix((np.random.randn(Nobj),(xix,xiy)),shape=(Py[i],Px[i]),dtype=complex)

        Xfull=Xsparse.todense()        # full form

        

        

        # benchmark the sparse FT

        t_start=time.time()

        Ysparse=rdft2.rdft2(Xsparse)

        trdft[i,j]=time.time()-t_start

        

        

        # benchmark dense FFTW

        t_start=time.time()

        Y=np.fft.fft2(Xfull)

        tfft[i,j]=time.time()-t_start

        

        # check the difference

        Err[i,j]=np.linalg.norm(Ysparse-Y[MaskFT['fk']] )/ np.linalg.norm(Y[MaskFT['fk']])



# Plot a comparison

plt.figure(figsize=(5,9))

plt.subplot(3,1,1)

plt.loglog(np.sqrt(Px*Py),1e3*np.mean(tfft,axis=1),'*r')

plt.loglog(np.sqrt(Px*Py),1e3*np.mean(trdft,axis=1),'*g')

#plt.legend('FFTW (full matrices)','RDFT (sparse matrices)')

plt.ylabel('Time (ms)')

#plt.xlabel('Matrix size np.sqrt(Nx\cdot Ny)')

#title(sprintf('Restricted Domain FT benchmark\n (#d Non-zero elements, #d Fourier coefficients)',Nobj,Nft))



plt.subplot(3,1,2)

plt.loglog(np.sqrt(Px*Py),np.mean(tfft,axis=1)/np.mean(trdft,axis=1),'*k')

plt.ylabel('Time ratio')

#plt.xlabel('Matrix size np.sqrt(Nx\cdot Ny)')



plt.subplot(3,1,3)

plt.loglog(np.sqrt(Px*Py),np.mean(Err,axis=1),'ok')

plt.ylabel('mean error (%)')

plt.xlabel('Matrix size $\sqrt{(Nx\cdot Ny)}$')



plt.tight_layout()