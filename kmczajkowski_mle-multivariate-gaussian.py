import numpy as np # linear algebra

import matplotlib.pyplot as plt

import itertools as it

#Normal_PDF=@(x,ep)exp( sum(-0.5*(abs(x)./max(std(x,1),ep).^2) ))  .*   (2*pi*max(std(x,1),ep).^2)^(-length(x)/2));

#SpCr.Crit_ML=@(SpVals)  Normal_PDF(  SpCr.Iout - abs(SpCr.Ugr+SpF.fft(SpVals)).^2 , 1e-5 *max(SpCr.Iout))   ;

def normal_pdf(x,mu,sig):

    N=x.shape[0]

    diff=(x-mu)

    prod=np.dot(diff,diff)

    var=sig**2   

    return np.exp(-0.5*prod/var)/((2*np.pi)**N*var)**0.5

def crit_ml(param,mu,sig):

    N=mu.shape[0]

    x=generate_signal(param,N)

    return 1-normal_pdf(x,mu,sig)

def crit_simple(param,mu):

    N=mu.shape[0]

    x=generate_signal(param,N)

    return np.sum(np.abs(x-mu)**2)
N=10

tst=[1,2]

corr=[3,4]

def generate_signal(vals,N=10):

    sig=np.zeros((N))

    sig[vals]=1

    return sig



signal=generate_signal(corr,N)

sigma_noise=0.6

noisy_signal=signal+sigma_noise*np.random.randn(N)

test_signal=generate_signal(tst,N)

plt.plot(signal)

plt.plot(noisy_signal,'--')

plt.plot(test_signal,'o')
likelihood=np.zeros((N,N))

simple=np.zeros((N,N))

for ii in range(N):

    for jj in range(N):

        likelihood[ii,jj]=crit_ml([ii,jj],noisy_signal,sigma_noise)

        simple[ii,jj]=crit_simple([ii,jj],noisy_signal)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.imshow(np.log(likelihood))

plt.plot(corr[0],corr[1],'ko')

plt.colorbar()

plt.subplot(1,2,2)

plt.imshow(np.log(simple))

plt.plot(corr[0],corr[1],'ko')

plt.colorbar()

plt.tight_layout()