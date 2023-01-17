import pandas as pd

import pylab as pl

import seaborn as sns

from scipy.stats import mode



def analyzer(n, Seed, Res, Plot):

    pl.seed(Seed)

    x = pl.arange(n)

    y = pl.zeros(n)  # let's make up our signal

    y += pl.sin(pl.pi * x / 100)  # slow oscillation

    y += pl.sin(pl.pi * x / 5)  # fast oscillation

    y += .01*x  # some trend

    y += .5*pl.cumsum(pl.randn(n))  # some randomness by randomwalk

    y += pl.exp(pl.randn(n))*pl.randn(n)  # some outliers



    Avg = pl.mean(y)

    Med = pl.median(y)

    bins = pl.linspace(y.min(), y.max(), 20)

    m = mode(pl.digitize(y, bins))

    Mod = bins[m[0]]-(bins[1]-bins[0])/2



    Res = Res.append(pd.DataFrame([[Seed,'Avg',((y-Avg)**2).mean(),abs(y-Avg).mean()]], columns=ResCols))

    Res = Res.append(pd.DataFrame([[Seed,'Med',((y-Med)**2).mean(),abs(y-Med).mean()]], columns=ResCols))

    Res = Res.append(pd.DataFrame([[Seed,'Mod',((y-Mod)**2).mean(),abs(y-Mod).mean()]], columns=ResCols))



    if Plot:

        pl.figure(figsize=(10,10))

        pl.subplot(2,1,1)

        pl.hist(y)

        pl.xlabel('y values')

        pl.ylabel('histogram')



        pl.subplot(2,1,2)

        pl.plot(x, y)

        pl.plot(x[[0,-1]], [Avg, Avg])

        pl.plot(x[[0,-1]], [Med, Med])

        pl.plot(x[[0,-1]], [Mod, Mod])

        pl.legend(('data','Mean','Median','Mode'))

        pl.xlabel('samples')

        pl.ylabel('y values')



    return Res





n = 500

Seed = 1

ResCols = ('Seed','Estimate','MSE','MAE')

Res = pd.DataFrame([],columns=ResCols)

Res = analyzer(n, Seed, Res, Plot=True)

print(Res)

Res = pd.DataFrame([],columns=ResCols)

for Seed in range(100):

    Res = analyzer(n, Seed, Res, Plot=False)



Res2=Res.pivot(columns='Estimate',index='Seed')

print(all(Res2[('MAE','Med')]<=Res2[('MAE','Avg')]))

print(all(Res2[('MSE','Med')]>=Res2[('MSE','Avg')]))

pl.figure(figsize=(10,10))

ax=pl.subplot(2,2,1)

sns.violinplot(x='Estimate', y='MSE', data=Res, ax=ax)

sns.pointplot(x='Estimate', y='MSE', data=Res, ax=ax, color='y', markers=".")



ax=pl.subplot(2,2,2)

sns.violinplot(x='Estimate', y='MAE', data=Res, ax=ax)

sns.pointplot(x='Estimate', y='MAE', data=Res, ax=ax, color='y', markers=".")



pl.subplot(2,2,3)

pl.plot(Res2[('MSE','Avg')],Res2[('MSE','Med')], '.')

pl.plot([0,Res.MSE.max()],[0,Res.MSE.max()],'r')

pl.xlabel('Median')

pl.ylabel('Average')

pl.title('MSE')



pl.subplot(2,2,4)

pl.plot(Res2[('MAE','Avg')],Res2[('MAE','Med')], '.')

pl.plot([0,Res.MAE.max()],[0,Res.MAE.max()],'r')

pl.xlabel('Median')

pl.ylabel('Average')

pl.title('MAE')
