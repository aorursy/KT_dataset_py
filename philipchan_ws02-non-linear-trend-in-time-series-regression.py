import pandas as pd #for data processing
import matplotlib.pyplot as plt #for scatterplot
import statsmodels.formula.api as sm #regression package
import numpy as np #for transformations on variables in Model B
dat=pd.DataFrame({"age":[5,7,6,6,5,4,7,6,5,5,1],"price":[80,57,58,55,70,88,43,60,69,63,128]})
dat.head()
plt.scatter(dat.age,dat.price) #the X variable goes along the horizontal axisl; the dependent variable on the vertical axis
plt.xlabel("Age(years)")
plt.ylabel("Price ($00's)")
#plt.legend(loc=2)
plt.show()
#Question 2
mA=sm.ols("price~age", data=dat).fit()
print(mA.summary())
print("The residual standard error is {0:.3f} ".format((mA.mse_resid)**0.5))
dat_pop=pd.read_csv('../input/population.csv')
print(dat_pop.head())
yr=dat_pop.year-1790 # recoding year

plt.scatter(yr,dat_pop.pp)
plt.xlabel("Years (1790=0)")
plt.ylabel("Population (000s)")
ln_pp=np.log(dat_pop.pp) # "log" from math module works on scalar, so not good here

#add log(pop) and yr to dataframe
dat_pop = dat_pop.assign(ln_pp=ln_pp.values, yr=yr.values)
dat_pop.head()
#Fit model on log transformed pop
mB=sm.ols("ln_pp~yr",data=dat_pop).fit()
mB.summary()
ln_pp1990=mB.predict(pd.DataFrame({'yr':[200]})) #predict takes a dataframe
print("Predicted ln(pop) for 1990 is {}".format(ln_pp1990))
print("Predicted population for 1990 is {}".format(np.exp(ln_pp1990)))
#1st get the predictions of price from the model
ln_pphat=mB.predict(dat_pop) #this is log price
pphat=np.exp(ln_pphat) #transform lnprice back to price
#print(pphat)
fig,f=plt.subplots()
f.plot(yr,dat_pop.pp, 'o',label='data')
f.plot(yr,pphat, label="Exponential trend model")
f.legend(loc='best')
f.set_xlabel("year (1790=0)")
f.set_ylabel("Population (000s)")