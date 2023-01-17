import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as st

from scipy.interpolate import interp1d

import pylab
pylab.rcParams['figure.figsize'] = (9, 6.0)
xs=np.linspace(-5,10,200)

ks= np.arange(50)
pmf_binom=st.binom.pmf(ks,50,0.25) # Modelling a Binomial distribution with probabilty of success = 25%

plt.bar(ks,pmf_binom,label="Binomial Distribution",alpha=0.8) 

pmf_poisson=st.poisson.pmf(ks,30) # Modelling a poisson distribution with average 30 (say, average support calls received every night)

plt.bar(ks,pmf_poisson,label="Poisson Distribution",alpha=0.8)

plt.xlabel("K value")

plt.ylabel("Probability")

plt.legend();


print(f"Probability of getting 10 successes: {st.binom.pmf(10,50,0.25):.3f} ")

print(f"Probability of getting upto 15 successes : {st.binom.cdf(15,50,0.25):.3f} ")

print(f"Probability of getting 45 calls in a night: {st.poisson.pmf(45,30):.4f}")

print(f"Probability of getting atleast 40 calls in a night: {st.poisson.sf(40,30):.4f}") #It's same as 1-CDF

unif_pdf=st.uniform.pdf(xs,-4,10)

plt.plot(xs,unif_pdf,label="Uniform(-4,10)",alpha=0.8,ls="--")

norm_pdf=st.norm.pdf(xs,5,2)

plt.plot(xs,norm_pdf,label="Normal(5,2)",alpha=0.8,ls="--")

exp_pdf=st.expon.pdf(xs,2)

plt.plot(xs,exp_pdf,label="Exponential(0.5)",alpha=0.8,ls="--")

st_pdf=st.t.pdf(xs,1)

plt.plot(xs,st_pdf,label="T (1)",alpha=0.8,ls="--")

pdf_lognorm=st.lognorm.pdf(xs,1)

plt.plot(xs,pdf_lognorm,label="Lognorm (1)",alpha=0.8,ls="--")



plt.legend()

plt.xlabel("xs")

plt.ylabel("Probability");
# Let's define some random xs and ys

xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 

      5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

ys = [0.2, 0.165, 0.167, 0.166, 0.154, 0.134, 0.117, 

      0.108, 0.092, 0.06, 0.031, 0.028, 0.048, 0.077, 

      0.103, 0.119, 0.119, 0.103, 0.074, 0.038, 0.003]
plt.scatter(xs,ys,label="Data")

plt.xlabel("Xs")

plt.ylabel("Observed PDF")

plt.title("Random Distribution");
x1=np.linspace(min(xs),max(xs),1000) # Defining an array of xs

y1=interp1d(xs,ys)(x1) # Interpolate function; Linear

y2=interp1d(xs,ys,kind="quadratic")(x1) #Quadratic
plt.plot(x1,y1,label="Linear")

plt.plot(x1,y2,label="Quadratic")

plt.scatter(xs,ys,c='black',s=30,label="Original")

plt.xlabel("Xs")

plt.ylabel("Observed PDF")

plt.title("Emperical Distribution");

plt.legend();
from scipy.integrate import simps



def get_pdf(x,y,a,b,res=1000):

    x_norm=np.linspace(min(x),max(x),res)

    y_norm=interp1d(x,y,kind="quadratic")(x_norm)

    normalization=simps(y_norm,x_norm)

    

    x1=np.linspace(a,b,res)

    y1=interp1d(x,y,kind="quadratic")(x1)

    pdf=simps(y1,x=x1)/normalization

    return pdf



def get_cdf(x,y,v):

    return get_pdf(x,y,min(x),v)

    

def get_sf(x,y,v):

    return 1- get_cdf(x,y,v)
def plot_pdf(x,y,a,b):

    x_interpolate=np.linspace(min(x),max(x),1000)

    y_interpolate=interp1d(x,y,kind="quadratic")(x_interpolate)

    plt.plot(x_interpolate,y_interpolate,label="Interpolation")

    plt.scatter(x,y,s=20,label="Original")

    z=plt.fill_between(x1,0,y_interpolate,where=(x1>=a) & (x1<=b),alpha=0.5)

    plt.annotate(f"p = {get_pdf(x,y,a,b):.3f}",(7,0.05))

    plt.xlabel("Values of Xs")

    plt.ylabel("Probability")

    plt.title("PDF of a Non-Parametrized Distribution")

    plt.legend()
print(f"Probability of getting a value between 6 and 9 is: {np.round(get_cdf(xs,ys,v=9)-get_cdf(xs,ys,v=6),3)}")

plot_pdf(xs,ys,a=6,b=9)
x1=np.linspace(min(xs),max(xs),1000)

y2=interp1d(xs,ys,kind="quadratic")(x1)

cdf=y2.cumsum()/y2.sum()

plt.plot(x1,cdf)

plt.xlabel("Values of xs")

plt.ylabel("Probability of X<=x")

plt.title("CDF of a Non-Parametrized Distribution");
plt.hist(st.norm.rvs(10,2,1000),bins=50);  # Random Sampling from a normal distribution. There arrays show the X and Y values

plt.xlabel("Xs")

plt.ylabel("PDF Outcome");

print(f"10 samples from a Normal distribution with mean 10 and sd 2:\n {st.norm.rvs(10,2,10)}")

samples=np.ceil(st.uniform.rvs(0,6,(1000,3))).sum(axis=1)

print(f"Number of times the rolled sum was greater than 16 is {np.sum(samples>16)}")

plt.hist(samples,bins=50);

plt.xlabel("Sum of three dice")

plt.ylabel("frequency")

plt.title("Sum of three dice in 1000 rolls");
xs=np.linspace(0,4,200) #Defining p(x)

def pdf(xs):

    return(np.sin(xs**2)+1)



px=pdf(xs)
plt.plot(xs,px,label="Actual curve")

plt.fill_between(xs,0,px,alpha=0.1)

plt.xlim(0,4)

plt.ylim(0,2)

plt.xlabel("Values of Xs")

plt.ylabel("P(x)");
random_x=st.uniform.rvs(0,4,100)

random_y=st.uniform.rvs(0,4,100)

plt.scatter(random_x,random_y,label="Random samples")

plt.plot(xs,px,label="Actual curve")

plt.fill_between(xs,0,px,alpha=0.1)

plt.xlim(0,4)

plt.ylim(0,2)

plt.xlabel("Values of Xs")

plt.ylabel("P(x)")

plt.title("Sampling from uniform distribution")

plt.legend(loc=2);
passed=random_y<pdf(random_x)

plt.scatter(random_x[passed],random_y[passed],label="Picked",c='g',s=20)

plt.scatter(random_x[~passed],random_y[~passed],label="rejected",c='r',s=20,marker='x')

plt.plot(xs,px,label="Distribution",ls="--")

plt.fill_between(xs,0,px,alpha=0.1)

plt.xlim(0,4)

plt.ylim(0,2)

plt.legend(loc=2)

plt.xlabel("Values of Xs")

plt.ylabel("P(x)")

plt.title("Rejecting the samples lying beyond the p(x) region");
n2=100000

x=st.uniform.rvs(scale=4,size=n2) #Scale is just the parameter of uniform dist(b-a)

y=st.uniform.rvs(scale=2,size=n2) #All the values 

x_final=x[y<=pdf(x)]  #Accepted values

print(f"Number of points selected: {len(x_final)}")

integral=simps(px,x=xs)

plt.plot(xs,px/integral,label="Actual curve")

plt.fill_between(xs,0,px/integral,alpha=0.1) #to make this an actual PDF between 0 and 1

plt.hist(x_final,density=True,histtype='step',bins=100,label='Sample PDF')

plt.legend()

plt.xlim(0,4)

plt.ylim(0,0.5)

plt.xlabel("Values of Xs")

plt.ylabel("Probability");