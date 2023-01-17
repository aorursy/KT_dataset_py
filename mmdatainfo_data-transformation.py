import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
%config InlineBackend.figure_format = "svg"
x0, y0 = np.meshgrid(np.linspace(0.0,200.0,num=30), np.linspace(-100,100,num=20))

# Convert to feature matrix

X = np.concatenate((np.asanyarray(x0).reshape(x0.size,1),

                    np.asanyarray(y0).reshape(y0.size,1)),axis=1);

y = 10.0+3.0*X[:,0]-2.0*X[:,1]+0.1*(X[:,0]**2)+0.4*(X[:,1]**2);

np.random.seed(123);

y[0:20] = y[0:20] + np.random.rand(20)*1e+4; # insert outlier

np.random.shuffle(y)

y = y.reshape(y.size,1)
def plotscaled(vec,i,ncol=2,nrow=6,title=""):

    plt.subplot(nrow,ncol,i);i+=1;

    plt.hist(vec,24)

    plt.legend([title])

    plt.subplot(nrow,ncol,i);

    plt.plot(vec,"k.");

    y_mean,y_med,y_std,y_min,y_max = np.mean(vec),np.median(vec),np.std(vec),np.min(vec),np.max(vec);

    plt.plot([0,vec.size],[y_mean,y_mean],"r-")

    plt.plot([0,vec.size],[y_med,y_med],"g--")

    plt.plot([0,vec.size],[y_mean-y_std*2,y_mean-y_std*2],"b-")

    plt.plot([0,vec.size],[y_mean+y_std*2,y_mean+y_std*2],"b-")

    if i == 2:

        plt.legend(["data","mean","median","2*$\sigma$"],ncol=4,fontsize=9);

    return i+1
plt.figure(figsize=(9.5,9));

i = 1;

for m in [StandardScaler(),RobustScaler(),MinMaxScaler(),

          PowerTransformer(),QuantileTransformer(output_distribution="normal")]:

    y_scaled = m.fit_transform(y);

    i = plotscaled(y_scaled,i,title=str(m).split("(")[0])
from scipy.stats import skewnorm
def plotskew(s):

    y = skewnorm.rvs(s, size=1000);

    plt.figure(figsize=(8,5));

    plt.subplot(2,2,1);

    plt.hist(y,20);

    if s > 0:

        plt.title("Positive skew");

    else:

        plt.title("Negative skew")

    plt.subplot(2,2,2);

    return y
np.random.seed(1)

y = plotskew(4)

plt.hist(np.log(y-np.min(y)+1),20);

plt.title("log-transform: log(const+y)");

y = plotskew(-4)

plt.hist((y-np.min(y))**2,20);

plt.title("square power-transform: (const+y)^2");