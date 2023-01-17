import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline

sns.set(style="white",palette='deep', font_scale=1.5, color_codes=True)

plt.rcParams['font.family'] = 'IPAPGothic'



plt.scatter(100,0.2,c="b",s=100)

plt.scatter(200,0.35,c="r",s=100)

plt.xlim(0,210)

plt.ylim(0,0.52)

plt.xlabel("試行回数")

plt.ylabel("表の出る割合")

plt.tight_layout()

plt.hlines(y=0.5,xmin=0,xmax=300, linestyles='dashed',colors='b', linewidths=1)

plt.hlines(y=0.35,xmin=0,xmax=300, linestyles='dashed',colors='r', linewidths=1)

plt.savefig("zu1")
x=np.array([10**2,10**3,10**4,10**5])

samples = np.random.binomial(x,0.5)

means=(samples+20)/(x+100)

plt.scatter(100,0.2,s=100,label="100")

for i in range(0,4):

    plt.scatter(x[i]+100, means[i],s=100,c="r",label=x[i]+100)

    plt.vlines(x+100,ymin=0,ymax=0.5,linestyles='dashed',colors='b', linewidths=1)

    

plt.hlines(y=0.5,xmin=0,xmax=10**6,linestyles='dashed',colors='b', linewidths=1)

plt.ylim(0,0.52)

plt.xscale("log")

plt.xticks([200,1100,10100,100100],["200","1100","10100","100100"])

plt.legend(loc="lower right")

plt.title("101回目からの試行をやってみた図")

plt.xlabel("試行回数")

plt.ylabel("表の出る割合")

plt.tight_layout()

plt.savefig("zu2")