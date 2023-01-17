# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
ad = pd.read_csv("../input/Advertising.csv", index_col=0)
ad.info()
ad.describe()
ad.head()
%matplotlib inline



import matplotlib.pyplot as plt



plt.scatter(ad.TV, ad.Sales, color='blue', label="TV")

plt.scatter(ad.Radio, ad.Sales, color='green', label='Radio')

plt.scatter(ad.Newspaper, ad.Sales, color='red', label='Newspaper')



plt.legend(loc="lower right")

plt.title("Sales vs. Advertising")

plt.xlabel("Advertising [1000 $]")

plt.ylabel("Sales [Thousands of units]")

plt.grid()

plt.show()
ad.corr()
plt.imshow(ad.corr(), cmap=plt.cm.Blues, interpolation='nearest')

plt.colorbar()

tick_marks = [i for i in range(len(ad.columns))]

plt.xticks(tick_marks, ad.columns, rotation='vertical')

plt.yticks(tick_marks, ad.columns)
import statsmodels.formula.api as sm
modelAll = sm.ols('Sales ~ TV + Radio + Newspaper', ad).fit()

modelAll.params
  # we need first to calculate the Residual Sum of Squares (RSS)

y_pred = modelAll.predict(ad)

import numpy as np

RSS = np.sum((y_pred - ad.Sales)**2)

RSS
y_mean = np.mean(ad.Sales) # mean of sales

TSS = np.sum((ad.Sales - y_mean)**2)

TSS
p=3 # we have three predictors: TV, Radio and Newspaper

n=200 # we have 200 data points (input samples)



F = ((TSS-RSS)/p) / (RSS/(n-p-1))

F
RSE = np.sqrt((1/(n-2))*RSS); 

RSE
np.mean(ad.Sales)
R2 = 1 - RSS/TSS; 

R2
modelAll.summary()
def evaluateModel (model):

    print("RSS = ", ((ad.Sales - model.predict())**2).sum())

    print("R2 = ", model.rsquared)
modelTV = sm.ols('Sales ~ TV', ad).fit()

modelTV.summary().tables[1]
evaluateModel(modelTV)
modelRadio = sm.ols('Sales ~ Radio', ad).fit()

modelRadio.summary().tables[1]
evaluateModel(modelRadio)
modelPaper = sm.ols('Sales ~ Newspaper', ad).fit()

modelPaper.summary().tables[1]
evaluateModel(modelPaper)
modelTVRadio = sm.ols('Sales ~ TV + Radio', ad).fit()

modelTVRadio.summary().tables[1]
evaluateModel(modelTVRadio)
modelTVPaper = sm.ols('Sales ~ TV + Newspaper', ad).fit()

modelTVPaper.summary().tables[1]
evaluateModel(modelTVPaper)
evaluateModel(modelAll)
modelTVRadio.summary()
modelTVRadio.params
normal = np.array([0.19,0.05,-1])

point  = np.array([-15.26,0,0])

# a plane is a*x + b*y +c*z + d = 0

# [a,b,c] is the normal. Thus, we have to calculate

# d and we're set

d = -np.sum(point*normal) # dot product

# create x,y

x, y = np.meshgrid(range(50), range(300))

# calculate corresponding z

z = (-normal[0]*x - normal[1]*y - d)*1./normal[2]
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

fig.suptitle('Regression: Sales ~ Radio + TV Advertising')

ax = Axes3D(fig)



ax.set_xlabel('Radio')

ax.set_ylabel('TV')

ax.set_zlabel('Sales')

ax.scatter(ad.Radio, ad.TV, ad.Sales, c='red')



ax.plot_surface(x,y,z, color='cyan', alpha=0.3)
modelSynergy = sm.ols('Sales ~ TV + Radio + TV*Radio', ad).fit()

modelSynergy.summary().tables[1]
evaluateModel(modelSynergy)
modelSynergy.params