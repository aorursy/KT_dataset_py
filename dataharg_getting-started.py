# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')

train['Src'] = "train"

test = pd.read_csv('../input/test.csv')

test['Src'] = "test"

comb = pd.concat([train,test],0)



print ("Stage0 shape: {0}".format(comb.shape))

#STEP1: Dummy set encoding of the categorical fields

catfields = []

frames = [comb]



for c in comb.drop(['ID','y','Src'],1).columns:

    if train[c].dtype == 'object':

        catfields.append(c)

        tempdf = pd.get_dummies(comb[c],prefix=c)

        frames.append(tempdf)

        

comb2 = pd.concat(frames,axis=1).drop(catfields, 1)

print ("Stage1 Shape: {0}".format(comb2.shape))



#STEP2: Remove fields with little or no information

#TFPE: Too few posititve entries (proportion)

tfpe = 0.05



problem_fields = []

for c in comb2.drop(['ID','y','Src'],1).columns:

    uniq = len(np.unique(comb2[c]))

    mv = comb2[c].mean()

    if ((uniq == 1) or (mv < tfpe)): problem_fields.append(c)



comb3 = comb2.drop(problem_fields, 1)

print ("Stage3 Shape: {0}".format(comb3.shape))
from sklearn.decomposition import PCA

pca2 = PCA(n_components=35)

pca2_results = pca2.fit_transform(comb3.drop(['ID','y','Src'],1))





eigvals = pca2.explained_variance_



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt





fig = plt.figure(figsize=(8,5))

sing_vals = np.arange(len(eigvals)) + 1

plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)

plt.title('Scree Plot')

plt.xlabel('Principal Component')

plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3, 

 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),

 markerscale=0.4)

leg.get_frame().set_alpha(0.4)

leg.draggable(state=True)

plt.show()



#Stage 4: Apply the PCA results

pcacols = ['ID','y','Src']

for i in range(16):

    cn = 'pca' + str(i)

    comb3[cn]=pca2_results[:,i]

    pcacols.append(cn)



comb4 = comb3[pcacols]

print ("Stage4 Shape: {0}".format(comb4.shape))
"""

Proposed Steps for the regression model

Box Cox model to assess if transformatins required

Build initial model

Calculate the pseudo cooks distance and look for over influential cases

VIF, Although we have orthogonal PCA so should not be a problem

test regression assumptions:

    Linear relationship (look at box cox results).

    Multivariate normality.

    No or little multicollinearity.

    No auto-correlation.

    Homoscedasticity.

Build final model

"""

from scipy import stats

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaled = scaler.fit_transform(comb4[pcacols[-16:]]) + 0.001

scaleddf = pd.DataFrame(scaled)                              



bclst = ['ID','y','Src']

lmbdalst = []

for p in range(16):

    X , ll = stats.boxcox(scaleddf[p])

    lmbdalst.append(ll)

    a = 'pca_bc' + str(p)

    bclst.append(a)

    comb4[a] = X 

print (lmbdalst)

comb5 = comb4[bclst]



print ("Stage5 Shape: {0}".format(comb5.shape))

print (comb5.columns)



x1=comb5['pca_bc14'].head(250)

x2=comb4['pca14'].head(250)

y=comb5['y'].head(250)



fig = plt.figure()

ax1 = fig.add_subplot(121)

ax1.scatter(x1, y)

ax2 = fig.add_subplot(122)

ax2.scatter(x2, y)
#Split back out the training and test set

train = comb5[comb4['Src'] == 'train'].drop('Src',1)

test = comb5[comb4['Src'] == 'test'].drop('Src',1)



print ("Stage5a; Train Shape: {0}, Test Shape: {1}".format(train.shape,test.shape))
#Evaluate an initial model



from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm



#Generate the model string

st = "y ~ "

for f in bclst[-16:]:  st = st + f + " + "

    

#Fit the model

model = ols(st[:-3], train).fit()



#Print the summary

print(model.summary())



print("\nparameter estimates:")

print(model._results.params)

#analysis of variance on fitted linear model

anova_results = anova_lm(model)

print('\nANOVA results')

print(anova_results)

#analysis of outliers

score = model.outlier_test()

outliers = (i for i,t in enumerate(score["bonf(p)"]) if t < 0.6 )

outlierslist = (list(outliers))

print ('\nOutliers: ', len(outlierslist))

print (outlierslist)

#And some regression plots to help us evaluate the model

%matplotlib inline

import statsmodels.api as sm

#Influence Plot

#--------------

fig, ax = plt.subplots(figsize=(12,8))

fig = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")



#This confirms what we saw in above bonf(p) list >> we are happy with out outlier list



#Homoscedasticity

#-------------

yhat = model.fittedvalues

ehat = model.resid

fig, ax1 = plt.subplots(figsize=(12,8))

ax1.scatter(yhat, ehat)



#Shows some evidence of homoscedasticity, but not severe enough to worry about

#More evidence of the outlier;
#Build a Second model.

#Not much work required, outliers mainly



train2 = train.drop(train.index[outlierslist])



#Fit the model

model2 = ols(st[:-3], train2).fit()



#Print the summary

print(model2.summary())



print("\nparameter estimates:")

print(model2._results.params)

#analysis of variance on fitted linear model

anova_results = anova_lm(model2)

print('\nANOVA results')

print(anova_results)

#analysis of outliers

score = model2.outlier_test()

outliers = (i for i,t in enumerate(score["bonf(p)"]) if t < 0.05 )

outlierslist2 = (list(outliers))

print ('\nOutliers: ', len(outlierslist2))

print (outlierslist2)

yhat = model2.fittedvalues

ehat = model2.resid

fig, ax1 = plt.subplots(figsize=(12,8))

ax1.scatter(yhat, ehat)
#Final Build



train3 = train2.drop(train.index[outlierslist2])



#Fit the model

model3 = ols(st[:-3], train3).fit()



#Print the summary

print(model3.summary())



print("\nparameter estimates:")

print(model3._results.params)
#Score up the test set 



y_pred = model3.predict(test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('RH_SUB1_Baseline.csv', index=False)