# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.formula.api as sm

import statsmodels.api as sma



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dfA = pd.read_csv("../input/catshw4/cats-hw4.csv")

dfA = dfA.drop([dfA.columns[0]],axis=1)

dfA.head()
dfA.describe()
dfA['SexNumber'] = dfA['Sex'].map({'F': 0, 'M': 1})

#Model Fit Without interaction

modelA1 = sm.ols(formula="Hwt ~ Bwt+SexNumber",data=dfA).fit()

print(modelA1.summary())
#Model Fit Without interaction

modelA2 = sm.ols(formula="Hwt ~ Bwt*SexNumber",data=dfA).fit()

print(modelA2.summary())

inputTestA = pd.DataFrame()

inputTestA["Bwt"] = [3.4]

inputTestA["SexNumber"] = [0]

print(inputTestA)
outputA1 = modelA1.predict(inputTestA)

print(outputA1)
outputA2 = modelA2.predict(inputTestA)

print(outputA2)
dfB = pd.read_csv("../input/treehw4/trees.csv")

dfB = dfB.drop([dfB.columns[0]],axis=1)

dfB.dropna()

dfB.head()
#main effect only

modelB1 = sm.ols(formula="Volume ~ Girth+Height",data=dfB).fit()

print(modelB1.summary())
#with interaction

modelB1B = sm.ols(formula="Volume ~ Girth*Height",data=dfB).fit()

print(modelB1B.summary())
#log without interaction

modelB2A = sm.ols(formula="Volume ~ np.log(Girth)+np.log(Height)",data=dfB).fit()

print(modelB2A.summary())
#log with interaction

modelB2A = sm.ols(formula="Volume ~ np.log(Girth)*np.log(Height)",data=dfB).fit()

print(modelB2A.summary())
dfC = pd.read_csv("../input/mtcarshw4/mtcars.csv")

dfC = dfC.dropna()

dfC.head()
modelCA = sm.ols(formula="mpg ~ hp*cyl+wt",data=dfC).fit()

print(modelCA.summary())
print("Car1", modelCA.predict(pd.DataFrame({"hp": [100],"cyl": [4], "wt": [2.1]})))

print("Car2", modelCA.predict(pd.DataFrame({"hp": [210],"cyl": [8], "wt": [3.9]})))

print("Car3", modelCA.predict(pd.DataFrame({"hp": [200],"cyl": [6], "wt": [2.9]})))
dfD = pd.read_csv("../input/diabethw4/diabetes.csv")

dfD.isnull().sum()

dfD.count()
dfD = dfD.dropna()

dfD.head()
#Model Intercept only

diaNull = sm.ols(formula="chol ~ 1",data=dfD).fit()

print(diaNull.summary())
#diaFull, Overly complex model with a four-way interaction (and all lower-order terms) among age, gender, weight, and frame; a three-way interaction (and all lower-order terms) among waist, height, and hip; and a main effect for location.

diaFull = sm.ols(formula="chol ~ age*gender*weight*frame+waist*height*hip+location",data=dfD).fit()

print(diaFull.summary())
chol1 = sm.ols(formula="chol ~ 1",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ age",data=dfD).fit()

chol3 = sm.ols(formula="chol ~ gender",data=dfD).fit()

chol4 = sm.ols(formula="chol ~ weight",data=dfD).fit()

chol5 = sm.ols(formula="chol ~ frame",data=dfD).fit()

chol6 = sm.ols(formula="chol ~ waist",data=dfD).fit()

chol7 = sm.ols(formula="chol ~ height",data=dfD).fit()

chol8 = sm.ols(formula="chol ~ hip",data=dfD).fit()

chol9 = sm.ols(formula="chol ~ location",data=dfD).fit()

print(sma.stats.anova_lm(chol1,chol2))

print(sma.stats.anova_lm(chol1,chol3))

print(sma.stats.anova_lm(chol1,chol4))

print(sma.stats.anova_lm(chol1,chol5))

print(sma.stats.anova_lm(chol1,chol6))

print(sma.stats.anova_lm(chol1,chol7))

print(sma.stats.anova_lm(chol1,chol8))

print(sma.stats.anova_lm(chol1,chol9))
chol1 = sm.ols(formula="chol ~ age",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ age+gender",data=dfD).fit()

chol3 = sm.ols(formula="chol ~ age+weight",data=dfD).fit()

chol4 = sm.ols(formula="chol ~ age+frame",data=dfD).fit()

chol5 = sm.ols(formula="chol ~ age+waist",data=dfD).fit()

chol6 = sm.ols(formula="chol ~ age+height",data=dfD).fit()

chol7 = sm.ols(formula="chol ~ age+hip",data=dfD).fit()

chol8 = sm.ols(formula="chol ~ age+location",data=dfD).fit()

print(sma.stats.anova_lm(chol1,chol2))

print(sma.stats.anova_lm(chol1,chol2)["Pr(>F)"][1])

print(sma.stats.anova_lm(chol1,chol3))

print(sma.stats.anova_lm(chol1,chol4))

print(sma.stats.anova_lm(chol1,chol5))

print(sma.stats.anova_lm(chol1,chol6))

print(sma.stats.anova_lm(chol1,chol7))

print(sma.stats.anova_lm(chol1,chol8))
chol1 = sm.ols(formula="chol ~ age+gender",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ age+gender+weight",data=dfD).fit()

chol3 = sm.ols(formula="chol ~ age+gender+frame",data=dfD).fit()

chol4 = sm.ols(formula="chol ~ age+gender+waist",data=dfD).fit()

chol5 = sm.ols(formula="chol ~ age+gender+height",data=dfD).fit()

chol6 = sm.ols(formula="chol ~ age+gender+hip",data=dfD).fit()

chol7 = sm.ols(formula="chol ~ age+gender+location",data=dfD).fit()

print(sma.stats.anova_lm(chol1,chol2))

print(sma.stats.anova_lm(chol1,chol3))

print(sma.stats.anova_lm(chol1,chol4))

print(sma.stats.anova_lm(chol1,chol5))

print(sma.stats.anova_lm(chol1,chol6))

print(sma.stats.anova_lm(chol1,chol7))
chol1 = sm.ols(formula="chol ~ age+gender+weight+frame+waist+height+hip+location",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender+weight+frame+waist+height+hip+location",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age+weight+frame+waist+height+hip+location",data=dfD).fit() #no gender

chol4 = sm.ols(formula="chol ~ age+gender+frame+waist+height+hip+location",data=dfD).fit() #no weight

chol5 = sm.ols(formula="chol ~ age+gender+weight+waist+height+hip+location",data=dfD).fit() #no frame

chol6 = sm.ols(formula="chol ~ age+gender+weight+frame+height+hip+location",data=dfD).fit() #no waist

chol7 = sm.ols(formula="chol ~ age+gender+weight+frame+waist+hip+location",data=dfD).fit() #no height

chol8 = sm.ols(formula="chol ~ age+gender+weight+frame+waist+height+location",data=dfD).fit() #no hip

chol9 = sm.ols(formula="chol ~ age+gender+weight+frame+waist+height+hip",data=dfD).fit() #no location

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))

print(sma.stats.anova_lm(chol4, chol1))

print(sma.stats.anova_lm(chol5, chol1))

print(sma.stats.anova_lm(chol6, chol1))

print(sma.stats.anova_lm(chol7, chol1))

print(sma.stats.anova_lm(chol8, chol1))

print(sma.stats.anova_lm(chol9, chol1))
chol1 = sm.ols(formula="chol ~ age+gender+frame+waist+height+hip+location",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender+frame+waist+height+hip+location",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age+frame+waist+height+hip+location",data=dfD).fit() #no gender

chol4 = sm.ols(formula="chol ~ age+gender+waist+height+hip+location",data=dfD).fit() #no frame

chol5 = sm.ols(formula="chol ~ age+gender+frame+height+hip+location",data=dfD).fit() #no waist

chol6 = sm.ols(formula="chol ~ age+gender+frame+waist+hip+location",data=dfD).fit() #no height

chol7 = sm.ols(formula="chol ~ age+gender+frame+waist+height+location",data=dfD).fit() #no hip

chol8 = sm.ols(formula="chol ~ age+gender+frame+waist+height+hip",data=dfD).fit() #no location

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))

print(sma.stats.anova_lm(chol4, chol1))

print(sma.stats.anova_lm(chol5, chol1))

print(sma.stats.anova_lm(chol6, chol1))

print(sma.stats.anova_lm(chol7, chol1))

print(sma.stats.anova_lm(chol8, chol1))
chol1 = sm.ols(formula="chol ~ age+gender+frame+waist+hip+location",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender+frame+waist+hip+location",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age+frame+waist+hip+location",data=dfD).fit() #no gender

chol4 = sm.ols(formula="chol ~ age+gender+waist+hip+location",data=dfD).fit() #no frame

chol5 = sm.ols(formula="chol ~ age+gender+frame+hip+location",data=dfD).fit() #no waist

chol6 = sm.ols(formula="chol ~ age+gender+frame+waist+location",data=dfD).fit() #no hip

chol7 = sm.ols(formula="chol ~ age+gender+frame+waist+hip",data=dfD).fit() #no location

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))

print(sma.stats.anova_lm(chol4, chol1))

print(sma.stats.anova_lm(chol5, chol1))

print(sma.stats.anova_lm(chol6, chol1))

print(sma.stats.anova_lm(chol7, chol1))
chol1 = sm.ols(formula="chol ~ age+gender+frame+waist+hip",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender+frame+waist+hip",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age+frame+waist+hip",data=dfD).fit() #no gender

chol4 = sm.ols(formula="chol ~ age+gender+waist+hip",data=dfD).fit() #no frame

chol5 = sm.ols(formula="chol ~ age+gender+frame+hip",data=dfD).fit() #no waist

chol6 = sm.ols(formula="chol ~ age+gender+frame+waist",data=dfD).fit() #no hip

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))

print(sma.stats.anova_lm(chol4, chol1))

print(sma.stats.anova_lm(chol5, chol1))

print(sma.stats.anova_lm(chol6, chol1))

chol1 = sm.ols(formula="chol ~ age+gender+waist+hip",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender+waist+hip",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age+waist+hip",data=dfD).fit() #no gender

chol4 = sm.ols(formula="chol ~ age+gender+hip",data=dfD).fit() #no waist

chol5 = sm.ols(formula="chol ~ age+gender+waist",data=dfD).fit() #no hip

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))

print(sma.stats.anova_lm(chol4, chol1))

print(sma.stats.anova_lm(chol5, chol1))
chol1 = sm.ols(formula="chol ~ age+gender+hip",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender+hip",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age+hip",data=dfD).fit() #no gender

chol4 = sm.ols(formula="chol ~ age+gender",data=dfD).fit() #no hip

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))

print(sma.stats.anova_lm(chol4, chol1))

chol1 = sm.ols(formula="chol ~ age+gender",data=dfD).fit()

chol2 = sm.ols(formula="chol ~ gender",data=dfD).fit() #no age

chol3 = sm.ols(formula="chol ~ age",data=dfD).fit() #no gender

print(sma.stats.anova_lm(chol2, chol1))

print(sma.stats.anova_lm(chol3, chol1))