#IMPORTING REQUIRED MODULES

import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.float_format', lambda x: '%.6f' % x)

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from IPython.display import Image
df = pd.read_csv('../input/train.csv')

df.columns = [x.lower() for x in df.columns]



#MINI FEATURE ENGINEERING

#transforming gender into numerical values

df['sex'].replace(['female','male'], [1,0],inplace=True)

#replacing nan age values by mean age

df.loc[np.isnan(df['age']), 'age'] = df['age'].mean()
Image(url= "http://www.tikalon.com/blog/2011/sigmoid.gif")
y = 1 - (1 / (1 + np.exp(0 + (1*2))))

round(y,2)
ind_vars = ['fare', 'age', 'sex']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(df[ind_vars], df['survived'], test_size=0.33)
logreg = LogisticRegression(solver='lbfgs')

logreg.fit(Xtrain[['fare']], Ytrain)

coef = logreg.coef_[0][0]

intercept = logreg.intercept_[0]

'intercept : ', round(intercept, 2), 'coef : ', round(coef, 2),
x = Xtrain[['fare']].values[0][0]

predicted_proba_y = 1- (1 / (1 + np.exp(intercept + (coef * x))))

'fare :', x, 'probability of surviving :', round(predicted_proba_y, 2)
Yprobas = logreg.predict_proba(Xtrain[['fare']])[:,1]

Xtrain['probas'] = Yprobas

Xtrain[['fare', 'probas']].head()
sns.scatterplot(Xtrain['fare'], Ytrain)

sns.lineplot('fare', 'probas', data=Xtrain.sort_values('fare'));
# y = 1 / (1 + np.exp(intercept + (coef * x)))

# 0.5 = 1 / (1 + np.exp(intercept + (coef * x)))

# 1 + np.exp(intercept + (coef * x)) = 1 / 0.5 

# np.exp(intercept + (coef * x)) = 1

# intercept + (coef * x) = np.log(1)

# coef * x = 0 - intercept

# x = - intercept / coef

x = - intercept / coef

round(x, 2)
sns.scatterplot(Xtrain['fare'], Ytrain)

sns.lineplot('fare', 'probas', data=Xtrain.sort_values('fare'));

plt.axvline(x, 0, 1, color='black');
Ypredicted = logreg.predict(Xtrain[['fare']])

Xtrain['prediction'] = Ypredicted

Xtrain[['fare', 'prediction']].head(20)
'accuracy', round(logreg.score(Xtrain[['fare']], Ytrain), 2)
ind_vars = ['sex', 'age', 'fare']

logreg.fit(Xtrain[ind_vars], Ytrain)

coefs = pd.DataFrame(logreg.coef_).transpose()

coefs.index = ind_vars

coefs.columns = ['coef']

coefs.abs().sort_values('coef', ascending=False)