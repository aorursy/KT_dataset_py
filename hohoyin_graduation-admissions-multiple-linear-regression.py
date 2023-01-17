import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model

import statsmodels.api as sm

from scipy import stats

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
df.shape
df = df.drop(['Serial No.'], axis=1)
df.isnull().sum()
df.columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA','Research', 'Chance of Admit']
DV = 'Chance of Admit'

IVs = list(df.columns)

IVs.remove('Chance of Admit')
def scatter_plot_with_admit(X,Y,df):

    fig = sns.regplot(x=X, y=Y, data=df)

    plt.title(str(X) + ' vs ' + str(Y))

    plt.show()



def plot_all(IVs, DV, df):

    for IV in IVs:

        scatter_plot_with_admit(IV,DV,df)
plot_all(IVs,DV,df)
corr = df.corr()

fig,ax = plt.subplots(figsize= (6, 6))

sns.heatmap(corr, ax= ax, annot= True,linecolor = 'white')

corr.style.background_gradient(cmap='coolwarm').set_precision(2)

plt.show()
x = df.drop(['Chance of Admit','GRE Score','TOEFL Score','University Rating','SOP'], axis=1)

y = df['Chance of Admit']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, shuffle=False)
mod = sm.OLS(y_train,X_train)

fii = mod.fit()

p_values = fii.summary2().tables[1]['P>|t|']

p_values <0.05
lm = linear_model.LinearRegression()

lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
for IV,coef in zip(X_train.columns,lm.coef_):

    print('coefficient of',IV,'is' ,coef)
lm.intercept_
yhat = lm.predict(X_train)

SS_Residual = sum((y_train-yhat)**2)       

SS_Total = sum((y_train-np.mean(y_train))**2)     

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

print(r_squared, adjusted_r_squared)
print(lm.score(X_train, y_train), 1 - (1-lm.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
sns.residplot(predictions.reshape(-1),y_test, data=df,lowess=True,

                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 1})

plt.xlabel("Fitted values")

plt.title('Residual plot')
residuals = y_test - predictions.reshape(-1)

plt.figure(figsize=(7,7))

stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Normal Q-Q Plot")
print('Mean Squared error:',np.sqrt(mean_squared_error(y_test, predictions)))

print('Mean Absolute error:',mean_absolute_error(y_test, predictions))