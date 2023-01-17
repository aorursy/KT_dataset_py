# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_csv(os.path.join(dirname, filename))
        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.head()
df.describe()
df.info()
df.columns
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
%matplotlib inline

cols_interest= ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Chance of Admit ']

#df = pd.DataFrame(np.random.randn(50, 7), columns=list('ABCDEFG'))

# initiate empty dataframe
corr = pd.DataFrame()
for a in cols_interest:
    for b in cols_interest:
        corr.loc[a, b] = df.corr().loc[a, b]

corr

sns.heatmap(corr)
df=df[cols_interest]

fig = plt.figure(figsize=(10,10))
fig.tight_layout(pad = 3.0)
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(423)
ax4 = fig.add_subplot(424)
ax5 = fig.add_subplot(425)
ax6 = fig.add_subplot(426)
ax7 = fig.add_subplot(427)

ax1.scatter(df['Chance of Admit '], df['GRE Score'], color='red')
#ax1.title.set_text('Chance of Admit Vs Gre Score')
ax1.set_xlabel('COA')
ax1.set_ylabel('GRE')

ax2.scatter(df['Chance of Admit '], df['TOEFL Score'], color='red')
#ax2.title.set_text('Chance of Admit Vs TOEFL')
ax2.set_xlabel('COA')
ax2.set_ylabel('Toefl')

ax3.scatter(df['Chance of Admit '], df['University Rating'], color='red')
#ax3.title.set_text('Chance of Admit Vs Ranking')
ax3.set_xlabel('COA')
ax3.set_ylabel('ranking')

ax4.scatter(df['Chance of Admit '], df['SOP'], color='red')
#ax4.title.set_text('Chance of Admit Vs SOP')
ax4.set_xlabel('COA')
ax4.set_ylabel('sop')

ax5.scatter(df['Chance of Admit '], df['LOR '], color='red')
#ax5.title.set_text('Chance of Admit Vs LOR')
ax5.set_xlabel('COA')
ax5.set_ylabel('lor')

ax6.scatter(df['Chance of Admit '], df['CGPA'], color='red')
#ax6.title.set_text('Chance of Admit Vs CGPA')
ax6.set_xlabel('COA')
ax6.set_ylabel('cgpa')

ax7.scatter(df['Chance of Admit '], df['Research'], color='red')
#ax7.title.set_text('Chance of Admit Vs Research')
ax7.set_xlabel('COA')
ax7.set_ylabel('research')

fig.subplots_adjust(top=0.92, bottom=0.04, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]
Y = df["Chance of Admit "]

from sklearn.model_selection import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred= regressor.predict(X_test)

print('Intercept: \n', regressor.intercept_)
#print('Coefficients: \n', regressor.coef_)
print("Coeffecients:")
print(list(zip(X.columns,regressor.coef_ )))

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
#Removing SOP as it has the lowest coeffecient and building the model again
cols_interest = ['GRE Score',  
       'LOR ', 'CGPA', 'Research']
X = df[['GRE Score',  
       'LOR ', 'CGPA', 'Research']]
Y = df["Chance of Admit "]

from sklearn.model_selection import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred= regressor.predict(X_test)

print('Intercept: \n', regressor.intercept_)
#print('Coefficients: \n', regressor.coef_)
print("Coeffecients:")
print(list(zip(cols_interest,regressor.coef_ )))

from sklearn import metrics
print("")
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# The above RMSE scores shows a decrease due to the removal of SOP variable from the list of columns, so we will not remove it and since RMSE is 0.063 which is low 
# I will consider this as my final model.
#But I will apply statsmodel on the data to see if my model is making the right predictions

# with statsmodels

import statsmodels.api as sm
X_train = sm.add_constant(X_train) # adding a constant
 
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_train) 
 
print_model = model.summary()
print(print_model)


GRE_Score = 325
Toefl = 115
Ranking = 10
SOP=4.0
LOR= 4.0 
CGPA =8.5
Research=1


print ('Predicted Stock Index Price: \n', regressor.predict([[GRE_Score ,Toefl,Ranking,SOP,LOR,CGPA,Research]])[0]*100)

