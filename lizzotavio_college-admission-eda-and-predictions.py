import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('seaborn-darkgrid')
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
max_nan = df.isnull().sum().max() # no data is missing

print('NaN values in our dataset: ' + str(max_nan))

print('\n')

sns.heatmap(abs(df.isnull()), cmap='viridis')
# Creating a new column just for EDA



# Assuming -> Chance of Admit >= 75% (Probably admitted)

#             Chance of Admit < 75% (Probably recused) 



df['probably_admitted'] = df['Chance of Admit '].apply(lambda x: 1 if x >= 0.75 else 0)
#Dataframes for continuous and discrete data

df_cont = df[['GRE Score', 'TOEFL Score', 'CGPA','Chance of Admit ']]

df_disc = df[['University Rating','SOP','LOR ', 'probably_admitted']]
for i in np.arange(0, len(df_cont.columns), 2):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))

    

    ax1.set_title('Distribution for %s' % (df_cont.columns[i]))    

    ax1 = sns.distplot(df_cont[df_cont.columns[i]],bins=30, kde=False, ax=ax1)

    

    ax2.set_title('Distribution for %s' % (df_cont.columns[i+1]))    

    ax2 = sns.distplot(df_cont[df_cont.columns[i+1]],bins=30, kde=False, ax=ax2)
for i in np.arange(0, len(df_disc.columns), 2):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))

   

    ax1.set_title('Distribution for %s' % (df_disc.columns[i]))    

    ax1 = sns.countplot(x=df_disc.columns[i], data=df_disc, ax=ax1)

    

    ax2.set_title('Distribution for %s' % (df_disc.columns[i+1]))    

    ax2 = sns.countplot(x=df_disc.columns[i+1],data=df_disc, ax=ax2)
fig, (ax1) = plt.subplots(figsize=(9,5))

ax1 = sns.heatmap(df.drop(['Serial No.'], axis=1).corr(), linewidths=0.5, square=True, cmap='viridis')
pd.DataFrame(df.drop(['Serial No.', 'probably_admitted'], axis=1).corr()['Chance of Admit '].sort_values(ascending=False)[1:])
from sklearn.model_selection import train_test_split
X = df.drop(['Serial No.', 'probably_admitted','Chance of Admit '], axis=1)

y = df['Chance of Admit '].values

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import cross_val_score # module import for model validation and comparison
lm = LinearRegression()

lm.fit(X_train, y_train)



print('Negative Mean Absolute Error for Linear Regression:')

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
alpha = []

error = []



for i in range(1, 100):

    alpha.append(i/2000)

    lml = Lasso(alpha=(i/2000))

    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

    

plt.grid(True)

plt.title('Lasso Regression Prediction Score')

plt.ylabel('Error')

plt.xlabel('Alpha')

plt.plot(alpha, error, label='Neg Mean Absolute Error')

plt.legend()
index = error.index(max(error))

best_alpha = alpha[index]



lml = Lasso(alpha=best_alpha)

lml.fit(X_train, y_train)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf.fit(X_train, y_train)

print('Negative Mean Absolute Error for Random Forest Regression:')

np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
tpred_lm = lm.predict(X_test)

tpred_lml = lml.predict(X_test)

tpred_rf = rf.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae_lm = '{:2.2f}'.format(mean_absolute_error(y_test, tpred_lm)*100) + '%'

mae_lml = '{:2.2f}'.format(mean_absolute_error(y_test, tpred_lml)*100) + '%'

mae_rf = '{:2.2f}'.format(mean_absolute_error(y_test, tpred_rf)*100) + '%'



print('Mean Absolute Error for Each Regression Type:')

print('\n')

pd.DataFrame({'Linear':[mae_lm], 'Lasso':[mae_lml], 'RdnForest':[mae_rf] })