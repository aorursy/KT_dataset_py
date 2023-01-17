# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
# Reading the dataset 

df = pd.read_csv('../input/covid-aggregated-data/Covid.csv')

df = df.iloc[:,:-2]
df.head()
df.info()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),cmap='coolwarm',annot =True)

plt.title('Heatmap of Correlation Matrix')
plt.figure(figsize=(10,5))

sns.lineplot(data = df[['TotalConfirmedCases','TotalDeaths','TotalClosedCases']],size=1000)

plt.title('Trends in Confirmed, Deaths and Closed Cases')

plt.xlabel('No of days after first case')

plt.ylabel('Total Cases')
plt.figure(figsize=(10,5))

sns.lineplot(data = df[['TotalSamples']],style='event',size=1000,palette='CMRmap_r')

plt.title('Total Samples recorded')

plt.xlabel('No of days after first case')

plt.ylabel('Total Cases')
dataset = pd.read_csv('../input/covid-aggregated-data/Covid.csv')

#dataset = dataset.iloc[:,:-3]

dataset.dropna(axis=0,inplace=True)

X = dataset.iloc[:, 0:2].values

y = dataset.iloc[:, 7:8].values

poly_reg = PolynomialFeatures(degree =3,include_bias =False)

X_poly = poly_reg.fit_transform(X)

lin_reg_2 =LinearRegression(fit_intercept = False,normalize = True)

lin_reg_2.fit(X_poly,y)
# Predicting the Missing value 

dataset = pd.read_csv('../input/covid-aggregated-data/Covid.csv')

#dataset = dataset.iloc[:,:-3]

y_pred1 = lin_reg_2.predict(poly_reg.fit_transform(np.array([range(45,49),range(14,18)]).transpose()) )

y_pred2 = lin_reg_2.predict(poly_reg.fit_transform(np.array([range(59,61),range(28,30)]).transpose()))
# Predicting the Total no of samples that should have been taken till 31st July 2020  

Samples_31_July =  lin_reg_2.predict(poly_reg.fit_transform(np.array([[184,31]])))

print('no. of samples till 31st July - {} '.format(Samples_31_July[0][0]))
# Replacing a missing values with predicted one's

missing = [0] * len(dataset)

missing[44:48] = [j for sub in y_pred1 for j in sub]

missing[58:60] = [j for sub in y_pred2 for j in sub]

missing = pd.DataFrame(np.array(missing))

missing = missing.rename(columns={0:'MissingSamplesValues'})

df = pd.concat([dataset,missing],axis=1)

df = df.fillna(0)

df['Samples'] = df['MissingSamplesValues'] + df['TotalSamples']

df.drop(['MissingSamplesValues','TotalSamples'],axis=1,inplace = True)
df.info()
fig = plt.figure(figsize=(15,4));   

ax1 = fig.add_subplot(1,5,1);

ax2 = fig.add_subplot(1,5,2);

ax3 = fig.add_subplot(1,5,3);

#ax4 = fig.add_subplot(1,5,4);

#ax5 = fig.add_subplot(1,5,5);



sns.lineplot(x=df['NoOfDays'],y=df['TotalConfirmedCases'],data=df,ax=ax1)

sns.lineplot(x=df['TotalClosedCases'],y=df['TotalConfirmedCases'],data=df,ax=ax2)

#sns.lineplot(x=df['DeathRate'],y=df['TotalConfirmedCases'],data=df,ax=ax3)

#sns.lineplot(x=df['RecoveryRate'],y=df['TotalConfirmedCases'],data=df,ax=ax4)

sns.lineplot(x=df['Samples'],y=df['TotalConfirmedCases'],data=df,ax=ax3)

plt.tight_layout()



# Relation of features with Total Coinfirmed cases 
X = df[['NoOfDays','Date','Month','TotalClosedCases','TotalPositiveCases']]

y = df['TotalConfirmedCases']



from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA

poly = PolynomialFeatures(degree = 3)

X_p = poly.fit_transform(X)

regressor =PCA(7)

regressor.fit(X_p,y)



print((regressor.explained_variance_ratio_)*100)
X = df.iloc[:,[0,2,-1]].values

y = df['TotalConfirmedCases']

poly2 = PolynomialFeatures(degree =2)

X_poly = poly2.fit_transform(X)

regressor2 =LinearRegression()

regressor2.fit(X_poly,y)



#visualising the polonomial regressor

plt.figure(figsize=(8,4))

plt.scatter(X[:,0],y,color = 'red')

plt.plot(X[:,0],regressor2.predict(poly2.fit_transform(X)),color='blue')

plt.title('Polynomial Regressor')

plt.xlabel('No of days after first case')

plt.ylabel('Total Confirmed cases')

plt.show()
Date = 31

Month =7

Days = 184

Samples = Samples_31_July[0,0]

X1 = np.array([[Days,Month,Samples]])

Total_infections  = regressor2.predict(poly2.fit_transform(X1))
print('Total No of Infection o 31st July -> {}'.format(int(Total_infections[0])))
# Predicting No of closed cases

X_closed = df.iloc[:,[0,1,2,3,-1]].values

y = df['TotalClosedCases']

poly3 = PolynomialFeatures(degree = 1)

X_poly = poly3.fit_transform(X_closed)

regressor3 =LinearRegression()

regressor3.fit(X_poly,y)



Date = 31

Month =7

Days = 184

Samples = Samples_31_July[0,0]

Infections = Total_infections[0]

X1 = np.array([[Days,Date,Month,Infections,Samples]])

Total_closed  = regressor3.predict(poly3.fit_transform(X1))
# predicting no of deaths 

X_death = df.iloc[:,[0,1,2,3,5,-1]].values

y = df['TotalDeaths']

poly4 = PolynomialFeatures(degree = 1)

X_poly = poly4.fit_transform(X_death)

regressor4 =LinearRegression()

regressor4.fit(X_poly,y)



plt.figure(figsize=(8,4))

plt.scatter(X_death[:,0],y,color = 'red')

plt.plot(X_death[:,0],regressor4.predict(poly4.fit_transform(X_death)),color='blue')

plt.title('Polynomial Regressor')

plt.xlabel('No of days after first case')

plt.ylabel('Total Deaths')

plt.show()



Date = 31

Month =7

Days = 184

Samples = Samples_31_July[0,0]

Infections = Total_infections[0]

Closed = Total_closed[0]

X1 = np.array([[Days,Date,Month,Infections,Closed,Samples]])

Total_deaths  = regressor4.predict(poly4.fit_transform(X1))
print('Total No of Deaths o 31st July -> {}'.format(int(Total_deaths[0])))
Date = np.concatenate((np.arange(15,31),np.arange(1,32)),axis=0)

Month =([6]*16) + ([7]*31)

Days = np.arange(138,185)

Samples_ = lin_reg_2.predict(poly_reg.fit_transform(np.array([Days,Date]).transpose()))

Samples_ = np.array([j for sub in Samples_ for j in sub])

X1 = np.array([Days,Month,Samples_]).transpose()

Total_infections_Range  = regressor2.predict(poly2.fit_transform(X1))

X2 = np.array([Days,Date,Month,Total_infections_Range,Samples_]).transpose()

Total_closed_Range  = regressor3.predict(poly3.fit_transform(X2))

X3 = np.array([Days,Date,Month,Total_infections_Range,Total_closed_Range,Samples_]).transpose()

Total_deaths_Range  = regressor4.predict(poly4.fit_transform(X3))



DeathRate = ((Total_deaths_Range)/(Total_closed_Range)) * 100

DeathRate = pd.Series(DeathRate)
DeathRate

# the values are in percenatge
# Now we have Total cases, we just have to Find total recovered cases: Closed - deaths

Total_recoverd_Range = Total_closed_Range -Total_deaths_Range

RecoveryRate = ((Total_recoverd_Range)/(Total_closed_Range)) * 100
RecoveryRate =pd.Series(RecoveryRate)

RecoveryRate