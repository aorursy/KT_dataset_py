import pandas as pd #for analysis

import numpy as np #linear algebra



#for visualization

import matplotlib.pylab as plt

from matplotlib import pyplot as plot

%matplotlib inline

import seaborn as sns
df= pd.read_csv('../input/carpriceprediction/CarPrice_Assignment.csv')
#take a look at a sample data to understand the dataset

df.head()
#let us know the types of the attributes

df.dtypes
#check the size of the data

df.shape
#let us know the descriptive statistics of the numrical variables

df.describe()
#let us know the descriptive statistics of the categorical variable

df.describe(include="object")
#checking for the missing values in the dataset

df.isnull().sum()
#removing 'car_ID'

df.drop('car_ID',axis=1,inplace=True)
df['CarName'].unique()
df['CarName'] = df['CarName'].str.split(' ',n=1,expand=True)

df.head(2)
df['CarName'].unique()
df['CarName'].replace({'vokswagen':'volkswagen','vw':'volkswagen', 'toyouta':'toyota','porcshce':'porsche',  'Nissan':'nissan','maxda':'mazda'},inplace =True)

df['CarName'].unique()
df.corr()['price']
sns.pairplot(data=df)
plot.figure(figsize=(20,15))

plt.subplot(3,3,1)

sns.boxplot(x= 'fueltype', y= 'price',data= df)

plt.subplot(3,3,2)

sns.boxplot(x= 'aspiration', y= 'price',data= df)

plt.subplot(3,3,3)

sns.boxplot(x= 'doornumber', y= 'price',data= df)

plt.subplot(3,3,4)

sns.boxplot(x= 'carbody', y= 'price',data= df)

plt.subplot(3,3,5)

sns.boxplot(x= 'drivewheel', y= 'price',data= df)

plt.subplot(3,3,6)

sns.boxplot(x= 'enginelocation', y= 'price',data= df)

plt.subplot(3,3,7)

sns.boxplot(x= 'enginetype', y= 'price',data= df)

plt.subplot(3,3,8)

sns.boxplot(x= 'cylindernumber', y= 'price',data= df)

plt.subplot(3,3,9)

sns.boxplot(x= 'fuelsystem', y= 'price',data= df)
#check the size before remove variables

df.shape
#remove the unnecessary variables from the data set

df.drop(['symboling','doornumber','carheight','fuelsystem','enginelocation','boreratio','stroke','compressionratio','peakrpm'],axis=1,inplace=True)
#check the size after remove variables

df.shape
#let us seperate the variables to change.

category = ['CarName','fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber']
#let us create dummy variables

dummy_var = pd.get_dummies(df[category])

dummy_var.head(2)
#add a dummy variables to the dataset

df = pd.concat([df,dummy_var],axis =1)

df.head(1)
#remove the original categorical variables

df.drop(category,axis =1,inplace=True)

df.shape
df.head()
# for model devolpment

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
#assign a variable for independent and dependent variables

X = df.drop('price',axis=1)

y = df['price']
#Feature Selection

X_selected = SelectKBest(f_regression,k=15).fit_transform(X,y)
#dividin into training and test set

X_train,X_test,y_train,y_test = train_test_split(X_selected,y,test_size =0.20)
#Alpha tuning

#create scoring list

fitscores =[]

predictscores =[]

#logspace list of alpha

alphas = np.logspace(-1, 4, num=25)

#alpha iteration and fit with Ridge regression

for alpha in alphas:

    estimator = Ridge(alpha = alpha)

    estimator.fit(X_train,y_train)

    fitscores.append(estimator.score(X_train,y_train))

    predictscores.append(estimator.score(X_test,y_test))
#look at actual score

fitscores[0]
#look at predictive score

predictscores[0]
#visualize a scores

ax = plt.gca()

ax.set_xscale('log')

ax.plot(alphas, fitscores,'g', label = 'Train')

ax.plot(alphas, predictscores,'b', label = 'Test')

plt.ylim([0,1])

plt.xlabel('alpha')

plt.ylabel('Correlation Coefficient')

plt.legend()

plt.title('Tune regularization')
#model on training and test data

ridgeEstimator = Ridge(alpha = 45)

ridgeEstimator.fit(X_train,y_train)

ridgeEstimator.score(X_train,y_train)

ridgeEstimator.score(X_test,y_test)

plt.plot(y_train,ridgeEstimator.predict(X_train),'r*', label = 'Train')

plt.plot(y_test,ridgeEstimator.predict(X_test),'bs', label = 'Test')

plt.legend()

plt.xlabel('Actual price')

plt.ylabel('Predicted price')

plt.title('Prediction Plot')