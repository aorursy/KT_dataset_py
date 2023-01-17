# Author       : AKASH DIXIT
# E-Mail       : akashdixit453@gmail.com
# Contact      : +91-7415770162
# Designation  : Robotics Engineer
# Regression Analysis for House Prices Prediction
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# Since we are going to make lot of visualization, let's set some visualization parameters in order to have same plots size
plt.rcParams['figure.figsize'] = [12,6]
sns.set_style('darkgrid')
#Read the data
house = pd.read_csv('/kaggle/input/maison/Maison.csv')
house.head()
# Since the columns are in french, in order to make them more readable, let's translate them into English
house = house.rename(index = str, columns = {'PRIX':'price','SUPERFICIE':'area','CHAMBERS':'rooms',
                                            'SDB':'bathroom','ETAGES':'floor','ALLEE':'driveway',
                                             'SALLEJEU':'game_room','CAVE':'cellar','GAZ':'gas',
                                             'AIR':'air','GARAGES':'garage','SITUATION':'suitation'})
house.head()
house.shape
house.info()
house_copy = house
house_copy.head(10)
print(house_copy['driveway'])
house_copy['garage'].unique()
house_copy.groupby('cellar').count()
house_copy.describe()
sns.boxplot(house_copy['price'])
sns.boxplot(house_copy['area'])    
#Remove outlier from area column
q1 = house_copy['area'].quantile(0.25)
q3 = house_copy['area'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)

house_filter_data = house_copy[house_copy['area'] < uppar_whisker]

house_filter_data.shape
sns.boxplot(house_copy['price'])    
house_filter_data.shape
sns.boxplot(house_copy['garage'])
#Remove outlier from garage column
q1 = house_copy['garage'].quantile(0.25)
q3 = house_copy['garage'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)

house_filter_data = house_copy[house_copy['garage'] < uppar_whisker]

house_copy.head()
house_data_new = house.copy()
house_data_new.head()
sns.boxplot(house_data_new['price'])    
sns.boxplot(house_data_new['area'])    
sns.boxplot(house_data_new['CHAMBRES'])    
sns.boxplot(house_data_new['bathroom'])    
sns.boxplot(house_data_new['floor'])    
sns.boxplot(house_data_new['driveway'])    
house_data_new = house_data_new.drop(columns = ['driveway'])
sns.boxplot(house_data_new['game_room'])    
house_data_new = house_data_new.drop(columns = ['game_room'])
house_data_new.head()
sns.boxplot(house_data_new['gas'])    
house_data_new = house_data_new.drop(columns = ['gas'])
house_data_new.head()
sns.boxplot(house_data_new['suitation'])    
house_data_new = house_data_new.drop(columns = ['suitation'])
house_data_new.head()
sns.boxplot(house_data_new['bathroom'])    
house_data_new.head()
house_data_new['cellar'].count()

house_data_new['cellar'].sum()
house_data_new.shape
sns.boxplot(house_data_new['price'])    
house_data_new.shape
#Remove outlier from price column
q1 = house_data_new['price'].quantile(0.25)
q3 = house_data_new['price'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)


house_data_new = house_data_new[house_data_new['price'] < uppar_whisker]
house_data_new.shape
sns.boxplot(house_data_new['area'])    
#Remove outlier from area column
q1 = house_data_new['area'].quantile(0.25)
q3 = house_data_new['area'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)

house_data_new = house_data_new[house_data_new['area'] < uppar_whisker]
house_data_new.shape
house_data_new.head()
sns.boxplot(house_data_new['CHAMBRES'])    
#Remove outlier from CHAMBRES column
q1 = house_data_new['CHAMBRES'].quantile(0.25)
q3 = house_data_new['CHAMBRES'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)
house_data_new = house_data_new[house_data_new['CHAMBRES'] < uppar_whisker]
house_data_new.shape
sns.boxplot(house_data_new['bathroom'])    
house_data_new.head()
house_data_new = house_data_new.drop(columns = ['bathroom'])
house_data_new.head()
sns.boxplot(house_data_new['floor'])    
#Remove outlier from floor column
q1 = house_data_new['floor'].quantile(0.25)
q3 = house_data_new['floor'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)
house_data_new = house_data_new[house_data_new['floor'] < uppar_whisker]
house_data_new.head()
sns.boxplot(house_data_new['garage'])    
#Remove outlier from garage column
q1 = house_data_new['garage'].quantile(0.25)
q3 = house_data_new['garage'].quantile(0.75)
IQR = q3-q1
print("q1:", q1)
print("q3:", q3)
print("IQR:", IQR)

lower_whisker = q1-1.5*IQR
uppar_whisker = q3+1.5*IQR

print("Lower_whisker: ",lower_whisker)
print("Uppar_wishker: ",uppar_whisker)
house_data_new = house_data_new[house_data_new['garage'] < uppar_whisker]
house_data_new.head()

house_data_new.shape
house_data_new['air'].count()
house_data_new['air'].sum()
house_data_new.head()
#Let's draw scatter plot between area & price
plt.scatter(house_data_new['area'], house_data_new['price'])
plt.show()
import warnings
warnings.filterwarnings('ignore')
sns.distplot(house_data_new['price'])
plt.show()
sns.distplot(house_data_new['area'])
plt.show()
#Import the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#we now instatiate a Linear Regression object
lm = LinearRegression()
# let's do the split of the dataset
house_data_new.columns

X = house_data_new[['area', 'CHAMBRES', 'floor', 'cellar', 'air', 'garage']]

Y = house_data_new['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 56)

X_test.head()
#Now let's build the model using sklearn
lm.fit(X_train,Y_train)
predictions = lm.predict(X_test)
sns.scatterplot(Y_test, predictions)
print("Accuracy --> ", lm.score(X_test, Y_test)*100)
import statsmodels.api as sm
X_endog = sm.add_constant(X_test)
res = sm.OLS(Y_test, X_endog)
res.fit()
res.fit().summary()
#correlation
train_corr = house_data_new.select_dtypes(include=[np.number])
train_corr.shape
#Coralation plot
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)
print(corr)
house_data_new.area.unique()
sns.barplot(house_data_new.price, house_data_new.area)
house_data_new = house_data_new.drop(columns = ['cellar'])

house_data_new.head()
house_data_new.shape
#we now instatiate a Linear Regression object
lm = LinearRegression()
# let's do the split of the dataset
house_data_new.columns

X = house_data_new[['area', 'CHAMBRES', 'air']]

Y = house_data_new['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#Now let's build the model using sklearn
lm.fit(X_train,Y_train)
predictions = lm.predict(X_test)
sns.scatterplot(Y_test, predictions)
print("Accuracy --> ", lm.score(X_train, Y_train)*100)