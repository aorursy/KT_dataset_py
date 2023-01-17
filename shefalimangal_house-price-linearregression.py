#loading need libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
%matplotlib inline
house = pd.read_csv('/kaggle/input/maison/Maison.csv')
house.head()


# Since the columns are in french, in order to make them more readable, let's translate them into English
house = house.rename(index = str, columns = {'PRIX':'price','SUPERFICIE': 'area','CHAMBRES': 'rooms', 
                         'SDB': 'bathroom', 'ETAGES': 'floors','ALLEE': 'driveway',
                         'SALLEJEU':'game_room', 'CAVE': 'cellar', 
                         'GAZ': 'gas', 'AIR':'air', 'GARAGES': 'garage', 'SITUATION': 'situation'})
house.head()
#shape of train data
house.shape
house.info()
house_c = house.copy()
house_c
house
house_c.head(10)
house_c.groupby('garage').count()
house_c.groupby('driveway').count()
house_c.groupby('situation').count()
house_c.describe()
house_c.drop('driveway', axis=1, inplace=True)
house_c.drop('game_room', axis=1, inplace=True)
house_c.drop('cellar', axis=1, inplace=True)
house_c.drop('gas', axis=1, inplace=True)
house_c.drop('air', axis=1, inplace=True)
house_c.drop('situation', axis=1, inplace=True)
house_c
x = sns.boxplot(house_c['price'])
Q1=house_c['price'].quantile(0.25)
Q3=house_c['price'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_c = house_c[house_c['price']< Upper_Whisker]
house_c.shape
x = sns.boxplot(house_c['bathroom'])
Q1=house_c['area'].quantile(0.25)
Q3=house_c['area'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_c = house_c[house_c['area']< Upper_Whisker]
house_c.shape
x = sns.boxplot(house_c['rooms'])
Q1=house_c['rooms'].quantile(0.25)
Q3=house_c['rooms'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_c = house_c[house_c['rooms']< Upper_Whisker]
house_c.shape
house_c
house_new_data2 = house.copy()
house_new_data2.head()
#checking box plot
sns.boxplot(house_new_data2['price'])
sns.boxplot(house_new_data2['area'])
sns.boxplot(house_new_data2['rooms'])
sns.boxplot(house_new_data2['bathroom'])
sns.boxplot(house_new_data2['floors'])
sns.boxplot(house_new_data2['driveway'])
house_new_data2 = house_new_data2.drop(columns = ['driveway'])
house_new_data2.head()
sns.boxplot(house_new_data2['game_room'])
house_new_data2 = house_new_data2.drop(columns = ['game_room'])
sns.boxplot(house_new_data2['cellar'])
sns.boxplot(house_new_data2['gas'])
house_new_data2 = house_new_data2.drop(columns = ['gas'])
sns.boxplot(house_new_data2['air'])
sns.boxplot(house_new_data2['garage'])
sns.boxplot(house_new_data2['situation'])
house_new_data2 = house_new_data2.drop(columns = ['situation'])
house_new_data2.head()
house_new_data2.shape
Q1=house_new_data2['price'].quantile(0.25)
Q3=house_new_data2['price'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_new_data2 = house_new_data2[house_new_data2['price']< Upper_Whisker]
house_new_data2.shape
Q1=house_new_data2['area'].quantile(0.25)
Q3=house_new_data2['area'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_new_data2 = house_new_data2[house_new_data2['area']< Upper_Whisker]
house_new_data2.shape
Q1=house_new_data2['rooms'].quantile(0.25)
Q3=house_new_data2['rooms'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_new_data2 = house_new_data2[house_new_data2['rooms']< Upper_Whisker]
house_new_data2.shape
sns.boxplot(house_new_data2['bathroom'])
house_new_data2 = house_new_data2.drop(columns = ['bathroom'])
house_new_data2.head()
house_new_data2.shape
sns.boxplot(house_new_data2['floors'])
Q1=house_new_data2['floors'].quantile(0.25)
Q3=house_new_data2['floors'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_new_data2 = house_new_data2[house_new_data2['floors']< Upper_Whisker]
house_new_data2.shape
sns.boxplot(house_new_data2['garage'])
Q1=house_new_data2['garage'].quantile(0.25)
Q3=house_new_data2['garage'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
house_new_data2 = house_new_data2[house_new_data2['garage']< Upper_Whisker]
house_new_data2.shape
house_new_data2.head()
plt.scatter(house_new_data2['area'], house_new_data2['price'])
plt.show()
import warnings
warnings.filterwarnings('ignore')
sns.distplot(house_new_data2['price'])
plt.show()
sns.distplot(house_new_data2['area'])
plt.show()
# Import the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lm = LinearRegression()
# let's do the split of the dataset
house_new_data2.columns

X = house_new_data2[['area', 'rooms', 'floors', 'cellar', 'air', 'garage']]

Y = house_new_data2['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.3, random_state = 42)
X_test.head()
# Now let's build the model using sklearn
lm.fit(X_train, Y_train)
#Prediction
lm.score(X_test, Y_test)
import statsmodels.api as sm
# Unlike sklearn that adds an intercept to our data for the best fit, statsmodel doesn't. We need to add it ourselves
# Remember, we want to predict the price based off our features.
# X represents our predictor variables, and y our predicted variable.
# We need now to add manually the intercepts
X_endog = sm.add_constant(X_test)
res = sm.OLS(Y_test, X_endog)
res.fit()
res.fit().summary()
#Separate variable into new dataframe from original dataframe which has only numerical values
#there is 38 numerical attribute from 81 attributes
train_corr = house_new_data2.select_dtypes(include=[np.number])
train_corr.shape
#Coralation plot
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)
house_new_data2.area.unique()
sns.barplot(house_new_data2.area, house_new_data2.price)
house_new_data2 = house_new_data2.drop(columns = ['floors'])
house_new_data2.head()
house_new_data2 = house_new_data2.drop(columns = ['garage'])
house_new_data2 = house_new_data2.drop(columns = ['cellar'])
house_new_data2 = house_new_data2.drop(columns = ['rooms'])
house_new_data2.head()
house_new_data2.shape
lm = LinearRegression()
# let's do the split of the dataset
house_new_data2.columns
X = house_new_data2[['area','air']]

Y = house_new_data2['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.3)
X_test.head()
# Now let's build the model using sklearn
lm.fit(X_train, Y_train)
lm.score(X_test, Y_test)