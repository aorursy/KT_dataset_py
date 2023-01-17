import pandas as pd

%pylab inline

import sklearn

import seaborn as sns

import statsmodels.regression.linear_model as sm

from sklearn.ensemble import RandomForestRegressor
#Import Data

data=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv',header=0)

data

#Checking 1st 5 row and if want to see more insert number into parentheses (10)

data.head()
#Checking last 10 rows of data but by deafult it will show you last 5 row

data.tail(10)
#checking the missing values and at the smae time you can chekc the type of data, Stroke and Horse powerbinnd 

#in which values/string are missing

data.info()
# At the same time you can check with below syntex as well where you can see the sum of missing values available into data

data.isnull().sum()
# You can check total number of rows and cloumns

data.shape
#you can check the name of cloumns present in data

data.columns
# You can check types of data available into data

data.dtypes
# you can see the max, min, mean, std, and quartiles of all numerice value

data.describe()
# histogram price range is b/w 50 to 160 and their milage is b/w 20 to 30 and so on 

data.hist(bins=50,figsize=(20,15))

plt.show()
#correlation matrix

corr_matrix=data.corr()
#correlation with Y=price with other independent variable (close to +1 strong postive correlation and -1 close to shows negative corelation which can bring the price of a car down,

# Close to '0' means no corelation )

corr_matrix['price'].sort_values(ascending=False)
#Symboling corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then,

#if it is more risky (or less),this symbol is adjusted by moving it up (or down) the scale.

data.plot(kind='scatter',x='symboling',y = 'price')
# If you want to extract particular coloum for a view and we have a missing value in this coloum

data.loc[:,'stroke']
# It will help you to replace missing value with the median value of stroke which we can see 3.29 in describe as mention on above syntex

data.stroke.fillna('3.29', inplace=True)
data
# If you want to check the number of object/string availabe in a particular coloum 

data.make.value_counts()
# Another way to check the count of string available into coloums

data["horsepower-binned"].value_counts()
# we have missing string in horesepower-binned, as I have selected 'LOW' because is highest frequency

#data.horsepower-binned.fillna('Low', inplace=True)

#converting coloum 'stroke' into float

data.stroke=data.stroke.astype(float)
# Extracting all numeric value (float, Int.) and creating diffrent variable 

data_number=data.select_dtypes(include=np.number)

data_number
# Extracting object and creating diffrent variable, with this we have not segragateed the original data into 2 part data_number & data_object

data_object=data.select_dtypes(include=np.object)

data_object
#Creating dummy variable for all clomum whose dtype is object so I have used data_object as dummy variable

#Latter on we will see how to use the dummy variable 

 

dummy=pd.get_dummies(data_object,drop_first=True)
dummy
# Droping the coloum 



data=data.drop(columns=['horsepower-binned'])
# Lets handel Text and categorical attribute by lable Encoder & OneHotEncoder & LabelBinarizer

# I have created data here to perform EDA and ML (data) (data_number/data_object)
# let see how label Encoder help us to convert our categorical data 

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

data['make']= label_encoder.fit_transform(data['make'])

data['aspiration']= label_encoder.fit_transform(data['aspiration'])

data['num-of-doors']= label_encoder.fit_transform(data['num-of-doors'])

data['body-style']= label_encoder.fit_transform(data['body-style'])

data['drive-wheels']= label_encoder.fit_transform(data['drive-wheels'])

data['engine-location']= label_encoder.fit_transform(data['engine-location'])

data['engine-type']= label_encoder.fit_transform(data['engine-type'])

data['num-of-cylinders']= label_encoder.fit_transform(data['num-of-cylinders'])

data['fuel-system']= label_encoder.fit_transform(data['fuel-system'])

#print(label_encoder.classes_)
data.shape
data.info()
#Declare the dependent variable and create your independent and dependent datasets

X = data.drop('price', axis=1).to_numpy()

Y = data['price'].to_numpy()
X.shape, Y.shape
#Split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.20, random_state=5)
from statsmodels.api import OLS

#Run model

lm = sm.OLS(Y_train,X_train).fit()

print(lm.summary())
# As you can see that we have selected all predetor in our feature/Predetaor X due to which we can see multicollinerity, in next step you can start 

#droping x variable which is less corelation to its Y (Price),to have high accuracy.

#When a statistical model is used to represent the process that generated the data, the representation will almost never be exact; so some information will be lost by using the model to represent the process. 

# AIC estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model.
# Let try random forest 

model = RandomForestRegressor(n_jobs=-1)

estimators = 240

scores = []

model.set_params(n_estimators=estimators)

model.fit(X_train, Y_train)

scores.append(model.score(X_test, Y_test))
scores