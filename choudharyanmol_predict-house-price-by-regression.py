#==============================================================================

# Import libraries

#==============================================================================

import numpy as np



import pandas as pd

import seaborn as sns

import matplotlib.ticker as mtick # For specifying the axes tick format 

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



#machine learning libraries

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

df = pd.read_csv ('../input/sample_submission.csv')


df.head()
actualsale=df['SalePrice']

actualsale
train=pd.read_csv('../input/train.csv')



train.head()
train.isnull().sum()
train.describe()
train.info()

train.columns

test=pd.read_csv('../input/test.csv')

test

test.shape

test.head()



test_features=test[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','EnclosedPorch','MSSubClass','OverallCond','YrSold','LowQualFinSF','MiscVal','BsmtHalfBath','BsmtFinSF2','Street','Neighborhood','ExterCond','Condition1','ExterQual']]

test_features.head()





from sklearn.preprocessing import LabelEncoder

X_labelencoder = LabelEncoder()

test_features.iloc[:, 21] = X_labelencoder.fit_transform(test_features.iloc[:, 21])

test_features.iloc[:,22] = X_labelencoder.fit_transform(test_features.iloc[:, 22])

test_features.iloc[:,23] = X_labelencoder.fit_transform(test_features.iloc[:, 23])

test_features.iloc[:, 24] = X_labelencoder.fit_transform(test_features.iloc[:, 24])

test_features.iloc[:, 25] = X_labelencoder.fit_transform(test_features.iloc[:, 25])



test_features.head()







from sklearn.preprocessing import Imputer

# First create an Imputer , Stratergy means what we want to write in place of missed value

missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 

                               axis = 0)  #if missing values are represented by 9999 then write same here

# Set which columns imputer should perform

missingValueImputer = missingValueImputer.fit (test_features.iloc[:,:])

# update values of X with new values

test_features.iloc[:,:] = missingValueImputer.transform(test_features.iloc[:,:])





test_features.info()
dv=train.iloc[:,80]

iv=train.iloc[:,:80]



X=iv.drop(["Utilities", "Condition2", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating", "Electrical", "GarageQual", "PoolQC", "MiscFeature"], axis=1)

X.head()

from sklearn.preprocessing import LabelEncoder

X_labelencoder = LabelEncoder()

X.iloc[:, 2] = X_labelencoder.fit_transform(X.iloc[:, 2])

X.iloc[:, 5] = X_labelencoder.fit_transform(X.iloc[:, 5])

X.iloc[:, 7] = X_labelencoder.fit_transform(X.iloc[:, 7])

X.iloc[:, 8] = X_labelencoder.fit_transform(X.iloc[:, 8])

X.iloc[:, 9] = X_labelencoder.fit_transform(X.iloc[:, 9])

X.iloc[:, 10] = X_labelencoder.fit_transform(X.iloc[:, 10])

X.iloc[:, 11] = X_labelencoder.fit_transform(X.iloc[:, 11])

X.iloc[:, 12] = X_labelencoder.fit_transform(X.iloc[:, 12])

X.iloc[:, 13] = X_labelencoder.fit_transform(X.iloc[:, 13])

X.iloc[:, 66] = X_labelencoder.fit_transform(X.iloc[:, 66])

X.iloc[:, 67] = X_labelencoder.fit_transform(X.iloc[:, 67])

X.iloc[:, 20] = X_labelencoder.fit_transform(X.iloc[:, 20])

X.iloc[:, 21] = X_labelencoder.fit_transform(X.iloc[:, 21])

X.iloc[:, 22] = X_labelencoder.fit_transform(X.iloc[:, 22])

                                            

X["MSZoning"]=X["MSZoning"].fillna('RL')

X["KitchenQual"]=X["KitchenQual"].fillna('TA')

X["Functional"]=X["Functional"].fillna('Typ')

X["SaleType"]=X["SaleType"].fillna('WD')

X.head()

X["Alley"]=X["Alley"].fillna('Grvl')



X['Fence']=X['Fence'].fillna('MnPrv')

X.iloc[:, 3] = X_labelencoder.fit_transform(X.iloc[:, 3])

X.iloc[:, 6] = X_labelencoder.fit_transform(X.iloc[:, 6])



X.iloc[:, 62] = X_labelencoder.fit_transform(X.iloc[:, 62])

X.iloc[:, 66] = X_labelencoder.fit_transform(X.iloc[:, 66])

 



X.head()


x=X.drop (['FireplaceQu'],axis=1)

x.columns
features=x[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','EnclosedPorch','MSSubClass','OverallCond','YrSold','LowQualFinSF','MiscVal','BsmtHalfBath','BsmtFinSF2','Street','Neighborhood','ExterCond','Condition1','ExterQual']]
from sklearn.preprocessing import Imputer

# First create an Imputer , Stratergy means what we want to write in place of missed value

missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 

                              axis = 0)  #if missing values are represented by 9999 then write same here

# Set which columns imputer should perform

missingValueImputer = missingValueImputer.fit (features.iloc[:,10:12])

# update values of X with new values

features.iloc[:,10:12] = missingValueImputer.transform(features.iloc[:,10:12])

#Splitting into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, dv, test_size=0.2)

X_train.shape#(1168, 26)

X_test.shape#(292, 26)

#y_train.shape#(1168,)

#y_test.shape#(292,)

#features.info()
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

#predictions

predictionprice=model.predict(test_features)
from sklearn.metrics import r2_score, mean_squared_error

print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))

print('RMSE : '+str(np.sqrt(mean_squared_error(actualsale, predictionprice))))
#explain equation now y=mx+c

m=model.coef_

c=model.intercept_

print(m) #slope

print(c)  

#y=m*X+c  #here i is x as defined above

#print(y)

#Visualising the results

plt.figure(figsize=(8,5))

sns.regplot(predictions,y_test,scatter_kws={'alpha':0.3,'color':'lime'},line_kws={'color':'red','alpha':0.5})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Linear Prediction of House Pricing")

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, dv, test_size=0.2)

X_train.shape#(1168, 26)

from sklearn.ensemble import RandomForestRegressor

modelrandom = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)



modelrandom.fit(X_train, y_train)

pred=modelrandom.predict(X_test)





predsale=modelrandom.predict(test_features)

predsale



#from sklearn.metrics import r2_score, mean_squared_error

#print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(actualsale, predsale))))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, pred))))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, dv, test_size=0.2)

X_train.shape#(1168, 26)

from sklearn.ensemble import RandomForestRegressor

modelrandom = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)



modelrandom.fit(X_train, y_train)

pred=modelrandom.predict(X_test)





predsale=modelrandom.predict(test_features)

predsale



#from sklearn.metrics import r2_score, mean_squared_error

#print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(actualsale, predsale))))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, pred))))

estimators=modelrandom.estimators_[5]



features=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','EnclosedPorch','MSSubClass','OverallCond','YrSold','LowQualFinSF','MiscVal','BsmtHalfBath','BsmtFinSF2','Street','Neighborhood','ExterCond','Condition1','ExterQual']



#labels=['density', 'pH', 'alcohol', 'quality']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=features

   , filled = True))

display(SVG(graph.pipe(format='svg')))