import pandas as pd

import numpy as np

import sklearn as sc

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import PolynomialFeatures



from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



from sklearn import metrics



from sklearn.model_selection import train_test_split
df_train = pd.read_csv("/kaggle/input/sales-prediction/train.csv")  #Import Data Set

df_test = pd.read_csv("/kaggle/input/sales-prediction/test.csv")
df_train.head()
y_train = df_train['SalePrice'] # Dependent variable
x_train = df_train.drop('SalePrice',1)

x_test = df_test
l_num_col = x_train.select_dtypes(exclude='O').columns # Assigning numeric features

l_cat_col = x_train.select_dtypes(include='O').columns # Assigning categorical features
print(x_train.shape)

print(x_test.shape)
(x_train[l_cat_col].isnull().sum()*100/x_train.shape[0]).apply(lambda col : col>0.7) # Check if feature have null values more than 50% of total volume
# Drop column which have missing values greater than 70

x_train[l_cat_col].isnull().sum()*100/x_train.shape[0] > 70 # Same as above without lambda
# Taking all the columns where percentage of missing values is less than 70%

l_cat_col_shortlisted = list(filter(lambda col : x_train[col].isnull().sum()*100/x_train.shape[0]<70, l_cat_col ))

l_num_col_shortlisted = list(filter(lambda col : x_train[col].isnull().sum()*100/x_train.shape[0]<70, l_num_col ))
### Fill missing values with most frequent count in categorical features

x_train_cat = x_train[l_cat_col_shortlisted].apply(lambda x: x.fillna(x.value_counts().index[0]))

x_test_cat = x_test[l_cat_col_shortlisted].apply(lambda x: x.fillna(x.value_counts().index[0]))
print(x_test_cat.isnull().sum()) # Check null values after fill
### Fill missing values with mean in numeric features

x_train_num = x_train[l_num_col_shortlisted].apply(lambda x: x.fillna(x.mean()))

x_test_num = x_test[l_num_col_shortlisted].apply(lambda x: x.fillna(x.mean()))
print(x_test_num.isnull().sum()) # Check null values after fill
###Check multi collinearity between independent features and remove features which have corelated value more than 0.7 i.e 70%

corr_matrix = x_train_num.corr().abs()

#print(corr_matrix)

col_corr = set()

for i in range(len(corr_matrix.columns)):

    for j in range(i):

        #print("row: "+ str(i),"column: " + str(j))

        if(corr_matrix.iloc[i,j] >= 0.7) and (corr_matrix.columns[j] not in col_corr):

            print("Correlated Columns are : {}, {}".format(corr_matrix.iloc[[i]].index[0], corr_matrix.columns[j]))

            col_name = corr_matrix.columns[i]

            col_corr.add(col_name)

print("Columns to be delete : {}".format(col_corr))
print(x_train_num.shape)

x_train_num.drop(columns = col_corr, inplace=True) # Remove Multi correlation feature from data frame

print(x_train_num.shape)
#Check correlation with dependent variable

correlation = pd.concat([x_train_num,pd.DataFrame(y_train)],axis=1).corr().abs()
### Correlation with dependent varaible

##Now we have to take only those variables which have high correlation with dependent varaible i.e. with sales price

print(correlation['SalePrice'])

threshold = 0.6

shortlisted_num_features = [correlation.index[i] for i in np.where(pd.DataFrame(correlation['SalePrice']) >threshold)[0] if correlation.index[i] !='SalePrice']

print("Shortlisted columns are : {}".format(shortlisted_num_features))

x_train[shortlisted_num_features].describe() # Check shortlisted data frame now
## From describe(), it is easily observable that we have values with different scales so need to do standardization 

## Need to starndarize the values

# Create scaler

scaler = preprocessing.StandardScaler()



# Transform the feature

standardized_scaler = scaler.fit(x_train_num[shortlisted_num_features])

x_train_standardized = standardized_scaler.transform(x_train_num[shortlisted_num_features])

x_test_standardized = standardized_scaler.transform(x_test_num[shortlisted_num_features]) # Use same scaler to transform test data

# Show feature

x_train_num_stardardized = pd.DataFrame(x_train_standardized, columns = shortlisted_num_features)

x_test_num_stardardized = pd.DataFrame(x_test_standardized, columns = shortlisted_num_features)



print(x_train_num_stardardized.shape)

print(x_test_num_stardardized.shape)
x_train_cat.describe(include='all').loc['unique', :] #Check unique values in each categorical feature
x_train_enc = pd.get_dummies(x_train_cat, drop_first=True) # Assign dummy variables and drop first coulumn to avoid dummy variable trap

x_test_enc = pd.get_dummies(x_test_cat, drop_first=True) 
## THis is very important step as we will get different unique value in train and test column

# e.g. Some values are present in train set but not available in test data set or vise versa

# This can be handle as below .If we use same model over test set then it will throw an error in further step because number of features would be different

print("Before alignment")

print(x_train_enc.shape)

print(x_test_enc.shape)

x_train_enc,x_test_enc = x_train_enc.align(x_test_enc, join='outer', axis=1, fill_value=0)



print("After alignment")

print(x_train_enc.shape)

print(x_test_enc.shape)
# Here we have categorical independent features and continous dependent varaible so use ANOVA (Analysis of varaince) method for feature selection

# Create an SelectKBest object to select features with two best ANOVA F-Values

fvalue_selector = SelectKBest(f_classif, k=3)



# Apply the SelectKBest object to the features and target

X_kbest = fvalue_selector.fit_transform(x_train_enc, y_train)
mask = fvalue_selector.get_support() #list of booleans

shortlisted_cat_features = [] # The list of your K best features



for bool, feature in zip(mask,  list(x_train_enc.columns.values)):

    if bool:

        shortlisted_cat_features.append(feature)
x_train_new = pd.concat([x_train_num_stardardized, x_train_enc[shortlisted_cat_features]], axis=1)

x_test_new = pd.concat([x_test_num_stardardized, x_test_enc[shortlisted_cat_features]], axis=1)
print(x_train_new.shape)

print(x_test_new.shape) # Now we have 7 independent features
X_train,X_test,Y_train,Y_test = train_test_split(x_train_new,y_train, test_size=0.2, random_state = 1)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
model = LinearRegression() 

fit = model.fit(X_train,Y_train)

Y_test_predict = fit.predict(X_test)

y_test_predict = fit.predict(x_test_new)
sns.residplot(Y_test, Y_test_predict, color="g")
print("Linear Regression")

print("R2 Score : {}".format(metrics.r2_score(Y_test, Y_test_predict)))

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_test_predict))

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_test_predict))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_predict)))
regressor = SVR()

fit = regressor.fit(X_train,Y_train)

Y_test_predict = fit.predict(X_test)

y_test_predict = fit.predict(x_test_new)
sns.residplot(Y_test, Y_test_predict, color="g")
print("SVR : ")

print("R2 Score : {}".format(metrics.r2_score(Y_test, Y_test_predict)))

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_test_predict))

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_test_predict))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_predict)))
regressor = DecisionTreeRegressor(random_state = 0)

fit = regressor.fit(X_train,Y_train)

Y_test_predict = fit.predict(X_test)

y_test_predict = fit.predict(x_test_new)
sns.residplot(Y_test, Y_test_predict, color="g")
print("Random Forest Regression : ")

print("R2 Score : {}".format(metrics.r2_score(Y_test, Y_test_predict)))

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_test_predict))

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_test_predict))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_predict)))
regressor = RandomForestRegressor(n_estimators=50, random_state=0)

fit = regressor.fit(X_train,Y_train)

Y_test_predict = fit.predict(X_test)

y_test_predict = fit.predict(x_test_new)
sns.residplot(Y_test, Y_test_predict, color="g")
print("Random Forest Regression : ")

print("R2 Score : {}".format(metrics.r2_score(Y_test, Y_test_predict)))

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_test_predict))

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_test_predict))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_predict)))