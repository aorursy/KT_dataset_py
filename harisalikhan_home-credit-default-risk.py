#from google.colab import drive
#drive.mount('/content/drive')
# Importing necessary modules first -> Numpy, Pandas and LabelEncoder
import numpy as np
import pandas as pd
# Label Encoder for dealing with categorical labels
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
#Matplot and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Now lets see the files in the data we are dealing with
print(os.listdir())
# We need to work our dataset on application_train.csv and application_test.csv . Lets import the data in a dataframe.
train_data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
test_data = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
# Visualize train data
train_data.head(20)
# There are some NaN in data which indicate missing values. Lets see the test data now.
test_data.head(20)
#Test data too contains some missing values which we will have to deal later.
#Lets see shape of both data.
print ("Shape of Training data is ",train_data.shape)
print ("Shape of Testing data is ",test_data.shape)
# We see that target variable is missing from the testing dataset. The target variable is what system needs to predict. 
# Therefore the task needs to be done using a supervised classification algorithm. We have thus established the boundry.
# For this task, I will use logistic Regression , Naive Bayes and a neural network classifier.
#Since this is a supervised classification task.We need to see distribution of class with respect to the label for our training data.
train_data['TARGET'].value_counts()
# We see many traning examples belong 0 (no) and less to 1(yes). Lets visualize it too.
train_data['TARGET'].astype(int).plot.hist();
# The dataset has high imbalance in classes. Therefore accuracy might be a good metric to evaluate the performance of our trained classifier.
# Lets see datatypes of our features.
train_data.dtypes.value_counts()
#Lets see which of features are object
print(train_data.dtypes == 'object')
# Lets now start dealing with missing values.
train_data.isnull()
# Can we do better to find count of null values .
train_data.isnull().sum().sort_values(ascending=False)
# Can we do more better by making it visualized using a table. For that I define a function like the one on the kaggle kernel followed
# by the professor.
def missing_values_table(df):
    missing_val= df.isnull().sum()
    
    # Percentage of missing values
    
    missing_per = 100 *df.isnull().sum() /len(df)
    
    mis_values_table = pd.concat([missing_val,missing_per],axis=1)
    
    missing_val_table_columns = mis_values_table.rename(columns={0 : 'Missing Values', 1 : '% of Missing Values'})
    
    missing_val_table_columns = missing_val_table_columns[
            missing_val_table_columns.iloc[:,1] != 0].sort_values(
        '% of Missing Values', ascending=False).round(1)
    
    return missing_val_table_columns
    
# Lets see the missing values visualization for our train_data.
missing_values_table(train_data)
# Lets check if I drop all Nan rows. How much data are we left to deal with ? 
train_data_modify = train_data
train_data_modify.shape
train_data_modify=train_data_modify.dropna()
train_data_modify.shape
# This shows its not better to drop all Nan values. This will significantly decrease our data. Other solution is to drop columns with 
# more missing data.
missing_values= missing_values_table(train_data)
missing_values.head(40)
(missing_values_table(train_data)).head(30)
train_data.drop(["COMMONAREA_AVG","COMMONAREA_MODE","NONLIVINGAPARTMENTS_MODE","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAPARTMENTS_AVG","FONDKAPREMONT_MODE","LIVINGAPARTMENTS_MODE","LIVINGAPARTMENTS_MEDI","LIVINGAPARTMENTS_AVG","FLOORSMIN_AVG","FLOORSMIN_MEDI","FLOORSMIN_MODE","YEARS_BUILD_AVG","YEARS_BUILD_MEDI","LANDAREA_MODE","LANDAREA_AVG","LANDAREA_MEDI","BASEMENTAREA_MEDI","BASEMENTAREA_MODE","BASEMENTAREA_AVG","EXT_SOURCE_1","NONLIVINGAREA_MODE","NONLIVINGAREA_AVG","NONLIVINGAREA_MEDI"],axis=1,inplace=True)
(missing_values_table(train_data)).head(30)
train_data.drop(["COMMONAREA_MEDI","YEARS_BUILD_MODE","OWN_CAR_AGE","ELEVATORS_MEDI","ELEVATORS_MODE","ELEVATORS_AVG","WALLSMATERIAL_MODE","APARTMENTS_MODE","APARTMENTS_MEDI","APARTMENTS_AVG","ENTRANCES_AVG","ENTRANCES_MEDI","ENTRANCES_MODE","LIVINGAREA_MODE","LIVINGAREA_AVG","LIVINGAREA_MEDI","HOUSETYPE_MODE","FLOORSMAX_AVG","FLOORSMAX_MEDI","FLOORSMAX_MODE","YEARS_BEGINEXPLUATATION_MEDI","YEARS_BEGINEXPLUATATION_AVG","YEARS_BEGINEXPLUATATION_MODE","TOTALAREA_MODE","EMERGENCYSTATE_MODE","OCCUPATION_TYPE"],axis=1,inplace=True)
train_data['EXT_SOURCE_3'].describe()
train_data['AMT_REQ_CREDIT_BUREAU_YEAR'].describe()
train_data['AMT_REQ_CREDIT_BUREAU_QRT'].describe()
train_data['NAME_TYPE_SUITE'].describe()
train_data['DEF_60_CNT_SOCIAL_CIRCLE'].describe()
train_data['EXT_SOURCE_2'].describe()
train_data['AMT_ANNUITY'].describe()
train_data['CNT_FAM_MEMBERS'].describe()
# in my opinion we have only one object dtype which has missing values and which can be very hard to replace in a dataset. Therefore
# I removed that feature.
train_data.drop(["NAME_TYPE_SUITE"],axis=1,inplace=True)
missing_values_test= missing_values_table(test_data)
missing_values_test.head(30)
test_data.drop(["COMMONAREA_MODE","COMMONAREA_MEDI","COMMONAREA_AVG","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAPARTMENTS_AVG","NONLIVINGAPARTMENTS_MODE","FONDKAPREMONT_MODE","LIVINGAPARTMENTS_MODE","LIVINGAPARTMENTS_MEDI","LIVINGAPARTMENTS_AVG","FLOORSMIN_MEDI","FLOORSMIN_MODE","FLOORSMIN_AVG","OWN_CAR_AGE","YEARS_BUILD_AVG","YEARS_BUILD_MEDI","YEARS_BUILD_MODE","LANDAREA_MODE","LANDAREA_AVG","LANDAREA_MEDI","BASEMENTAREA_MEDI","BASEMENTAREA_AVG","BASEMENTAREA_MODE","NONLIVINGAREA_AVG","NONLIVINGAREA_MODE","NONLIVINGAREA_MEDI","ELEVATORS_MEDI","ELEVATORS_MODE","ELEVATORS_AVG","WALLSMATERIAL_MODE"],axis=1,inplace=True)
missing_values_test= missing_values_table(test_data)
missing_values_test.head(40)
test_data.drop(["APARTMENTS_AVG","APARTMENTS_MEDI","APARTMENTS_MODE","HOUSETYPE_MODE","ENTRANCES_MODE","ENTRANCES_MEDI","ENTRANCES_AVG","LIVINGAREA_MEDI","LIVINGAREA_AVG","LIVINGAREA_MODE","FLOORSMAX_MODE","FLOORSMAX_MEDI","FLOORSMAX_AVG","YEARS_BEGINEXPLUATATION_MEDI","YEARS_BEGINEXPLUATATION_MODE","TOTALAREA_MODE","EMERGENCYSTATE_MODE","EXT_SOURCE_1"],axis=1,inplace=True)
missing_values_test= missing_values_table(test_data)
missing_values_test.head(20)
test_data.drop(["YEARS_BEGINEXPLUATATION_AVG"],axis=1,inplace=True)
missing_values_test= missing_values_table(test_data)
# Lets now check dimensions of our test and train data
train_data.shape
test_data.shape
test_data.head(20)
train_data.head(20)
# okay perfect . Now its time to do some encoding of our categorical labels.
list(train_data.select_dtypes(['object']).columns)
from sklearn.preprocessing import LabelEncoder
train_data_encode = train_data
test_data_encode = test_data
encoder=LabelEncoder()
train_data_encode['NAME_CONTRACT_TYPE'] = encoder.fit_transform(train_data_encode['NAME_CONTRACT_TYPE'])
# Now lets replicate it for all object types.
train_data_encode['CODE_GENDER'] = encoder.fit_transform(train_data_encode['CODE_GENDER'])
train_data_encode['FLAG_OWN_CAR'] = encoder.fit_transform(train_data_encode['FLAG_OWN_CAR'])
train_data_encode['FLAG_OWN_REALTY'] = encoder.fit_transform(train_data_encode['FLAG_OWN_REALTY'])
train_data_encode['NAME_INCOME_TYPE'] = encoder.fit_transform(train_data_encode['NAME_INCOME_TYPE'])
train_data_encode['NAME_EDUCATION_TYPE'] = encoder.fit_transform(train_data_encode['NAME_EDUCATION_TYPE'])
train_data_encode['NAME_FAMILY_STATUS'] = encoder.fit_transform(train_data_encode['NAME_FAMILY_STATUS'])
train_data_encode['NAME_HOUSING_TYPE'] = encoder.fit_transform(train_data_encode['NAME_HOUSING_TYPE'])
train_data_encode['WEEKDAY_APPR_PROCESS_START'] = encoder.fit_transform(train_data_encode['WEEKDAY_APPR_PROCESS_START'])
train_data_encode['ORGANIZATION_TYPE'] = encoder.fit_transform(train_data_encode['ORGANIZATION_TYPE'])
train_data_encode['NAME_CONTRACT_TYPE'].head()
train_data_encode['ORGANIZATION_TYPE'].head()
# now over to our test set
list(test_data.select_dtypes(['object']).columns)
# I see some differences in categorical columns in both dataset.
missing_values_test.head(20)
# I see I can remove OCCUPATION TYPE AND NAME_TYPE_SUITE FROM MY TEST DATA.
test_data.drop(["OCCUPATION_TYPE","NAME_TYPE_SUITE"],axis=1,inplace=True)
test_data_encode = test_data
list(test_data_encode.select_dtypes(['object']).columns)
# Encoding test data
test_data_encode['NAME_CONTRACT_TYPE'] = encoder.fit_transform(test_data_encode['NAME_CONTRACT_TYPE'])
test_data_encode['CODE_GENDER'] = encoder.fit_transform(test_data_encode['CODE_GENDER'])
test_data_encode['FLAG_OWN_CAR'] = encoder.fit_transform(test_data_encode['FLAG_OWN_CAR'])
test_data_encode['FLAG_OWN_REALTY'] = encoder.fit_transform(test_data_encode['FLAG_OWN_REALTY'])
test_data_encode['NAME_INCOME_TYPE'] = encoder.fit_transform(test_data_encode['NAME_INCOME_TYPE'])
test_data_encode['NAME_EDUCATION_TYPE'] = encoder.fit_transform(test_data_encode['NAME_EDUCATION_TYPE'])
test_data_encode['NAME_FAMILY_STATUS'] = encoder.fit_transform(test_data_encode['NAME_FAMILY_STATUS'])
test_data_encode['NAME_HOUSING_TYPE'] = encoder.fit_transform(test_data_encode['NAME_HOUSING_TYPE'])
test_data_encode['WEEKDAY_APPR_PROCESS_START'] = encoder.fit_transform(test_data_encode['WEEKDAY_APPR_PROCESS_START'])
test_data_encode['ORGANIZATION_TYPE'] = encoder.fit_transform(test_data_encode['ORGANIZATION_TYPE'])
train_data.head(20)
test_data.head(20)
train_data = train_data_encode
test_data = test_data_encode
train_data.head(20)
test_data.head(20)
train_data.shape
test_data.shape
# Back to EDA , lets do some feature visualization and engineering
correlations=train_data.corr()['TARGET'].sort_values()
correlations.tail(20)
# We see that it has very high correlations with DAYS_BIRTH
train_data['DAYS_BIRTH'].describe()
# The values should not be negative 
train_data['DAYS_BIRTH'] = train_data['DAYS_BIRTH']/-365
# Check similar for the test data
test_data['DAYS_BIRTH'].describe()
# Again same issue
test_data['DAYS_BIRTH'] = test_data['DAYS_BIRTH']/-365
test_data['DAYS_BIRTH'].head()
test_data['DAYS_LAST_PHONE_CHANGE'].describe()
train_data['DAYS_LAST_PHONE_CHANGE'].describe()
# As its highly correlated feature, we must also transform it from being negative

train_data['DAYS_LAST_PHONE_CHANGE']=train_data['DAYS_LAST_PHONE_CHANGE']/-1
test_data['DAYS_LAST_PHONE_CHANGE']=test_data['DAYS_LAST_PHONE_CHANGE']/-1
train_data['DAYS_ID_PUBLISH'].describe()
train_data['DAYS_ID_PUBLISH']=train_data['DAYS_ID_PUBLISH']/-1
test_data['DAYS_ID_PUBLISH']=test_data['DAYS_ID_PUBLISH']/-1
train_data['DAYS_LAST_PHONE_CHANGE'].plot.hist()
train_data['DAYS_BIRTH'].plot.hist()
# lets now again see correlations
correlations=train_data.corr()['TARGET'].sort_values()
correlations.tail(10)
# Surprising ! :D 
# Now lets see our both datasets after transformations.

print ("Shape of training data is : ", train_data.shape)
print ("Shape of testing data is :",test_data.shape)



# now lets begin model development and tuning. # I will use three algorithms logistic regression , naive bayes and neural networks.
# Create label to predict
y_train = train_data.TARGET
# Create X_train for our data
X_train = train_data.drop(columns=['TARGET'])
# Create X_test for our data
X_test = test_data
#train_data_reg = train_data
#train_data_nb = train_data
#train_data_svm = train_data
#test_data_reg = test_data
#test_data_nb = test_data
#test_data_svm = test_data

X_train,X_test=X_train.align(X_test, join= 'inner', axis=1)
X_train.shape
X_test.shape
#Check for null-values
print (X_test.isnull().values.any())
print (X_train.isnull().values.any())
# Use imputer for null-values ! 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(X_train)
X_train.loc[:] = imputer.transform(X_train)
X_test.loc[:] = imputer.transform(X_test)
# Perfect. Now that we are here, before regression. I would like to check which features are important for my data and model so
# that I can use them in future and store in an array.
# Check out feature importances and save important features for later use and improvement.
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Necesary to seperate data for scaling and apply regression model
X_train_reg=X_train
X_test_reg=X_test
y_train_reg=y_train
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(X_train_reg)
X_train_reg = scaler.transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)
print('Training data shape: ', X_train_reg.shape)
print('Testing data shape: ', X_test_reg.shape)
np.isnan(X_test_reg).any()
np.isnan(X_train_reg).any()
# Lets apply our first model - Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_reg,y_train_reg)
pred = logreg.predict_proba(X_test_reg)[:,1]
pred
result = test_data[['SK_ID_CURR']]
result['TARGET']=pred
result.head(40)
(result['TARGET']).describe()
result.to_csv('logisticRegression1.csv',index=False)
predictions = logreg.predict_proba(X_train_reg)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score
print(roc_auc_score(y_train_reg,predictions))
fpr, tpr, thr = roc_curve(y_train_reg,predictions)
plt.figure()
plt.plot(fpr, tpr)
# Some hypermeter tuning and trying improvement on results
logreg = LogisticRegression(C=0.1,max_iter=1000)
logreg.fit(X_train_reg,y_train_reg)
pred = logreg.predict_proba(X_test_reg)[:,1]
pred
result = test_data[['SK_ID_CURR']]
result['TARGET']=pred
(result['TARGET']).describe()
result.to_csv('logisticRegression2.csv',index=False)
predictions = logreg.predict_proba(X_train_reg)[:,1]
accuracy = logreg.score(X_train_reg,y_train_reg)
print("Accuracy is : ",accuracy)
print(roc_auc_score(y_train_reg,predictions))
fpr, tpr, thr = roc_curve(y_train_reg,predictions)
plt.figure()
plt.plot(fpr, tpr)
# Lets now work with Guassain Naive Bayes and see results

X_train_nb = X_train
X_test_nb = X_test
y_train_nb = y_train
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
X_train_nb
X_test_nb.head()
X_train_nb.shape
X_test_nb.shape
print (X_test_nb.isnull().values.any())
# Good ! :D
model.fit(X_train_nb,y_train_nb)
accuracy = model.score(X_train_nb,y_train_nb)
print("Accuracy of model is : " , accuracy)
predictions = model.predict_proba(X_test_nb)[:,1]
predictions
submission = test_data[['SK_ID_CURR']]
submission['TARGET']=predictions
submission.head()
(submission['TARGET']).describe()
submission.to_csv('NaiveBayes1.csv',index=False)
predict = model.predict_proba(X_train_nb)[:,1]
print(roc_auc_score(y_train_nb,predict))
# Let me try randomn forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc_rain = RandomForestClassifier(n_estimators=100, random_state=13).fit(X_train_nb, y_train_nb)
rfc_predict = rfc_rain.predict_proba(X_test_nb)[:, 1]
submission1 = test_data[['SK_ID_CURR']]
submission1['TARGET']=rfc_predict
submission.head()
submission1.to_csv('NaiveBayes2.csv',index=False)
predict = rfc_rain.predict_proba(X_train_nb)[:, 1]
print(roc_auc_score(y_train_nb,predict))
# Thats great .
# Now I move onto my final algorithm MLPClassifier.

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
X_train_mlp=X_train_reg
X_test_mlp=X_test_reg
y_train_mlp=y_train_reg
mlp.fit(X_train_mlp,y_train_mlp)
pred=mlp.predict_proba(X_test_mlp)[:,1]
res = test_data[['SK_ID_CURR']]
res['TARGET']=pred
res.head()
res.to_csv('NeuralNetwork.csv',index=False)
(res['TARGET']).describe()
list(X_train.columns) 
# As per feature importances curve 
# Feature 37,38,15,18,0,17,43,6,29,8 are 10 most important features.
feature_columns=['ORGANIZATION_TYPE','EXT_SOURCE_2','REGION_POPULATION_RELATIVE','DAYS_REGISTRATION','DAYS_EMPLOYED','DEF_60_CNT_SOCIAL_CIRCLE','CNT_CHILDREN','WEEKDAY_APPR_PROCESS_START','AMT_CREDIT']
X_train_f = X_train[feature_columns]
X_test_f=X_test[feature_columns]
y_train_f = y_train
from sklearn.preprocessing import MinMaxScaler
scaler.fit(X_train_f)
X_train_f = scaler.transform(X_train_f)
X_test_f = scaler.transform(X_test_f)
print('Training data shape: ', X_train_f.shape)
print('Testing data shape: ', X_test_f.shape)
logreg = LogisticRegression()
logreg.fit(X_train_f,y_train_f)
pred = logreg.predict_proba(X_test_f)[:,1]
result = test_data[['SK_ID_CURR']]
result['TARGET']=pred
result.head()
result.to_csv('LogisticRegressionImp.csv',index=False)