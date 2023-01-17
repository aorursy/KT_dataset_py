import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#Feature  scaling
#Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
#Standard scaler
from sklearn.preprocessing import StandardScaler
#normalizarion
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
#To encode categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Regressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#PCA
from sklearn.decomposition import PCA
# Ignore warnings
import warnings    # To suppress warnings
warnings.filterwarnings("ignore")

#metrics
from scipy import sparse
from scipy.sparse import  hstack
from sklearn import metrics

#sns
import seaborn as sns
housing_data_read = pd.read_csv("../input/train.csv")
housing_data = housing_data_read.iloc[:,housing_data_read.columns != 'SalePrice']
housing_data.head()
housing_data.dtypes
fig=plt.figure(figsize=(500,100))
housing_data.hist( figsize=[30,20])
plt.show()
#housing_data.dtypes!='Object'
plt.figure(figsize=(8,6))
plt.scatter(housing_data_read['LotArea'],housing_data_read['SalePrice'])
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.title("LotArea vs SalePrice")
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(housing_data_read['YrSold'],housing_data_read['SalePrice'],cmap='plasma',alpha=0.5)
plt.xlabel('YrSold')
plt.ylabel('SalePrice')
plt.title("YrSold vs SalePrice")
plt.show()
#housing_data_read[['BldgType','SalePrice']].head()
sns.pairplot(housing_data_read[['BldgType','SalePrice','YrSold']],hue='BldgType');
#Divide dependent independent variable to training and testing set
def data_split(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test
def find_column_na(dataframe,na_limit):
    dataframe_null_check = dataframe.isnull()
    dataframe_null_check_sum = dataframe_null_check.sum()
    columnlist=[]
    for i in range(0,len(dataframe_null_check_sum)):
        if(dataframe_null_check_sum[i]>na_limit):
            print(dataframe_null_check_sum.index[i],dataframe_null_check_sum[i])
            columnlist.append(dataframe.columns[i])
    return columnlist
def drop_na_column(na_columns,dataframe):
     return dataframe.drop(columns=na_columns)
na_columns = find_column_na(housing_data,1000)
housing_data_drop_na = drop_na_column(na_columns,housing_data)
#Function to find Numerical columns from our data frame
def find_numerical_col(dataframe):
    columnlist_num=[]
    #fig =  plt.figure(figsize=(10,10))
    for i in range(1,len(dataframe.columns)):
           if dataframe[dataframe.columns[i]].dtypes=='int64' or dataframe[dataframe.columns[i]].dtypes=='float64':
                    columnlist_num.append(dataframe.columns[i])
    return columnlist_num
#Function to replace blank numerical values with mean of that column
def replaceNumBlankwithMeans(dataframe,num_columns):
    data_no_blank = dataframe
    for col in num_columns:
            data_no_blank[col].fillna(data_no_blank[col].mean(), inplace=True)
    
    return data_no_blank
num_columns = find_numerical_col(housing_data_drop_na)
housing_data_drop_na_no_num_blank = replaceNumBlankwithMeans(housing_data_drop_na,num_columns)
#This is the data frame with NA columns with more than thousands rows removed and blank numerical values repalced with eman
housing_data_drop_na_no_num_blank.head()
#Find categorical columns
def find_categ_column(dataframe):
    return dataframe.select_dtypes(exclude=["number","bool_"]).columns
#This function will replace NA with 'Other' value in categorical columns
def replaceNumBlankwithString(dataframe,cat_columns):
    data_no_blank = dataframe
    for col in cat_columns:
            data_no_blank[col].fillna('Other', inplace=True)
    
    return data_no_blank
cat_columns = find_categ_column(housing_data_drop_na_no_num_blank)
#print(cat_columns)
#find_column_na(housing_data_drop_na_no_num_blank,0)
housing_data_drop_na_no_num_cat_blank = replaceNumBlankwithString(housing_data_drop_na_no_num_blank,cat_columns)
#This data frame has no numerical NA,no categorical NA
housing_data_drop_na_no_num_cat_blank.head()
# Standard Scaling of data
def do_standardscaling(X):
    X_scaled = StandardScaler().fit_transform(X)
    #X_scaled = preprocessing.scale(X)
    return X_scaled
# Normalization of data
def do_normalize(X):
    #normalized_X = preprocessing.normalize(X)
    X_normalized = Normalizer().fit_transform(X)
    return X_normalized
# Min Max scaler
def do_minmaxscale(X):
    min_max=MinMaxScaler()
    X_minmax = min_max.fit_transform(X)
    return X_minmax
# Unscaled data
housing_data_drop_na_no_num_cat_blank.dtypes

# Selecting numerical columns for feature scaling
housing_num_data_feature_scale=housing_data_drop_na_no_num_cat_blank[num_columns]
# min max scaling of numerical data
housing_data_minmax=do_minmaxscale(housing_num_data_feature_scale)
# normalized scaling of numerical data
housing_data_normalize=do_normalize(housing_num_data_feature_scale)
# standard scaling of numerical data
housing_data_standardscale=do_standardscaling(housing_num_data_feature_scale)
# Label encoding categorical data
def cat_label_encode(dataframe):
    le=LabelEncoder()
    dataencoded = dataframe
    for col in dataencoded.columns:
          if dataencoded[col].dtypes=='object':
              le.fit(dataencoded[col].values)
              dataencoded[col]=le.transform(dataencoded[col])
    return dataencoded
#https://stackoverflow.com/questions/46406720/labelencoder-typeerror-not-supported-between-instances-of-float-and-str/46406995
# One hot coding categorical data
#https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
def one_hot_encode(dataframe):
        enc = OneHotEncoder(categorical_features = "all",sparse=False) 
        onehotencoded = dataframe
        for col in onehotencoded.columns:
            
            enc.fit(onehotencoded[[col]]) # use two square brackets to make it as 2 dimensional
            #print(col)
            temp = enc.transform(onehotencoded[[col]])
            temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in onehotencoded[col].value_counts().index])
            temp=temp.set_index(onehotencoded.index.values)
            dataframe=pd.concat([dataframe,temp],axis=1)
            dataframe=dataframe.drop(col,axis=1)
        return dataframe
#UnEncoded data
cat_columns = find_categ_column(housing_data_drop_na_no_num_cat_blank)
housing_data_cat_col = housing_data_drop_na_no_num_cat_blank[cat_columns]
#Label Encoded Data Frame
dataframe_labelenc = cat_label_encode(housing_data_cat_col)
#One Hot Encoded Data Frame
dataframe_ohenc = one_hot_encode(housing_data_cat_col)
dataframe_ohenc.head()
def do_pca(dataframe):
    model_pca = PCA()
    pca = PCA(n_components=4)
    pca.fit(dataframe)
    num_pca = pca.transform(dataframe)
    return num_pca
# Selecting  feature scaled data for pcs
housing_num_data_feature_scale_pca=do_pca(housing_num_data_feature_scale)
PCADataDF_scaled = pd.DataFrame(housing_num_data_feature_scale_pca, columns = ['P1','P2','P3','P4'])
# min max scaled data pca
housing_data_minmax_pca=do_pca(housing_data_minmax)
PCADataDF_minmax = pd.DataFrame(housing_data_minmax_pca, columns = ['P1','P2','P3','P4'])
# normalized scaled data for pca
housing_data_normalize_pca=do_pca(housing_data_normalize)
PCADataDF_normal = pd.DataFrame(housing_data_normalize_pca, columns = ['P1','P2','P3','P4'])
# standard scaled data for pca
housing_data_standardscale_pca=do_pca(housing_data_standardscale)
PCADataDF_standard = pd.DataFrame(housing_data_standardscale_pca, columns = ['P1','P2','P3','P4'])
X = pd.concat([housing_num_data_feature_scale,dataframe_labelenc],axis=1)
#Y = housing_data_read.iloc[:,housing_data_read.columns == 'SalePrice']
Y = housing_data_read.SalePrice#We want original sale price data to work on
X_train, X_test, y_train, y_test = data_split(X, Y)
#X_train.head()
features = housing_data_read.columns[housing_data_read.columns != 'SalePrice']
rfc  = ExtraTreesClassifier(n_estimators=250,random_state=0)
#rfc = KNeighborsClassifier(n_estimators=100, random_state=42)
# Fit the model
rfc .fit(X_train, y_train)
#Features importance
importances = rfc.feature_importances_
#Assign indices to relative importance based upon importance
indices = np.argsort(importances)
# Plot the feature importances of the forest
plt.figure(figsize=(15,20))
plt.title("Feature importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.show()
def regressionEvaluationMetrics(predictions,y_test,title):
    
    plt.figure(figsize=(8,6))
    plt.scatter(predictions,y_test,cmap='plasma',alpha=0.5)
    plt.xlabel('Prediction')
    plt.ylabel('Real value')
    plt.title(title)
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test), np.log1p(predictions))))

df_sp = hstack((dataframe_ohenc, PCADataDF_standard),  format = "csr")
df_sp.shape
X = df_sp
Y = housing_data_read.SalePrice#We want original sale price data to work on
Xtrain, Xtest, ytrain, ytest = data_split(X, Y)
estimator = LinearRegression()
estimator.fit(Xtrain, ytrain)
y_pred = estimator.predict(Xtest)
#print(ytest)
regressionEvaluationMetrics(y_pred,ytest,'LinearRegression')
df_sp = hstack((dataframe_ohenc, PCADataDF_standard),  format = "csr")
df_sp.shape
X = df_sp
Y = housing_data_read.SalePrice#We want original sale price data to work on
Xtrain, Xtest, ytrain, ytest = data_split(X, Y)
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
estimator.fit(Xtrain, ytrain)
y_pred = estimator.predict(Xtest)
#print(ytest)
regressionEvaluationMetrics(y_pred,ytest,'RandomForestRegressor')
df_sp = hstack((dataframe_ohenc, PCADataDF_standard),  format = "csr")
df_sp.shape
X = df_sp
Y = housing_data_read.SalePrice#We want original sale price data to work on
Xtrain, Xtest, ytrain, ytest = data_split(X, Y)
estimator = RidgeCV()
estimator.fit(Xtrain, ytrain)
y_pred = estimator.predict(Xtest)
#print(ytest)
regressionEvaluationMetrics(y_pred,ytest,'RidgeCV')
df_sp = hstack((dataframe_ohenc, PCADataDF_standard),  format = "csr")
df_sp.shape
X = df_sp
Y = housing_data_read.SalePrice#We want original sale price data to work on
Xtrain, Xtest, ytrain, ytest = data_split(X, Y)
estimator = ExtraTreesRegressor(n_estimators=10, max_features=32,random_state=0)
estimator.fit(Xtrain, ytrain)
y_pred = estimator.predict(Xtest)
#print(ytest)
regressionEvaluationMetrics(y_pred,ytest,'ExtraTreesRegressor')