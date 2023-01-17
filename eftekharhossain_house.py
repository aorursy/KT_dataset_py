from google.colab import drive

drive.mount('/content/gdrive/')
import pandas as pd 

import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns



train_data_path = '/content/gdrive/My Drive/Colab Notebooks/Kaggle Project/Housing Price/train.csv'

test_data_path = '/content/gdrive/My Drive/Colab Notebooks/Kaggle Project/Housing Price/test.csv'

saved_file_path = '/content/gdrive/My Drive/Colab Notebooks/Kaggle Project/Housing Price/submission.csv'
# Read the data

X_full = pd.read_csv(train_data_path,index_col='Id')

X_test_full = pd.read_csv(test_data_path,index_col='Id')

print("Shape of the Training Set:",X_full.shape)

print("Shape of the Test Set",X_test_full.shape)
col_with_missing = (X_full.isnull().sum())

list_features_with_na = col_with_missing[col_with_missing>0]

print(list_features_with_na)
#find the column names which have missing values

features_with_na = [feature for feature in X_full.columns if X_full[feature].isnull().sum()>0]

#print(features_with_na)



# Find the percentage of missing values in each feature

for feature in features_with_na:

  print("{} ----- {:,.4f}".format(feature,X_full[feature].isnull().mean()))
for feature in features_with_na:

  data = X_full.copy()

  # make a variable that indicates '1' if the observation is missing otherwise '0'

  data[feature] = np.where(data[feature].isnull(),1,0)

  # Calculate the mean sales price where the information is missing

  data.groupby(feature).SalePrice.median().plot.bar()

  plt.title(feature)

  plt.show()
numerical_features = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64','float64']]

print("Number of numerical Features: ",len(numerical_features))

X_full[numerical_features].head()
year_features = [cname for cname in numerical_features if 'Yr' in cname or 'Year' in cname]

#print(temp_features)

# Let's explore the contents of these features

for feature in year_features:

  print(feature,X_full[feature].unique())
X_full.groupby("YrSold")['SalePrice'].median().plot()

plt.xlabel('Year Sold')

plt.ylabel('Sale Price')

plt.title("Price vs YrSold ")

## Here we will compare the difference between All years feature with SalePrice



for feature in year_features:

    if feature!='YrSold':

        data=X_full.copy()

        ## We will capture the difference between year variable and year the house was sold for

        data[feature]=data['YrSold']-data[feature]



        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()
## Numerical variables are usually of 2 type

## 1. Continous variable and Discrete Variables



discrete_feature=[feature for feature in numerical_features if len(X_full[feature].unique())<25 and feature not in year_features]

print("Discrete Variables Count: {}".format(len(discrete_feature)))

X_full[discrete_feature].head()
## Lets Find the realtionship between them and Sale PRice



for feature in discrete_feature:

    data=X_full.copy()

    data.groupby(feature).SalePrice.median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_features]

print("Continuous feature Count {}".format(len(continuous_feature)))

X_full[continuous_feature].head()
## Lets analyse the continuous values by creating histograms to understand the distribution



for feature in continuous_feature:

    data=X_full.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()

for feature in continuous_feature:

  data = X_full.copy()

  if 0 in data[feature].unique():

    pass

  else:

    data[feature] = np.log(data[feature])  

    data['SalePrice'] = np.log(data['SalePrice'])

    plt.scatter(x = data[feature],y = data['SalePrice'])

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
for feature in continuous_feature:

  data = X_full.copy()

  if 0 in data[feature].unique():

    pass

  else:

    data[feature] = np.log(data[feature])  

    data.boxplot(column=feature)

    plt.title(feature)

    plt.show()
for feature in continuous_feature:

  data = X_full.copy()

  if 0 in data[feature].unique():

    pass

  else:

    Q1 = data[feature].quantile(0.25)

    Q3 = data[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5*IQR

    upper_limit = Q3 + 1.5*IQR

    df_no_outlier = data[(data[feature]>lower_limit)&(data[feature]<upper_limit)]

    plt.boxplot(df_no_outlier[feature])

    #data.boxplot(column=feature)

    plt.title("After Removing Outlier "+feature)

    plt.show()
#X_full['GrLivArea'] = np.log(X_full['GrLivArea'])  

X_full.boxplot(column='GrLivArea')

plt.title('GrLivArea')

plt.show()
Q1 = X_full['GrLivArea'].quantile(0.25)

Q3 = X_full['GrLivArea'].quantile(0.75)

Q1, Q3

IQR = Q3 - Q1

IQR
lower_limit = Q1 - 1.5*IQR

upper_limit = Q3 + 1.5*IQR

lower_limit, upper_limit
df_no_outlier = X_full[(X_full['GrLivArea']>lower_limit)&(X_full['GrLivArea']<upper_limit)]

plt.boxplot(df_no_outlier.GrLivArea)

plt.show()
categorical_features = [cname for cname in X_full.columns if X_full[cname].dtype =='object']

print("Number of Categorical Features:",len(categorical_features))
# we have to find the cardinality in each categorical features

for feature in categorical_features:

  print("{}--Cardinality: {}".format(feature,len(X_full[feature].unique())))
# The relationship between the categorical features and SalePrice

for feature in categorical_features:

  data = X_full.copy()

  #data.groupby(feature).SalePrice.median().plot.bar()

  sns.swarmplot(x= data[feature],y = data['SalePrice'])

  plt.title(feature)

  plt.xlabel(feature)

  plt.ylabel('SalePrice')

  plt.show()
droped_columns = [cname for cname in X_test_full.columns if X_test_full[cname].isnull().sum()>1000 ]

droped_columns
# Always remember there must be some data leakage so do split the dataset first

# Remove rows with missing sale price

X_full.dropna(subset=['SalePrice'],axis = 0,inplace = True)

y = X_full.SalePrice

X_full.drop(columns=droped_columns,axis =1,inplace=True )

X_test_full.drop(columns=droped_columns,axis =1,inplace=True )

X_full.drop(columns=['SalePrice'],axis =1,inplace=True )

# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print("After Removing size of Train Set:",X_train.shape)

print("After Remving Size of Validation Set:",X_valid.shape)

print("After Removing Size of Test Set:",X_test_full.shape)
X_train_full =X_train

X_train_full['SalePrice'] = y_train 

X_valid_full =X_valid

X_valid_full['SalePrice'] = y_valid 
## Let us capture all the nan values

## First lets handle Categorical features which are missing

features_nan=[feature for feature in X_full.columns if X_full[feature].isnull().sum()>0 and X_full[feature].dtypes=='O']

for feature in features_nan:

    print("{}: {}% missing values".format(feature,np.round(X_full[feature].isnull().mean(),4)))
categorical_features=[feature for feature in X_full.columns if X_full[feature].dtypes=='O']

## Replace missing value with the most-frequent value

def replace_cat_feature(dataset,features_nan):

    data=dataset.copy()

    for feature in features_nan:

      value = data[feature].value_counts().to_dict()

        # use most frequent value

      most_frequent = max(value,key=value.get)

      #data[feature+' nan']=np.where(data[feature].isnull(),1,0)

      data[feature]=data[feature].fillna(most_frequent)

    return data



print("Number of Categorical Features: ",len(categorical_features))

X_train_full=replace_cat_feature(X_train_full,categorical_features)

X_valid_full=replace_cat_feature(X_valid_full,categorical_features)

X_test_full=replace_cat_feature(X_test_full,categorical_features)

print("After Removing Missing Values:\n")

print("Size of Train Set:",X_train_full.shape)

print("Size of Validation Set:",X_valid_full.shape)

print("Size of Test Set:",X_test_full.shape)
#X_train_full[categorical_features].isnull().sum()

X_test_full.head()
## Now lets check for numerical variables the contains missing values

numerical_with_nan=[feature for feature in X_full.columns if X_full[feature].isnull().sum()>1 and X_full[feature].dtypes!='O']



## We will print the numerical nan variables and percentage of missing values



for feature in numerical_with_nan:

    print("{}: {}% missing value".format(feature,np.around(X_full[feature].isnull().mean(),4)))
numerical_features = [feature for feature in X_full.columns if X_full[feature].dtypes!='O']

# Replace nan values with the median value

def replace_num_feature(dataset,numerical_with_nan):

  data = dataset.copy()

  for feature in numerical_with_nan:

    ## We will replace by using median since there are outliers

    median_value=data[feature].median()

    ## create a new feature to capture nan values

    #data[feature+' nan']=np.where(data[feature].isnull(),1,0)

    data[feature].fillna(median_value,inplace=True)

  return data



print("Number of Numerical Features: ",len(numerical_features))

X_train_full=replace_num_feature(X_train_full,numerical_features)

X_valid_full=replace_num_feature(X_valid_full,numerical_features)

X_test_full=replace_num_feature(X_test_full,numerical_features)  

print("After Removing Missing Values:\n")

print("Size of Train Set:",X_train_full.shape)

print("Size of Validation Set:",X_valid_full.shape)

print("Size of Test Set:",X_test_full.shape)
X_valid_full[numerical_features].isnull().sum()
X_test_full[numerical_features].isnull().sum()    
# Temporal Varaibles Handling

temp_features = ['YearBuilt','YearRemodAdd','GarageYrBlt']

def temp_variables_handling(dataset,temp_features):

  data = dataset.copy()

  for feature in temp_features:

    data[feature] = data['YrSold']-data[feature]

  return data  



###

X_train_full = temp_variables_handling(X_train_full,temp_features)

X_valid_full = temp_variables_handling(X_valid_full,temp_features)

X_test_full = temp_variables_handling(X_test_full,temp_features)  

print("After Adding Temporal Variables:\n")

print("Size of Train Set:",X_train_full.shape)

print("Size of Validation Set:",X_valid_full.shape)

print("Size of Test Set:",X_test_full.shape)
correlated_features = set()

correlation_matrix = X_train_full.corr()

#plt.figure(figsize=(30,20))

#sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)

#plt.show()
for i in range(len(correlation_matrix .columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)



print("Number of Correlated Independent Features:",len(correlated_features))

print(correlated_features)            
#Correlation with output variable

cor_target = abs(correlation_matrix["SalePrice"])

#Selecting low correlated features

non_relevant_features = cor_target[cor_target < 0.045]

non_relevant_features = [feature for feature,value in non_relevant_features.to_dict().items()]

print("Number of non correlated features with Output:",len(non_relevant_features))

print(non_relevant_features)
non_correlated_features = non_relevant_features + list(correlated_features)

print("Total Non Correlated Features are:",len(non_correlated_features))

print(non_correlated_features)
X_train_full_dropped = X_train_full.drop(non_correlated_features,axis = 1)

X_valid_full_dropped = X_valid_full.drop(non_correlated_features,axis = 1)

#non_correlated_features.remove('SalePrice')

X_test_full_dropped = X_test_full.drop(non_correlated_features,axis = 1)

print("After Removing Non Correlated Variables:\n")

print("Size of Train Set:",X_train_full_dropped.shape)

print("Size of Validation Set:",X_valid_full_dropped.shape)

print("Size of Test Set:",X_test_full_dropped.shape)
import numpy as np

num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

def log_transformation(dataset,num_features):

  data = dataset.copy()

  for feature in num_features:

      data[feature]=np.log(data[feature])

  return data    

X_train_full = log_transformation(X_train_full_dropped,num_features)

X_valid_full = log_transformation(X_valid_full_dropped,num_features)

X_test_full = log_transformation(X_test_full_dropped,num_features)
X_test_full.head(10)
## Bring it all together in a function

categorical_features=[feature for feature in X_full.columns if X_full[feature].dtype=='O']

def rare_category_handling(dataset,categorical_features):

  data = dataset.copy()

  for feature in categorical_features:

      temp=X_train_full.groupby(feature)['SalePrice'].count()/len(data)

      temp_df=temp[temp>0.01].index

      data[feature]=np.where(data[feature].isin(temp_df),data[feature],'Rare_var')

  return data    





X_valid_full = rare_category_handling(X_valid_full,categorical_features)

X_test_full = rare_category_handling(X_test_full,categorical_features)

X_train_full = rare_category_handling(X_train_full,categorical_features)

X_train_full.head(20)  
# All categorical columns

object_cols = [col for col in X_train_full_dropped.columns if X_train_full_dropped[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train_full_dropped[col]) == set(X_valid_full_dropped[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
from sklearn.preprocessing import LabelEncoder

def label_encoding(X_train,X_valid,X_test,good_label_cols,bad_label_cols):

  

# Drop categorical columns that will not be encoded

  label_X_train = X_train.drop(bad_label_cols, axis=1)

  label_X_valid = X_valid.drop(bad_label_cols, axis=1)

  label_X_test  = X_test.drop(bad_label_cols, axis=1)



  # Apply label encoder only to good_label_cols 

  

  l_encoder = LabelEncoder()

  for col in good_label_cols:

      label_X_train[col] = l_encoder.fit_transform(X_train[col])

      label_X_valid[col] = l_encoder.transform(X_valid[col])

      label_X_test[col] = l_encoder.fit_transform(X_test[col])

      

  return label_X_train,label_X_valid,label_X_test   

X_t_full,X_va_full,X_tes_full = label_encoding(X_train_full_dropped,X_valid_full_dropped,X_test_full_dropped,good_label_cols,bad_label_cols)

print("After Label Encoding:\n")

print("Size of Train Set:",X_t_full.shape)

print("Size of Validation Set:",X_va_full.shape)

print("Size of Test Set:",X_tes_full.shape)
# All categorical columns

object_cols = [col for col in X_train_full_dropped.columns if X_train_full_dropped[col].dtype == "object"]



# Taking the columns with low cardinality

low_cardinality_cols = [cname for cname in X_train_full_dropped.columns if X_train_full_dropped[cname].nunique() < 10 and

                        X_train_full_dropped[cname].dtype =='object']



high_cardinality_cols = [cname for cname in X_train_full_dropped.columns if X_train_full_dropped[cname].nunique() >10 and

                        X_train_full_dropped[cname].dtype =='object']

                        

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder



def onehot_encoding(X_train,X_valid,X_test,low_cardinality_cols,object_cols):

  OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

  OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

  OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

  OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))



  # One-hot encoding removed index; put it back

  OH_cols_train.index = X_train.index

  OH_cols_valid.index = X_valid.index

  OH_cols_test.index = X_test.index



  # Remove categorical columns (will replace with one-hot encoding)

  num_X_train = X_train.drop(object_cols, axis=1)

  num_X_valid = X_valid.drop(object_cols, axis=1)

  num_X_test = X_test.drop(object_cols, axis=1)



  # Add one-hot encoded columns to numerical features

  OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

  OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

  OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

  return  OH_X_train,OH_X_valid,OH_X_test



X_t_full,X_va_full,X_tes_full = onehot_encoding(X_train_full_dropped,X_valid_full_dropped,X_test_full_dropped,low_cardinality_cols,object_cols)

print("After One Hot Encoding:\n")

print("Size of Train Set:",X_t_full.shape)

print("Size of Validation Set:",X_va_full.shape)

print("Size of Test Set:",X_tes_full.shape)
X_t_full.head()
scaling_feature=[feature for feature in X_train_full.columns if feature not in ['SalePrice'] ]

len(scaling_feature)
from sklearn.preprocessing import MinMaxScaler

feature_scale=[feature for feature in X_t_full.columns if feature not in ['SalePrice']]

scaler=MinMaxScaler()

scaler.fit(X_t_full[feature_scale])

# Bring it all together in a function

def feature_scaling(dataset,feature_scale):

  data = dataset.copy()

  scaled_data = pd.concat([data[['SalePrice']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(data[feature_scale]), columns=feature_scale)],

                    axis=1)

  return scaled_data
X_train_scaled = feature_scaling(X_t_full,feature_scale)

X_valid_scaled = feature_scaling(X_va_full,feature_scale)
X_tes_scaled = pd.DataFrame(scaler.transform(X_tes_full[feature_scale]), columns=feature_scale,index=X_tes_full.index)

X_tes_scaled
X_tes_full.isnull().sum()
## for feature slection



from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
y_train=X_t_full[['SalePrice']]

## drop dependent feature from dataset

X_train=X_t_full.drop(['SalePrice'],axis=1)
### Apply Feature Selection

# first, I specify the Lasso Regression model, and I

# select a suitable alpha (equivalent of penalty).

# The bigger the alpha the less features that will be selected.



# Then I use the selectFromModel object from sklearn, which

# will select the features which coefficients are non-zero



feature_sel_model = SelectFromModel(Lasso(alpha=0.1, random_state=0)) # remember to set the seed, the random state in this function

feature_sel_model.fit(X_train, y_train)

feature_sel_model.get_support()
# let's print the number of total and selected features



# this is how we can make a list of the selected features

selected_feat = X_train.columns[(feature_sel_model.get_support())]



# let's print some stats

print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(

    np.sum(feature_sel_model.estimator_.coef_ == 0)))
selected_feat
X_train=X_train[selected_feat]

X_train.head()

X_valid = X_va_full[selected_feat]

y_valid = X_va_full[['SalePrice']]
# this function will determine the appropriate n_estimator

def find_estimator(n_estimator):

  model =  XGBRegressor(n_estimators=n_estimator,learning_rate=0.05,random_state=0)

                    

  scores = -1 * cross_val_score(model, X_train, y_train,

                                  cv=5,

                                  scoring='neg_mean_absolute_error')

  return scores.mean()

results = {i:find_estimator(i) for i in [700,705,710,715,720,725,730,735,740,745,750]}
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(list(results.keys()), list(results.values()))

plt.show()



print("Optimum estimator number:{}".format(min(results, key= results.get )))
y_train = np.log(y_train)
y_train = X_t_full['SalePrice']

y_valid = X_va_full['SalePrice']

X_train = X_t_full.drop('SalePrice',axis =1)

X_valid = X_va_full.drop('SalePrice',axis = 1)

#X_train = X_t_full

#X_valid = X_va_full



X_train.head()
# Final Model

model = XGBRegressor(n_estimators=600,learning_rate=0.05,random_state=0)



# Preprocessing of training data, fit model 

model.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = model.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
X_train.head()
# Submission

preds_test = model.predict(X_tes_full)

output = pd.DataFrame({'Id': X_tes_full.index,

                       'SalePrice': preds_test})

output.to_csv(saved_file_path, index=False)