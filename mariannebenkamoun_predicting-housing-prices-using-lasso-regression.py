#Import the imported libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#read the data 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.head())
print(train.shape)
#Looking now at the number of non-NaN elements in each columns
notnullcount = train.count()
notnullcount

disposable_columns = []
for n in range (0,train.shape[1]):
    if notnullcount[n] <0.3*train.shape[0]:
        disposable_columns = np.append(disposable_columns,[notnullcount.index[n]])
        train.drop([notnullcount.index[n]],1,inplace=True)
disposable_columns
col = train.columns
col.shape
# let's check for columns contraining null values
train.isnull().any()
# let's now have a look at a ranking of the tops values in a column train['Neighborhood'].value_counts()
from collections import Counter

columns = train.columns
for col in columns:
    if train[col].dtype == np.dtype('O'):
        count = Counter(train[col])
        if (count.most_common(1)[0][0]) is np.nan:
            train[col].fillna(count.most_common(2)[1][0],inplace=True)
        else: 
            train[col].fillna(count.most_common(1)[0][0],inplace=True)
    else:
        train[col] = train[col].fillna(train[col].mean())
        
#we count the columns having missing values.
train.isnull().any().value_counts()
#let's now drop the column Id as it is irrelevant for the analysis 
train.drop(['Id'],1,inplace=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dataset_numeric = train.select_dtypes(include=numerics)

dataset_numeric.shape

nonnumeric = ['object']
dataset_nonnumeric = train.select_dtypes(include=nonnumeric)
dataset_numeric.skew()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
cols = dataset_numeric.columns
for c in cols:
    sns.violinplot(dataset_numeric[c])
    plt.xlabel(c)
    plt.show()
skew = dataset_numeric.skew()
skew #check the skew of the numeric values. 

skewedfeatures = [s for s in skew if(s > 5.0)]
skewedfeatures
for skf in skewedfeatures:
    sk = skew[skew == skf].index[0]
    dataset_numeric[sk] = np.log1p(dataset_numeric[sk])


test.drop(disposable_columns,1,inplace=True)

columns = test.columns
for col in columns:
    if test[col].dtype == np.dtype('O'):
        count = Counter(test[col])
        if (count.most_common(1)[0][0]) is np.nan:
            test[col].fillna(count.most_common(2)[1][0],inplace=True)
        else: 
            test[col].fillna(count.most_common(1)[0][0],inplace=True)
    else:
        test[col] = test[col].fillna(test[col].mean())
        
        
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
test_numeric = test.select_dtypes(include=numerics)

for skf in skewedfeatures:
    sk = skew[skew == skf].index[0]
    test_numeric[sk] = np.log1p(test_numeric[sk])
test_numeric.shape



nonnumeric = ['object']
test_nonnumeric = test.select_dtypes(include=nonnumeric)

# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore categorical data

# Calculates pearson co-efficient for all combinations
all_numeric = [test_numeric,dataset_numeric]
all_numeric = pd.concat(all_numeric)
all_numeric.shape
data_corr = all_numeric.loc[:, all_numeric.columns != 'SalePrice'].corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.7

# List of pairs along with correlation above threshold
corr_list = []

size = 36

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
for v,i,j in s_corr_list:
    test.drop([cols[j]],1,inplace=True)
    train.drop([cols[j]],1,inplace=True)

# plot the heatmap
sns.heatmap(data_corr, 
        xticklabels=data_corr.columns,
        yticklabels=data_corr.columns)
#let's proceed to one hot encoding of our categorical variables. 
non_numeric_train = pd.get_dummies(dataset_nonnumeric)

non_numeric_test = pd.get_dummies(test_nonnumeric)


#now adding the missing dummy variables present in train_nonnumeric in test_nonnumeric
missing_cols = set( non_numeric_train.columns ) - set( test_nonnumeric.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    non_numeric_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
non_numeric_test = non_numeric_test[non_numeric_train.columns]
#let's now print the shape of the encoded categorical variables 
print(non_numeric_train.shape)
print(non_numeric_test.shape)
train_dataset_encoded = [non_numeric_train,dataset_numeric]
train_dataset_encoded = pd.concat(train_dataset_encoded,axis=1)
train_dataset_encoded.shape
test_dataset_encoded = [non_numeric_test,test_numeric]
test_dataset_encoded = pd.concat(test_dataset_encoded, axis = 1)
test_dataset_encoded.shape
ID = test_dataset_encoded['Id']
test_dataset_encoded.drop(['Id'],1,inplace=True)
X_test = test_dataset_encoded
train_dataset_encoded["SalePrice"] = np.log1p(train["SalePrice"])

X_train = train_dataset_encoded.loc[:, train_dataset_encoded.columns != 'SalePrice']
y_train = train_dataset_encoded.loc[:, train_dataset_encoded.columns == 'SalePrice']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.8)

lasso_params_housing = {"alpha": np.arange(0.0, 1.0, 0.01)}
gs_params_housing = {"cv": 3, "n_jobs": -1, "verbose": 1}

scores = []
for name, clf, params in [("Lasso", Lasso(), lasso_params_housing)]:
    grid = GridSearchCV(estimator=clf, param_grid=params, **gs_params_housing)
    grid.fit(X_train2, y_train2)
    scores.append((name, grid.score(X_test2, y_test2), grid.best_params_))
    

for name, score, params in scores:
    print("Score {0}: {1:0.2f}\t".format(name, score), params)
lasso = Lasso(alpha= 0.01)
lasso.fit(X_train, y_train)

predictions=np.expm1(lasso.predict(X_test))

results_dataframe = pd.DataFrame({
    "Id" : ID,
    "SalePrice": predictions
})


results_dataframe.to_csv('ypred-housingprices.csv', index=False)  # Save prediction

#pred = pd.read_csv('ypred-housingprices.csv')
#pred