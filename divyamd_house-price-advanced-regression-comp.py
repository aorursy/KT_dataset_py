import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
data.head(6)
data.shape
data.columns
#find percentage of missing values in columns
perct = data.isnull().sum()/data.shape[0] *100
perct[perct!=0]
# Let's drop the columns with over 70% missing values
print(perct[perct>70])

data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"],axis = 1, inplace = True )
plt.figure(figsize = (20,4))
sns.heatmap(data.isnull(),yticklabels = False, cbar = False)
miss_val_col = data.columns[data.isnull().any()]
print(data[miss_val_col].dtypes)
miss_cat_col = data[miss_val_col].select_dtypes(include = 'object').columns

# Impute each missing value in categorical feature with most freq value
for each_col in miss_cat_col:
    data[each_col] = data[each_col].fillna(data[each_col].mode()[0])
cols = data.isnull().sum()
cols[cols>0]
data["GarageYrBlt"] = data["GarageYrBlt"].fillna(data["GarageYrBlt"].mean())
data["MasVnrArea"] = data["MasVnrArea"].fillna(data["MasVnrArea"].mean())
data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].mean())
cols = data.isnull().sum()
cols[cols>0]
catergorical_cols = data.select_dtypes(include = 'object').columns
catergorical_cols
# Lets write a function to perform 1 hot encoding on all the categorical features

def one_hot_encode(cols):
    data1 = big_dataset
    i = 0
    
    for each_col in cols:
        print(each_col)
        df = pd.get_dummies(big_dataset[each_col], drop_first = True)
        big_dataset.drop([each_col], axis = 1, inplace = True)
        
        if i==0:
            data1 = df.copy()
        else:
            data1 = pd.concat([data1, df], axis = 1)
        i = i + 1
        
    data1 = pd.concat([data1, big_dataset], axis = 1)
    
    return(data1)
# pd.get_dummies(data["MSZoning"], drop_first = True).head(4)
## There are several features whose categories are different in test and train dataset.
## In order to handle this, lets combine test and train 
test_df = pd.read_csv("./formulatedtest.csv")

big_dataset = pd.concat([data, test_df], axis = 0)
big_dataset.head()
big_dataset.shape
# Perform one hot encoding on the categorical columns
big_dataset = one_hot_encode(list(catergorical_cols))
big_dataset.shape 
# Observe that there are 238 columns in the latest dataset. More columns are created due to dummies
# Lets remove the duplicate columns
big_dataset = big_dataset.loc[:,~big_dataset.columns.duplicated()]
big_dataset.shape
# split train and test data
train_dataset = big_dataset[:1460]
test_dataset = big_dataset[1460:]

test_dataset.drop(["SalePrice"], axis = 1, inplace = True)

print(train_dataset.shape)
print(test_dataset.shape)
# To train the model, Lets choose X and y
X_train = train_dataset.drop(["SalePrice"], axis = 1)
y_train = train_dataset.SalePrice
import xgboost

model = xgboost.XGBRegressor()
model.fit(X_train, y_train)
# from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor()
# model.fit(X_train, y_train)

import pickle

filename = "final_model.pkl"
pickle.dump(model, open(filename,'wb'))
# Create sample submission file and submit

y_pred = pd.DataFrame(model.predict(test_dataset))

other_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
datasets = pd.concat([other_df['Id'], y_pred], axis = 1)

datasets.columns = ['Id', 'SalePrice']
datasets.to_csv("sample_submission.csv", index = False)
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 1]


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2,3,5,10,15]
learning_rate = [0.05, 0.1, 0.15, 0.20]
min_child_weight = [1,2,3,4]