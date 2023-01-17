import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics  # accuracy score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #decision trees learn from data to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model.
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.feature_selection import chi2, SelectKBest
df = pd.read_csv('train.csv')
df.head()
df.drop(['Id'],axis=1,inplace = True)
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
# see the null values
df.isnull().sum()
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage']).mean()
df.drop(['Alley'],axis=1,inplace=True)
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df.shape
df.isnull().sum()
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False, cbar = False)
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)
df.shape
df.head()


datacorr = df.corr()
datacorr
plt.figure(figsize = (20,20))
sns.heatmap(datacorr,annot = True)
plt.show()
# categorical feat 
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
def category_multcols(col):
    df_final = final_df
    i=0
    for fields in col:
        print(fields)
        df1 = pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace = True)
        if i==0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final,df1],axis=1)
        i=i+1
        
    df_final = pd.concat([final_df,df_final],axis =1)
    return df_final

# all the df concated with the categorical
main_df = df.copy()
test_df = pd.read_csv('formulatedtest.csv')
test_df.shape
test_df.head()
# combine the train data with test data
final_df = pd.concat([df,test_df],axis = 0)
final_df.shape
final_df = category_multcols(columns)
final_df.shape
final_df = final_df.loc[:,~final_df.columns.duplicated()]

# to remove duplicate columns 
final_df.shape
final_df.head()
df_Train = final_df.iloc[:1422,:]
df_Test = final_df.iloc[1422:,:]
df_Test.drop(['SalePrice'],axis=1,inplace = True)
df_Test.shape
x = df_Train.drop(['SalePrice'],axis=1)
y = df_Train['SalePrice']
import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(x,y)
from sklearn.ensemble import RandomForestRegressor
import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename,'wb'))
y_pred = classifier.predict(df_Test)
y_pred
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('sample_submission.csv')
datasets = pd.concat([sub_df['Id'],pred],axis = 1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index = False)
