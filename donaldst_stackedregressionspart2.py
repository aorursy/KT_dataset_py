#import some necessary libraries

import datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import time



from scipy import stats

from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics

from scipy.special import boxcox1p, inv_boxcox, inv_boxcox1p

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

#print(check_output(["ls", "-rlt", "../StackedRegression"]).decode("utf8")) #check the files available in the directory

#print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory



StartTime = datetime.datetime.now()
class MyTimer():

    # usage:

    #with MyTimer():                            

    #    rf.fit(X_train, y_train)

    

    def __init__(self):

        self.start = time.time()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        end = time.time()

        runtime = end - self.start

        msg = 'The function took {time} seconds to complete'

        print(msg.format(time=runtime))
a='''

#train = pd.read_csv('input/train.csv')

#test = pd.read_csv('input/test.csv')



# default competition

competition = 'SRP_2' # StackedRegression



try:

    a = check_output(["ls", "../input"]).decode("utf8") # new Kaggle competition

except:

    a=''

finally:

    print('')

try:

    b = check_output(["ls", "-rlt", "../StackedRegression"]).decode("utf8")

except:

    b=''

finally:

    print('')    

#if (competition == 'SRP_2'): # Stacked Regressions Part 2

if (len(a) > 0): # new competition

    competition = 'SR'

    train = pd.read_csv('../input/home-data-for-ml-course/train.csv')#,index_col='Id')

    test = pd.read_csv('../input/home-data-for-ml-course/test.csv')#,index_col='Id')

elif (len(b)): # run locally

    competition = 'SR'

    train = pd.read_csv('input/train.csv')

    test = pd.read_csv('input/test.csv')

else: # old competition

    competition = 'SRP_2'

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')'''
competition = 'SRP_2'

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
##display the first five rows of the train dataset.

train.head(5)
##display the first five rows of the test dataset.

test.head(5)

fig, ax = plt.subplots()

#ax.scatter(x = train['GrLivArea'], y = train['SalePrice']

ax.scatter(x = train['GrLivArea'], y = np.log(train['SalePrice']))

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)



#m, b = np.polyfit(train['GrLivArea'], train['SalePrice'], 1)

m, b = np.polyfit(train['GrLivArea'], np.log(train['SalePrice']), 1)

#m = slope, b=intercept

plt.plot(train['GrLivArea'], m*train['GrLivArea'] + b)



plt.show()

train.shape[1]

#a = int(np.sqrt(train.shape[1]))

a = 4

b = int(train.shape[1]/4)

r = int(train.shape[1]/a)

c = int(train.shape[1]/b)

i = 0

fig, ax = plt.subplots(nrows=r, ncols=c, figsize=(15, 60))

for row in ax:

    for col in row:

        #print(train.columns[i])

        #print(train[train.columns[i]].dtype)

        #col.plot()

        #ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

        try:

            #col.scatter(x = train[train.columns[i]], y = train['SalePrice'])

            col.scatter(x = train[train.columns[i]], y = np.log(train['SalePrice']))

            col.title.set_text(train.columns[i])

            #col.title(train.columns[i])

        except:

            temp=1

        #except Exception as e:

        #    print(e.message, e.args)

        finally:

            temp=1

        i = i + 1

        

plt.show()
#Deleting outliers



#train = train.drop(train[(train['LotFrontage']>300) & (train['SalePrice']<400000)].index)

#train = train.drop(train[(train['LotArea']>100000) & (train['SalePrice']<400000)].index)

#train = train.drop(train[(train['BsmtFinSF1']>4000) & (train['SalePrice']<250000)].index)

#train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<250000)].index)



train = train.drop(train[(train['OverallQual']>9) & (train['SalePrice']<220000)].index)

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

test
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']





#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
#Check the graphic again

fig, ax = plt.subplots()

#ax.scatter(train['GrLivArea'], train['SalePrice'])

ax.scatter(train['GrLivArea'], np.log(train['SalePrice']))

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

#m, b = np.polyfit(train['GrLivArea'], train['SalePrice'], 1)

m, b = np.polyfit(train['GrLivArea'], np.log(train['SalePrice']), 1)

#m = slope, b=intercept

plt.plot(train['GrLivArea'], m*train['GrLivArea'] + b)

plt.show()
# linear 

x_data = train['GrLivArea']

y_data = train['SalePrice']

log_y_data = np.log(train['SalePrice'])



curve_fit = np.polyfit(x_data, log_y_data, 1)

print(curve_fit)

y = np.exp(1.11618915e+01) * np.exp(5.70762509e-04*x_data)

plt.plot(x_data, y_data, "o")

plt.scatter(x_data, y,c='red')
# linear with log y

y = np.exp(1.11618915e+01) * np.exp(5.70762509e-04*x_data)

plt.plot(x_data, np.log(y_data), "o")

plt.scatter(x_data, np.log(y),c='red')
# polynomial

x_data = train['GrLivArea']

y_data = train['SalePrice']

log_y_data = np.log(train['SalePrice'])

curve_fit_gla = np.polyfit(x_data, y_data, 2)

y = curve_fit_gla[2] + curve_fit_gla[1]*x_data + curve_fit_gla[0]*x_data**2
plt.plot(x_data, y_data, "o")

plt.scatter(x_data, y,c='red')
# polynomial with log y

plt.plot(x_data, np.log(y_data), "o")

plt.scatter(x_data, np.log(y),c='red')
y_data #HERE
# polynomial with log y

plt.plot(x_data, np.log(y_data), "o")

plt.scatter(x_data, np.log(y),c='red')
sns.distplot(train['SalePrice'] , fit=norm)



_skew = skew(train['SalePrice'])

_kurtosis = kurtosis(train['SalePrice'])



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}, skew = {:.2f} kurtosis = {:.2f}\n'.format(mu, sigma, _skew, _kurtosis))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()

#We try the numpy function log1p which  applies log(1+x) to all elements of the column



# option 1 - original

#train["SalePrice"] = np.log1p(train["SalePrice"])



# Option 2: use box-cox transform - this performs worse than the log(1+x), reverting back...now it seems to be scoring better. Maybe my earlier evaluation was incorrect

# try different alpha values  between 0 and 1

lam_l = 0.35

train["SalePrice"] = boxcox1p(train["SalePrice"], lam_l) 



# Option 3: boxcox letting the algorithm select lmbda based on least-likelihood calculation

#train["SalePrice"], lam_l = boxcox1p(x=train["SalePrice"], lmbda=None)



# option 4 - compare to log1p => score is same

#train["SalePrice"] = np.log(train["SalePrice"])
x = np.linspace(0, 20)

y1 = np.log(x)

y2 = np.log1p(x)

y3 = boxcox1p(x, 0.35)

y4 = boxcox1p(x, 0.10)

y5 = boxcox1p(x, 0.50)

plt.plot(x,y1,label='log') 

plt.plot(x,y2,label='log1p') 

plt.plot(x,y3,label='boxcox .35') 

plt.plot(x,y4,label='boxcox .10') 

plt.plot(x,y5,label='boxcox .50') 

plt.legend()

plt.show()



x = np.linspace(0, 100000)

y1 = np.log(x)

y2 = np.log1p(x)

y3 = boxcox1p(x, 0.35)

y4 = boxcox1p(x, 0.10)

y5 = boxcox1p(x, 0.50)

plt.plot(x,y1,label='log') 

plt.plot(x,y2,label='log1p') 

plt.plot(x,y3,label='boxcox .35') 

plt.plot(x,y4,label='boxcox .10') 

plt.plot(x,y5,label='boxcox .50') 

plt.legend()

plt.show()
# to revert back use y = inv_boxcox1p(x, lam_l) => train["SalePrice"] = inv_boxcox1p(train["SalePrice"], lam_l)

# think the underlying formula is this: # pred_y = np.power((y_box * lambda_) + 1, 1 / lambda_) - 1





#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



_skew = skew(train['SalePrice'])

_kurtosis = kurtosis(train['SalePrice'])



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}, skew = {:.2f} kurtosis = {:.2f}\n'.format(mu, sigma, _skew, _kurtosis))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
def add_gla(row, p):

    return (p[2] + p[1]*row.GrLivArea + p[0]*(row.GrLivArea**2))



#all_data['GrLivAreaPoly'] = all_data.apply(lambda row: add_gla(row,curve_fit_gla), axis=1)

#all_data[['GrLivAreaPoly','GrLivArea']].head()
all_data.GrLivArea[:ntrain]
correlation_matrix = np.corrcoef(all_data.GrLivArea[:ntrain], y_train)

correlation_xy = correlation_matrix[0,1]

r_squared = correlation_xy**2



print(r_squared)
a='''

correlation_matrix = np.corrcoef(all_data.GrLivAreaPoly[:ntrain], y_train)

correlation_xy = correlation_matrix[0,1]

r_squared = correlation_xy**2



print(r_squared)'''
for i in range(1,11,1):

    j = i/10

    correlation_matrix = np.corrcoef(all_data.GrLivArea[:ntrain]**j, y_train)

    correlation_xy = correlation_matrix[0,1]

    r_squared = correlation_xy**2



    print(j, r_squared)
def add_gla2(row, p):

    return (row.GrLivArea**p)



#all_data['GrLivAreaRoot'] = all_data.apply(lambda row: add_gla2(row,0.3), axis=1)
all_data.head()
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(30)
all_numerical = all_data.select_dtypes(include=['number']).columns.tolist()

missing_data.index.values.tolist()

missing_df = all_data[missing_data.index.values.tolist()]

missing_numerical = missing_df.select_dtypes(include=['number']).columns.tolist()
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = train.corr(method='kendall')

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
print (corrmat['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corrmat['SalePrice'].sort_values(ascending=False)[-5:])
all_data.shape
ImputeToNone = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for col in ImputeToNone:  

    all_data[col].fillna("None", inplace=True)
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

#all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

#    lambda x: x.fillna(x.median()))
# selecting another column

all_data[missing_numerical]
corrmat_all = all_data[all_numerical].corr(method='kendall')

corrmat_all
corrmat2 = all_data[missing_numerical].corr(method='kendall')

corrmat2

#print (corrmat2['LotFrontage'].sort_values(ascending=False), '\n')
all_data
!pip install scikit-learn==0.20
!conda install -c anaconda scikit-learn==0.20
def ImputeData(all_data, numerical_input, col_to_impute):

    from sklearn.impute import KNNImputer

    

    Missing = all_data[numerical_input]

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

    imputer.fit(Missing)

    Xtrans = imputer.transform(Missing)

    df_miss = pd.DataFrame(Xtrans,columns = Missing.columns)

    all_data[col_to_impute] = df_miss[col_to_impute]

    return (all_data)

    
all_data = ImputeData(all_data, all_numerical, 'LotFrontage')
all_data
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'].value_counts()
# this one may be a bit dangerous, maybe try to get zone from neighborhood most common value, similar to LotFrontage previously

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



# NEW, slightly worse score

#all_data["MSZoning"] = all_data.groupby("Neighborhood")["MSZoning"].transform(

#    lambda x: x.fillna(x.mode()))
all_data['Utilities'].value_counts()
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
all_data['MSSubClass'].value_counts()

all_data['OverallCond'].value_counts()
all_data.shape
import datetime

Yr = all_data['YrSold'].min()

Mo = all_data['MoSold'].min()

t = datetime.datetime(Yr, Mo, 1, 0, 0)



def calculateYrMo (row):   

    return int((datetime.datetime(row.YrSold,row.MoSold,1) - t).total_seconds())
# either way will work

#all_data['YrMoSold'] = all_data.apply(lambda row: int((datetime.datetime(row.YrSold,row.MoSold,1) - t).total_seconds()), axis=1)



all_data['YrMoSold'] = all_data.apply(lambda row: calculateYrMo(row), axis=1)
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder

#cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

#        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

#        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

#        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

#        'YrSold', 'MoSold', 'YrMoSold')



# Edit: Dropping PoolQC for missing values => makes the model worse, reverting

#all_data = all_data.drop(['PoolQC'], axis=1)



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'YrMoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape

print('Shape all_data: {}'.format(all_data.shape))

#all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# feature engineering add new features 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd'] # A-

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']) # B-

#all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])) # C-

#all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']) # D-

#all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0) # E-

#all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0) # F-

#all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0) # G-

#all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0) # H-

#all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0) # I-

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index



lam_f = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam_f)

    

    #all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data)

print(all_data.shape)
correlations = corrmat['SalePrice'].sort_values(ascending=False)

df_corr = correlations.to_frame()

print(df_corr.query("abs(SalePrice) < 0.011"))

low_corr = df_corr.query("abs(SalePrice) < 0.011").index.values.tolist()

#print('dropping these columns for low correlation', low_corr)

#for i in low_corr: 

#    all_data = all_data.drop([i], axis=1)
# to choose number of components, look at this chart. Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

from sklearn.decomposition import PCA



pca = PCA().fit(all_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')
#do PCA/StandardScaler+clip here or before the skew boxcox1p transform



n_components = 50

pca = PCA(n_components=n_components)

all_data_pca = pca.fit_transform(all_data)
print(all_data.shape)

print(all_data_pca.shape)
weights = np.round(pca.components_, 3) 

ev = np.round(pca.explained_variance_ratio_, 3)

print('explained variance ratio',ev)

pca_wt = pd.DataFrame(weights)#, columns=all_data.columns)

pca_wt.head()
pca_wt.shape 
#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = pd.DataFrame(all_data_pca).corr(method='kendall')

plt.subplots(figsize=(12,9))

plt.title("Kendall's Correlation Matrix PCA applied", fontsize=16)

sns.heatmap(corrmat, vmax=0.9, square=True)





#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = train.corr(method='kendall')

plt.subplots(figsize=(12,9))

plt.title("Kendall's Correlation Matrix Initial Train Set", fontsize=16)

sns.heatmap(corrmat, vmax=0.9, square=True);

train_orig = train.copy()

train_orig['SalePrice'] = y_train

corrmat = train_orig.corr(method='kendall')

print (corrmat['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corrmat['SalePrice'].sort_values(ascending=False)[-5:])
a=''' not working now

correlations = corrmat["SalePrice"].sort_values(ascending=False)

features = correlations.index[0:10]



sns.pairplot(train[features], size = 2.5)

plt.show();'''
print(type(all_data))

print(type(pd.DataFrame(all_data_pca)))
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

rc = RobustScaler()



use_pca = 0 # using PCA currently hurts the score

use_normalization = 0 # using StandardScaler doesn't work, try RobustScaler now



if (use_pca == 1):

    all_data_pca = pd.DataFrame(all_data_pca)

    train = all_data_pca[:ntrain]

    test = all_data_pca[ntrain:]

    all_data_pca.head()

elif (use_normalization == 1):

    train = all_data[:ntrain]

    test = all_data[ntrain:]

    scaled_features = sc.fit_transform(train.values)

    train = pd.DataFrame(scaled_features, index=train.index, columns=train.columns)

    scaled_features = sc.transform(test.values)

    test = pd.DataFrame(scaled_features, index=test.index, columns=test.columns)   

elif (use_normalization == 2):

    train = all_data[:ntrain]

    test = all_data[ntrain:]

    scaled_features = rc.fit_transform(train.values)

    train = pd.DataFrame(scaled_features, index=train.index, columns=train.columns)

    scaled_features = rc.transform(test.values)

    test = pd.DataFrame(scaled_features, index=test.index, columns=test.columns)  

else:

    # back to original splits (from train.csv and test.csv)

    train = all_data[:ntrain]

    test = all_data[ntrain:]

save_data = 0

if (save_data == 1):

    df1 = train.copy()

    df1['SalePrice'] = inv_boxcox1p(y_train, lam_l)

    df1.insert(0, 'Id', list(train_ID), allow_duplicates=False)

    df1.to_csv('HousePricesTrain.csv', index=False)  

    df2 = test.copy()

    df2.insert(0, 'Id', list(test_ID), allow_duplicates=False)

    df2.to_csv('HousePricesTest.csv', index=False) 
#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = train.corr(method='kendall')

plt.subplots(figsize=(24,18))

sns.heatmap(corrmat, vmax=0.9, square=True)
#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = test.corr(method='kendall')

plt.subplots(figsize=(24,18))

sns.heatmap(corrmat, vmax=0.9, square=True)
train.hist(bins=20, figsize=(30,20))

plt.show()
test.hist(bins=20, figsize=(30,20))

plt.show()
train.describe()
test.describe()
a='''

from sklearn.model_selection import KFold, cross_val_score, train_test_split

X = train

y = y_train

train, val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=201)#, stratify=y)'''
print("Modelling: ", datetime.datetime.now())
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression, Ridge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor



#from sklearn.metrics import mean_squared_log_error

# to run locally: conda install -c anaconda py-xgboost

import xgboost as xgb

import lightgbm as lgb

#Validation function

# train.values and y_train are both log scaled so just need to take the square of the delta between them to calculate the error, then take the sqrt to get rmsle

# but for now y_train is boxcox1p(), not log(). Use this to convert back: inv_boxcox1p(y_train, lam_l)

n_folds=5 # was 5 => better score but twice as slow now



def rmsle_cv(model):

    print("running rmsle_cv code")

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42

    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf)) # also r2

    print("raw rmse scores for each fold:", rmse)

    return(rmse)



def r2_cv(model):

    print("running r2_cv code")

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42

    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    r2= cross_val_score(model, train.values, y_train, scoring="r2", cv = kf) # also r2

    print("raw r2 scores for each fold:", r2)

    return(r2)



# used for another competition

def mae_cv(model):

    print("running mae_cv code")

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42

    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    mae = -cross_val_score(model, train.values, y_train, scoring="neg_mean_absolute_error", cv = kf) # also r2

    print("raw mae scores for each fold:", mae)

    return(mae)



def all_cv(model, n_folds, cv):

    print("running cross_validate")

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42

    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    scorers = {

        'r2': 'r2',

        'nmsle': 'neg_mean_squared_log_error', 

        'nmse': 'neg_mean_squared_error',

        'mae': 'neg_mean_absolute_error'

    }

    scores = cross_validate(model, train.values, y_train, scoring=scorers,

                           cv=kf, return_train_score=True)

    return(scores)
print(y_train.mean())

print(inv_boxcox1p(y_train, lam_l).mean())
def runGSCV(num_trials, features, y_values):

    non_nested_scores = np.zeros(num_trials) # INCREASES BIAS

    nested_scores = np.zeros(num_trials)

    # Loop for each trial

    for i in range(num_trials):

        print("Running GridSearchCV:")

        with MyTimer():    

            #grid_result = gsc.fit(train, y_train)  

            grid_result = gsc.fit(features, y_values)  

        non_nested_scores[i] = grid_result.best_score_

        if (competition == 'SR'):

            print("Best mae %f using %s" % ( -grid_result.best_score_, grid_result.best_params_))

        else:

            print("Best rmse %f using %s" % ( np.sqrt(-grid_result.best_score_), grid_result.best_params_))

        

        # nested/non-nested cross validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

        with MyTimer():    

            #nested_score = cross_val_score(gsc, X=train, y=y_train, cv=outer_cv, verbose=0).mean() 

            nested_score = cross_val_score(gsc, X=features, y=y_values, cv=outer_cv, verbose=0).mean() 

            # source code for cross_val_score is here: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py#L137

        if (competition == 'SR'):

            print("nested mae score from KFold %0.3f" % -nested_score)

        else:

            print("nested rmse score from KFold %0.3f" % np.sqrt(-nested_score))

        

        nested_scores[i] = nested_score

        print('grid_result',grid_result)

        #if (competition == 'SR'):

        print("mean scores: r2(%0.3f) mae(%0.3f) nmse(%0.3f) nmsle(%0.3f)" % (grid_result.cv_results_['mean_test_r2'].mean(), -grid_result.cv_results_['mean_test_mae'].mean(),  np.sqrt(-grid_result.cv_results_['mean_test_nmse'].mean()), grid_result.cv_results_['mean_test_nmsle'].mean() ))

        #print("mean scores: r2(%0.3f) nmse(%0.3f) mae(%0.3f)" % (grid_result.cv_results_['mean_test_r2'].mean(), np.sqrt(-grid_result.cv_results_['mean_test_nmse'].mean()), grid_result.cv_results_['mean_test_mae'].mean()))

    return grid_result

def calc_scores(model):

    score_mae = mae_cv(model)

    print("\n mae_cv score: {:.4f} ({:.4f})\n".format(score_mae.mean(), score_mae.std()))

    score_rmsle = rmsle_cv(model)

    print("\n rmsle_cv score: {:.4f} ({:.4f})\n".format(score_rmsle.mean(), score_rmsle.std()))

    score_r2 = r2_cv(model)

    print("\n r2_cv score: {:.4f} ({:.4f})\n".format(score_r2.mean(), score_r2.std()))

    return (score_mae, score_rmsle, score_r2)



# calculate all 3 at once, takes 1/3 the time as calc_scores

def calc_all_scores(model, n_folds=5, cv=5):

    scores = all_cv(model, n_folds, cv)

    #scores['train_<scorer1_name>'']

    #scores['test_<scorer1_name>'']

    print("\n mae_cv score: {:.4f} ({:.4f})\n".format( (-scores['test_mae']).mean(), scores['test_mae'].std() ))

    print("\n rmsle_cv score: {:.4f} ({:.4f})\n".format( (np.sqrt(-scores['test_nmse'])).mean(), scores['test_nmse'].std() ))

    print("\n r2_cv score: {:.4f} ({:.4f})\n".format( scores['test_r2'].mean(), scores['test_r2'].std() ))

    return (scores)



# useful when you can't decide on parameter setting from best_params_

# result_details(grid_result,'mean_test_nmse',100)

def result_details(grid_result,sorting='mean_test_nmse',cols=100):

    param_df = pd.DataFrame.from_records(grid_result.cv_results_['params'])

    param_df['mean_test_nmse'] = np.sqrt(-grid_result.cv_results_['mean_test_nmse'])

    param_df['std_test_nmse'] = np.sqrt(grid_result.cv_results_['std_test_nmse'])

    param_df['mean_test_mae'] = -grid_result.cv_results_['mean_test_mae']

    param_df['std_test_mae'] = -grid_result.cv_results_['std_test_mae']

    param_df['mean_test_r2'] = -grid_result.cv_results_['mean_test_r2']

    param_df['std_test_r2'] = -grid_result.cv_results_['std_test_r2']

    return param_df.sort_values(by=[sorting]).tail(cols)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def mae(y, y_pred):

    return mean_absolute_error(y,y_pred)
# initialize the algorithm for the GridSearchCV function

lasso = Lasso()

tuningLasso = 1 # takes 2 minutes to complete



if (tuningLasso == 1):

    # use this when tuning

    param_grid={

        'alpha':[0.01,], # done, lower keeps getting better, but don't want to go too low and begin overfitting (alpha is related to L1 reg)

        'fit_intercept':[True], # done, big difference

        'normalize':[False], # done, big difference

        'precompute':[False], # done, no difference

        'copy_X':[True], # done, no difference

        'max_iter':[200], # done

        'tol':[0.005], # done, not much difference

        'warm_start':[False], # done, no difference

        'positive':[False], # done, big difference

        'random_state':[1],

        'selection':['cyclic'] # done both are same, cyclic is default

    }



else:

    # use this when not tuning

    param_grid={

        'alpha':[0.2],

        'fit_intercept':[True],

        'normalize':[False],

        'precompute':[False],

        'copy_X':[True],

        'max_iter':[200],

        'tol':[0.0001],

        'warm_start':[False],

        'positive':[False],

        'random_state':[None],

        'selection':['cyclic']

    }



scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error', 

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=lasso,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(2, train, y_train)



rd = result_details(grid_result,'random_state',100)

rd[['random_state','alpha','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['random_state','alpha'])

tuning_lasso = 1

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) # was 1

lasso_new = make_pipeline(RobustScaler(), Lasso(**grid_result.best_params_))

#l = "{'alpha': 0.2, 'copy_X': True, 'fit_intercept': True, 'max_iter': 200, 'normalize': False, 'positive': False, 'precompute': False, 'random_state': 1, 'selection': 'cyclic', 'tol': 0.005, 'warm_start': False}"

#Lasso_new = make_pipeline(RobustScaler(), Lasso(**l))

#lasso_ss = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1)) # was 1 => worse score
if (tuning_lasso == 1):

    #TEMP

    model_results = [] # model flow, mae, rmsle

    models = [lasso, lasso_new]



    for model in models:

        #print(model)

        with MyTimer(): 

            scores = calc_all_scores(model,5,5)

        #print("------------------------------------------")

        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

    df_mr.sort_values(by=['rmsle'])
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) # was 1



if (tuning_lasso == 1):

    for i in [2,5,20,42,99]:

        from sklearn.linear_model import Lasso

        print('random_state =',i)



        l = {'alpha': 0.01, 'copy_X': True, 'fit_intercept': True, 'max_iter': 200, 'normalize': False, 'positive': False, 'precompute': False, 'selection': 'cyclic', 'tol': 0.005, 'warm_start': False}

        lasso_new = make_pipeline(RobustScaler(), Lasso(**l, random_state=i))

        #lasso_new = Lasso(**l, random_state=i)



        model_results = [] # model flow, mae, rmsle

        models = [lasso, lasso_new]



        for model in models:

            #print(model)

            with MyTimer(): 

                scores = calc_all_scores(model,5,5)

            #print("------------------------------------------")

            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

        print(df_mr.sort_values(by=['rmsle']))

else:

    lasso_new = make_pipeline(RobustScaler(), Lasso(**grid_result.best_params_, random_state=i))
model_results
# initialize the algorithm for the GridSearchCV function

ENet = ElasticNet()

tuningENet = 0 # takes 2 minutes to complete



if (tuningENet == 1):

    # use this when tuning

    param_grid={

        'alpha':[0.005,0.01,0.05,0.1],

        'l1_ratio':[.6,.65,.7,.75,.8,.85,.9],

        'fit_intercept':[True], # ,False

        'normalize':[False], # True,

        'max_iter':range(50,500,50),

        'selection':['random'], # 'cyclic',

        'random_state':[None]

    }



else:

    # use this when not tuning

    param_grid={

        'alpha':[0.05],

        'l1_ratio':[.85],

        'fit_intercept':[True],

        'normalize':[False],

        'max_iter':[100], # default 1000

        'selection':['random']

    }



scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error',

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=ENet,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(2, train, y_train)



rd = result_details(grid_result,'mean_test_nmse',100)

rd[['alpha','l1_ratio','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])

#ENet_orig = make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

ENet = make_pipeline(StandardScaler(), ElasticNet(**grid_result.best_params_, random_state=3))

ENet_new = make_pipeline(RobustScaler(), ElasticNet(**grid_result.best_params_, random_state=3))

# initialize the algorithm for the GridSearchCV function

KRR = KernelRidge()

tuningKRR = 0 # this took 40 mins, 20 per iteration



if (tuningKRR == 1):

    # use this when tuning

    param_grid={

        'alpha':[2.0,2.2,2.4,2.6], 

        'kernel':['polynomial'], #for entire list see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics

        'gamma':[0.0001,0.001,0.01,0.1],

        'degree':[1,2,3,4,5,6], 

        'coef0':[0.1,0.3,0.5,1.0,2.0]

    }



else:

    # use this when not tuning

    # nmse: Best mae 583416973.611280 using {'alpha': 2.2, 'coef0': 0.5, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}

    # mae: Best mae 15805.764347 using {'alpha': 2.0, 'coef0': 0.1, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}

    param_grid={

        'alpha':[2.2], 

        'kernel':['polynomial'], # 'linear', 'rbf'

        'gamma':[0.001],

        'degree':[5], 

        'coef0':[0.5]

    }

scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error',

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=KRR,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(2, train, y_train)



rd = result_details(grid_result,'mean_test_nmse',100)

rd[['alpha','degree','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])

#KRR_orig = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

KRR = KernelRidge(**grid_result.best_params_)
# initialize the algorithm for the GridSearchCV function

rf = RandomForestRegressor()

tuningRF = 0 # this took 2 hours last time, 1 hour per iteration



if (tuningRF == 1):

    # use this when tuning

    param_grid={

        'max_depth':[3,4,5],

        'max_features':[None,'sqrt','log2'], 

        # 'max_features': range(50,401,50),

        # 'max_features': [50,100], # can be list or range or other

        'n_estimators':range(25,100,25), 

        #'class_weight':[None,'balanced'],  

        'min_samples_leaf':range(5,15,5), 

        'min_samples_split':range(10,30,10), 

        'criterion':['mse', 'mae'] 

    }



else:

    # use this when not tuning

    param_grid={

        'max_depth':[5],

        'max_features':[None], # max_features is None is default and works here, removing 'sqrt','log2'

        # 'max_features': range(50,401,50),

        # 'max_features': [50,100], # can be list or range or other

        'n_estimators': [50], # number of trees selecting 100, removing range(50,126,25)

        #'class_weight':[None], # None was selected, removing 'balanced'

        'min_samples_leaf': [5], #selecting 10, removing range 10,40,10)

        'min_samples_split': [10], # selecting 20, removing range(20,80,10),

        'criterion':['mse'] # remove gini as it is never selected

    }



scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error',

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(2, train, y_train)



rd = result_details(grid_result,'mean_test_nmse',100)

rd[['n_estimators','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])

#RF_orig = make_pipeline(StandardScaler(), RandomForestRegressor(max_depth=3,n_estimators=500))

RF = make_pipeline(StandardScaler(), RandomForestRegressor(**grid_result.best_params_)) # better than default, but still not good

RF_new = make_pipeline(RobustScaler(), RandomForestRegressor(**grid_result.best_params_)) # better than default, but still not good
print("Optimize GBoost: ", datetime.datetime.now())
GBoost_orig = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state=5) # was 5



tuning_gb = 0

if (tuning_gb == 1):

    # initialize the algorithm for the GridSearchCV function

    GBoost = GradientBoostingRegressor()

    tuningGB = 0



    if (tuningGB == 1):

        # use this when tuning

        param_grid={

            #'loss':['ls','lad','huber','quantile'],

            'loss':['huber'], # done

            'learning_rate':[0.05],

            'n_estimators':[2000], # done

            'subsample':[1.0],

            #'criterion':['friedman_mse','mse','mae'],

            'criterion':['friedman_mse'], # done

            'min_samples_split':[10],

            'min_samples_leaf':[15],

            'min_weight_fraction_leaf':[0.0],

            'max_depth':[2], # done

            'min_impurity_decrease':[0.0],

            'min_impurity_split':[None],

            'init':[None],

            'random_state':[None],

            #'max_features':[None,'auto','sqrt','log2'],

            'max_features':['sqrt'], # done

            'alpha':[0.60], # done

            'verbose':[0],

            'max_leaf_nodes':[None],

            'warm_start':[False],

            'presort':['deprecated'],

            'validation_fraction':[0.1],

            'n_iter_no_change':[None],

            'tol':[0.0001],

            'ccp_alpha':[0.0],

            'random_state':[5]

        }



    else:

        # use this when not tuning

        param_grid={

            'loss':['huber'], 

            'learning_rate':[0.05],

            'n_estimators':[2000], 

            'subsample':[1.0],

            'criterion':['friedman_mse'], 

            'min_samples_split':[10],

            'min_samples_leaf':[15],

            'min_weight_fraction_leaf':[0.0],

            'max_depth':[2], 

            'min_impurity_decrease':[0.0],

            'min_impurity_split':[None],

            'init':[None],

            'random_state':[None],

            'max_features':['sqrt'], 

            'alpha':[0.60], 

            'verbose':[0],

            'max_leaf_nodes':[None],

            'warm_start':[False],

            'presort':['deprecated'],

            'validation_fraction':[0.1],

            'n_iter_no_change':[None],

            'tol':[0.0001],

            'ccp_alpha':[0.0],

            'random_state':[5]

        }

    scorers = {

        'r2': 'r2',

        'nmsle': 'neg_mean_squared_log_error',

        'nmse': 'neg_mean_squared_error',

        'mae': 'neg_mean_absolute_error'

    }

    # To be used within GridSearch 

    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

    # To be used in outer CV 

    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



    #inner loop KFold example:

    gsc = GridSearchCV(

        estimator=GBoost,

        param_grid=param_grid,

        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

        verbose=0,

        return_train_score=True, # keep the other scores

        refit='nmse' # use this one for optimizing

    )



    grid_result = runGSCV(2, train, y_train)



rd = result_details(grid_result,'mean_test_nmse',100)

rd[['n_estimators','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])



if (tuning_gb == 1):

    GBoost = GradientBoostingRegressor(**grid_result.best_params_)

else:

    gbr = gbr  = {'alpha': 0.6, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'huber', 'max_depth': 2, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 15, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2000, 'n_iter_no_change': None, 'presort': 'deprecated', 'random_state': None, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    GBoost = GradientBoostingRegressor(**gbr)
if (tuning_gb == 1):

    #TEMP

    model_results = [] # model flow, mae, rmsle

    models = [GBoost_orig, GBoost]



    for model in models:

        #print(model)

        with MyTimer(): 

            scores = calc_all_scores(model,5,5)

        #print("------------------------------------------")

        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

    df_mr.sort_values(by=['rmsle'])
if (tuning_gb == 1):

    #TEMP



    # random_state=None

    #new  = {'alpha': 0.6, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'huber', 'max_depth': 2, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 15, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2000, 'n_iter_no_change': None, 'presort': 'deprecated', 'random_state': 5, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    #GBoost_new = GradientBoostingRegressor(**new)



    model_results = [] # model flow, mae, rmsle

    models = [GBoost_orig, GBoost]



    for model in models:

        #print(model)

        with MyTimer(): 

            scores = calc_all_scores(model,10,10)

        #print("------------------------------------------")

        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

    df_mr.sort_values(by=['rmsle'])
print("Optimize XGB: ", datetime.datetime.now())
tuning_xgb = 0

if (tuning_xgb == 1):

    # initialize the algorithm for the GridSearchCV function# initialize the algorithm for the GridSearchCV function

    xgb1 = xgb.XGBRegressor()

    tuningXGB = 1 # this took 2 hours last time, 1 hour per iteration



    if (tuningXGB == 1):

        # use this when tuning

        param_grid={

            'colsample_bytree':[0.4603],

            'gamma':[0.0468], # done - all values almost identical results

            'colsample_bylevel':[0.3], # done - all give same result

            'objective':['reg:squarederror'], # done - Default:'reg:squarederror', None, reg:pseudohubererror, reg:squaredlogerror, reg:gamma

            'booster':['gbtree'], # done - Default: 'gbtree', 'gblinear' or 'dart'

            'learning_rate':[0.04], # done

            'max_depth':[3], # - done

            'importance_type':['gain'], # done - all give same value, Default:'gain', 'weight', 'cover', 'total_gain' or 'total_cover'

            'min_child_weight':[1.7817], # done - no difference with several values

            'n_estimators':[1000], # done

            'reg_alpha':[0.4], # done

            'reg_lambda':[0.8571], # done

            'subsample':[0.5], # done

            'silent':[1],

            'random_state':[7],

            'scale_pos_weight':[1],

            'eval_metric':['rmse'], # done - all options have same results  Default:rmse for regression rmse, mae, rmsle, logloss, cox-nloglik

            #'nthread ':[-1],

            'verbosity':[0]

        }



    else:

        # use this when not tuning

        param_grid={

            'colsample_bytree':[0.4603],

            'gamma':[0.0468],

            'colsample_bylevel':[0.3],

            'objective':['reg:squarederror'], # 'binary:logistic', 'reg:squarederror', 'rank:pairwise', None

            'booster':['gbtree'], # 'gbtree', 'gblinear' or 'dart'

            'learning_rate':[0.04],

            'max_depth':[3],

            'importance_type':['gain'], # 'gain', 'weight', 'cover', 'total_gain' or 'total_cover'

            'min_child_weight':[1.7817],

            'n_estimators':[1000],

            'reg_alpha':[0.4],

            'reg_lambda':[0.8571],

            'subsample':[0.5],

            'silent':[1],

            'random_state':[7],

            'scale_pos_weight':[1],

            'eval_metric':['rmse'],

            #'nthread ':[-1],

            'nthread ':[-1],

            'verbosity':[0]

        }



    scorers = {

        'r2': 'r2',

        'nmsle': 'neg_mean_squared_log_error',

        'nmse': 'neg_mean_squared_error',

        'mae': 'neg_mean_absolute_error'

    }

    # To be used within GridSearch 

    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



    # To be used in outer CV 

    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



    #inner loop KFold example:

    gsc = GridSearchCV(

        estimator=xgb1,

        param_grid=param_grid,

        #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

        #scoring='neg_mean_squared_error', # or look here for other choices 

        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        #cv=5,

        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

        verbose=0,

        return_train_score=True, # keep the other scores

        refit='nmse' # use this one for optimizing

    )



    grid_result = runGSCV(2, train, y_train)

if (tuning_xgb == 1):

    rd = result_details(grid_result,'mean_test_nmse',100)



    rd[['random_state','eval_metric','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['random_state','mean_test_nmse'])

model_xgb = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'

                             learning_rate=0.05, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state=7) # was random_state=7, cannot set to None 

# maybe an issue with silent=1...



if (tuning_xgb == 1):

    import xgboost as xgb

    for i in [2,5,20,42,99]:

        print('random_state =',i)



        model_xgb_new = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'

                                 learning_rate=0.04, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)

                                 min_child_weight=1.7817, n_estimators=1000,

                                 reg_alpha=0.4, reg_lambda=0.8571,

                                 subsample=0.45, silent=1,

                                 random_state=i) # was random_state=7, cannot set to None 



        model_xgb_new2 = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'

                                 learning_rate=0.04, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)

                                 min_child_weight=1.7817, n_estimators=1000,

                                 reg_alpha=0.4, reg_lambda=0.8571,

                                 subsample=0.5, silent=1,

                                 random_state=i) # was random_state=7, cannot set to None 



        model_xgb_new3 = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'

                                 learning_rate=0.04, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)

                                 min_child_weight=1.7817, n_estimators=1000,

                                 reg_alpha=0.4, reg_lambda=0.8571,

                                 subsample=0.5213, silent=1,

                                 random_state=i) # was random_state=7, cannot set to None



        model_results = [] # model flow, mae, rmsle

        models = [model_xgb, model_xgb_new, model_xgb_new2, model_xgb_new3]



        for model in models:

            #print(model)

            with MyTimer(): 

                scores = calc_all_scores(model,5,5)

            #print("------------------------------------------")

            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

        print(df_mr.sort_values(by=['rmsle']))

else:

    model_xgb_new = xgb.XGBRegressor(**grid_result.best_params_)
import graphviz

model_xgb.fit(train, y_train,  verbose=False) #  eval_set=[(X_test, y_test)]

xgb.plot_importance(model_xgb)

xgb.to_graphviz(model_xgb, num_trees=20)
print("Optimize LightGBM: ", datetime.datetime.now())
tuning_lgb = 0

if (tuning_lgb == 1):

    # initialize the algorithm for the GridSearchCV function

    #model_lgb = lgb.LGBMRegressor(boosting_type='gbdt', max_depth=- 1, 

    #                            subsample_for_bin=200000, 

    #                            objective='regression',num_leaves=5,

    #                            learning_rate=0.05, n_estimators=720,

    #                            class_weight=None, min_split_gain=0.0, 

    #                            min_child_weight=0.001, min_child_samples=20, subsample=1.0, 

    #                            subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, 

    #                            random_state=None, n_jobs=- 1, silent=True, importance_type='split')



    lgb1 = lgb.LGBMRegressor()

    tuningLGB = 0



    if (tuningLGB == 1):

        # use this when tuning

        param_grid={

            'objective':['regression'], # - only one option for regression

            'boosting_type':['gbdt'], # - done gbdt dart goss rf

            'num_leaves':[5,6], # - done

            'learning_rate':[0.05], # - done

            'n_estimators':[650,750], # - done

            'max_bin':[45,55], # - done

            'bagging_fraction':[0.85], # - done

            'bagging_freq':[5], # - done

            'feature_fraction':[0.2319], # - done

            'feature_fraction_seed':[9], 

            'bagging_seed':[9],

            'min_data_in_leaf':[9], # - done

            'min_sum_hessian_in_leaf':[11], # - done

            'max_depth':[-1], # - -1 means no limit

            'subsample_for_bin':[500,1000], # - done

            'class_weight':[None],

            'min_split_gain':[0.0],

            'min_child_weight':[0.001],

            'min_child_samples':[5], # - done

            'subsample':[1.0],

            'subsample_freq':[0],

            'colsample_bytree':[1.0],

            'reg_alpha':[0.0], # - l1 regularization done

            'reg_lambda':[0.0], # - L2 regularization done

            'random_state':[1],

            'importance_type':['split'] # - done

        }

    else:

        # use this when not tuning

        param_grid={

            'objective':['regression'], # - only one option for regression

            'boosting_type':['gbdt'], # - done gbdt dart goss rf

            'num_leaves':[5], # - done, maybe 5 is okay too

            'learning_rate':[0.05], # - done

            'n_estimators':[650], # - done

            'max_bin':[55], # - done

            'bagging_fraction':[0.85], # - done

            'bagging_freq':[5], # - done

            'feature_fraction':[0.2319], # - done

            'feature_fraction_seed':[9], 

            'bagging_seed':[9],

            'min_data_in_leaf':[9], # - done

            'min_sum_hessian_in_leaf':[11], # - done

            'max_depth':[-1], # - -1 means no limit

            'subsample_for_bin':[1000], # - done

            'class_weight':[None],

            'min_split_gain':[0.0],

            'min_child_weight':[0.001],

            'min_child_samples':[5], # - done

            'subsample':[1.0],

            'subsample_freq':[0],

            'colsample_bytree':[1.0],

            'reg_alpha':[0.0], # - l1 regularization done

            'reg_lambda':[0.0], # - L2 regularization done

            'random_state':[1],

            'importance_type':['split'] # - done

        }



    scorers = {

        'r2': 'r2',

        'nmsle': 'neg_mean_squared_log_error',

        'nmse': 'neg_mean_squared_error',

        'mae': 'neg_mean_absolute_error'

    }

    # To be used within GridSearch 

    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



    # To be used in outer CV 

    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



    #inner loop KFold example:

    gsc = GridSearchCV(

        estimator=lgb1,

        param_grid=param_grid,

        #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

        #scoring='neg_mean_squared_error', # or look here for other choices 

        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        #cv=5,

        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

        verbose=0,

        return_train_score=True, # keep the other scores

        refit='nmse' # use this one for optimizing

    )



    grid_result = runGSCV(2, train, y_train)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

if (tuning_lgb == 1):

    model_lgb_op = lgb.LGBMRegressor(**grid_result.best_params_)

else:

    lgbm = {'bagging_fraction': 0.85, 'bagging_freq': 5, 'bagging_seed': 9, 'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'feature_fraction': 0.2319, 'feature_fraction_seed': 9, 'importance_type': 'split', 'learning_rate': 0.05, 'max_bin': 55, 'max_depth': -1, 'min_child_samples': 5, 'min_child_weight': 0.001, 'min_data_in_leaf': 9, 'min_split_gain': 0.0, 'min_sum_hessian_in_leaf': 11, 'n_estimators': 650, 'num_leaves': 5, 'objective': 'regression', 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 1000, 'subsample_freq': 0}

    model_lgb_op = lgb.LGBMRegressor(**lgbm)
# initialize the algorithm for the GridSearchCV function

br = BayesianRidge()

tuningBR = 1 # this took 2 hours last time, 1 hour per iteration



if (tuningBR == 1):

    # use this when tuning

    param_grid={

        'n_iter':[50],

        'tol':[0.001],

        'alpha_1':[1e-06],

        'alpha_2':[1e-05],

        'lambda_1':[1e-05],

        'lambda_2':[1e-06],

        'alpha_init':[None],

        'lambda_init':[None],

        'compute_score':[True,False],

        'fit_intercept':[True,False],

        'normalize':[False,True],

        'copy_X':[True],

        'verbose':[False]

    }



else:

    # use this when not tuning

    param_grid={

        'n_iter':[50],

        'tol':[0.001],

        'alpha_1':[1e-06],

        'alpha_2':[1e-05],

        'lambda_1':[1e-05],

        'lambda_2':[1e-06],

        'alpha_init':[None],

        'lambda_init':[None],

        'compute_score':[True,False],

        'fit_intercept':[True,False],

        'normalize':[False,True],

        'copy_X':[True],

        'verbose':[False]

    }



scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error',

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=br,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(2, train, y_train)
tuning_br = 0

BR = BayesianRidge()

if (tuning_br == 1):

    BR_new = BayesianRidge(**grid_result.best_params_)
rd = result_details(grid_result,'alpha_1',100)

rd[['alpha_1','alpha_2','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]
if (tuning_br == 1):

    model_results = [] # model flow, mae, rmsle

    models = [BR, BR_new]



    for model in models:

        #print(model)

        with MyTimer(): 

            scores = calc_all_scores(model,10,10)

        #print("------------------------------------------")

        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

    df_mr.sort_values(by=['rmsle'])
# initialize the algorithm for the GridSearchCV function

# defaults are best

ET = ExtraTreesRegressor()

tuningET = 0 # this took 2 hours last time, 1 hour per iteration



if (tuningET == 1):

    # use this when tuning

    param_grid={

        'n_estimators':[100], # revisit at possibly 2000, but this algorithm becomes really slow for large values

        'criterion':['mse'], # done Default: mse, mae

        'max_depth':[None], # done - above 30 result converges to None

        'min_samples_split':[2], # done - inconsistently better

        'min_samples_leaf':[1],

        'min_weight_fraction_leaf':[0.0],

        'max_features':['auto'], # done - Default:“auto”, “sqrt”, “log2”, None

        'max_leaf_nodes':[None],

        'min_impurity_decrease':[0.0],

        'min_impurity_split':[None],

        'bootstrap':[False],

        'oob_score':[False], # done - True doesn't work, results in a nan value

        'n_jobs':[None],

        'random_state':[1,5,42,55,98],

        'verbose':[0],

        'warm_start':[False],

        'ccp_alpha':[0.0],

        'max_samples':[None] # done - no difference

    }



else:

    # use this when not tuning

    param_grid={

        'n_estimators':[100],

        'criterion':['mse'],

        'max_depth':[None],

        'min_samples_split':[2],

        'min_samples_leaf':[1],

        'min_weight_fraction_leaf':[0.0],

        'max_features':['auto'],

        'max_leaf_nodes':[None],

        'min_impurity_decrease':[0.0],

        'min_impurity_split':[None],

        'bootstrap':[False],

        'oob_score':[False],

        'n_jobs':[None],

        'random_state':[None],

        'verbose':[0],

        'warm_start':[False],

        'ccp_alpha':[0.0],

        'max_samples':[None]

    }



scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error',

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=ET,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(2, train, y_train)
rd = result_details(grid_result,'mean_test_nmse',100)

rd[['random_state','n_estimators','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]
#ET = make_pipeline(RobustScaler(), ExtraTreesRegressor()) # was 1 Tree algorithms don't need scaling

tuning_et = 0

if (tuning_et == 1):

    for i in [2,5,20,42,99]:

        print('random_state =',i)

        ET = ExtraTreesRegressor(random_state=i)

        #ET2 = make_pipeline(StandardScaler(), ExtraTreesRegressor(random_state=i))



        e = {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 6, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'verbose': 0, 'warm_start': False}

        ET_new = ExtraTreesRegressor(**e, random_state=i)

        #ET_new2 = make_pipeline(StandardScaler(), ExtraTreesRegressor(**e, random_state=i))



        model_results = [] # model flow, mae, rmsle

        models = [ET, ET_new]



        for model in models:

            #print(model)

            with MyTimer(): 

                scores = calc_all_scores(model,5,5)

            #print("------------------------------------------")

            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

        print(df_mr.sort_values(by=['rmsle']))

else:

    ET_new = make_pipeline(RobustScaler(), ExtraTreesRegressor(**grid_result.best_params_))
# initialize the algorithm for the GridSearchCV function

R = Ridge(alpha=1.0)

tuningR = 1 # this took 2 hours last time, 1 hour per iteration



if (tuningR == 1):

    # use this when tuning

    param_grid={

        'alpha':[8], # done

        'fit_intercept':[True], # done

        'normalize':[False], # done

        'copy_X':[True],

        'max_iter':[None], # done - no difference

        'tol':[0.001],

        'solver':['auto'], # done - Default:‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’

        'random_state':[1,10,42,99,127]

    }



else:

    # use this when not tuning

    param_grid={

        'alpha':[1.0],

        'fit_intercept':[True],

        'normalize':[False],

        'copy_X':[True],

        'max_iter':[None],

        'tol':[0.001],

        'solver':['auto'],

        'random_state':[None]

    }



scorers = {

    'r2': 'r2',

    'nmsle': 'neg_mean_squared_log_error',

    'nmse': 'neg_mean_squared_error',

    'mae': 'neg_mean_absolute_error'

}

# To be used within GridSearch 

inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)



# To be used in outer CV 

outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    



#inner loop KFold example:

gsc = GridSearchCV(

    estimator=R,

    param_grid=param_grid,

    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc

    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated

    #scoring='neg_mean_squared_error', # or look here for other choices 

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    #cv=5,

    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly

    verbose=0,

    return_train_score=True, # keep the other scores

    refit='nmse' # use this one for optimizing

)



grid_result = runGSCV(5, train, y_train)
rd = result_details(grid_result,'mean_test_nmse',100)

summary = rd[['alpha','random_state','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['random_state','mean_test_nmse'])

#summary.groupby(['fit_intercept'], as_index=False).agg({'mean_test_nmse': 'mean', 'mean_test_nmse': 'std', 'mean_test_mae': 'mean', 'mean_test_r2': 'mean'}).sort_values(by=['mean_test_mae'])

summary.groupby(['alpha'], as_index=False).agg({'mean_test_nmse': 'mean', 'mean_test_mae': 'mean', 'mean_test_r2': 'mean'}).sort_values(by=['mean_test_mae'])

#R = make_pipeline(RobustScaler(), Ridge(alpha =0.0005, random_state=1)) # was 1

tuning_r = 1

if (tuning_r == 1):

    for i in [2,5,20,42,99]:

        print('random_state =',i)



        r= {'alpha': 8, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'solver': 'auto', 'tol': 0.001}

        #R_new = make_pipeline(RobustScaler(), Ridge(**r, random_state=i))

        R_new = Ridge(**r, random_state=i)



        model_results = [] # model flow, mae, rmsle

        models = [R, R_new]



        for model in models:

            #print(model)

            with MyTimer(): 

                scores = calc_all_scores(model,5,5)

            #print("------------------------------------------")

            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

        print(df_mr.sort_values(by=['rmsle']))

else:

    #R_new = make_pipeline(RobustScaler(), Ridge(**grid_result.best_params_))

    R_new = Ridge(**grid_result.best_params_)
AB = AdaBoostRegressor()

SVR = SVR()

DT = DecisionTreeRegressor()

KN = KNeighborsRegressor()

B = BaggingRegressor()
if (tuning_gb == 1):

    model_results = [] # model flow, mae, rmsle

    models = [GBoost, GBoost_orig]#, GBoost] # model_lgb_op, lasso_ns, 



    for model in models:

        #print(model)

        with MyTimer(): 

            scores = calc_all_scores(model)

        #print("------------------------------------------")

        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

    df_mr.sort_values(by=['rmsle'])
print("Calculate Metrics: ", datetime.datetime.now())
compare_models = 0

if (compare_models == 1):

    model_results = [] # model flow, mae, rmsle

    # GBoost is better than GBoost_orig, but has lower final score, think this may be  overfitting

    # BR_new has same results, here, but better final score

    models = [lasso, lasso_new, ENet, ENet_new, KRR, GBoost_orig, GBoost, model_xgb, model_lgb, BR, ET, ET_new, RF, RF_new, AB, SVR, DT, KN, B, R, R_new] # worse or same: BR_new, model_lgb_op, lasso_ns, model_xgb_new,



    for model in models:

        #print(model)

        with MyTimer(): 

            scores = calc_all_scores(model)

        #print("------------------------------------------")

        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])



    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])

    df_mr.sort_values(by=['rmsle'])
print("Stacking and Ensembling: ", datetime.datetime.now())
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        # IDEA: return weighted means

        return np.mean(predictions, axis=1)

#AveragingModels will fit and predict each model and predict using the mean of the individual predictions

with MyTimer():

    averaged_models = AveragingModels(models = (model_lgb, model_xgb, GBoost_orig, KRR, BR)) # Adding ENet and RF is worse, model_xgb_new is worse

    #averaged_models = AveragingModels(models = (GBoost))



if (competition == 'SR'):

    score = mae_cv(averaged_models)

    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

else:

    score = rmsle_cv(averaged_models)

    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models.fit(train.values, y_train)

averaged_train_pred = averaged_models.predict(train.values)

averaged_pred = inv_boxcox1p(averaged_models.predict(test.values), lam_l)



if (competition == 'SR'):

    print(mae(y_train, averaged_train_pred))

else:

    print(rmsle(y_train, averaged_train_pred))
averaged_pred
use_average = 0 # AveragingModels Averages

#use_average = 1 # VotingRegressor Averages

if (use_average == 1):

    from sklearn.ensemble import VotingRegressor



    e = {'alpha': 0.05, 'fit_intercept': True, 'l1_ratio': 0.85, 'max_iter': 100, 'normalize': False, 'selection': 'random'}

    r = {'alpha': 8, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'solver': 'auto', 'tol': 0.001, 'cv': None, 'gcv_mode': 'auto', 'random_state': 99} #  

    k = {'alpha': 2.2, 'coef0': 0.5, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}



    # ('enet', make_pipeline(StandardScaler(), ElasticNet(**e, random_state=3)))

    estimator_list = [('xgb', xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7)),

                  ('lgb', lgb.LGBMRegressor(objective='regression',num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)),

                  ('gboost', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,  loss='huber', random_state=5)),

                  ('krr', KernelRidge(**k)),

                  ('br', BayesianRidge())]



    ereg = VotingRegressor(estimators=estimator_list, weights=[1,1,1,1,1]) 

    ereg = ereg.fit(train, y_train)

    averaged_pred = inv_boxcox1p(ereg.predict(test), lam_l)

    averaged_pred
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=10): # increasing this value should give a more accurate prediction, averaged over n_fold iterations

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156) # was 156

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models): # for each model passed in

            for train_index, holdout_index in kfold.split(X, y): # create train,test for the number of folds

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index]) # fit the model for this fold

                y_pred = instance.predict(X[holdout_index]) # predict values for this fold

                out_of_fold_predictions[holdout_index, i] = y_pred # think we either use all of these values as features later, or the mean value?

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new and only feature

        print("out_of_fold_predictions", out_of_fold_predictions)

        self.meta_model_.fit(out_of_fold_predictions, y) # need to see out_of_fold_predictions feature set

        return self

   

    # Calculate the predictions of all base models on the test data and use the averaged predictions as 

    # meta-features for the final prediction which is calculated by the meta-model

    def predict(self, X):

        # column_stack() function is used to stack 1-D arrays as columns into a 2-D array.

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
# ~ 10 minutes to run



#stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR), # adding RF here did not help

#                                                 meta_model = lasso)

# verify: this class uses out of fold predictions in the stacking method, so rows in dataset are split up betwen models and each row in dataset is only used once

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost_orig, KRR, BR), # adding RF here did not help

                                                 meta_model = R_new)



if (compare_models == 1):

    with MyTimer():

        if (competition == 'SR'):

            score = mae_cv(stacked_averaged_models)

            print("Stacking Averaged models score mean and std: {:.4f} ({:.4f})".format(score.mean(), score.std()))

        else:

            score = rmsle_cv(stacked_averaged_models)

            print("Stacking Averaged models score mean and std: {:.4f} ({:.4f})".format(score.mean(), score.std()))
stacked_averaged_models
use_Regressor = 0 # default - best score

#use_Regressor = 1 # StackingRegressor - worst score

#use_Regressor = 2 # StackingCVRegressor - middle score



if (use_Regressor == 1):

    from sklearn.ensemble import StackingRegressor

    from sklearn.linear_model import RidgeCV, LassoCV

    from sklearn.linear_model import ElasticNetCV



    use_cv = 1

    k = {'alpha': 2.2, 'coef0': 0.5, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}

    if (use_cv == 1):

        e = {'fit_intercept': True, 'l1_ratio': 0.85, 'max_iter': 100, 'normalize': False, 'selection': 'random', 'cv': 10} # 'alpha': 0.05,

        r = {'fit_intercept': True, 'normalize': False, 'cv': None, 'gcv_mode': 'auto'} # cv value has no effect

        estimators = [('enet', make_pipeline(StandardScaler(), ElasticNetCV(**e, random_state=3))),

                      ('gboost', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,  loss='huber', random_state=5)),

                      ('krr', KernelRidge(**k)),

                      ('br', BayesianRidge())]

        reg = StackingRegressor(

            estimators=estimators,

            final_estimator=RidgeCV(**r))

    else:

        e = {'alpha': 0.05, 'fit_intercept': True, 'l1_ratio': 0.85, 'max_iter': 100, 'normalize': False, 'selection': 'random'}

        r = {'alpha': 8, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'solver': 'auto', 'tol': 0.001, 'cv': None, 'gcv_mode': 'auto', 'random_state': 99} #  

        estimators = [('enet', make_pipeline(StandardScaler(), ElasticNet(**e, random_state=3))),

                      ('gboost', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,  loss='huber', random_state=5)),

                      ('krr', KernelRidge(**k)),

                      ('br', BayesianRidge())]

        reg = StackingRegressor(

            estimators=estimators,

            final_estimator=Ridge(**r))



    reg.fit(train, y_train)

    stacked_pred = inv_boxcox1p(reg.predict(test.values), lam_l)

    #reg.transform(inv_boxcox1p(stacked_averaged_models2, lam_l))

    print(stacked_pred)

    print(reg)
a='''

# Note Multiple stacking layers can be achieved by assigning final_estimator to a StackingClassifier or StackingRegressor:



final_layer = StackingRegressor(

    estimators=[('rf', RandomForestRegressor(random_state=42)),

                ('gbrt', GradientBoostingRegressor(random_state=42))],

    final_estimator=RidgeCV()

    )

multi_layer_regressor = StackingRegressor(

    estimators=[('ridge', RidgeCV()),

                ('lasso', LassoCV(random_state=42)),

                ('svr', SVR(C=1, gamma=1e-6, kernel='rbf'))],

    final_estimator=final_layer

)

multi_layer_regressor.fit(X_train, y_train)'''
if (use_Regressor == 2):

    from mlxtend.regressor import StackingCVRegressor

    from sklearn.linear_model import RidgeCV, LassoCV

    from sklearn.linear_model import ElasticNetCV



    use_cv = 0

    k = {'alpha': 2.2, 'coef0': 0.5, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}

    if (use_cv == 1):

        e = {'fit_intercept': True, 'l1_ratio': 0.85, 'max_iter': 100, 'normalize': False, 'selection': 'random', 'cv': 10} # 'alpha': 0.05,

        r = {'fit_intercept': True, 'normalize': False, 'cv': None, 'gcv_mode': 'auto'} # cv value has no effect

        enet = make_pipeline(StandardScaler(), ElasticNetCV(**e, random_state=3))

        gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,  loss='huber', random_state=5)

        krr = KernelRidge(**k)

        br = BayesianRidge()

        r = RidgeCV(**r)

        reg = StackingCVRegressor(

            regressors=(enet, gboost, krr, br),

            meta_regressor=r)

    else:

        e = {'alpha': 0.05, 'fit_intercept': True, 'l1_ratio': 0.85, 'max_iter': 100, 'normalize': False, 'selection': 'random'}

        r = {'alpha': 8, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'solver': 'auto', 'tol': 0.001, 'random_state': 99} #  

        enet = make_pipeline(StandardScaler(), ElasticNet(**e, random_state=3))

        gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,  loss='huber', random_state=5)

        krr = KernelRidge(**k)

        br = BayesianRidge()

        r = Ridge(**r)

        reg = StackingCVRegressor(

            regressors=(enet, gboost, krr, br),

            meta_regressor=r)



    reg.fit(train, y_train)

    stacked_pred = inv_boxcox1p(reg.predict(test.values), lam_l)

    print(stacked_pred)

    print(reg)

    

    print('5-fold cross validation scores:\n')

    for clf, label in zip([enet, gboost, krr, br], ['enet', 'gboost', 

                                                'krr', 'br',

                                                'StackingCVRegressor']):

        scores = cross_val_score(clf, train, y_train, cv=5, scoring='neg_mean_squared_error')

        print("Neg. MSE Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

        

        #scores = cross_val_score(clf, train, y_train, cv=5)

        #print("R^2 Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
averaged_models.fit(train.values, y_train)

averaged_train_pred = averaged_models.predict(train.values)

if (use_average == 0):

    averaged_pred = inv_boxcox1p(averaged_models.predict(test.values), lam_l)



if (competition == 'SR'):

    print(mae(y_train, averaged_train_pred))

else:

    print(rmsle(y_train, averaged_train_pred))
#pre adjustment

averaged_pred
plt.scatter(averaged_train_pred, y_train, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Averaged Model')

plt.show()
averaged_models.predict(test.values)
averaged_train_pred
#post adjustment

averaged_pred
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

if (use_Regressor == 0):

    stacked_pred = inv_boxcox1p(stacked_averaged_models.predict(test.values), lam_l)

    

with MyTimer():

    if (competition == 'SR'):

        print(mae(y_train, stacked_train_pred))

    else:

        print(rmsle(y_train, stacked_train_pred))
stacked_pred
print(y_train)

print(stacked_train_pred)

print(stacked_pred)
def fit_pred(train, ytrain, test, model):

        model.fit(train, y_train)

        model_train_pred = model.predict(train)

        model_pred = inv_boxcox1p(model.predict(test), lam_l)

        return(model_train_pred, model_pred)

    

models = [lasso, lasso_new, ENet, KRR, GBoost_orig, GBoost, model_xgb, model_xgb_new, BR, ET, ET_new, RF, RF_new, AB, SVR, DT, KN, B] # model_lgb,

model_names = ['lasso', 'lasso_new', 'ENet', 'KRR', 'GBoost_orig', 'GBoost', 'model_xgb', 'model_xgb_new', 'BR', 'ET', 'ET_new', 'RF', 'RF_new', 'AB', 'SVR', 'DT', 'KN', 'B']



with MyTimer():

    for i in range(0,len(models)):

        mn = model_names[i]+"_pred"

        train_pred, test_pred = fit_pred(train, y_train, test, models[i])

        print(mn, test_pred)    

# values are not normalized

train[train.columns].mean().head()
# are train and test normalized? between -1 and 1

replace_xgb = 0 # new optimized model is worse, was overfit

if (replace_xgb == 1):

    model_xgb_new.fit(train, y_train)

    xgb_train_pred = model_xgb_new.predict(train)

    xgb_pred = inv_boxcox1p(model_xgb_new.predict(test), lam_l)

else:

    model_xgb.fit(train, y_train)

    xgb_train_pred = model_xgb.predict(train)

    xgb_pred = inv_boxcox1p(model_xgb.predict(test), lam_l)



if (competition == 'SR'):

    print(mae(y_train, xgb_train_pred))

else:

    print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = inv_boxcox1p(model_lgb.predict(test.values), lam_l)



if (competition == 'SR'):

    print(mae(y_train, lgb_train_pred))

else:

    print(rmsle(y_train, lgb_train_pred))
if (tuning_lgb == 1):

    model_lgb_op.fit(train, y_train)

    lgb_train_pred_op = model_lgb_op.predict(train)

    lgb_pred_op = inv_boxcox1p(model_lgb_op.predict(test.values), lam_l)



    if (competition == 'SR'):

        print(mae(y_train, lgb_train_pred_op))

    else:

        print(rmsle(y_train, lgb_train_pred_op))
# compare values with optimization

print(lgb_train_pred)

if (tuning_lgb == 1):

    print(lgb_train_pred_op)

print(lgb_pred)

if (tuning_lgb == 1):

    print(lgb_pred_op)
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')





stk = 0.70

xgb = 0.15

lgb = 0.15





if (competition == 'SR'):

    print(mae(y_train,stacked_train_pred*stk + xgb_train_pred*xgb + lgb_train_pred*lgb ))

else:

    print(rmsle(y_train,stacked_train_pred*stk + xgb_train_pred*xgb + lgb_train_pred*lgb ))
#method = 'ensemble'

method = 'stacked'

if (method == 'stacked'):

    #stacked_pred => ENet, GBoost_orig, KRR

    ensemble = stacked_pred*stk + xgb_pred*xgb + lgb_pred*lgb  # if using averaged_pred, need to add averaged_pred here

else:

    ensemble = averaged_pred

print(lgb_pred)

print(xgb_pred)

print(stacked_pred)

print(averaged_pred)

print(ensemble)
print(y_train,stacked_train_pred * stk + xgb_train_pred * xgb + lgb_train_pred * lgb) # if using averaged_pred, need to add averaged_pred here

print(y_train,stacked_train_pred)
sub_train = pd.DataFrame()

sub_train['Id'] = train_ID

sub_train['SalePrice'] = inv_boxcox1p(stacked_train_pred, lam_l)
Predicted = sub_train['SalePrice']

Actual = inv_boxcox1p(y_train, lam_l)

plt.scatter(sub_train['SalePrice'], Actual, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Averaged Model')



m, b = np.polyfit(Predicted, Actual, 1)

#m = slope, b=intercept

plt.plot(Predicted, m*Predicted+b,c='red')

plt.show()
# Pre-adjustment score

print("mae for boxcox(SalePrice)",mae(y_train, sub_train['SalePrice']))

print("mse for boxcox(SalePrice)",rmsle(y_train, sub_train['SalePrice']))

print("mae for SalePrice",mae(Actual, Predicted))

print("mse for SalePrice",rmsle(Actual, Predicted))
AdjustedScores = []



for i in np.arange(.994, .996, 0.001):

    for j in np.arange(1.06, 1.08, .01):



        q1 = sub_train['SalePrice'].quantile(0.0025)

        q2 = sub_train['SalePrice'].quantile(0.0045)

        q3 = sub_train['SalePrice'].quantile(i)



        #Verify the cutoffs for the adjustment

        print(q1,q2,q3)

        # adjust at low end

        #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q1 else x*0.79)

        #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)



        # adjust at high end

        sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x < q3 else x*j)



        Predicted = sub_train['SalePrice2']

        Actual = inv_boxcox1p(y_train, lam_l)



        # Pre-adjustment score

        print("mae for boxcox(SalePrice)",mae(y_train, sub_train['SalePrice2']))

        print("mse for boxcox(SalePrice)",rmsle(y_train, sub_train['SalePrice2']))

        print("mae for SalePrice",mae(Actual, Predicted))

        print("mse for SalePrice",rmsle(Actual, Predicted))



        AdjustedScores.append([i, j, mae(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), rmsle(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), mae(Actual, Predicted), rmsle(Actual, Predicted)])



df_adj = pd.DataFrame(AdjustedScores, columns=["QUANT","COEF","MAE_BC","RMSE_BC","MAE_SP","RMSE_SP"])
q1 = sub_train['SalePrice'].quantile(0.0015)

q2 = sub_train['SalePrice'].quantile(0.01)

q3 = sub_train['SalePrice'].quantile(0.995)



#Verify the cutoffs for the adjustment

print(q1,q2,q3)

# adjust at low end

#sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q1 else x*0.79)

#sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)



# adjust at high end

sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x < q3 else x*1.07)
plt.scatter(sub_train['SalePrice'], sub_train['SalePrice2'], alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Adjusted Predicted Price')

plt.title('Averaged Model')

plt.show()
sub_train.query("SalePrice != SalePrice2")
Predicted = sub_train['SalePrice2']

Actual = inv_boxcox1p(y_train, lam_l)

plt.figure(figsize=(15,15))

plt.scatter(sub_train['SalePrice2'], Actual, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Adj Predicted Price')

plt.ylabel('Actual Price')

plt.title('Averaged Model')



m, b = np.polyfit(Predicted, Actual, 1)

#m = slope, b=intercept

plt.plot(Predicted, m*Predicted+b,c='red')

plt.show()
# Post adjustment for high score

print("mae for boxcox(SalePrice2)",mae(y_train, sub_train['SalePrice2']))

print("mse for boxcox(SalePrice2)",rmsle(y_train, sub_train['SalePrice2']))

print("mae for SalePrice2",mae(Actual, Predicted))

print("mse for SalePrice2",rmsle(Actual, Predicted))
#sub_train = pd.DataFrame()

#sub_train['Id'] = train_ID

#sub_train['SalePrice'] = inv_boxcox1p(stacked_train_pred, lam_l)

q1 = sub_train['SalePrice'].quantile(0.013)

q2 = sub_train['SalePrice'].quantile(0.10)

q3 = sub_train['SalePrice'].quantile(0.995)

print(q1,q2,q3)
sub_train.min()
AdjustedScores = []



for i in np.arange(.005, .015, 0.001):

    for j in np.arange(.90, 1.00, 0.01):



        q1 = sub_train['SalePrice'].quantile(i)

        q2 = sub_train['SalePrice'].quantile(0.1)

        q3 = sub_train['SalePrice'].quantile(.995)



        #Verify the cutoffs for the adjustment

        print(q1,q2,q3)

        # adjust at low end

        sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q1 else x*j)

        #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)



        # adjust at high end

        #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x < q3 else x*j)



        Predicted = sub_train['SalePrice2']

        Actual = inv_boxcox1p(y_train, lam_l)



        # Pre-adjustment score

        print("mae for boxcox(SalePrice)",mae(y_train, sub_train['SalePrice2']))

        print("mse for boxcox(SalePrice)",rmsle(y_train, sub_train['SalePrice2']))

        print("mae for SalePrice",mae(Actual, Predicted))

        print("mse for SalePrice",rmsle(Actual, Predicted))



        AdjustedScores.append([i, j, mae(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), rmsle(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), mae(Actual, Predicted), rmsle(Actual, Predicted)])



df_adj = pd.DataFrame(AdjustedScores, columns=["QUANT","COEF","MAE_BC","RMSE_BC","MAE_SP","RMSE_SP"])
df_adj.sort_values(by=['RMSE_BC'])
q1 = sub_train['SalePrice'].quantile(0.01)

q2 = sub_train['SalePrice'].quantile(0.1)

q3 = sub_train['SalePrice'].quantile(0.995)



#Verify the cutoffs for the adjustment

print(q1,q2,q3)

# adjust at low end

sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q1 else x*0.91) # also try .94

#sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)



# adjust at high end

sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x < q3 else x*1.07)
plt.figure(figsize=(15,15))

plt.scatter(sub_train['SalePrice'], sub_train['SalePrice2'], alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Adjusted Predicted Price')

plt.title('Averaged Model')

plt.show()
Predicted = sub_train['SalePrice2']

Actual = inv_boxcox1p(y_train, lam_l)

plt.figure(figsize=(15,15))

plt.scatter(sub_train['SalePrice2'], Actual, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Adj Predicted Price')

plt.ylabel('Actual Price')

plt.title('Averaged Model')



m, b = np.polyfit(Predicted, Actual, 1)

#m = slope, b=intercept

plt.plot(Predicted, m*Predicted+b,c='red')



plt.show()
sub_train.query("SalePrice != SalePrice2")
# Post adjustment for low and high score

print("mae for boxcox(SalePrice2)",mae(y_train, sub_train['SalePrice2']))

print("mse for boxcox(SalePrice2)",rmsle(y_train, sub_train['SalePrice2']))

print("mae for SalePrice2",mae(Actual, Predicted))

print("mse for SalePrice2",rmsle(Actual, Predicted))
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

#sub.to_csv('submission.csv',index=False)
q1 = sub['SalePrice'].quantile(0.01)

q2 = sub['SalePrice'].quantile(0.1)

q3 = sub['SalePrice'].quantile(0.995)



# adjust at low end

#sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x > q1 else x*0.99) # didn't help, with several values

#sub['SalePrice2'] = sub['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)



# adjust at high end

sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x < q3 else x*1.07)
sub.to_csv('submission.csv',index=False)
sub.head()
!pwd
ls -arlt
!head -10 submission.csv
print("Start: ", StartTime)

print("End: ", datetime.datetime.now())
from mlens.ensemble import SuperLearner

ensemble = SuperLearner(random_state=1, verbose=2)