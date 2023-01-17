import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import sklearn

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso

from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF

from scipy.stats import uniform, reciprocal, randint

import xgboost

from sklearn.decomposition import KernelPCA

import seaborn as sns





scale = True

n_bins = 10

n_components = 2



scaler = StandardScaler()



def cross_val_score_log_form(estimator, X, y, cv = 5):

    from sklearn.model_selection import KFold

    from sklearn.metrics import mean_squared_error

    kf = KFold(n_splits=cv)

    errs = list()

    for train_index, test_index in kf.split(X): #

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

        y_train, y_test = y[train_index], y[test_index]

        y_hat = estimator.fit(X_train,y_train).predict(X_test)

        errs.append(np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_hat))))

    return(errs)

data = pd.read_csv("../input/BlackFriday.csv")

# Replace gender with 1 and -1

data['Gender'] = data['Gender'].replace({'M':1,'F':-1})

data = pd.concat([data,pd.get_dummies(data['City_Category'])],axis=1)

data = data.rename(index=str, columns = {'A':'City A', 'B':'City B', 'C': 'City C'})

data['Age group'] = data['Age'].replace(list(data['Age'].value_counts().sort_index().index), 1+np.arange(len(list(data['Age'].value_counts().sort_index().index))))

data = data.drop(['Age'], axis = 1)

# Create dummies

# Fill product category 2 and 3 with 0s

data['Product_Category_2'] = data['Product_Category_2'].fillna(0)

data['Product_Category_3'] = data['Product_Category_3'].fillna(0)

data['Product_Category_1'] = data['Product_Category_1'].astype("int").astype("category")

data['Product_Category_2'] = data['Product_Category_2'].astype("int").astype("category")

data['Product_Category_3'] = data['Product_Category_3'].astype("int").astype("category")

data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].replace({"4+":"4"}).astype("int")

data.head()


# Create customers dataframe

customers = pd.DataFrame()

customers['User_ID'] = data['User_ID'].unique()



customers = pd.DataFrame(data[['User_ID','Gender','Age group','Occupation','Stay_In_Current_City_Years', 'Marital_Status','City_Category']].drop_duplicates(subset=['User_ID','Gender'],keep="first").sort_values(by='User_ID')).reset_index().drop(["index"],axis = 1)







customers['N buys'] = data.groupby("User_ID").count().reset_index()['Gender']

customers = customers.reset_index().iloc[:,1:]

customers['Purchase'] = data.groupby('User_ID').sum().sort_values(by="User_ID")['Purchase'].values

customers.Stay_In_Current_City_Years = customers.Stay_In_Current_City_Years.replace(list(customers.Stay_In_Current_City_Years.value_counts().sort_index().index), np.arange(customers.Stay_In_Current_City_Years.value_counts().sort_index().shape[0]))

customers.Occupation = customers.Occupation.astype("str")

customers = pd.get_dummies(customers)



dividing_bins = np.concatenate((np.arange(min(np.log(customers['Purchase'])),max(np.log(customers['Purchase'])),(max(np.log(customers['Purchase'])) - min(np.log(customers['Purchase'])))/n_bins),np.array([20])))

customers['Purchase_group'] = pd.cut(np.log(customers['Purchase']), bins = dividing_bins)

customers['logged_purchase'] = np.log(customers['Purchase'])





targets_customers = ["logged_purchase", "Purchase_group", "Purchase", "N buys"]



features = customers.drop(targets_customers+["User_ID"],axis=1).columns





customers.head()
products = pd.DataFrame()

n_prod_categories = 18



prod_IDs_by_customer = pd.concat([pd.DataFrame(data['User_ID']),pd.get_dummies(data[['Product_Category_1','Product_Category_2','Product_Category_3']])], axis = 1)

cols_to_drop = prod_IDs_by_customer.columns

for j in range(n_prod_categories):

    prod_IDs_by_customer['Prod_Cat_'+str(j+1)] = prod_IDs_by_customer['Product_Category_1_'+str(j+1)]

    try:

        prod_IDs_by_customer['Prod_Cat_'+str(j+1)] = prod_IDs_by_customer['Prod_Cat_'+str(j+1)] + prod_IDs_by_customer['Product_Category_2_'+str(j+1)]

    except:

        print('Product_Category_2_'+str(j+1)+" not found")

    try:

        prod_IDs_by_customer['Prod_Cat_'+str(j+1)] = prod_IDs_by_customer['Prod_Cat_'+str(j+1)] + prod_IDs_by_customer['Product_Category_3_'+str(j+1)]

    except:

        print('Product_Category_3_'+str(j+1)+" not found")

prod_IDs_by_customer = prod_IDs_by_customer.groupby("User_ID").sum().reset_index()

prod_IDs_by_customer = prod_IDs_by_customer.drop(cols_to_drop, axis=1)





prod_IDs_by_customer.columns

n_prod_categories = 18

cols_to_drop = prod_IDs_by_customer.columns

products = data.groupby("Product_ID").sum().reset_index()[['Product_ID','Purchase']]

products['N sales'] = data.groupby("Product_ID").count().reset_index()['User_ID']



products['Perc males buying'] = (data.groupby("Product_ID").mean()['Gender']/2 + 0.5).reset_index()['Gender']





products['Avg_City_Tenure'] = data.groupby("Product_ID").mean().reset_index()['Stay_In_Current_City_Years']

products['Avg_Age'] = data.groupby("Product_ID").mean().reset_index()['Age group']

products['Buys City A'] = data.groupby("Product_ID").sum().reset_index()['City A']

products['Buys City B'] = data.groupby("Product_ID").sum().reset_index()['City B']

products['Buys City C'] = data.groupby("Product_ID").sum().reset_index()['City C']

products['Married fraction'] = data.groupby("Product_ID").mean().reset_index()['Marital_Status']

products.shape

products['Buys City A'] = products['Buys City A']/products['Buys City A'].sum()

products['Buys City B'] = products['Buys City B']/products['Buys City B'].sum()

products['Buys City C'] = products['Buys City C']/products['Buys City C'].sum()

prod_occupations = pd.concat([data[["User_ID","Product_ID"]],pd.get_dummies(data['Occupation'].astype("category"))], axis = 1).groupby("Product_ID").sum().reset_index().iloc[:,2:].apply(lambda x : x/sum(x))

prod_occupations.columns = ["Fraction Occupation "+str(x) for x in range(prod_occupations.shape[1])]

products = pd.concat([products,prod_occupations],axis = 1)

targets_products = ['N sales', "Purchase"]

products.head()
customers['Purchase'].hist(bins=100)
np.log(customers['Purchase']).hist(bins=100)
target = 'N buys'



scaler = StandardScaler()

kernel_pca = KernelPCA(n_components=n_components, kernel = "linear",degree = 2)


n_components = 2



principalComponents = kernel_pca.fit_transform(pd.DataFrame(scaler.fit_transform(customers.drop(targets_customers+["User_ID"], axis = 1))))

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PC'+str(i+1) for i in range(n_components)])

finalDf = pd.concat([principalDf, np.log(customers[target])], axis = 1)

finalDf.columns = ['PC'+str(i+1) for i in range(n_components)]+['target']

finalDf.head()





cmap = matplotlib.cm.get_cmap('magma')

normalize = matplotlib.colors.Normalize(vmin=min(finalDf['target']), vmax=max(finalDf['target']))

colors = [cmap(normalize(value)) for value in finalDf['target']]





fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(finalDf['PC1'], finalDf['PC2'], color=colors)

cax, kw = matplotlib.colorbar.make_axes(ax)

cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

plt.show()
n_components = 3

target = 'logged_purchase'



scaler = StandardScaler()

kernel_pca = KernelPCA(n_components=n_components, kernel = "linear",degree = 2)





principalComponents = kernel_pca.fit_transform(pd.DataFrame(scaler.fit_transform(customers.drop(targets_customers+["User_ID"], axis = 1))))

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PC'+str(i+1) for i in range(n_components)])

finalDf = pd.concat([principalDf, customers[target]], axis = 1)

finalDf.columns = ['PC'+str(i+1) for i in range(n_components)]+['target']

finalDf.head()

# Now we try to picture it



cmap = matplotlib.cm.get_cmap('magma')

normalize = matplotlib.colors.Normalize(vmin=min(finalDf['target']), vmax=max(finalDf['target']))

colors = [cmap(normalize(value)) for value in finalDf['target']]





fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(finalDf['PC1'], finalDf['PC2'], finalDf['PC3'], c=colors)

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

ax.set_zlabel('PC3')

cax, kw = matplotlib.colorbar.make_axes(ax)

cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

plt.show()


# Taken from Yassine Ghouzam@Kaggle

def multiplot(data,features,plottype,nrows,ncols,figsize,y=None,colorize=False):

    """ This function draw a multi plot for 3 types of plots ["regplot","distplot","coutplot"]"""

    n = 0

    plt.figure(1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    

    if colorize:

        colors = sns.color_palette(n_colors=(nrows*ncols))

    else :

        colors = [None]*(nrows*ncols)

        

    for row in range(ncols):

        for col in range(nrows):

            

            if plottype == 'regplot':

                if y == None:

                    raise ValueError('y value is needed with regplot type')

                

                sns.regplot(data = data, x = features[n], y = y ,ax=axes[row,col], color = colors[n])

                correlation = np.corrcoef(data[features[n]],data[y])[0,1]

                axes[row,col].set_title("Correlation {:.2f}".format(correlation))

            

            elif plottype == 'distplot':

                sns.distplot(a = data[features[n]],ax = axes[row,col],color=colors[n])

                skewness = data[features[n]].skew()

                axes[row,col].legend(["Skew : {:.2f}".format(skewness)])

            

            elif plottype in ['countplot']:

                g = sns.countplot(x = data[features[n]], y = y, ax = axes[row,col],color = colors[n])

                g = plt.setp(g.get_xticklabels(), rotation=45)

                

            n += 1

    plt.tight_layout()

    plt.show()

    plt.gcf().clear()
multiplot(data = products, features = products.drop(targets_products+["Product_ID"], axis = 1).columns, plottype = "distplot", nrows = 5, ncols = 5, figsize = (20,20),colorize = True)


features = products.drop(targets_products+["Product_ID"],axis=1).columns





for feature in features:

    if feature not in targets_products+["Product_ID"] and np.abs(products[feature].skew()) > 1:

        products[feature] = np.log1p(products[feature])

target = 'N sales'



scaler = StandardScaler()



X_products = products.drop(targets_products+["Product_ID"],axis=1)
n_components = 2

kernel_pca = KernelPCA(n_components=n_components, kernel = "sigmoid")

principalComponents = kernel_pca.fit_transform(pd.DataFrame(scaler.fit_transform(X_products)))

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PC'+str(i+1) for i in range(n_components)])

finalDf = pd.concat([principalDf, products[target]], axis = 1)

finalDf.columns = ['PC'+str(i+1) for i in range(n_components)]+['target']

# Now we try to picture it



cmap = matplotlib.cm.get_cmap('magma')

normalize = matplotlib.colors.Normalize(vmin=min(finalDf['target']), vmax=max(finalDf['target']))

colors = [cmap(normalize(value)) for value in finalDf['target']]



fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(finalDf['PC1'], finalDf['PC2'], color=colors)

cax, _ = matplotlib.colorbar.make_axes(ax)

cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

plt.show()
n_components = 3

target = 'N sales'



kernel_pca = KernelPCA(n_components=n_components, kernel = "sigmoid")

X_products.columns



principalComponents = kernel_pca.fit_transform(pd.DataFrame(scaler.fit_transform(X_products)))

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PC'+str(i+1) for i in range(n_components)])

finalDf = pd.concat([principalDf, products[target]], axis = 1)

finalDf.columns = ['PC'+str(i+1) for i in range(n_components)]+['target']

finalDf.head()



cmap = matplotlib.cm.get_cmap('magma')

normalize = matplotlib.colors.Normalize(vmin=min(finalDf['target']), vmax=max(finalDf['target']))

colors = [cmap(normalize(value)) for value in finalDf['target']]





fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(finalDf['PC1'], finalDf['PC2'], finalDf['PC3'], c=colors)

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

ax.set_zlabel('PC3')

cax, _ = matplotlib.colorbar.make_axes(ax)

cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

plt.show()


target = "N buys"



X = customers.drop(targets_customers+["User_ID"],axis=1)

y = customers[target]



X_scaled = pd.DataFrame(scaler.fit_transform(X))



print("Linear regression log form: ",np.mean(cross_val_score_log_form(LinearRegression(),X,np.log(y),cv=10)))

print("XGBoost log form: ",np.mean(cross_val_score_log_form(XGBRegressor(),X_scaled,np.log(y),cv=10)))

print("Huber regression log form: ",np.mean(cross_val_score_log_form(HuberRegressor(),X,np.log(y),cv=10)))

print("Linear regression: ",np.mean(np.sqrt(np.abs(cross_val_score(LinearRegression(),X_scaled,y,cv=10,scoring='neg_mean_squared_error')))))

print("Huber regression log form: ",np.mean(np.sqrt(np.abs(cross_val_score(HuberRegressor(),X,y,cv=10,scoring='neg_mean_squared_error')))))

print("XGBoost: ",np.mean(np.sqrt(np.abs(cross_val_score(XGBRegressor(),X,y,cv=10,scoring='neg_mean_squared_error')))))



target = "N sales"



X_products = products.drop(targets_products+["Product_ID"],axis=1)

y_products = products[target]



print("Linear Regression: ",np.mean(np.sqrt(np.abs(cross_val_score(LinearRegression(),X_products,y_products,cv=10,scoring='neg_mean_squared_error')))))

print("XGBoost: ",np.mean(np.sqrt(np.abs(cross_val_score(XGBRegressor(),X_products,y_products,cv=10,scoring='neg_mean_squared_error')))))

print("XGBoost log form: ",np.mean(np.abs(cross_val_score_log_form(XGBRegressor(),X_products,np.log(y_products),cv=10))))

products.columns
to_drop = ['Buys City A', 'Buys City B', 'Buys City C'] + ['Fraction Occupation '+str(i) for i in range(21)]

X_products = products.drop(targets_products+["Product_ID"]+to_drop,axis=1)
print("Linear Regression: ",np.mean(np.sqrt(np.abs(cross_val_score(LinearRegression(),X_products,y_products,cv=10,scoring='neg_mean_squared_error')))))

print("XGBoost: ",np.mean(np.sqrt(np.abs(cross_val_score(XGBRegressor(),X_products,y_products,cv=10,scoring='neg_mean_squared_error')))))

print("XGBoost log form: ",np.mean(np.abs(cross_val_score_log_form(XGBRegressor(),X_products,np.log(y_products),cv=10))))
