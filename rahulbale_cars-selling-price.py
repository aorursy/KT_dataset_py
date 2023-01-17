# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # Setting a default seaborn setting in plots

from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the Dataset

df=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')

df.head()
#Shape of the overall dataset(row,columns)

df.shape
# Let's get some information from dataset



df.info()
#Checking weither the dataset contains any null values



df.isnull().mean()


df.describe(include='all')
df.describe(include='object')
#Let's check some unique values in data set using unique()



print(df['Fuel_Type'].unique())

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
df.columns
#Distribution plot of Year



sns.distplot(df.Year)
# Exploring PDF(probability density function) of features

fig = df.hist(figsize=(18,18))
sns.barplot('Seller_Type','Selling_Price',data=df,palette='twilight')
sns.barplot('Fuel_Type','Selling_Price',data=df,palette='twilight')
sns.regplot('Selling_Price','Present_Price',data=df)
sns.regplot('Selling_Price','Kms_Driven',data=df)
sns.barplot('Transmission','Selling_Price',data=df,palette='spring')
sns.barplot('Owner','Selling_Price',data=df,palette='ocean')
plt.figure(figsize=(15,8))

df.boxplot()

plt.show()
plt.figure(figsize=(10,5))

sns.lineplot(df['Year'],df['Selling_Price'])
# Creating a function to make a countplot



def plot_categorical(feature , dataset):

    ax = sns.countplot(y=feature, data=dataset)

    plt.title('Distribution of ' + feature)

    plt.xlabel('Count')



    total = len(dataset[feature])

    for p in ax.patches:

            percentage = '{:.1f}%'.format(100 * p.get_width()/total)

            x = p.get_x() + p.get_width() + 0.02

            y = p.get_y() + p.get_height()/2

            ax.annotate(percentage, (x, y))



    plt.show()
plot_categorical('Fuel_Type' , df)
plot_categorical('Seller_Type' , df)
plot_categorical('Transmission' , df)
#Sellection of our final dataset



final_set=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]



#We have droped Car_Name from dataset
#It's important to know how many years old the car is.



final_set['Current_Year']=2020  #Adding the Current_Year in dataset



final_set['No_of_total_years']=final_set['Current_Year']-final_set['Year'] 



final_set.head()
# It's time to drop the Year column after the needed info is derived.



final_set.drop(['Year','Current_Year'],axis=1,inplace=True)
plt.figure(figsize=(10,5))

sns.barplot('No_of_total_years','Selling_Price',data=final_set)
final_set.head()
sns.pairplot(final_set)
final_set=pd.get_dummies(final_set,drop_first=True) #drop_first drops the first feature 

final_set.head()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(final_set.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.show()
X = final_set.drop(['Selling_Price'] , axis = 1)

y = final_set['Selling_Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

print("X train: ",X_train.shape)

print("X test: ",X_test.shape)

print("Y train: ",y_train.shape)

print("Y test: ",y_test.shape)
#Thanks to : https://www.kaggle.com/cagkanbay/car-price-prediction



r_2 = [] # List for r 2 score

MSE = [] # list for mean_squared_error scores mean



# Main function for models

def model(algorithm,X_train,y_train,X_test,y_test): 

    algorithm.fit(X_train,y_train)

    predicts=algorithm.predict(X_test)

    prediction=pd.DataFrame(predicts)

    R_2=r2_score(y_test,prediction)

    error=mean_squared_error(y_test,prediction)

    

    

    # Appending results to Lists 

    r_2.append(R_2)

    MSE.append(error)

    

    # Printing results  

    print(algorithm,"\n") 

    print("r_2 score :",R_2,"\n")

    print("MSE:",error)

    

    # Plot for prediction vs originals

    test_index=y_test.reset_index()["Selling_Price"]

    ax=test_index.plot(label="originals",figsize=(12,6),linewidth=2,color="r")

    ax=prediction[0].plot(label = "predictions",figsize=(12,6),linewidth=2,color="g")

    plt.legend(loc='upper right')

    plt.title("ORIGINALS VS PREDICTIONS")

    plt.xlabel("index")

    plt.ylabel("values")

    plt.show()
lr = LinearRegression()

model(lr,X_train,y_train,X_test,y_test)
alpha=[0.001,0.1,1,10,100,1000]

normalize=['True',"False"]



parameters={

    'alpha':alpha,

    'normalize':normalize

}



rv_rid=RandomizedSearchCV(Ridge(),parameters,cv=6,

                       n_iter=10,scoring='neg_mean_squared_error',random_state=5,n_jobs=1)

rv_rid.fit(X_train,y_train)





print(rv_rid.best_estimator_)
ridge = Ridge(alpha = 0.01, normalize = True) # applied the best estimator

model(ridge,X_train,y_train,X_test,y_test)
parameters={

    'alpha':np.logspace(-3,3,num=14)   # range for alpha

}



rv_rid=RandomizedSearchCV(Lasso(),parameters,cv=6,

                       n_iter=10,scoring='neg_mean_squared_error',random_state=5,n_jobs=1)

rv_rid.fit(X_train,y_train)





print(rv_rid.best_estimator_)
ls = Lasso(alpha = rv_rid.best_estimator_.alpha, normalize = True) # applied the best estimator

model(ls,X_train,y_train,X_test,y_test)
#Randomized Search CV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
rf_para = {

    "n_estimators":n_estimators,

    "max_depth":max_depth,

    "min_samples_split":min_samples_split,

    "min_samples_leaf":min_samples_leaf,

    "max_features":max_features    

}
rf_reg=RandomForestRegressor()

rf_rand=RandomizedSearchCV(estimator=rf_reg,param_distributions=rf_para,cv=6,

                       n_iter=10,scoring='neg_mean_squared_error',random_state=5,n_jobs=1)



rf_rand.fit(X_train,y_train)

print(rf_rand.best_estimator_)
rf = RandomForestRegressor(max_depth=25, min_samples_leaf=2, n_estimators=300, random_state = 42)

model(rf,X_train,y_train,X_test,y_test)
dtr = DecisionTreeRegressor()

model(dtr,X_train,y_train,X_test,y_test)
Model = ["LinearRegression","Ridge","Lasso","RandomForestRegressor","DecisionTreeRegressor"]

results=pd.DataFrame({'Model': Model,'R Squared': r_2,'MSE': MSE})

results