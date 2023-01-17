# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##Loading appropriate libraries##

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/Insurance.csv")



#Check top 5 data.....

df.head()
#Checking the objects available.....

df.info()
#Check the statistical analysis.........

df.describe()


#Checking the shape of dataframe.......

df.shape
#Separating the dtypes based on objects and integer or float.......

g = df.columns.to_series().groupby(df.dtypes).groups

print(g)


#Checking the misiing values...

df.isnull().sum()

#Visualizing the "Sex"

df['sex'].value_counts().plot(kind='bar')

#Visualizing the "Smoker"

df['smoker'].value_counts().plot(kind='bar')
#Visualizing the "Region"....

df['region'].value_counts().plot(kind='bar')
#Checking the unique values in the column["Southeast","Southwest","Northwest","Northeast"]..............

df['region'].unique()
#Visualizing the "Age"

import seaborn as sns

sns.boxplot(df["age"])
#Visualizing the "bmi-body mass index"

import seaborn as sns

sns.boxplot(df["bmi"])
#Checking the correlation...........................

plt.figure(figsize=(9,3)) #7 is the size of the width and 4 is parts.... 

sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()



#Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in df.columns:

    df[col] = labelencoder.fit_transform(df[col])





#Now one hot encoding

df=pd.get_dummies(df, columns=["sex","smoker","region"],drop_first=False)



print(df)
df.rename(columns={'sex_0':'Sex_Female','sex_1':'Sex_Male','smoker_0':'Smoker_No','smoker_1':'Smoker_Yes','region_0':'Region_Northeast','region_1':'Region_Northwest','region_2':'Region_Southeast','region_3':'Region_Southwest'}, inplace=True)





# iterating the columns 

for col in df.columns: 

    print(col)

#Rearranged the order of the dataframe....

df = df[['age','bmi','children','Sex_Female','Sex_Male','Smoker_No','Smoker_Yes','Region_Northeast','Region_Northwest','Region_Southeast','Region_Southwest','charges']]
#Separating features and label

X = df.iloc[:,0:11].values

y = df.iloc[:,-1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)



# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_



print(explained_variance)





with plt.style.context('dark_background'):

    plt.figure(figsize=(16, 8))

    

    plt.bar(range(11), explained_variance, alpha=0.5, align='center',label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()

    

    

# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 7)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print(explained_variance)


#Model comparison

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor







#Fit Decision_tree

tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)
#Fit Random_forest

forest = RandomForestRegressor(n_jobs=-1)

forest.fit(X_train, y_train)
#Fit linear_regression

lin_reg = LinearRegression(n_jobs=-1)

lin_reg.fit(X_train, y_train)

#Fit Ada_Boost_Regressor..........

Ada_boost = AdaBoostRegressor()

Ada_boost.fit(X_train, y_train)


#Fit Bagging_Regressor..........

Bagging = BaggingRegressor()

Bagging.fit(X_train, y_train)
#Fit Extra_tree_regressor........

Extra_trees = ExtraTreesRegressor()

Extra_trees.fit(X_train, y_train)





#Fit Gradient_Boosting_Regressor........

Gradient_boosting = GradientBoostingRegressor()

Gradient_boosting.fit(X_train, y_train)

models= [('lin_reg', lin_reg), ('forest', forest), ('dt', tree),('Ada_boost',Ada_boost),('Bagging',Bagging),('Extra_trees',Extra_trees),('Gradient_boosting',Gradient_boosting)]

scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']





#for each model I want to test three different scoring metrics. Therefore, results[0] will be lin_reg x MSE, 

# results[1] lin_reg x MSE and so on until results [8], where we stored dt x r2



results= []

metric= []

for name, model in models:

    for i in scoring:

        scores = cross_validate(model, X_train, y_train, scoring=i, cv=10, return_train_score=True)

        results.append(scores)



print(results[20])





###############################################################################



#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

LR_RMSE_mean = np.sqrt(-results[0]['test_score'].mean())

LR_RMSE_std= results[0]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

LR_MAE_mean = -results[1]['test_score'].mean()

LR_MAE_std= results[1]['test_score'].std()

LR_r2_mean = results[2]['test_score'].mean()

LR_r2_std = results[2]['test_score'].std()



#THIS IS FOR RF

RF_RMSE_mean = np.sqrt(-results[3]['test_score'].mean())

RF_RMSE_std= results[3]['test_score'].std()

RF_MAE_mean = -results[4]['test_score'].mean()

RF_MAE_std= results[4]['test_score'].std()

RF_r2_mean = results[5]['test_score'].mean()

RF_r2_std = results[5]['test_score'].std()



#THIS IS FOR DT

DT_RMSE_mean = np.sqrt(-results[6]['test_score'].mean())

DT_RMSE_std= results[6]['test_score'].std()

DT_MAE_mean = -results[7]['test_score'].mean()

DT_MAE_std= results[7]['test_score'].std()

DT_r2_mean = results[8]['test_score'].mean()

DT_r2_std = results[8]['test_score'].std()









#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

ADA_RMSE_mean = np.sqrt(-results[9]['test_score'].mean())

ADA_RMSE_std= results[9]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

ADA_MAE_mean = -results[10]['test_score'].mean()

ADA_MAE_std= results[10]['test_score'].std()

ADA_r2_mean = results[11]['test_score'].mean()

ADA_r2_std = results[11]['test_score'].std()







#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

BAGGING_RMSE_mean = np.sqrt(-results[12]['test_score'].mean())

BAGGING_RMSE_std= results[12]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

BAGGING_MAE_mean = -results[13]['test_score'].mean()

BAGGING_MAE_std= results[13]['test_score'].std()

BAGGING_r2_mean = results[14]['test_score'].mean()

BAGGING_r2_std = results[14]['test_score'].std()





#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

ET_RMSE_mean = np.sqrt(-results[15]['test_score'].mean())

ET_RMSE_std= results[15]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

ET_MAE_mean = -results[16]['test_score'].mean()

ET_MAE_std= results[16]['test_score'].std()

ET_r2_mean = results[17]['test_score'].mean()

ET_r2_std = results[17]['test_score'].std()





#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

GB_RMSE_mean = np.sqrt(-results[18]['test_score'].mean())

GB_RMSE_std= results[18]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

GB_MAE_mean = -results[19]['test_score'].mean()

GB_MAE_std= results[19]['test_score'].std()

GB_r2_mean = results[20]['test_score'].mean()

GB_r2_std = results[20]['test_score'].std()







modelDF = pd.DataFrame({

    'Model'       : ['Linear Regression', 'Random Forest', 'Decision Trees','Ada Boosting','Bagging','Extra trees','Gradient Boosting'],

    'RMSE_mean'    : [LR_RMSE_mean, RF_RMSE_mean, DT_RMSE_mean,ADA_RMSE_mean,BAGGING_RMSE_mean,ET_RMSE_mean,GB_RMSE_mean],

    'RMSE_std'    : [LR_RMSE_std, RF_RMSE_std, DT_RMSE_std,ADA_RMSE_std,BAGGING_RMSE_std,ET_RMSE_std,GB_RMSE_std],

    'MAE_mean'   : [LR_MAE_mean, RF_MAE_mean, DT_MAE_mean,ADA_MAE_mean,BAGGING_MAE_mean,ET_MAE_mean,GB_MAE_mean],

    'MAE_std'   : [LR_MAE_std, RF_MAE_std, DT_MAE_std, ADA_MAE_std, BAGGING_MAE_std, ET_MAE_std, GB_MAE_std],

    'r2_mean'      : [LR_r2_mean, RF_r2_mean, DT_r2_mean, ADA_r2_mean,BAGGING_r2_mean, ET_r2_mean, GB_r2_mean],

    'r2_std'      : [LR_r2_std, RF_r2_std, DT_r2_std, ADA_r2_std,BAGGING_r2_std, ET_r2_std, GB_r2_std],

    }, columns = ['Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'r2_mean', 'r2_std'])



    

modelDF.sort_values(by='r2_mean', ascending=False)





import seaborn as sns



sns.factorplot(x= 'Model', y= 'RMSE_mean', data= modelDF, kind='bar',size=6, aspect=4)



from sklearn.model_selection import GridSearchCV



GBR =GradientBoostingRegressor()



parameters = {

    

    "learning_rate": [0.01,0.05, 0.1, 0.15, 0.2],

    "min_samples_split": [100,150],

    "min_samples_leaf":[100,150],

    "max_depth":[3,5,8],

    "max_features":[0.3, 0.1],

    "n_estimators":[10,15,20,25,30,35,40]

    }



clf = GridSearchCV(GBR, parameters, cv=10, n_jobs=-1)



clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.best_params_)
GBR = GradientBoostingRegressor(learning_rate= 0.2,max_depth=5,max_features=0.3,min_samples_leaf=100,min_samples_split = 100, n_estimators=35)

GBR.fit(X_train, y_train)





#predicting the test set

y_pred = GBR.predict(X_test)

from sklearn import metrics

print("MAE:", metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))