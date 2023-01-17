import warnings

warnings.simplefilter(action='ignore')



import numpy as np

np.set_printoptions(suppress=True)

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing



import seaborn as sns



import sklearn

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, RidgeCV

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn import linear_model

from sklearn.feature_selection import chi2

import statsmodels.api as sm

from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures





from sklearn.feature_selection import RFE
def understand_variables(dataset):

    print(type(dataset))

    print(dataset.shape)

    print(dataset.head())

    print(dataset.columns)

    print(dataset.nunique(axis=0))

    print(dataset.describe())

    print(dataset.describe(exclude=[np.number]))

    print("\nNull count :\n"+str(dataset.isnull().sum()))

    

    

def EDA(dataset,feature_type):

    

    if feature_type == "Categorical":

        

        categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']   

        dataframes=[]

        for feature in categorical_features:

            dataframe=dataset[feature].value_counts().rename_axis(feature).reset_index(name='counts')

            dataframes.append(dataframe)



        for i in range(len(dataframes)):

            print(dataframes[i],'\n')

            

    elif feature_type == "Numeric":

        

        numerical_features=[feature for feature in dataset.columns if dataset[feature].dtype!='O']

        

        for feature in numerical_features:

            sns.distplot(dataset[feature])

            plt.show()





        sns.pairplot(dataset,kind="reg")

        plt.show()





def outlier_processing(dataset):

    # Using IQR



    Q1 = dataset.quantile(0.25)

    Q3 = dataset.quantile(0.75)

    IQR = Q3 - Q1

    

    #outlier_col = ['year']

    

    print("\n-------------\n% of outliers\n")    

    print(((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).sum()/len(dataset)*100)

    

    for col in list(IQR.index): 

        

        if col!='price':

        

            dataset.loc[dataset[col] < (Q1 - 1.5 * IQR)[col],[col]] = (Q1 - 1.5 * IQR)[col]

            dataset.loc[dataset[col] > (Q3 + 1.5 * IQR)[col],[col]] = (Q3 + 1.5 * IQR)[col]

            

            dataset[col] = dataset[col].round(0).astype(int)

    

    

    for col in ['price']:

        dataset = dataset[(dataset[col] <= (Q3 + 1.5 * IQR)[col]) & (dataset[col] >= (Q1 - 1.5 * IQR)[col])]

        #dataset[col] = dataset[col].round(0).astype(int)

        

    ## We eliminate rows with price as outliers, rest we replace with upper/lower boundary



    return dataset
cars_dataset = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv",index_col="Unnamed: 0")

cars_dataset = cars_dataset.drop(["lot","vin"],axis=1)

understand_variables(cars_dataset)
EDA(cars_dataset,feature_type="Categorical")
EDA(cars_dataset,feature_type="Numeric")
################ feature engineering ###########



######### convert year to age (2020 - year)

cars_dataset.year = 2021 - cars_dataset.year



######## condition column : [Listings expired = 0, remove 'left' from others, convert everything to minutess ]



cars_dataset.loc[cars_dataset.condition == "Listing Expired", 'condition'] = "0 minutes left"

cars_dataset['condition'] = cars_dataset.condition.str.replace("left","")

cars_dataset.loc[cars_dataset.condition.str.contains("minutes"),'condition'] = (cars_dataset.loc[cars_dataset.condition.str.contains("minutes"),'condition'].astype(str).str.split().str[0].astype(int)).astype(str)

cars_dataset.loc[cars_dataset.condition.str.contains("hours"),'condition'] = (cars_dataset.loc[cars_dataset.condition.str.contains("hours"),'condition'].astype(str).str.split().str[0].astype(int) * 60).astype(str)

cars_dataset.loc[cars_dataset.condition.str.contains("days"),'condition'] = (cars_dataset.loc[cars_dataset.condition.str.contains("days"),'condition'].astype(str).str.split().str[0].astype(int) * 60*24).astype(str)

cars_dataset.condition = cars_dataset.condition.astype(int)



######## dealing with outliers ########



cars_dataset = outlier_processing(cars_dataset)



#cars_dataset = cars_dataset[cars_dataset.price>0]





############## Correlation check ############



corr = cars_dataset.corr()

#sns.heatmap(corr, annot=True)





####### get dummies ########



cars_dataset = pd.get_dummies(cars_dataset, dummy_na=True)

############## Feauture Selection (using Correlation) ######



corr = cars_dataset.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = cars_dataset.columns[columns]

cars_dataset = cars_dataset[selected_columns]





############## Feauture Selection (using p-value) ######



X = cars_dataset.drop("price",axis=1)

y = cars_dataset["price"]





X = sm.add_constant(X)

mod = sm.OLS(y,X)

fii = mod.fit()

sm_p_value = fii.summary2().tables[1]['P>|t|']

pvalues = pd.Series(sm_p_value)





sig_p_val = pvalues[pvalues<=0.05]

sig_p_val.drop("const", inplace=True)

cars_col_index = sig_p_val.index



cars_col_index = pd.Series(cars_col_index)



cars_col_index = list(cars_col_index)



cars_dataset = cars_dataset[cars_col_index] 

cars_dataset = pd.concat([cars_dataset,y], axis=1)



print("Retained columns : " +  str(cars_dataset.columns))
############ Training the model ##############



X = cars_dataset.drop("price",axis=1)

y = cars_dataset["price"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



regressor = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1,1e1,1e2,1e3,1e4,1e5,1e6], store_cv_values=True)

regressor.fit(X_train, y_train)

cv_mse = np.mean(regressor.cv_values_, axis=0)

#print([0.0001,0.001,0.01,0.1,1,1e1,1e2,1e3,1e4,1e5,1e6])

#print(cv_mse)



# Best alpha

print("Best alpha = " + str(regressor.alpha_))



y_pred = pd.Series(regressor.predict(X_test))



y_pred[y_pred<0]=0



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("R_2 = {:.2f} %".format((r2_score(y_test, y_pred)*100)))

print("Variance score = {:.2f} %".format((regressor.score(X_test, y_test)*100)))



#print('10% of Mean Price:', cars_dataset['price'].mean() * 0.1)



sns.regplot(y_test,y_pred)

plt.show()