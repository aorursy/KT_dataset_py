import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model as lm

import statsmodels.api as sm



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from statsmodels.tools.eval_measures import mse, rmse



import warnings

warnings.filterwarnings(action= "ignore")
from matplotlib import style

style.use('fivethirtyeight')
from subprocess import check_output



print(check_output(["ls", "../input/allcsv"]).decode("utf8"))
LifeExpectancyData = pd.read_csv('../input/life-expectancy-who/led.csv')

regions = pd.read_csv('../input/allcsv/all.csv')
LifeExpectancyData.head()
LifeExpectancyData.info()
LifeExpectancyData.isnull().sum()
LifeExpectancyData.columns 
LifeExpectancyData.columns= ['Country', 'Year', 'Status', 'Life_Expectancy', 'Adult_Mortality',

       'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',

       'Measles', 'BMI', 'under_five_deaths', 'Polio', 'Total_Expenditure',

       'Diphtheria', 'HIV/AIDS', 'GDP','Population', 'thinness_1_19_years', 'thinness_5_9_years',

       'Income_composition_of_resources', 'Schooling']
total_missing_values = LifeExpectancyData.isnull().sum()

missing_values_per = LifeExpectancyData.isnull().sum()/LifeExpectancyData.isnull().count()

null_values = pd.concat([total_missing_values, missing_values_per], axis=1, keys=['total_null', 'total_null_perc'])

null_values = null_values.sort_values('total_null', ascending=False)
def null_cell(LifeExpectancyData):

    total_missing_values = LifeExpectancyData.isnull().sum()

    missing_values_per = LifeExpectancyData.isnull().sum()/LifeExpectancyData.isnull().count()

    null_values = pd.concat([total_missing_values, missing_values_per], axis=1, keys=['total_null', 'total_null_perc'])

    null_values = null_values.sort_values('total_null', ascending=False)

    return null_values[null_values['total_null'] > 0]
plt.figure(figsize=(10,8))

sns.heatmap(LifeExpectancyData.isnull(), cmap='viridis')
#regions = pd.read_csv('./data/all.csv')
regions.head()
regions[['name', 'region', 'sub-region']].isnull().sum()
regions.columns
LifeExpectancyData_merged = pd.merge(LifeExpectancyData, regions[['name', 'region', 'sub-region']],

                                     left_on='Country', right_on='name')
null_cell(LifeExpectancyData_merged)
LifeExpectancyData_merged.head()
LifeExpectancyData_merged.drop('Population', inplace=True, axis=1)
LifeExpectancyData_merged.columns
fill_list = (null_cell(LifeExpectancyData_merged)).index
df_interpolate = LifeExpectancyData_merged.copy()



for col in fill_list:

    df_interpolate[col] = df_interpolate.groupby(['Country'])[col].transform(lambda x: x.interpolate(limit_direction = 'both'))
null_cell(df_interpolate)
df_interpolate[df_interpolate['Adult_Mortality'].isna()]
for col in fill_list:

    df_interpolate[col] = df_interpolate.groupby(['sub-region', 'Year'])[col].transform(lambda x: x.interpolate(limit_direction='both'))
null_cell(df_interpolate)
LifeExpectancyData_num = df_interpolate._get_numeric_data() 
corr_matrix = LifeExpectancyData_num.corr()

corr_list = corr_matrix.Life_Expectancy.abs().sort_values(ascending=False).index[1:]
corr_list
plt.figure(figsize=(15,15))

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')

plt.title('Correlation Matrix')
corr_matrix = LifeExpectancyData_num[['Hepatitis_B','Measles', 'Polio','Diphtheria','HIV/AIDS', 'thinness_1_19_years',

                                      'thinness_5_9_years','Life_Expectancy']].corr()

corr_list = corr_matrix.Life_Expectancy.abs().sort_values(ascending=False).index[1:]
corr_list
plt.figure(figsize=(10,10))

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')

plt.title('Correlation Matrix')
plt.figure(figsize=(20,10))

sns.violinplot(x=df_interpolate["Year"], y=df_interpolate["Life_Expectancy"], data=df_interpolate)

plt.title('General Looking on Life Expectancy in Years')
plt.figure(figsize=(20,10))

sns.violinplot(x=df_interpolate.loc[df_interpolate['Year']>2009]["Year"], 

               y=df_interpolate["Life_Expectancy"],

               hue=df_interpolate["region"], 

               data=df_interpolate.loc[df_interpolate['Year']>2010], 

               palette="muted")



plt.title('Life Expectancy Values in Years by Regions')
plt.figure(figsize=(20,10))

sns.scatterplot(x='Life_Expectancy', 

                y='Alcohol', 

                hue='region',

                data=df_interpolate, 

                s=df_interpolate.GDP/100);

plt.xlabel('Life_expectancy',size=15)

plt.ylabel('Alcohol', size =10)

plt.show()
plt.rcParams['figure.dpi'] = 60

plt.rcParams['figure.figsize'] = (8,5.5)
outliers_by_nineteen_variables = ['Year', 'Life_Expectancy','Adult_Mortality', 'infant_deaths', 'Alcohol', 'percentage_expenditure',

                                    'Hepatitis_B','Measles', 'BMI',

                                    'under_five_deaths', 'Polio', 'Total_Expenditure','Diphtheria', 'HIV/AIDS', 'GDP',

                                    'thinness_1_19_years', 'thinness_5_9_years', 'Income_composition_of_resources', 'Schooling'] 

plt.figure(figsize=(25,25))



for i in range(0,19):

    plt.subplot(5, 4, i+1)

    plt.boxplot(df_interpolate[outliers_by_nineteen_variables[i]])

    plt.title(outliers_by_nineteen_variables[i])
from scipy.stats.mstats import winsorize
def winsor(x, multiplier=3): 

    upper= x.median() + x.std()*multiplier

    for limit in np.arange(0.001, 0.20, 0.001):

        if np.max(winsorize(x,(0,limit))) < upper:

            return limit

    return None 
#An example to get limit value for winsorization

limit= winsor(df_interpolate['infant_deaths'])

print(limit)
df_interpolate["Adult_Mortality"]        = winsorize(df_interpolate["Adult_Mortality"], (0, 0.018))

df_interpolate["infant_deaths"]          = winsorize(df_interpolate["infant_deaths"], (0, 0.018))

df_interpolate["percentage_expenditure"] = winsorize(df_interpolate["percentage_expenditure"], (0, 0.036))

df_interpolate["Hepatitis_B"]            = winsorize(df_interpolate["Hepatitis_B"], (0,0.001))

df_interpolate["Measles"]                = winsorize(df_interpolate["Measles"], (0, 0.018))

df_interpolate["under_five_deaths"]      = winsorize(df_interpolate["under_five_deaths"], (0, 0.013))

df_interpolate["Polio"]                  = winsorize(df_interpolate["Polio"], (0, 0.001))

df_interpolate["Total_Expenditure"]      = winsorize(df_interpolate["Total_Expenditure"], (0, 0.011))

df_interpolate["Diphtheria"]             = winsorize(df_interpolate["Diphtheria"], (0, 0.001))

df_interpolate["HIV/AIDS"]               = winsorize(df_interpolate["HIV/AIDS"], (0, 0.030))

df_interpolate["GDP"]                    = winsorize(df_interpolate["GDP"], (0, 0.43))

df_interpolate["thinness_1_19_years"]    = winsorize(df_interpolate["thinness_1_19_years"], (0, 0.026))

df_interpolate["thinness_5_9_years"]     = winsorize(df_interpolate["thinness_5_9_years"], (0, 0.27))

df_interpolate["Income_composition_of_resources"] = winsorize(df_interpolate["Income_composition_of_resources"], (0, 0.001))

df_interpolate["Schooling"]              = winsorize(df_interpolate["Schooling"], (0, 0.001))

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
LifeExpectancyData_num = df_interpolate._get_numeric_data() 
LifeExpectancyData_num = LifeExpectancyData_num.dropna()



X = StandardScaler().fit_transform(LifeExpectancyData_num) #standardize the feature matrix



pca = PCA(n_components=0.90, whiten=True)



X_pca = pca.fit_transform(X)
print (pca.explained_variance_ratio_)
print('Original Number of Features', X.shape[1]) 

print('Reduced Number of Features',X_pca.shape[1])
#Creating a scaler object

sc = StandardScaler()



#fit the scaler to the features and transform

X_std = sc.fit_transform(X)



# Fit the PCA and transform the data

X_std_pca = pca.fit_transform(X_std)



# View the new feature data's shape

X_std_pca.shape
from sklearn.decomposition import PCA

from sklearn import decomposition, datasets
#Creating a PCA object with 12 components as a parameter

pca = decomposition.PCA(n_components=12) 

# Fit the PCA and transform the data

X_std_pca = pca.fit_transform(X_std)



# View the new feature data's shape

X_std_pca.shape
plt.figure(figsize = (10,5))

plt.plot(pca.explained_variance_ratio_)

plt.title('Total variance explained: {}'.format(pca.explained_variance_ratio_.sum()))

plt.show()
df_interpolate.info()
df_dummies = pd.get_dummies(df_interpolate)

df_dummies.head()
df_dummies = df_dummies.dropna()



X = StandardScaler().fit_transform(df_dummies)#standardize the feature matrix



pca = PCA(n_components=0.95, whiten=True)



X_pca = pca.fit_transform(X)
print('Original Number of Features', X.shape[1]) 

print('Reduced Number of Features',X_pca.shape[1])
#Creating a scaler object

sc = StandardScaler()



#fit the scaler to the features and transform

X_std = sc.fit_transform(X)
#Creating a PCA object with 178 components as a parameter

pca = decomposition.PCA(n_components=178) 

# Fit the PCA and transform the data

X_std_pca = pca.fit_transform(X_std)



# View the new feature data's shape

X_std_pca.shape
plt.figure(figsize = (10,5))

plt.plot(pca.explained_variance_ratio_)

plt.title('Total variance explained: {}'.format(pca.explained_variance_ratio_.sum()))

plt.show()
y_allValues = LifeExpectancyData_num['Life_Expectancy']

X_allValues = LifeExpectancyData_num[corr_list]
X_train, X_test, y_train, y_test = train_test_split(X_allValues, y_allValues, test_size = 0.2, random_state = 101)



print(" Observations in Training Group : {}".format(X_train.shape[0]))

print(" Observations in Test Group     : {}".format(X_test.shape[0]))
X_train = sm.add_constant(X_train)



Model_all = sm.OLS(y_train, X_train).fit()



Model_all.summary()
pValue = Model_all.pvalues

significant_values = list(pValue[pValue<= 0.05].index)
from sklearn import linear_model
Model_all = linear_model.LinearRegression()

Model_all.fit(X_allValues, y_allValues)
pred = Model_all.predict(X_allValues)

Residuals = y_allValues - pred
from statsmodels.tsa.stattools import acf



acf_data = acf(Residuals)



plt.figure(figsize=(9,6))

plt.plot(acf_data[1:])

plt.show()
rand_nums = np.random.normal(np.mean(Residuals), np.std(Residuals), len(Residuals))



plt.figure(figsize=(12,5))



plt.subplot(1,2,1)

plt.scatter(np.sort(rand_nums), np.sort(Residuals))

plt.xlabel("Normally Distributed Random Variable")

plt.ylabel("Residuals")

plt.title("QQ Plot")



plt.subplot(1,2,2)

plt.hist(Residuals)

plt.xlabel("Residuals")

plt.title("Residuals Histogram")



plt.tight_layout()

plt.show()
from scipy.stats import jarque_bera

from scipy.stats import normaltest
jb_stats = jarque_bera(Residuals)

norm_stats = normaltest(Residuals)



print("Jarque-Bera test value : {0} ve p değeri : {1}".format(jb_stats[0], jb_stats[1]))

print("Normal test value      : {0}  ve p değeri : {1:.30f}".format(norm_stats[0], norm_stats[1]))
df = LifeExpectancyData_num.drop(["Life_Expectancy", "Year"], axis=1)
df.shape
from sklearn.preprocessing import PolynomialFeatures 
def polynomial(df,pol):

    poly = PolynomialFeatures(pol)

    poly_array = poly.fit_transform(df.drop('Life_Expectancy', axis=1))

    df_dropped = df.drop('Life_Expectancy', axis=1)

    df_pol = pd.DataFrame(poly_array, columns= poly.get_feature_names(df_dropped.columns))

    df_pol = pd.concat([df_pol, df['Life_Expectancy']], axis=1)

    Feature_list = df_pol.corr()['Life_Expectancy'].abs().sort_values(ascending = False)[1:].index

    return pd.concat([df_pol[Feature_list], df['Life_Expectancy']], axis=1)
df_pol1 = polynomial(LifeExpectancyData_num,1)
def model_pol(df,pol):

    y = df['Life_Expectancy']

    Feature_list = Feature_list = df.columns[:500] #Having overfitting after 200 variables I prefer to limit until 500

    MSE_list_test=[]

    R_list=[]

    number_of_variables=[]

    MAE_list=[]

    RMSE_list=[]

    MAPE_list=[]

    R_train_list=[]

    MSE_train_list=[]

    adj_R_test=[]

    adj_R_train=[]

    for variable in range(1,len(Feature_list)-1, pol**pol*2):

        selected_features =  Feature_list[:(-1*variable)]

        X_poly=df[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)

        

        

        model_poly = LinearRegression()

        results = model_poly.fit(X_train, y_train)

        y_pred  = model_poly.predict(X_test)

        y_pred_train = model_poly.predict(X_train)



        MSE_list_test.append(mse(y_test, y_pred))

        MSE_train_list.append(mse(y_train, y_pred_train))



        R_list.append(model_poly.score(X_test, y_test))

        adj_R_test.append(1 - (1-model_poly.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

        

        R_train_list.append(model_poly.score(X_train, y_train))

        adj_R_train.append(1 - (1-model_poly.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))





        number_of_variables.append(len(selected_features))



        MAE_list.append(mean_absolute_error(y_test, y_pred))



        RMSE_list.append(rmse(y_test, y_pred))



        MAPE_list.append(np.mean(np.abs((y_test-y_pred) / y_test)) * 100)

        

    model_means = list(zip(number_of_variables, R_list,R_train_list,MSE_list_test,MSE_train_list,MAE_list,RMSE_list,MAPE_list,adj_R_test,adj_R_train))

    poly_means = pd.DataFrame(model_means, columns= ['number_of_variables','R_list','R_train_list',

                                                            'MSE_list_test','MSE_train_list','MAE_list', 'RMSE_list', 'MAPE_list','adj_R_test', 'adj_R_train'])

    

    return poly_means
df_poly_transform1 = polynomial(LifeExpectancyData_num,1)

df_pol1 = model_pol(df_poly_transform1,2)
df_poly_transform2 = polynomial(LifeExpectancyData_num,2)

df_pol2 = model_pol(df_poly_transform2,2)
#%%time #checking total time of process in Pyhton

df_poly_transform3 = polynomial(LifeExpectancyData_num,3)

df_pol3 = model_pol(df_poly_transform3,3)
display(df_pol1.sort_values(by='MSE_list_test').head())

display(df_pol2.sort_values(by='MSE_list_test').head())

display(df_pol3.sort_values(by='MSE_list_test').head())
plt.figure(1, figsize = (25,10))

plt.suptitle('MSE TEST TRAIN VALUES', size=20)







plt.subplot(1,3,1)

plt.plot(df_pol1.number_of_variables,df_pol1.MSE_list_test, label  = 'MSE Values', color='blue', linewidth=5)

plt.plot(df_pol1.number_of_variables,df_pol1.MSE_train_list, label = 'MSE_train Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values ')

plt.title('POL 1 MSE Test/Train Values')

plt.ylim(0,30)

plt.legend()



plt.subplot(1,3,2)

plt.plot(df_pol2.number_of_variables, df_pol2.MSE_list_test, label = 'MSE Values', color='blue', linewidth=5)

plt.plot(df_pol2.number_of_variables, df_pol2.MSE_train_list,label = 'MSE_train Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values ')

plt.ylim(0,30)

plt.title('POL 2 MSE Test/Train Values')

plt.legend()



plt.subplot(1,3,3)

plt.plot(df_pol3.number_of_variables, df_pol3.MSE_list_test, label = 'MSE Values', color='blue', linewidth=5)

plt.plot(df_pol3.number_of_variables, df_pol3.MSE_train_list,label = 'MSE_train Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values ')

plt.ylim(0,30)

plt.title('POL 3 MSE Test/Train Values')





plt.subplots_adjust()

plt.legend()

plt.show()



plt.figure(figsize=(15,10))

objects = ('df_pol1', 'df_pol2', 'df_pol3')



y_pos = np.arange(len(objects)) 

performance  =[df_pol1.MSE_list_test.min() ,df_pol2.MSE_list_test.min(), df_pol3.MSE_list_test.min()]

performance2 =[df_pol1.MSE_train_list.min(), df_pol2.MSE_train_list.min(), df_pol3.MSE_train_list.min()]



plt.subplot(121)

plt.bar(y_pos, performance, align='center')

plt.xticks(y_pos, objects,size=10)

plt.xlabel('Model',size=10)

plt.ylabel('MSE Values',size=10)

plt.title('MSE TEST Values \n', fontsize=10)



plt.subplot(122)

plt.bar(y_pos, performance2, align='center')

plt.xticks(y_pos, objects,size=10)

plt.title('MSE TRAIN Values \n', size = 10)





plt.xlabel('Model',size=10)

plt.ylabel('MSE Values',size=10)



plt.show()



plt.figure(figsize=(15,10))

objects = ('df_pol1', 'df_pol2', 'df_pol3')



y_pos = np.arange(len(objects)) 

performance  =[df_pol1.adj_R_test.max() ,df_pol2.adj_R_test.max(), df_pol3.adj_R_test.max()]

performance2 =[df_pol1.adj_R_train.max(), df_pol2.adj_R_train.max(), df_pol3.adj_R_train.max()]



plt.subplot(121)

plt.bar(y_pos, performance, align='center')

plt.xticks(y_pos, objects,size=10)

plt.xlabel('Model',size=10)

plt.ylabel('Adj R Squared Test Values',size=10)

plt.title('Adj R Squared Test Values \n', fontsize=10)



plt.subplot(122)

plt.bar(y_pos, performance2, align='center')

plt.xticks(y_pos, objects,size=10)

plt.ylabel('Adj R Squared Train Values',size=10)

plt.title('Adj R Squared Train Values \n', size = 10)

plt.xlabel('Model',size=10)





plt.show()
df = LifeExpectancyData_num.drop(["Life_Expectancy", "Year"], axis=1)
poly = PolynomialFeatures(2)

poly_array = poly.fit_transform(df)
df_poly2 = pd.DataFrame(poly_array, columns= poly.get_feature_names())
y = LifeExpectancyData_num['Life_Expectancy']

X = df_poly2



X_train_pol2, X_test_pol2, y_train_pol2, y_test_pol2 = train_test_split(X, y, test_size = 0.2, random_state = 101)



print("Eğitim kümesindeki gözlem sayısı : {}".format(X_train.shape[0]))

print("Test kümesindeki gözlem sayısı   : {}".format(X_test.shape[0]))



X_train = sm.add_constant(X_train)



poly_model_2 = sm.OLS(y_train_pol2, X_train_pol2).fit()

y_preds_pol2 = poly_model_2.predict(X_test_pol2)

y_preds_train_pol2 = poly_model_2.predict(X_train_pol2)
poly = PolynomialFeatures(3)

poly_array = poly.fit_transform(df)

df_poly3 = pd.DataFrame(poly_array, columns= poly.get_feature_names())



y = LifeExpectancyData_num['Life_Expectancy']

X = df_poly3



X_train_pol3, X_test_pol3, y_train_pol3, y_test_pol3 = train_test_split(X, y, test_size = 0.2, random_state = 101)



print("Observations in Train Group : {}".format(X_train.shape[0]))

print("Observations in Test Group  : {}".format(X_test.shape[0]))



X_train = sm.add_constant(X_train)



poly_model_3 = sm.OLS(y_train_pol3, X_train_pol3).fit()

y_preds_pol3 = poly_model_3.predict(X_test_pol3)

y_preds_train_pol3 = poly_model_3.predict(X_train_pol3)
poly = PolynomialFeatures(1)

poly_array = poly.fit_transform(df)

df_poly1 = pd.DataFrame(poly_array, columns= poly.get_feature_names())



y = LifeExpectancyData_num['Life_Expectancy']

X = df_poly1



X_train_pol1, X_test_pol1, y_train_pol1, y_test_pol1 = train_test_split(X, y, test_size = 0.2, random_state = 101)



print("Observations in Train Group : {}".format(X_train.shape[0]))

print("Observations in Test Group  : {}".format(X_test.shape[0]))



X_train = sm.add_constant(X_train)



poly_model_1 = sm.OLS(y_train_pol1, X_train_pol1).fit()

y_preds_pol1 = poly_model_1.predict(X_test_pol1)

y_preds_train_pol1 = poly_model_1.predict(X_train_pol1)

plt.figure(figsize=(18,8))

plt.suptitle('Scatter Plots of Life Expectancy Predictions', size = 16)



plt.subplot(1,3,1)

plt.title('Poly 1 Model \n', size = 14)

plt.scatter(y_test_pol1, y_preds_pol1)

plt.scatter(y_train_pol1, y_preds_train_pol1,alpha=0.10)

plt.plot(y_test_pol1, y_test_pol1, color="red")

plt.ylim(0,90)

plt.xlabel("True Values")

plt.ylabel("Predictions")



plt.subplot(1,3,2)

plt.title('Poly 2 Model \n', size = 14)

plt.scatter(y_test_pol2, y_preds_pol2 )

plt.scatter(y_train_pol2, y_preds_train_pol2,alpha=0.10)

plt.plot(y_test_pol2, y_test_pol2, color="red")

plt.xlabel("True Values")

plt.ylabel("Predictions")



plt.subplot(1,3,3)

plt.title('Poly 3 Model \n', size = 14)

plt.scatter(y_test_pol3, y_preds_pol3)

plt.scatter(y_train_pol3, y_preds_train_pol3,alpha=0.10)

plt.plot(y_test_pol3, y_test_pol3, color="red")

plt.ylim(0,90)

plt.xlabel("True Values")

plt.ylabel("Predictions")









plt.subplots_adjust()

plt.show()
from sklearn.linear_model import Ridge
def Ridge_model(df,pol, alpha, col=None):



    y = df['Life_Expectancy']

    Feature_list = df.columns[:500]

    

    MSE_list_test=[]

    R_list=[]

    adj_R_test=[]

    number_of_variables=[]

    MAE_list=[]

    RMSE_list=[]

    MAPE_list=[]

    R_train_list=[]

    adj_R_train=[]

    MSE_train_list=[]

    model_list=[]

    feature_list=[]

        

    

    for variable in range(1,len(Feature_list)-1, pol**pol*2):

        selected_features =  Feature_list[:(-1*variable)]

        X_poly=df[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)

                

        model_poly = Ridge(alpha= alpha) 

        model_poly.fit(X_train, y_train)

        results = model_poly.fit(X_train, y_train)

               

        y_pred  = model_poly.predict(X_test)

        

        y_pred_train = model_poly.predict(X_train)

      

        MSE_list_test.append(mse(y_test, y_pred))

        

        MSE_train_list.append(mse(y_train, y_pred_train))

        R_list.append(model_poly.score(X_test, y_test))

        adj_R_test.append(1 - (1-model_poly.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

        

        R_train_list.append(model_poly.score(X_train, y_train))

        adj_R_train.append(1 - (1-model_poly.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

        

        number_of_variables.append(len(selected_features))

        MAE_list.append(mean_absolute_error(y_test, y_pred))

        

        RMSE_list.append(rmse(y_test, y_pred))

        

        MAPE_list.append(np.mean(np.abs((y_test-y_pred) / y_test)) * 100)

        model_list.append(model_poly)

        feature_list.append(selected_features)

        

        

    

        

        

    model_means = list(zip(number_of_variables, R_list, adj_R_test, R_train_list, adj_R_train, MSE_list_test,

                           MSE_train_list,MAE_list,RMSE_list,MAPE_list,model_list,feature_list))

    

    poly_means = pd.DataFrame(model_means, columns= ['number_of_variables', 'R_list','adj_R_test',

                                                     'R_train_list','adj_R_train',

                                                     'MSE_list_test','MSE_train_list','MAE_list','RMSE_list','MAPE_list',

                                                     'model_list', 'feature_list'])

    

    

    return poly_means, (y_pred,y_pred_train, X_train,y_train, X_test, y_test, MSE_list_test,MSE_train_list)
%%time

for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]: 

    df, _  = Ridge_model(df_poly_transform2,2,alpha)

    print(alpha, df.MSE_list_test.min())
#The Best Model option with minimum MSE_test Value on Alpha 10-⁵ and polynomial 2nd degree.



df_Ridge_alpha_pol2, degerler1_2 = Ridge_model(df_poly_transform2,2,0.000001)
df_Ridge_alpha_pol2.head()
%%time

for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]:

    df, _  = Ridge_model(df_poly_transform3,3,alpha)

    print(alpha, df.MSE_list_test.min())
#The Best Model option with minimum MSE_test Value on Alpha 10³ and polynomial 3rd degree.

df_Ridge_alpha_pol3, degerler1_3 = Ridge_model(df_poly_transform3,3,1000)
df_Ridge_alpha_pol3.head()
MSE_list_test_alpha_pol2  = df_Ridge_alpha_pol2['MSE_list_test']

MSE_train_test_alpha_pol2 = df_Ridge_alpha_pol2['MSE_train_list']

MSE_list_test_alpha_pol3  = df_Ridge_alpha_pol3['MSE_list_test']

MSE_train_test_alpha_pol3 = df_Ridge_alpha_pol3['MSE_train_list']
plt.figure(1, figsize = (15,8))



plt.subplot(1,2,1)

plt.plot(df_Ridge_alpha_pol2.number_of_variables, MSE_list_test_alpha_pol2,label  = 'MSE Test Alpha Pol2 Values', color='blue', linewidth=5)

plt.plot(df_Ridge_alpha_pol2.number_of_variables, MSE_train_test_alpha_pol2,label = 'MSE Train  Alpha Pol2 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values')

plt.title('POLY 2 MSE Test/Train Values')

plt.legend()



plt.subplot(1,2,2)

plt.plot(df_Ridge_alpha_pol3.number_of_variables,MSE_list_test_alpha_pol3,label  = 'MSE Test Alpha Pol3 Values', color='blue', linewidth=5)

plt.plot(df_Ridge_alpha_pol3.number_of_variables, MSE_train_test_alpha_pol3,label = 'MSE Train Alpha Pol3 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values')

plt.title('POLY 3 MSE Test/Train Values')





plt.subplots_adjust()

plt.legend()

plt.show()
adj_R_test_alpha_pol2  = df_Ridge_alpha_pol2['adj_R_test']

adj_R_train_alpha_pol2 = df_Ridge_alpha_pol2['adj_R_train']

adj_R_test_alpha_pol3  = df_Ridge_alpha_pol3['adj_R_test']

adj_R_train_alpha_pol3 = df_Ridge_alpha_pol3['adj_R_train']









plt.figure(1, figsize = (15,8))



plt.subplot(1,2,1)

plt.plot(df_Ridge_alpha_pol2.number_of_variables, adj_R_test_alpha_pol2,label  = 'Adjusted R² Test Alpha Pol2 Values', color='blue', linewidth=5)

plt.plot(df_Ridge_alpha_pol2.number_of_variables, adj_R_train_alpha_pol2,label = 'Adjusted R² Train  Alpha Pol2 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Adjusted Values')

plt.title('Ridge POLY 2 Adjusted R² Test/Train Values')

plt.legend()



plt.subplot(1,2,2)

plt.plot(df_Ridge_alpha_pol3.number_of_variables,adj_R_test_alpha_pol3,label  = 'Adjusted R² Test Alpha Pol3 Values', color='blue', linewidth=5)

plt.plot(df_Ridge_alpha_pol3.number_of_variables, adj_R_train_alpha_pol3,label = 'Adjusted R² Train Alpha Pol3 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Adjusted Values')

plt.title('Ridge POLY 3 Adjusted R² Test/Train Values')





plt.subplots_adjust()

plt.legend()

plt.show()
from sklearn.linear_model import Lasso
def Lasso_model(df,pol, alpha):



    y = df['Life_Expectancy']

    Feature_list = df.columns[:500]

    

    MSE_list_test=[]

    R_list=[]

    adj_R_test=[]

    number_of_variables=[]

    MAE_list=[]

    RMSE_list=[]

    MAPE_list=[]

    R_train_list=[]

    adj_R_train=[]

    MSE_train_list=[]

    

    for variable in range(1,len(Feature_list)-1, pol**pol*2):

        selected_features =  Feature_list[:(-1*variable)]

        X_poly=df[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)

                

        model_poly = Lasso(alpha= alpha) 

        model_poly.fit(X_train, y_train)

        results = model_poly.fit(X_train, y_train)

               

        y_pred  = model_poly.predict(X_test)

        

        y_pred_train = model_poly.predict(X_train)

      

        MSE_list_test.append(mse(y_test, y_pred))

        

        MSE_train_list.append(mse(y_train, y_pred_train))

        

        R_list.append(model_poly.score(X_test, y_test))

        adj_R_test.append(1 - (1-model_poly.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

        

        R_train_list.append(model_poly.score(X_train, y_train))

        adj_R_train.append(1 - (1-model_poly.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

        

        number_of_variables.append(len(selected_features))

        MAE_list.append(mean_absolute_error(y_test, y_pred))

        

        RMSE_list.append(rmse(y_test, y_pred))

        

        MAPE_list.append(np.mean(np.abs((y_test-y_pred) / y_test)) * 100)

        

        

    model_means = list(zip(number_of_variables, R_list, adj_R_test, R_train_list, adj_R_train, MSE_list_test,MSE_train_list,MAE_list,RMSE_list,MAPE_list))

    

    poly_means = pd.DataFrame(model_means, columns= ['number_of_variables', 'R_list', 'adj_R_test', 'R_train_list', 'adj_R_train','MSE_list_test','MSE_train_list','MAE_list','RMSE_list','MAPE_list'])

    

    

    return poly_means, (y_pred,y_pred_train, X_train,y_train, X_test, y_test, model_poly, MSE_list_test,MSE_train_list)
%%time

for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]:

    df, _  = Lasso_model(df_poly_transform2,2,alpha)

    print(alpha, df.MSE_list_test.min())
#The Best Model option with minimum MSE_test Value on Alpha 10-⁵ and polynomial 2 degree



df_Lasso_alpha_pol2, degerler1_2 = Lasso_model(df_poly_transform2,2,0.000001)
df_Lasso_alpha_pol2.head()
%%time

for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]:

    df, _  = Lasso_model(df_poly_transform3,3,alpha)

    print(alpha, df.MSE_list_test.min())
# The Best Model option with minimum MSE_test Value on Alpha 10³ and polynomial 3 degree.



df_Lasso_alpha_pol3, degerler1_3 = Lasso_model(df_poly_transform3,3,1000)
df_Lasso_alpha_pol3.head()
MSE_list_test_Lasso_alpha_pol2  = df_Lasso_alpha_pol2['MSE_list_test']

MSE_train_test_Lasso_alpha_pol2 = df_Lasso_alpha_pol2['MSE_train_list']

MSE_list_test_Lasso_alpha_pol3  = df_Lasso_alpha_pol3['MSE_list_test']

MSE_train_test_Lasso_alpha_pol3 = df_Lasso_alpha_pol3['MSE_train_list']
plt.figure(1, figsize = (15,8))



plt.subplot(1,2,1)

plt.plot(df_Lasso_alpha_pol2.number_of_variables,MSE_list_test_Lasso_alpha_pol2, label = 'MSE Test  Alpha Pol2 Values', color='blue', linewidth=5)

plt.plot(df_Lasso_alpha_pol2.number_of_variables,MSE_train_test_Lasso_alpha_pol2,label = 'MSE Train  Alpha Pol2 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values')

plt.title('Lasso POLY 2 MSE Test/Train Values')

plt.legend()



plt.subplot(1,2,2)

plt.plot(df_Lasso_alpha_pol3.number_of_variables, MSE_list_test_Lasso_alpha_pol3,label = 'MSE Alpha1 Pol3 Values', color='blue', linewidth=5)

plt.plot(df_Lasso_alpha_pol3.number_of_variables, MSE_train_test_Lasso_alpha_pol3,label = 'MSE Train  Alpha Pol3 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values')

plt.title('Lasso POLY 3 MSE Test/Train Values')

plt.legend()



plt.subplots_adjust()



plt.show()
adj_R_test_Lasso_alpha_pol2  = df_Lasso_alpha_pol2['adj_R_test']

adj_R_train_Lasso_alpha_pol2 = df_Lasso_alpha_pol2['adj_R_train']

adj_R_test_Lasso_alpha_pol3  = df_Lasso_alpha_pol3['adj_R_test']

adj_R_train_Lasso_alpha_pol3 = df_Lasso_alpha_pol3['adj_R_train']



plt.figure(1, figsize = (15,8))



plt.subplot(1,2,1)

plt.plot(df_Lasso_alpha_pol2.number_of_variables,adj_R_test_Lasso_alpha_pol2, label = 'Adjusted R² Test  Alpha Pol2 Values', color='blue', linewidth=5)

plt.plot(df_Lasso_alpha_pol2.number_of_variables,adj_R_train_Lasso_alpha_pol2,label = 'Adjusted R² Train  Alpha Pol2 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Adjusted R² Values')

plt.title('Lasso POLY 2 Adjusted R² Test/Train Values')

plt.legend()



plt.subplot(1,2,2)

plt.plot(df_Lasso_alpha_pol3.number_of_variables, adj_R_test_Lasso_alpha_pol3,label = 'Adjusted R² Alpha1 Pol3 Values', color='blue', linewidth=5)

plt.plot(df_Lasso_alpha_pol3.number_of_variables, adj_R_train_Lasso_alpha_pol3,label = 'Adjusted R² Train  Alpha Pol3 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Adjusted R² Values')

plt.title('Lasso POLY 3 Adjusted R² Test/Train Values')

plt.legend()



plt.subplots_adjust()



plt.show()



from sklearn.linear_model import ElasticNet
def ElasticNet_model(df,pol, alpha):



    y = df['Life_Expectancy']

    Feature_list = df.columns[:500]

    

    MSE_list_test=[]

    R_list=[]

    adj_R_test=[]

    number_of_variables=[]

    MAE_list=[]

    RMSE_list=[]

    MAPE_list=[]

    R_train_list=[]

    adj_R_train=[]

    MSE_train_list=[]

    

    for variable in range(1,len(Feature_list)-1, pol**pol*2):

        selected_features =  Feature_list[:(-1*variable)]

        X_poly=df[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)

                

        model_poly = ElasticNet(alpha=alpha, l1_ratio=0.5)

        model_poly.fit(X_train, y_train)

        results = model_poly.fit(X_train, y_train)

               

        y_pred  = model_poly.predict(X_test)

        

        y_pred_train = model_poly.predict(X_train)

      

        MSE_list_test.append(mse(y_test, y_pred))

        

        MSE_train_list.append(mse(y_train, y_pred_train))

        

        R_list.append(model_poly.score(X_test, y_test))

        adj_R_test.append(1 - (1-model_poly.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

        

        R_train_list.append(model_poly.score(X_train, y_train))

        adj_R_train.append(1 - (1-model_poly.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

                

        number_of_variables.append(len(selected_features))

        MAE_list.append(mean_absolute_error(y_test, y_pred))

        

        RMSE_list.append(rmse(y_test, y_pred))

        

        MAPE_list.append(np.mean(np.abs((y_test-y_pred) / y_test)) * 100)

        

        

    model_means = list(zip(number_of_variables, R_list, adj_R_test, R_train_list, adj_R_train, MSE_list_test,MSE_train_list,MAE_list,RMSE_list,MAPE_list))

    

    poly_means = pd.DataFrame(model_means, columns= ['number_of_variables', 'R_list', 'adj_R_test', 'R_train_list', 'adj_R_train', 'MSE_list_test','MSE_train_list','MAE_list','RMSE_list','MAPE_list'])

    

    

    return poly_means, (y_pred,y_pred_train, X_train,y_train, X_test, y_test, model_poly, MSE_list_test,MSE_train_list)
%%time

for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]:

    df, _  = ElasticNet_model(df_poly_transform3,3,alpha)

    print(alpha, df.MSE_list_test.min())
#The Best Model with minimum MSE_test Value on Alpha 10⁴ and polynomial 3 degree 

df_ElasticNet_alpha_pol3, degerler1_3 = ElasticNet_model(df_poly_transform3,3,0.00001)
%%time

for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]:

    df, _  = ElasticNet_model(df_poly_transform2,2,alpha)

    print(alpha, df.MSE_list_test.min())
#The Best Model with minimum MSE_test Value on Alpha 10-⁵ and polynomial 2 degree 



df_ElasticNet_alpha_pol2, degerler1_2 = ElasticNet_model(df_poly_transform2,2,0.000001)
MSE_list_test_ElasticNet_alpha_pol2  = df_ElasticNet_alpha_pol2['MSE_list_test']

MSE_list_train_ElasticNet_alpha_pol2 = df_ElasticNet_alpha_pol2['MSE_train_list']

MSE_list_test_ElasticNet_alpha_pol3  = df_ElasticNet_alpha_pol3['MSE_list_test']

MSE_list_train_ElasticNet_alpha_pol3 = df_ElasticNet_alpha_pol3['MSE_train_list']
plt.figure(1, figsize = (15,8))



plt.subplot(1,2,1)

plt.plot(df_ElasticNet_alpha_pol2.number_of_variables,MSE_list_test_ElasticNet_alpha_pol2, label = 'MSE Test  Alpha Pol2 Values', color='blue', linewidth=5)

plt.plot(df_ElasticNet_alpha_pol2.number_of_variables,MSE_list_train_ElasticNet_alpha_pol2,label = 'MSE Train  Alpha Pol2 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values')

plt.title('POLY 2 MSE Test/Train Values')

plt.legend()



plt.subplot(1,2,2)

plt.plot(df_ElasticNet_alpha_pol3.number_of_variables, MSE_list_test_ElasticNet_alpha_pol3,label = 'MSE Alpha1 Pol3 Values', color='blue', linewidth=5)

plt.plot(df_ElasticNet_alpha_pol3.number_of_variables, MSE_list_train_ElasticNet_alpha_pol3,label = 'MSE Train  Alpha Pol3 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Values')

plt.title('POLY 3 MSE Test/Train Values')

plt.legend()



plt.subplots_adjust()



plt.show()
adj_R_test_ElasticNet_alpha_pol2  = df_ElasticNet_alpha_pol2['adj_R_test']

adj_R_train_ElasticNet_alpha_pol2 = df_ElasticNet_alpha_pol2['adj_R_train']

adj_R_test_ElasticNet_alpha_pol3  = df_ElasticNet_alpha_pol3['adj_R_test']

adj_R_train_ElasticNet_alpha_pol3 = df_ElasticNet_alpha_pol3['adj_R_train']





plt.figure(1, figsize = (15,8))



plt.subplot(1,2,1)

plt.plot(df_ElasticNet_alpha_pol2.number_of_variables,adj_R_test_ElasticNet_alpha_pol2, label = 'Adjusted R² Test  Alpha Pol2 Values', color='blue', linewidth=5)

plt.plot(df_ElasticNet_alpha_pol2.number_of_variables,adj_R_train_ElasticNet_alpha_pol2,label = 'Adjusted R² Train  Alpha Pol2 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Adjusted R²')

plt.title('Elastic Net POLY 2 Adjusted R² Test/Train Values')

plt.legend()



plt.subplot(1,2,2)

plt.plot(df_ElasticNet_alpha_pol3.number_of_variables, adj_R_test_ElasticNet_alpha_pol3,label = 'Adjusted R² Alpha1 Pol3 Values', color='blue', linewidth=5)

plt.plot(df_ElasticNet_alpha_pol3.number_of_variables, adj_R_train_ElasticNet_alpha_pol3,label = 'Adjusted R² Train  Alpha Pol3 Values', color='red', linewidth=5)

plt.xlabel('Number of Variable')

plt.ylabel('Adjusted R²')

plt.title('Elastic Net POLY 3 Adjusted R² Test/Train Values')

plt.legend()



plt.subplots_adjust()



plt.show()
plt.figure(figsize=(25,20))



objects=('df_pol1', 'df_pol2', 'df_pol3',

           'df_Ridge_alpha_pol2', 'df_Ridge_alpha_pol3',

           'df_Lasso_alpha_pol2', 'df_Lasso_alpha_pol3',

           'df_ElasticNet_alpha_pol2', 'df_ElasticNet_alpha_pol3' )



y_pos = np.arange(len(objects)) 

performance  =[df_pol1.MSE_list_test.min() ,df_pol2.MSE_list_test.min(), df_pol3.MSE_list_test.min(),

               df_Ridge_alpha_pol2.MSE_list_test.min(),df_Ridge_alpha_pol3.MSE_list_test.min(),

               df_Lasso_alpha_pol2.MSE_list_test.min(), df_Lasso_alpha_pol3.MSE_list_test.min(),

               df_ElasticNet_alpha_pol2.MSE_list_test.min(), df_ElasticNet_alpha_pol3.MSE_list_test.min()]



performance2 =[df_pol1.MSE_train_list.min(), df_pol2.MSE_train_list.min(), df_pol3.MSE_train_list.min(),

               df_Ridge_alpha_pol2.MSE_train_list.min(),df_Ridge_alpha_pol3.MSE_train_list.min(),

               df_Lasso_alpha_pol2.MSE_train_list.min(), df_Lasso_alpha_pol3.MSE_train_list.min(),

               df_ElasticNet_alpha_pol2.MSE_train_list.min(), df_ElasticNet_alpha_pol3.MSE_train_list.min()]

               

               

performance3 = [df_pol1.R_list.max() ,df_pol2.R_list.max(), df_pol3.R_list.max(),

               df_Ridge_alpha_pol2.R_list.max(),df_Ridge_alpha_pol3.R_list.max(),

               df_Lasso_alpha_pol2.R_list.max(), df_Lasso_alpha_pol3.R_list.max(),

               df_ElasticNet_alpha_pol2.R_list.max(), df_ElasticNet_alpha_pol3.R_list.max()]



performance4 = [df_pol1.adj_R_test.max() ,df_pol2.adj_R_test.max(), df_pol3.adj_R_test.max(),

               df_Ridge_alpha_pol2.adj_R_test.max(),df_Ridge_alpha_pol3.adj_R_test.max(),

               df_Lasso_alpha_pol2.adj_R_test.max(), df_Lasso_alpha_pol3.adj_R_test.max(),

               df_ElasticNet_alpha_pol2.adj_R_test.max(), df_ElasticNet_alpha_pol3.adj_R_test.max()]



plt.subplot(411)

plt.bar(y_pos, performance, align='center')

plt.xticks(y_pos, objects,size=13)



plt.ylabel('MSE Values',size=15)

plt.title('MSE TEST Values \n', fontsize=15)





plt.subplots_adjust()

plt.subplot(412)

plt.bar(y_pos, performance2, align='center')

plt.xticks(y_pos, objects,size=13)



plt.ylabel('MSE TRAIN Values',size=15)

plt.title('MSE  Values \n', size = 15)



plt.subplot(413)

plt.bar(y_pos, performance3, align='center')

plt.xticks(y_pos, objects,size=13)

plt.title('R Squared Values \n', size = 15)



plt.ylabel('R Squared Values',size=15)



plt.subplot(414)

plt.bar(y_pos, performance4, align='center')

plt.xticks(y_pos, objects,size=13)

plt.title('Adjusted R Squared Values \n', size = 15)



plt.ylabel('Adjusted R Squared Values',size=15)





plt.subplots_adjust()

plt.show()
objects =(df_pol1, df_pol2, df_pol3,

             df_Ridge_alpha_pol2, df_Ridge_alpha_pol3,

             df_Lasso_alpha_pol2, df_Lasso_alpha_pol3,

             df_ElasticNet_alpha_pol2, df_ElasticNet_alpha_pol3)



df_results = pd.DataFrame()

for df in objects:

    df_results= df_results.append(df.sort_values(by='MSE_list_test').head(1), ignore_index=True)

    



df_results['Model'] = ['Linear Regression (Polynomial 1)',

                           'Linear Regression (Polynomial 2)',

                           'Linear Regression (Polynomial 3)',

                           'Ridge Regression (Polynomial 2)',

                           'Ridge Regression (Polynomial 3)',

                           'Lasso Regression (Polynomial 2)',

                           'Lasso Regression (Polynomial 3)',                           

                           'ElasticNet Regression(Polynomial 2)',

                           'ElasticNet Regression(Polynomial 3)']

    

df_results.sort_values('MSE_list_test')[['Model', 'number_of_variables', 'MSE_list_test','MSE_train_list', 'R_list','adj_R_test', 'adj_R_train']]
#As we see on the graph of this model, best performance is starting after 125th variable.

#Thus, I selected the first 126 variables from our model.



df_Ridge_alpha_pol2[df_Ridge_alpha_pol2['number_of_variables']== 126 ]
# A rondom row[5] of our data set to find values for each columns as an example:



Selected_Model = df_Ridge_alpha_pol2.iloc[5].model_list
#Here are the first 5 coeficiants from our model. 



Selected_Model.coef_[:5]
#Switching our values to doctionary for the further step.

LifeExpectancyData_num.iloc[5].to_dict()
#Creating a dictionary to have values for each variables.



dictionary = {'Year': 2010.0,

 'Adult_Mortality': 279.0,

 'infant_deaths': 74.0,

 'Alcohol': 0.01,

 'percentage_expenditure': 79.67936736,

 'Hepatitis_B': 66.0,

 'Measles': 1989.0,

 'BMI': 16.7,

 'under_five_deaths': 102.0,

 'Polio': 66.0,

 'Total_Expenditure': 9.2,

 'Diphtheria': 66.0,

 'HIV/AIDS': 0.1,

 'GDP': 553.32894,

 'thinness_1_19_years': 16.6,

 'thinness_5_9_years': 6.9,

 'Income_composition_of_resources': 0.45,

 'Schooling': 9.2}
Example = np.array(list(dictionary.values())).reshape(1,-1)

poly = PolynomialFeatures(2)

df = LifeExpectancyData_num.drop('Life_Expectancy', axis=1)

poly.fit_transform(df)



df_example = pd.DataFrame(poly.transform(Example), columns= poly.get_feature_names(df.columns))



df_Ridge_alpha_pol2, degerler1_2 = Ridge_model(df_poly_transform2,2,0.000001)

selected_fetures = df_Ridge_alpha_pol2.iloc[5]['feature_list']

selected_model = df_Ridge_alpha_pol2.iloc[5]['model_list']



Selected_Model.predict(df_example[selected_fetures]) 