import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
train = pd.read_csv("../input/train2csv/train2.csv", header=0)
df_train = train.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
df_train.head(10)
#Combine WoodDeck SF, OpenPorchSF, and EnclosedPorch, 3SsnPorch, and ScreenPorch into one variable: PorchSF
df_train.loc[:,'PorchSF'] = df_train.loc[:,'OpenPorchSF'].add(df_train.loc[:,'EnclosedPorch']).add(df_train.loc[:,'WoodDeckSF']).add(df_train.loc[:,'3SsnPorch']).add(df_train.loc[:,'ScreenPorch'])

#make new dataframe with only numerical variables and dropping the variables that we added to make PorchSF

df_train2 = df_train.drop(['HouseStyle', 'ExterQual', 'WoodDeckSF', 'OpenPorchSF',
                           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'Neighborhood'], axis=1)

df_train2 = df_train2.apply (pd.to_numeric, errors='coerce')
df_train2 = df_train2.dropna()
df_train2 = df_train2.reset_index(drop=True)
#finding the descriptive statistics of our variables
df_train2.describe()
#Searching for any abnormalities with SalePrice distribution using histograms and experimenting with bins.

plt.rcParams['figure.figsize'] = (15,4)
plt.subplot(1,2,1)
sns.distplot(df_train2['SalePrice'])

plt.subplot(1,2,2)
sns.distplot(df_train2['SalePrice'], bins=100)

print("When looking at the histogram we see the data for SalePrice skews to the right. This shows us that the mean is greater than the median SalePrice because of the few homes' prices that push the average higher.")
print()
print("IMPORTANT: Analyzing the second histogram where we used our own bins instead of using the default calculated by Seaborn, we notice highs and lows in the distribution. This may be explained by psychological price points the houses were listed under in order to entice home buyers.")
plt.subplot(1,1,1)
plt.rcParams['figure.figsize'] = (15,4)
plt.xticks(rotation='45')
sns.boxplot(data=df_train2)
plt.title("Boxplot for all attributes")
plt.show()

print('We look at the boxplots for all attributes and visually determine which variables we assume are most likely correlated to SalePrice')

BoxPlotFilt = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', 'TotalBsmtSF', 'FullBath', 'YrSold']
plt.subplot(2,1,2)
plt.rcParams['figure.figsize'] = (15,4)
plt.xticks(rotation='45')
sns.boxplot(data=df_train2[BoxPlotFilt])
plt.title("Boxplot for variables most likely related to SalePrice.")
plt.show()

print("There seems to be a lot of outliers in SalePrice. Let's remove them and see if there are enough remaining data points to continue our analysis without the outliers.")
Q1 = df_train2.quantile(0.25)
Q3 = df_train2.quantile(0.75)

IQR = Q3-Q1

df_No_Outliers = df_train2[~((df_train2 < (Q1 - 1.5 * IQR)) |(df_train2 > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_No_Outliers.shape[0])

merged_dataframe_proportion = df_No_Outliers.shape[0]/df_train2.shape[0]

print("When we have not removed any outliers from the dataset, we have " + str(df_train2.shape[0]) + " entries.") 

print("When we remove outliers from the dataset, we have " + str(df_No_Outliers.shape[0]) + " entries.")
print("The proportion of entries that are not outliers compared to the dataframe is " + str(round(merged_dataframe_proportion*100,2))+"%")
#creating scatter plots to see if we can see correlation via this type of visual

x_col = df_train2['SalePrice'].reset_index(drop=True)

y_columns = df_train2.drop(columns=['SalePrice']).reset_index(drop=True)

for i in y_columns:

    figure = plt.figure
    ax = plt.gca()
    ax.scatter(df_train2['SalePrice'], df_train2[i])
    ax.set_xlabel('SalePrice')
    ax.set_ylabel(i)
    plt.show()

print("There are some variables that look to be correlated and some that don't. Using a correlation matrix will give us the best visualization to determine the relationships between SalePrice and the other variables.")

#building correlation matrix heatmap

df_train2 = df_train2.drop(['YrSold'], axis=1)

df_train2 = df_train2.apply (pd.to_numeric, errors='coerce')
df_train2 = df_train2.dropna()
df_train2 = df_train2.reset_index(drop=True)

df_train2_corr = df_train2.corr()

plt.figure(figsize=(20,16))
ax = sns.heatmap(df_train2_corr, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#Order the correlation of each variable to SalePrice in descending order and create a bar chart to display the degree of correlation

row = df_train2_corr.loc[:,['SalePrice']]
row = row.sort_values(by='SalePrice', ascending=False).drop('SalePrice')

plt.figure(figsize=(20,8))
sns.barplot(data=row, x = row.index, y = 'SalePrice')
plt.xticks(rotation='45')

print(row)
X = df_train2.drop(['SalePrice'], axis=1) #We now have only independent variables here without any dependent variables. 
X = sm.add_constant(X) #We've added our constant intercept to our independent variable dataframe

#We've specified SalePrice to the Y Dataframe
#We fit the model below using the statsmodel (denoted by SM) OLS function.
OLSmodel = sm.OLS(df_train2['SalePrice'], X)
OLSmodelResult = OLSmodel.fit()
OLSmodelResult.summary()
#create a filter that only has variables with a correlation greater than .50 to SalePrice

filt1 = ['OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'BsmtUnfSF', 'GrLivArea',
         'BsmtFullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'SalePrice'] 

X = df_train2[filt1].drop(['SalePrice'], axis=1) #We now have only independent variables here without any dependent variables. 
X = sm.add_constant(X) #We've added our constant intercept to our independent variable dataframe

#We've specified SalePrice to the Y Dataframe
#We fit the model below using the statsmodel (denoted by SM) OLS function.
OLSmodel = sm.OLS(df_train2['SalePrice'], X)
OLSmodelResult = OLSmodel.fit()
OLSmodelResult.summary()
filt2 = ['OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'GrLivArea',
         'BsmtFullBath', 'TotRmsAbvGrd', 'GarageCars', 'SalePrice'] 

X = df_train2[filt2].drop(['SalePrice'], axis=1) #We now have only independent variables here without any dependent variables. 
X = sm.add_constant(X) #We've added our constant intercept to our independent variable dataframe

#We've specified SalePrice to the Y Dataframe
#We fit the model below using the statsmodel (denoted by SM) OLS function.
OLSmodel = sm.OLS(df_train2['SalePrice'], X)
OLSmodelResult = OLSmodel.fit()
OLSmodelResult.summary()
filt2 = ['OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'GrLivArea',
         'BsmtFullBath', 'GarageCars', 'SalePrice'] 

X = df_train2[filt2].drop(['SalePrice'], axis=1) #We now have only independent variables here without any dependent variables. 
X = sm.add_constant(X) #We've added our constant intercept to our independent variable dataframe

#We've specified SalePrice to the Y Dataframe
#We fit the model below using the statsmodel (denoted by SM) OLS function.
OLSmodel = sm.OLS(df_train2['SalePrice'], X)
OLSmodelResult = OLSmodel.fit()
OLSmodelResult.summary()

plt.rcParams['figure.figsize'] = (15,4)


RegressionResults=OLSmodelResult.predict(X)
print(RegressionResults)

fig, ax1 = plt.subplots()

ax1.get_xaxis().set_visible(False)
ax1.set_ylabel('Regression Result', color='navy')
ax1.plot(RegressionResults, color='navy') 
ax1.set_ylabel('SalePrice', color='red')
ax1.plot(df_train2['SalePrice'], color='red')
#reload data so we can have a clean start again with the same raw data
#adding new variables from our EDA into the table 

train = pd.read_csv("../input/train2csv/train2.csv", header=0)

df_train2.loc[:,'RegressionResults'] = RegressionResults
df_train2 = pd.merge(df_train2,train,on='Id')

df_train2["RegressionResults"].astype(int)
df_train2["SalePrice_x"].astype(int)

Variance = df_train2["RegressionResults"] - df_train2["SalePrice_x"]
df_train2.loc[:,'Variance'] = Variance
df_train2["Variance"].astype(int)

Variance_PCT = Variance/df_train2["SalePrice_x"]
df_train2.loc[:,'Variance_PCT'] = Variance_PCT
df_train2['Variance_PCT'].astype(int)

df_train2.head()
filt = ['Id','Neighborhood','RegressionResults', 'SalePrice_x', 'Variance', 'Variance_PCT']
df_train3 = df_train2[filt]

df = df_train3[df_train3['Variance_PCT'] > .10] 

df_NBHD = df.groupby(['Neighborhood']).sum().reset_index()
df_NBHD
df_NBHD.plot(x='Neighborhood', y=['RegressionResults', 'SalePrice_x', 'Variance'], kind='bar')