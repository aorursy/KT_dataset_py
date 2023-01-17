%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
population_df = pd.read_csv('../input/PopulationPerCountry.csv', skiprows = range(0,4))
population_df.head()
GDP_df = pd.read_csv('../input/GDP by Country.csv', skiprows = range(0,4))
GDP_df.head()
GDPData_df = pd.merge(GDP_df, population_df, on= ['Country Code','Country Name'], how='inner')
GDPData_df.head()
GDPDataCurated_df = GDPData_df.drop(['Indicator Name_x','Indicator Code_x','Indicator Name_y','Indicator Code_y','Unnamed: 62_y','Unnamed: 62_x'], axis = 1)
GDPDataCurated_df.head()
# GDP_df is the X column, Population_df is the Y column.
GDPperCapita_df = pd.DataFrame()
for col in GDPDataCurated_df.columns:
    if col.endswith("Name"):
        country = col[:]
        GDPperCapita_df[country] = GDPDataCurated_df[country]
    if col.endswith("_x"):
        year = col[:4]
        GDPperCapita_df[year] = GDPDataCurated_df[year + '_x']/GDPDataCurated_df[year + '_y']
    if col.endswith("Code"):
        code = col[:]
        GDPperCapita_df['Units:- US$/person' + code] = GDPDataCurated_df[code]
        
GDPDataCurated_df.head()
GDPperCapita_df.head()      
GDP_Stacked_df = pd.melt(GDPperCapita_df,id_vars=['Country Name','Units:- US$/personCountry Code'])
GDP_Stacked_df.head()
WomenMakingInformedChoices_df = pd.read_csv('../input/WomenMakingInformedChoicestoReproductiveHealthCare.csv', skiprows = range(0,4))
WomenMakingInformedChoices_df = pd.melt(WomenMakingInformedChoices_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
WomenMakingInformedChoices_df
RuralPopulationPerCent_df = pd.read_csv('../input/RuralPopulationofTotalPopulation.csv', skiprows = range(0,4))
RuralPopulationPerCent_df = RuralPopulationPerCent_df.drop(['Unnamed: 62'],axis = 1)
RuralPopulationPerCent_df = pd.melt(RuralPopulationPerCent_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
RuralPopulationPerCent_df.head()
PublicEduRatioGDP_df = pd.read_csv('../input/public-education-expenditure-as-share-of-gdp.csv')
#PublicEduRatioGDP_df = pd.melt(PublicEduRatioGDP_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
#PublicEduRatioGDP_df

#IGNORE THIS DATA SET IN THE CURRENT SET OF RESULTS.
LegalRightsStrength_df = pd.read_csv('../input/LegalRightsStrengthIndex.csv', skiprows = range(0,4))
LegalRightsStrength_df = pd.melt(LegalRightsStrength_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
LegalRightsStrength_df.head()
CreditToPrivateSector_df = pd.read_csv('../input/DomesticCreditToPrivateSector.csv', skiprows = range(0,4))
CreditToPrivateSector_df = CreditToPrivateSector_df.drop(['Unnamed: 62'],axis = 1)
CreditToPrivateSector_df = pd.melt(CreditToPrivateSector_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
CreditToPrivateSector_df.head()
BirthsAttendedbySkilledStaff_df = pd.read_csv('../input/BirthsAttendedbySkilledHealthStaffofTotal.csv', skiprows = range(0,4))
BirthsAttendedbySkilledStaff_df = pd.melt(BirthsAttendedbySkilledStaff_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
BirthsAttendedbySkilledStaff_df.head()
ATMMachinesRatio_df = pd.read_csv('../input/ATMMachines_Per100000Adults.csv', skiprows = range(0,4))
ATMMachinesRatio_df = pd.melt(ATMMachinesRatio_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
ATMMachinesRatio_df.head()
AgriculturalMachines_df = pd.read_csv('../input/AgriculturalMachinery_PerUnitofArableLand.csv', skiprows = range(0,4))
AgriculturalMachines_df = AgriculturalMachines_df.drop(['Unnamed: 62'],axis = 1)
AgriculturalMachines_df = pd.melt(AgriculturalMachines_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
AgriculturalMachines_df.head()
LiteracyRateAdult_df = pd.read_csv('../input/AdultPopulation_Literate.csv', skiprows = range(0,4))
#AgriculturalMachines_df = AgriculturalMachines_df.drop(['Unnamed: 62'],axis = 1)
LiteracyRateAdult_df = pd.melt(LiteracyRateAdult_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
LiteracyRateAdult_df.head()
AccountsRatioFinancialInst_df = pd.read_csv('../input/AccountAtaFinancialInstitutionMale15Adults.csv', skiprows = range(0,4))
AccountsRatioFinancialInst_df = AccountsRatioFinancialInst_df.drop(['Unnamed: 62'],axis = 1)
AccountsRatioFinancialInst_df = pd.melt(AccountsRatioFinancialInst_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
AccountsRatioFinancialInst_df.head()

GDP_Stacked_df['WomenMakingInformedChoices_df'] = WomenMakingInformedChoices_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['RuralPopulationPerCent_df'] = RuralPopulationPerCent_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['LegalRightsStrength_df'] = LegalRightsStrength_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['CreditToPrivateSector_df'] = CreditToPrivateSector_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'] = BirthsAttendedbySkilledStaff_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['ATMMachinesRatio_df'] = ATMMachinesRatio_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['AgriculturalMachines_df'] = AgriculturalMachines_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['LiteracyRateAdult_df'] = LiteracyRateAdult_df.value
GDP_Stacked_df.head()
GDP_Stacked_df['AccountsRatioFinancialInst_df'] = AccountsRatioFinancialInst_df.value
GDP_Stacked_df
sns.regplot(x="value", y="WomenMakingInformedChoices_df", data=GDP_Stacked_df)
sns.regplot(x="value", y="WomenMakingInformedChoices_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="RuralPopulationPerCent_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="LegalRightsStrength_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="CreditToPrivateSector_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="BirthsAttendedbySkilledStaff_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="ATMMachinesRatio_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="AgriculturalMachines_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="LiteracyRateAdult_df", data=GDP_Stacked_df);
sns.regplot(x="value", y="AccountsRatioFinancialInst_df", data=GDP_Stacked_df);
print(GDP_Stacked_df.isnull().any())
# where value = GDP_per_Capita
# Counting missing values in a column

GDP_Stacked_df.dropna(subset=['value'],inplace = True)
print(GDP_Stacked_df['value'].isnull().sum())
print(GDP_Stacked_df['value'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['WomenMakingInformedChoices_df'].fillna(value=GDP_Stacked_df['WomenMakingInformedChoices_df'].mean(),inplace=True)
print(GDP_Stacked_df['WomenMakingInformedChoices_df'].isnull().sum())
print(GDP_Stacked_df['WomenMakingInformedChoices_df'].notnull().sum())

# Counting missing values in a column
GDP_Stacked_df['RuralPopulationPerCent_df'].fillna(value=GDP_Stacked_df['RuralPopulationPerCent_df'].mean(),inplace=True)
print(GDP_Stacked_df['RuralPopulationPerCent_df'].isnull().sum())
print(GDP_Stacked_df['RuralPopulationPerCent_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['LegalRightsStrength_df'].fillna(value=GDP_Stacked_df['LegalRightsStrength_df'].mean(),inplace=True)
print(GDP_Stacked_df['LegalRightsStrength_df'].isnull().sum())
print(GDP_Stacked_df['LegalRightsStrength_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['CreditToPrivateSector_df'].fillna(value=GDP_Stacked_df['CreditToPrivateSector_df'].mean(),inplace=True)
print(GDP_Stacked_df['CreditToPrivateSector_df'].isnull().sum())
print(GDP_Stacked_df['CreditToPrivateSector_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].fillna(value=GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].mean(),inplace=True)
print(GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].isnull().sum())
print(GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['ATMMachinesRatio_df'].fillna(value=GDP_Stacked_df['ATMMachinesRatio_df'].mean(),inplace=True)
print(GDP_Stacked_df['ATMMachinesRatio_df'].isnull().sum())
print(GDP_Stacked_df['ATMMachinesRatio_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['AgriculturalMachines_df'].fillna(value=GDP_Stacked_df['AgriculturalMachines_df'].mean(),inplace=True)
print(GDP_Stacked_df['AgriculturalMachines_df'].isnull().sum())
print(GDP_Stacked_df['AgriculturalMachines_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['LiteracyRateAdult_df'].fillna(value=GDP_Stacked_df['LiteracyRateAdult_df'].mean(),inplace=True)
print(GDP_Stacked_df['LiteracyRateAdult_df'].isnull().sum())
print(GDP_Stacked_df['LiteracyRateAdult_df'].notnull().sum())
# Counting missing values in a column
GDP_Stacked_df['AccountsRatioFinancialInst_df'].fillna(value=GDP_Stacked_df['AccountsRatioFinancialInst_df'].mean(),inplace=True)
print(GDP_Stacked_df['AccountsRatioFinancialInst_df'].isnull().sum())
print(GDP_Stacked_df['AccountsRatioFinancialInst_df'].notnull().sum())
X = GDP_Stacked_df[['CreditToPrivateSector_df','WomenMakingInformedChoices_df','RuralPopulationPerCent_df','BirthsAttendedbySkilledStaff_df','ATMMachinesRatio_df','AgriculturalMachines_df','LiteracyRateAdult_df','AccountsRatioFinancialInst_df','LegalRightsStrength_df']]
X = np.array(X)
X
y = GDP_Stacked_df['value']
y = np.array(y)
y
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, Y_train)
y_pred = model.predict(X_val)
y_actual = Y_val
mean_squared_error(y_actual, y_pred)
# Test R^2
print(model.score(X_val, y_actual))
plt.scatter(y_pred, y_actual, marker='.')
plt.xlabel('Predicted y')
plt.ylabel('Actual y')
plt.show()
y_pred_test = model.predict(X_test)
y_actual_test = Y_test
mean_squared_error(y_actual_test, y_pred_test)
# Test R^2
print(model.score(X_test, y_actual_test))
plt.scatter(y_pred_test, y_actual_test, marker='.')
plt.xlabel('Predicted y')
plt.ylabel('Actual y')
plt.show()
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
X = GDP_Stacked_df[['CreditToPrivateSector_df','WomenMakingInformedChoices_df','RuralPopulationPerCent_df','BirthsAttendedbySkilledStaff_df','ATMMachinesRatio_df','AgriculturalMachines_df','LiteracyRateAdult_df','AccountsRatioFinancialInst_df','LegalRightsStrength_df']]
# splitting into training and val sets for cross validation
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)
#training the model
lreg.fit(X_train,Y_train)
pred = lreg.predict(X_val)
# calculating MSE
mse = np.mean((pred - Y_val)**2)
from pandas import Series, DataFrame
lreg.score(X_val,Y_val)
x_plot = plt.scatter(pred, (pred - Y_val), c='b')
plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')
predictors = X_train.columns
coef = Series(lreg.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
#Ridge Regression 

from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.0001, normalize=True)
ridgeReg.fit(X_train,Y_train)
pred = ridgeReg.predict(X_val)
mse = np.mean((pred - Y_val)**2)
ridgeReg.score(X_train,Y_train)
ridgeReg.score(X_test,Y_test)
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)
# Fit the regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor()
regr_1.fit(X_train, Y_train)
regr_2.fit(X_train, Y_train)
regr_1.score(X_val,Y_val)
regr_2.score(X_val,Y_val)
regr_2.score(X_test,Y_test)
regr_2.feature_importances_
list(zip(regr_2.feature_importances_,X.columns))
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)
regr_1 = RandomForestRegressor(n_estimators=500, min_samples_leaf=1)
regr_2 = RandomForestRegressor()
regr_1.fit(X_train, Y_train)
regr_2.fit(X_train, Y_train)
regr_1.score(X_val,Y_val)
regr_2.score(X_val,Y_val)
regr_2.feature_importances_
list(zip(regr_2.feature_importances_,X.columns))
regr_1.score(X_test,Y_test)
regr_2.feature_importances_
list(zip(regr_1.feature_importances_,X.columns))
