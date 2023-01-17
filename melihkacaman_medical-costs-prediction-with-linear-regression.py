import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/insurance/insurance.csv')
data.head() 
data.count()
pd.isnull(data).any() 
data.info() 
skw = round(data['charges'].skew(), 2)
plt.figure(figsize=(10,6))
sns.distplot(data['charges'], bins=50, norm_hist=False)
plt.title(f'The Distribution of Medical Charges Skew:{skw}', fontsize=14)
plt.xlabel('Charges', fontsize=14)
plt.show() 

log_charges = np.log(data['charges'])
skw = round(log_charges.skew(), 2)
plt.figure(figsize=(10,6))
sns.distplot(log_charges, bins=50, norm_hist=False, color='indigo')
plt.title(f'The Distribution of Medical Charges Skew:{skw}', fontsize=14)
plt.xlabel('Log Charges', fontsize=14)
plt.show() 
pd.get_dummies(data['smoker'])['yes']
data['smoker'] = pd.get_dummies(data['smoker'])['yes']
data.head() 
plt.figure(figsize=(10,6))
plt.bar(x=['No','Yes'], height=[data.loc[data['smoker'] != 1].shape[0],data.loc[data['smoker'] == 1].shape[0]],color="indigo" )
plt.title("Smooker Bar Chart")
plt.show() 
charges_smoker = data.loc[data['smoker'] == 1, ['charges']].mean() 
charges_nosmoker = data.loc[data['smoker'] == 0, ['charges']].mean() 

plt.figure(figsize=(10,6))
plt.bar(x=['No','Yes'], height=[float(charges_nosmoker),float(charges_smoker)],color="purple" )
plt.title("Smooker Bar Chart")
plt.show() 
ages = data['age']
ages.describe() 
plt.figure(figsize=(10,6))
plt.hist(data['age'], bins=50, color='skyblue')
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(x=data['age'], y=data['charges'])
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Charges vs Age')
plt.show() 
categorical_ages = []
for index,value in ages.items():
    if value >= 18 and value <=35:
        categorical_ages.append('young')
    elif value >= 36 and value <=55:
        categorical_ages.append('senior')
    else: 
        categorical_ages.append('elder')

categorical_ages = pd.Series(np.asanyarray(categorical_ages), index=ages.index)
categorical_ages.value_counts()

plt.figure(figsize=(10,8))
plt.pie(x=categorical_ages.value_counts(),labels=['Young','Senior', 'Elder'],shadow=True,autopct='%1.1f%%')
plt.title("Categorical Ages", fontsize=16)
plt.show() 
age_cty_data = data.copy() 
age_cty_data['age'] = categorical_ages 


young_mean=age_cty_data.loc[age_cty_data['age'] == 'young']['charges'].mean() 
senior_mean = age_cty_data.loc[age_cty_data['age'] == 'senior']['charges'].mean()
elder_mean = age_cty_data.loc[age_cty_data['age'] == 'elder']['charges'].mean()


young_med = np.median(age_cty_data.loc[age_cty_data['age'] == 'young']['charges'])
senior_med = np.median(age_cty_data.loc[age_cty_data['age'] == 'senior']['charges'])
elder_med = np.median( age_cty_data.loc[age_cty_data['age'] == 'elder']['charges'])

df = pd.DataFrame({'mean': [young_mean, senior_mean,elder_mean], 'median': [young_med, senior_med, elder_med]}, 
                  index=['Young Adult', 'Senior Adult', 'Elder'])


ax = df.plot.bar(rot=0, figsize=(10,6))
young = age_cty_data.loc[age_cty_data['age'] == 'young'].loc[age_cty_data['smoker'] == 1]['charges']
senior = age_cty_data.loc[age_cty_data['age'] == 'senior'].loc[age_cty_data['smoker'] == 0]['charges']
elder = age_cty_data.loc[age_cty_data['age'] == 'elder'].loc[age_cty_data['smoker'] == 0]['charges']

dic = {
    'mean': [np.mean(arr) for arr in [young, senior, elder]], 
    'median': [np.median(arr) for arr in [young, senior, elder]], 
}

df = pd.DataFrame(dic, index=['Young', 'Senior Adult', 'Elder'])
ax = df.plot.bar(rot=0, figsize=(10,6))
data['age_category'] = categorical_ages 
data.tail() 
sex = pd.get_dummies(data['sex'])
data['sex'] = sex['male']
data.head() 
data['sex'].value_counts()
skw = round(data['bmi'].skew(), 2)
plt.figure(figsize=(10,6))
sns.distplot(data['bmi'], bins=50, norm_hist=False)
plt.title(f'The Distribution of BMI Skew:{skw}', fontsize=14)
plt.xlabel('BMI', fontsize=14)
plt.show()
data['bmi'].describe() 
plt.figure(figsize=(10,6))
plt.scatter(x=data['bmi'], y=data['charges'], alpha=0.7, color='indigo')
plt.xlabel('BMI', fontsize=14)
plt.ylabel('Charges', fontsize=14)
plt.title('BMI vs Charges')
plt.show()
bmis = data['bmi']

categorical_bmi = []
for index,value in bmis.items():
    if value <=18.5: 
        categorical_bmi.append('underweight')
    elif value > 18.5 and value <= 24.9: 
        categorical_bmi.append('normalweight')
    elif value > 24.9 and value < 29.9: 
        categorical_bmi.append('overweight')
    else: 
        categorical_bmi.append('obese')

categorical_bmi = pd.Series(np.asanyarray(categorical_bmi), index=ages.index)

plt.figure(figsize=(10,6))
plt.pie(x=categorical_bmi.value_counts().values, 
        labels=[str.capitalize(i) for i in categorical_bmi.value_counts().index], shadow=True,autopct='%1.1f%%')
plt.title('The Pop. of BMI')
plt.show() 
bmi_charges = pd.concat([categorical_bmi, data['charges']], axis=1)
bmi_charges.columns = ['bmi', 'charges']

obese_charges = bmi_charges.loc[bmi_charges['bmi'] == 'obese', ['charges']]
overweight_charges = bmi_charges.loc[bmi_charges['bmi'] == 'overweight', ['charges']]
normalweight_charges = bmi_charges.loc[bmi_charges['bmi'] == 'normalweight', ['charges']]
underweight_charges = bmi_charges.loc[bmi_charges['bmi'] == 'underweight', ['charges']]

array = [obese_charges, overweight_charges, normalweight_charges,underweight_charges]

dic = {
    'mean': [np.mean(arr).values[0] for arr in array], 
    'median': [np.median(arr) for arr in array], 
    
}

index=categorical_bmi.value_counts().index


df = pd.DataFrame(dic, index=index)
ax = df.plot.bar(rot=0, figsize=(10,6))
data['bmi_category'] = categorical_bmi
data.head() 
f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,8))

sns.stripplot(y='charges', x='age_category', data=data, ax=ax1)
ax1.set_title('Stripplot of charges vs age_category')

sns.stripplot(y='charges', x='age_category', data=data, hue='bmi_category', ax=ax2)
ax2.set_title('Stripplot of charges vs age_category')

sns.stripplot(y='charges', x='smoker', data=data, hue='bmi_category', ax=ax3)
ax3.set_title('Stripplot of charges vs Smoker')
plt.show()
data.head() 
%%time
dfr = data.loc[:, ['charges', 'age', 'bmi', 'children', 'smoker']] 

sns.set(style="ticks")
pal = ["#BDBDBD", "#E91E63"]

sns.pairplot(dfr, hue="smoker", palette=pal)
plt.title("Smokers")
plt.show()
obese_smoker = data.loc[data['bmi_category'] == 'obese'].loc[data['smoker'] == 1]
obese_nosmoker = data.loc[data['bmi_category'] == 'obese'].loc[data['smoker'] == 0]

obese_smoker_skew = round(obese_smoker['charges'].skew(), 3) 
just_obese_skew = round( obese_nosmoker['charges'].skew(), 3) 

f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,8))

sns.distplot(obese_smoker['charges'], bins=50, ax=ax1, color='skyblue')
ax1.set_title(f'Smoker Obese {obese_smoker_skew}',fontsize=14)

sns.distplot(obese_nosmoker['charges'], bins=50, ax=ax2, color='#E91E63')
ax2.set_title(f'Just Obese {just_obese_skew}', fontsize=14)

plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='charges', hue='smoker', data=data, palette=pal,legend="full",s=80, alpha=.7)
plt.title('Age vs Charges', fontsize=14)


plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='charges', hue='bmi_category', data=data, palette='muted',legend="full",s=80, alpha=.7)
plt.title('Age vs Charges', fontsize=14)
plt.show()
data['children'].describe() 
data['children'].value_counts()
data['region'].value_counts()
unique_region = dict(pd.Series([0,1,2,3],index=data['region'].unique()))
data['region'] = data['region'].replace(unique_region)
data.head() 
mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
mask
plt.figure(figsize=(16,10))
sns.heatmap(data.corr(), mask=mask, annot=True,annot_kws={
    'size': 14
})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
data_log_charges = data 
data_log_charges['charges'] = np.log(data_log_charges['charges'])

plt.figure(figsize=(16,10))
sns.heatmap(data_log_charges.corr(), mask=mask, annot=True,annot_kws={
    'size': 14
})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
%%time

sns.pairplot(data_log_charges, kind='reg', plot_kws={'line_kws':{'color': 'cyan'}})
plt.show()
data['bmi_category'] = data['bmi_category'].astype('category')
data['bmi_category'] = data['bmi_category'].cat.codes

data['age_category'] = data['age_category'].astype('category')
data['age_category'] = data['age_category'].cat.codes
data.head() 

# Model 1: log charges 
data_log = data.drop(['age_category', 'bmi_category'], axis=1) 

log_target = data_log['charges']
features = data_log.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)
regr = LinearRegression().fit(X=X_train, y=y_train)

log_target_caty_bmi_score = regr.score(X_train, y_train)


print('Intercept is', round(regr.intercept_,3))
print('R-squared for training set is', regr.score(X_train, y_train))
print('R-squared for testing set is', regr.score(X_test, y_test))

pd.DataFrame(regr.coef_, columns=['coef'], index=features.columns)
# Model 2 log target with categorical bmi 

data_log = data.drop(['age_category', 'bmi'], axis=1) 

log_target = data_log['charges']
features = data_log.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)
regr = LinearRegression().fit(X=X_train, y=y_train)

log_target_score = regr.score(X_train, y_train)


print('Intercept is', round(regr.intercept_,3))
print('R-squared for training set is', regr.score(X_train, y_train))
print('R-squared for testing set is', regr.score(X_test, y_test))

pd.DataFrame(regr.coef_, columns=['coef'], index=features.columns)
# Model 1: log charges 
data_log = data.drop(['age_category', 'bmi_category'], axis=1) 

log_target = data_log['charges']
features = data_log.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)

X_incl_const_log_target =sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const_log_target) 
results_log_target = model.fit() 
round(results_log_target.pvalues, 3)
# Model 2 log target with categorical bmi 
data_log = data.drop(['age_category', 'bmi'], axis=1) 

log_target = data_log['charges']
features = data_log.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)

X_incl_const_categorical_bmi =sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const_categorical_bmi) 
results_categorical_bmi = model.fit() 
round(results_categorical_bmi.pvalues, 3)
variance_inflation_factor(exog=np.asanyarray(X_incl_const_log_target), exog_idx=1)
vifs_log_target = [variance_inflation_factor(exog=np.asanyarray(X_incl_const_log_target), exog_idx=i) 
        for i in range(len(X_incl_const_log_target.columns))]

vifs_categorical = [variance_inflation_factor(exog=np.asanyarray(X_incl_const_categorical_bmi), exog_idx=i) 
        for i in range(len(X_incl_const_categorical_bmi.columns))]

pd.DataFrame(np.asanyarray([np.asanyarray(vifs_log_target).
                            reshape(1,7), np.asanyarray(vifs_categorical).reshape(1,7)]).reshape(2,7),
            columns=X_incl_const_log_target.columns, index=['Just Log Charges', 'Log Charges and Categorical BMI']
            )
dict_results = dict({
    'Log Charges': {
        'R-Squared': round( results_log_target.rsquared,3),
        'BIC': round(results_log_target.bic, 3)
    },
    'Log Charges with Categorical BMI': {
        'R-Squared': round( results_categorical_bmi.rsquared,3),
        'BIC': round(results_categorical_bmi.bic, 3)
    }
})
# Model 3 log charges without sex
data_log = data.drop(['age_category', 'bmi_category', 'sex'], axis=1) 

log_target = data_log['charges']
features = data_log.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)

X_incl_const_witout_sex =sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const_witout_sex) 
results_witoutsex = model.fit() 
round(results_witoutsex.pvalues, 3)

dict_results['Log Charges Without Sex'] = {
        'R-Squared': round( results_witoutsex.rsquared,3),
        'BIC': round(results_witoutsex.bic, 3)
    }
pd.DataFrame(dict_results)
# Predicted charges vs Actual Charges for Final Model 

data_log = data.drop(['age_category', 'bmi_category', 'sex'], axis=1) 

log_target = data_log['charges']
features = data_log.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, log_target, test_size=0.2)

regr = LinearRegression().fit(X_train, y_train) 

predicted_values = pd.Series(regr.predict(X_train))
corr = np.round(y_train.corr(predicted_values), 3)

plt.figure(figsize=(10,6))

plt.scatter(x=predicted_values, y=y_train)
plt.plot(y_train, y_train, c='red')
plt.title(f"Predicted log prices vs Actual Log prices {corr}", fontsize=14)
plt.xlabel('Predicted Price',fontsize=14)
plt.ylabel('Actual Price', fontsize=14) 


# residual vs predicted values 
plt.figure(figsize=(10,6))
y = np.asanyarray(y_train)
y_hat = np.asanyarray(predicted_values)
resi = y - y_hat
resi_mean = round(np.average(resi), 5)
plt.scatter(x=predicted_values, y=resi, c="skyblue",alpha=0.7)
plt.ylabel('Residual', fontsize=14)
plt.xlabel('Predicted Values', fontsize=14)
plt.title("Residual vs Predicted Values", fontsize=14)


plt.figure(figsize=(10,6))
sns.distplot(resi)
plt.title(f'The Distribution of the Residuals Skew:{round(pd.Series(resi).skew(), 2)} Residual Mean {resi_mean}', fontsize=14)

plt.show() 