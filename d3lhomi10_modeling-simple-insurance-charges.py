import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import seaborn as sns
import xgboost as xgb

import matplotlib.pyplot as plt


import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings as ws
ws.defaultaction = "ignore"
data = pd.read_csv('../input/medical-cost-personal-dataset/insurance.csv')
data
print('Data shape: ',data.shape, '\n')
print('*******************************')
print('Data means:\n',data.mean(), '\n')
print('*******************************')
print('Data features count:\n',data.count(), '\n')
print('*******************************')
print('Data Info about null vals:\n',data.info(), '\n')
print('*******************************')
print('Data Features null vals:\n',data.isnull().sum(), '\n')
# Insurance charges histogram (How good is its skew value?)
plt.figure(figsize=(10, 8))
plt.hist(data['charges'], bins = 50 ,color='#3f4c6b', ec='#606c88')
plt.title('Insurance charges in $ vs Nr. of people', fontsize=18)
plt.ylabel('Nr. of people', fontsize=14)
plt.xlabel('Prices in $', fontsize=14)
plt.show()
# Changing "sex" feature to 0s and 1s => 0s: female; 1s: male
data['sex'] = data.sex.replace({"female" :0, "male" : 1 })

# Changing"smoker" features to 0s and 1s => 0s: no; 1s: yes
data['smoker'] = data.smoker.replace({"yes": 1, "no" : 0 })

data['region'] = data.region.replace({"southeast": 0, "southwest" : 1,
                                     "northeast":2, "northwest":3})

# Extracting relevant data and ignoring repetitive correlations
mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
data.corr()
# Correlations value graph
plt.figure(figsize=(10, 8))

sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size":14})

#Analysis: We can clearly notice that there is a strong correlation between the age and the charges
sns.set_context('talk')
sns.set_style('darkgrid')
g = sns.FacetGrid(data, row="smoker", col="sex", margin_titles=True, height=5, )
g.map(sns.regplot, "bmi", "charges", color="#12c2e9", x_jitter=.1, line_kws={"color": "#f64f59"})
region_charges = sns.catplot(x="region", y='charges', data=data, legend_out = False,
            height=8, hue="sex", kind='bar', palette=["#f64f59", "#12c2e9"]);

# region_charges.set_title('Region vs. Charges by gender')
leg = region_charges.axes.flat[0].get_legend()
region_charges.set(xlabel='Regions', ylabel='Charges', 
                   title='Regions vs. Insurance Charges')

region_charges.set_xticklabels(['Southeast','Southwest','Northeast','Northwest'])


leg.set_title('Gender')
new_labels = ['Felmale', 'Male']
for t, l in zip(leg.texts, new_labels): t.set_text(l)
plt.show()



child_charges = sns.catplot(x="children", y='charges', data=data, height=8, legend_out = False,
           kind='bar', palette=["#aa4b6b", "#3b8d99"]);

child_charges.set(xlabel='# of Children', ylabel='Charges', 
                   title='Nr. of Children vs. Insurance Charges')


charges = data['charges']
features = data.drop(['charges'], axis=1) #Dropping charges collumn

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    charges, 
                                                    test_size= 0.2, 
                                                    random_state=42)

regression = LinearRegression()
model = regression.fit(X_train, y_train)
prediction = regression.predict(X_test)

print('Test Data r-Squared score: ', regression.score(X_test, y_test))
print('Train Data r-Squared score: ', regression.score(X_train, y_train))

pd.DataFrame(data=regression.coef_, index=X_train.columns, columns=['coef'])

# Pre-transformation skew val
pre_trans = round(data['charges'].skew(), 3)
print('Pre-transformation skew val: ', pre_trans)
sns.distplot(data['charges'])
plt.title(f'Original Charges with skew {pre_trans}')
plt.show()
# Post-transformation skew val
post_trans = round(np.log(data['charges'].skew()), 3)
print('Post-transformation skew val: ', post_trans)

y_log = np.log(data['charges'])
sns.distplot(y_log)
plt.title(f'Log Charges with skew {post_trans}')
# Apply the transformation.
log_charges = np.log(data['charges'])

transformed_data = data.drop('charges', axis=1)


X_train, X_test, y_train, y_test = train_test_split(transformed_data, 
                                                    log_charges, 
                                                    test_size= 0.2, 
                                                    random_state=42)

regression_t = LinearRegression()
model_t = regression_t.fit(X_train, y_train)
prediction_t = regression_t.predict(X_test)

pd.DataFrame(data=regression_t.coef_, index=X_train.columns, columns=['coef'])

plt.scatter(y_test, prediction_t)
plt.plot(y_test, y_test, color='red')
rmse = np.sqrt(mean_squared_error(y_test, prediction_t))


print('Intercept: ', regression_t.intercept_)
print('Coef: ', regression_t.coef_)
print('rmse: ', rmse)
print('Test Data r-Squared score: ', regression_t.score(X_test, y_test))
print('Train Data r-Squared score: ', regression_t.score(X_train, y_train))



x_include_const = sm.add_constant(X_train) #Adding an intercept

model = sm.OLS(y_train, x_include_const) 
results = model.fit()


# Graph of Actual vs. Predicted Prices
plt.figure(figsize=(10, 8))
corr = round(y_train.corr(results.fittedvalues), 2)
plt.scatter(x=y_train, y=results.fittedvalues, c='black', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')

plt.xlabel('Actual log prices $y _i$', fontsize=14)
plt.ylabel('Predicted log prices $\hat y _i$', fontsize=14)
plt.title(f'Actual vs Predicted log prices $y _i$ vs $\hat y _i$ (Corr: {corr})', 
          fontsize=18)

plt.show()


pd.DataFrame({'Coef' : results.params, 
             'P-values' : round(results.pvalues, 3)})
#Hence, all the features are statistically significant
