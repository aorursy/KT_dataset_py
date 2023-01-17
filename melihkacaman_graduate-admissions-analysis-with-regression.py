import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt  

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
data = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

data.head() 
data.tail() 
data = data.drop('Serial No.', axis=1)

data.head() 
data.count() 
pd.isnull(data).any() 
data.info() 
data.columns = ['GRE_SCORE', 'TOEFL_SCORE', 'UNIVERSITY_RATING', 'SOP', 'LOR', 'CGPA',

       'RESEARCH', 'CHANCE_OF_ADMIT']

data.columns
plt.figure(figsize=(10,6))

sns.distplot(data['CHANCE_OF_ADMIT'],bins=50, color='#ff7c43')

plt.xlabel('Chance of Admit', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"Chance of Admit Distribution Skew:{round(data['CHANCE_OF_ADMIT'].skew())}", fontsize=14)

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(data['GRE_SCORE'],bins=50, color='#f95d6a')

plt.xlabel('GRE SCORE', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"GRE SCORE Distribution Skew:{round(data['GRE_SCORE'].skew(),2)}", fontsize=14)

plt.show()
data['GRE_SCORE'].max() 
data['GRE_SCORE'].min()
data['GRE_SCORE'].mean() 
plt.figure(figsize=(10,6))

sns.distplot(data['TOEFL_SCORE'],bins=50, color='#2f4b7c')

plt.xlabel('TOEFL SCORE', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"TOEFL SCORE Distribution Skew:{round(data['TOEFL_SCORE'].skew(),2)}", fontsize=14)

plt.show()
plt.figure(figsize=(10,6))

plt.hist(data['UNIVERSITY_RATING'],bins=50, color='#a05195')

plt.xlabel('UNIVERSITY RATING', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"UNIVERSITY RATING Distribution", fontsize=14)

plt.show()
data.UNIVERSITY_RATING.value_counts() 
x = data.UNIVERSITY_RATING.value_counts().values 

labels = data.UNIVERSITY_RATING.value_counts().index

custom_colours = ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7', '#d45087']

offset = [0.05, 0.05, 0.05, 0.05]



plt.figure(figsize=(2, 2), dpi=227)

plt.pie(x, labels=labels, textprops={'fontsize': 6}, startangle=90, 

       autopct='%1.0f%%', colors=custom_colours, pctdistance=0.8)

# draw circle

centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')

plt.gca().add_artist(centre_circle)



plt.show()
plt.figure(figsize=(10,6))

plt.hist(data['SOP'],bins=50, color='#d45087')

plt.xlabel('SOP', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"SOP Distribution", fontsize=14)

plt.show()
data.SOP.value_counts() 
plt.figure(figsize=(10,6))

plt.hist(data['LOR'],bins=50, color='#a05195')

plt.xlabel('LOR', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"LOR Distribution", fontsize=14)

plt.show()
data.LOR.value_counts() 
plt.figure(figsize=(10,6))

sns.distplot(data.CGPA,bins=50, color='#a05195')

plt.xlabel('CGPA', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"CGPA Distribution Skew:{round(data['CGPA'].skew(), 2)}", 

          fontsize=14)

plt.show()
data.CGPA.max()
data.CGPA.min() 
plt.figure(figsize=(10,6))



y = data.RESEARCH.value_counts().values

x = [0,1]



sns.barplot(x=x, y=y, color='#a05195')

plt.xlabel('Research', fontsize=14)

plt.ylabel('Number', fontsize=14)

plt.title(f"Research Distribution", 

          fontsize=14)

plt.show()
mask = np.zeros_like(data.corr())

triangle_indices = np.triu_indices_from(mask)

mask[triangle_indices] = True

mask
plt.figure(figsize=(16,16))

sns.heatmap(data.corr(), mask=mask, annot=True,annot_kws={

    'size': 14

})

plt.xticks(rotation=90, fontsize=14)

plt.yticks(rotation=1, fontsize=14)

plt.show()
%%time



sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color': 'cyan'}})

plt.show()
target = data.CHANCE_OF_ADMIT

features = data.drop('CHANCE_OF_ADMIT', axis=1)



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
regr = LinearRegression().fit(X=X_train, y=y_train)



print('Intercept is', round(regr.intercept_,3))

print('R-squared for training set is', regr.score(X_train, y_train))

print('R-squared for testing set is', regr.score(X_test, y_test))



pd.DataFrame(regr.coef_, columns=['coef'], index=features.columns)
X_incl_const =sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const) 

results = model.fit() 



vifs = [variance_inflation_factor(exog=np.asanyarray(X_incl_const), exog_idx=i) 

        for i in range(len(X_incl_const.columns))]

vifs = pd.DataFrame(np.asanyarray(vifs).reshape(8,1), index=X_incl_const.columns)

vifs
pvals = pd.DataFrame(round(results.pvalues, 3))

pvals
pd.concat([vifs, pvals], axis=1)
# Model 1 without SOP, UNIVERSITY_RATING, CGPA 



features = X_train.drop(['SOP', 'CGPA', 'UNIVERSITY_RATING'], axis=1)



X_incl_const =sm.add_constant(features)

model = sm.OLS(y_train, X_incl_const) 

results = model.fit() 



print("MODEL 1 R-SQUARED:", results.rsquared)

print("MODEL 1 BIC:", results.bic)



vifs = [variance_inflation_factor(exog=np.asanyarray(X_incl_const), exog_idx=i) 

        for i in range(len(X_incl_const.columns))]

vifs = pd.DataFrame(np.asanyarray(vifs).reshape(5,1), index=X_incl_const.columns)

vifs
# Model 2 without UNIVERSITY_RATING, SOP



features = X_train.drop(['UNIVERSITY_RATING', 'SOP'], axis=1)



X_incl_const =sm.add_constant(features)

model = sm.OLS(y_train, X_incl_const) 

results2 = model.fit() 



print("MODEL 2 R-SQUARED:", results2.rsquared)

print("MODEL 2 BIC:", results2.bic)



vifs = [variance_inflation_factor(exog=np.asanyarray(X_incl_const), exog_idx=i) 

        for i in range(len(X_incl_const.columns))]

vifs = pd.DataFrame(np.asanyarray(vifs).reshape(6,1), index=X_incl_const.columns)

vifs
actual_target = y_train

predicted_target = results2.predict(X_incl_const)

residuals = results2.resid 



plt.figure(figsize=(10,6))

plt.scatter(x=actual_target, y=predicted_target, alpha=0.5, color='purple')

plt.plot(actual_target, actual_target, color='cyan')

plt.title(f'Actual vs Predicted Corr: {actual_target.corr(predicted_target)}',

         fontsize=14)

plt.xlabel('Actual',fontsize=14)

plt.ylabel('Predicted',fontsize=14)

plt.show() 



plt.figure(figsize=(10,6))

plt.scatter(x=predicted_target, y=residuals)

plt.title('Predicted vs Residuals', fontsize=14)

plt.xlabel('Predicted',fontsize=14)

plt.ylabel('Residuals',fontsize=14)

plt.show()



plt.figure(figsize=(10,6))

sns.distplot(residuals, bins=50)

plt.title(f'Dist. of Residuals {round(residuals.skew(),2)}', fontsize=14)

plt.xlabel('Residuals',fontsize=14)

plt.ylabel('Nr',fontsize=14)

plt.show()
