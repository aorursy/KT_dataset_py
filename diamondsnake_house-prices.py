import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
#pd.set_option('display.max_columns', None)
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



import scipy.stats as ss

def theils_u(x, y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x



def correlation_ratio(categories, measurements):

    fcat, _ = pd.factorize(categories)

    cat_num = np.max(fcat)+1

    y_avg_array = np.zeros(cat_num)

    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):

        cat_measures = measurements[np.argwhere(fcat == i).flatten()]

        n_array[i] = len(cat_measures)

        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))

    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:

        eta = 0.0

    else:

        eta = np.sqrt(numerator/denominator)

    return eta
train_data.describe()
factors_num = list(pd.DataFrame(train_data.dtypes)[(train_data.dtypes=='int64') | (train_data.dtypes=='float64')].index)

factors_num.remove('MSSubClass')

no_factors_num = len(factors_num)

fig, axs = plt.subplots(no_factors_num, 1, figsize=(10,100))

for i in range(no_factors_num):

    factor = factors_num[i]

    axs[i].plot(train_data['SalePrice'], train_data[factor], 'o')
corr_matrix_num = train_data.corr(method ='spearman')

cm = sns.diverging_palette(20, 133, sep=20, as_cmap=True)

corr_matrix_num.style.background_gradient(cmap=cm)
factors_cat = [i for i in list(train_data) if i not in list(factors_num)]

no_factors_cat = len(factors_cat)

sns.set(style="whitegrid")

fig, axs = plt.subplots(no_factors_cat, 1, figsize=(20,400), sharex=False)

for i in range(no_factors_cat):

    factor = factors_cat[i]

    sns.violinplot(x=factor, y="SalePrice", data=train_data, ax=axs[i])
cat_factors = train_data[factors_cat].fillna('Unknown')

cat_factors.head()
corr_matrix_cat = pd.DataFrame(index=factors_cat, columns=factors_cat)

for i in range(0, len(factors_cat)):

    for j in range(0, len(factors_cat)):

        corr_matrix_cat.iloc[i][j] = cramers_v(cat_factors[factors_cat[i]],cat_factors[factors_cat[j]])

corr_matrix_cat = corr_matrix_cat.astype(float)

cm = sns.diverging_palette(20, 133, sep=20, as_cmap=True)

corr_matrix_cat.style.background_gradient(cmap=cm)
corr_cat = pd.DataFrame(index=factors_cat, columns=['SalePrice'])

for i in range(0, len(factors_cat)):

    corr_cat.iloc[i][0] = correlation_ratio(cat_factors[factors_cat[i]], train_data['SalePrice'])

corr_cat = corr_cat.astype(float)

cm = sns.diverging_palette(20, 133, sep=20, as_cmap=True)

corr_cat.style.background_gradient(cmap=cm)
regression_factors_num = list(corr_matrix_num[corr_matrix_num['SalePrice']> 0.5].index.values)

regression_factors_cat = list(corr_cat[corr_cat['SalePrice']> 0.5].index.values)

regression_factors_num.remove('SalePrice')

regression_factors = regression_factors_num+regression_factors_cat

print(regression_factors)
regression_train_num = train_data[regression_factors_num]

regression_train_num = regression_train_num.fillna(regression_train_num.median())

regression_train_cat = train_data[regression_factors_cat]

regression_train_cat = regression_train_cat.fillna('Unknown') #.fillna(regression_train_cat.mode())

regression_train = regression_train_num.merge(regression_train_cat, left_index=True, right_index=True)

regression_train.head()
#from sklearn.compose import ColumnTransformer 

#ct = ColumnTransformer([("categorical_ohe", OneHotEncoder(),regression_factors_cat)], remainder="passthrough")

#regression_train_ct = ct.fit_transform(regression_train)  
regression_train = pd.get_dummies(regression_train, prefix_sep='_', drop_first=True)

regression_train.head()
X = regression_train.values

y = np.log(train_data['SalePrice']).values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = LinearRegression()  

regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(list(np.transpose(regressor.coef_)[:,0]), [list(regression_train)], columns=['Coefficient']) 

coeff_df['Std Dev'] = X_train.std(0)

coeff_df['Importance'] = round(abs(1000000*coeff_df['Coefficient']/coeff_df['Std Dev']))

coeff_df
y_pred = regressor.predict(X_test)

validation = pd.DataFrame({'Real SalePrice': y_test.flatten(), 'Predicted SalePrice': y_pred.flatten()})

plt.plot(validation['Real SalePrice'], validation['Predicted SalePrice'], 'o')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
validation.sort_values(by=['Predicted SalePrice'], ascending=False)
regression_test_num = train_data[regression_factors_num]

regression_test_num = regression_test_num.fillna(regression_train_num.median())

regression_test_cat = test_data[regression_factors_cat]

regression_test_cat = regression_test_cat.fillna('Unknown')

regression_test = regression_test_num.merge(regression_test_cat, left_index=True, right_index=True)

regression_test = pd.get_dummies(regression_test, prefix_sep='_', drop_first=True)

regression_test = regression_test[list(regression_train)]
regressor = LinearRegression()  

regressor.fit(X, y)

predictions = regressor.predict(regression_test.values)

predictions = np.exp(predictions)
Id = test_data['Id']

SalePrice = predictions[:,0]

submission = pd.DataFrame(data={'Id': Id, 'SalePrice': SalePrice})

submission.to_csv("submission.csv", index=False)