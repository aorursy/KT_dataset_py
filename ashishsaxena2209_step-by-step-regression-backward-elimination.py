import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import statsmodels.formula.api as smf

from sklearn.preprocessing import PolynomialFeatures

import random

from sklearn.metrics import mean_squared_error, mean_absolute_error
odf = pd.read_csv('../input/diamonds.csv')
odf.head()
# Dropping first column of dataframe

odf.drop(columns=odf.columns[0], axis = 1, inplace=True)

odf.head(10)
odf.describe()
# Categorizing columns according to data types

categorical = odf.columns[[1,2,3]]

to_predict = 'price'

numeric = odf.columns[[0,4,5,6,7,8,9]]
# Checking for missing values in dataset

for c in odf.columns:

    print('Total Missing values in \'{}\' are {}.'.format(c,odf[c].isna().sum()))
# Replacing zero values of x, y and z with NaNs. After then NaN will be dropped



odf[['x','y','z']] = odf[['x','y','z']].replace(0,np.NaN)



odf.dropna(inplace=True)
plt.figure(figsize=(15,12))

for i in range(1,8):

    plt.subplot(3, 3, i)

    plt.scatter(odf['price'], odf[numeric[i-1]], s= 1)

    plt.xlabel(to_predict)

    plt.ylabel(numeric[i-1])
odf.drop(odf[odf.y > 20].index, inplace=True) # Dropping outliers from 'y' column

odf.drop(odf[odf.z > 20].index, inplace=True) # Dropping outliers from 'z' column



plt.figure(figsize=(15,12))

for i in range(1,8):

    plt.subplot(3, 3, i)

    plt.scatter(odf['price'], odf[numeric[i-1]], s= 1)

    plt.xlabel(to_predict)

    plt.ylabel(numeric[i-1])
p = sns.heatmap(odf.corr('spearman'), center=0, cmap = 'RdYlGn')
# Making dummy variables for categorical Columns

odfd = pd.get_dummies(data=odf, columns=categorical)
# Dropping Extra caegoricals

lst = ['color_D','cut_Fair','clarity_IF']

idx = []

for i in lst:

    # Removing D color from color column, Fair cut and IF clarity column

    idx.append(odfd.columns.get_loc(i))

    

odfd.drop(columns=odfd.columns[idx], axis = 1, inplace=True)
# Rearranging columns of dataframe



col = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_E', 

       'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',

       'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2', 'price']



odfd = odfd[col]
plt.figure(figsize=(25,12))

p = sns.heatmap(data=odfd.corr(method='spearman'), annot=True, cmap='RdYlGn', center=0)
# Automating backward elimination technique



def DoBackwardElimination(the_regressor, X, y, minP2eliminate):

    

    assert np.shape(X)[0] == np.shape(y)[0], 'Length of X and y do not match'

    assert minP2eliminate > 0, 'Minimum P value to eliminate cannot be zero or negative'

    

    original_list = list(range(0, np.shape(the_regressor.pvalues)[0]))

    

    max_p = 10        # Initializing with random value of maximum P value

    i = 0

    r2adjusted = []   # Will store R Square adjusted value for each loop

    r2 = []           # Will store R Square value  for each loop

    list_of_originallist = [] # Will store modified index of X at each loop

    classifiers_list = [] # fitted classifiers at each loop

    

    while max_p >= minP2eliminate:

        

        p_values = list(the_regressor.pvalues)

        r2adjusted.append(the_regressor.rsquared_adj)

        r2.append(the_regressor.rsquared)

        list_of_originallist.append(original_list)

        

        max_p = max(p_values)

        max_p_idx = p_values.index(max_p)

        

        if max_p_idx == 0:

            

            temp_p = set(p_values)

            

            # removing the largest element from temp list

            temp_p.remove(max(temp_p))

            

            max_p = max(temp_p)

            max_p_idx = p_values.index(max_p)

            

            print('Index value 0 found!! Next index value is {}'.format(max_p_idx))

            

            if max_p < minP2eliminate:

                

                print('Max P value found less than 0.1 with 0 index ...Loop Ends!!')

                

                break

                

        if max_p < minP2eliminate:

            

            print('Max P value found less than 0.1 without 0 index...Loop Ends!!')

            

            break

        

        val_at_idx = original_list[max_p_idx]

        

        idx_in_org_lst = original_list.index(val_at_idx)

        

        original_list.remove(val_at_idx)

        

        print('Popped column index out of original array is {} with P-Value {}'.format(val_at_idx, np.round(np.array(p_values)[max_p_idx], decimals= 4)))

        

        X_new = X[:, original_list]

        

        the_regressor = smf.OLS(endog = y, exog = X_new).fit()

        classifiers_list.append(the_regressor)

        

        print('==================================================================================================')

        

    return classifiers_list, r2, r2adjusted, list_of_originallist
# Preprocessing data



X = odfd.iloc[:,:-1].values          # Selecting all columns except last one that is 'price'.

y = odfd['price'].values



# # Adding constant values at start of array X

# X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
# Scaling input data



from sklearn.preprocessing import MinMaxScaler



mscalar = MinMaxScaler()



X_minmax_scaled = mscalar.fit_transform(X)
# Adding constant values at start of array X

X_minmax_scaled = np.append(arr = np.ones((X_minmax_scaled.shape[0], 1)).astype(int), values=X_minmax_scaled, axis=1)
X_lst = list(range(0, X.shape[1]))

X_opt = X_minmax_scaled[:,X_lst]

flag = 2 # Done intentionally in order to stop popping 5th column from input array X
# To be used when doing manual elimination

# Eliminating columns according to P values from summary to make new X_opt

idx_to_pop = 5

if flag == 1:

    X_lst.pop(element_to_pop)

    flag = 99

    

X_opt = X_minmax_scaled[:,X_lst]
# Splitting data for taining and testing

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.25, random_state=22)
# Fitting the regressor

regressor_SLR_OLS = smf.OLS(endog = y_train, exog = X_train).fit()



# Looking at the summary of regressor

print(regressor_SLR_OLS.summary())
sns.set(style='ticks')

plt.figure(figsize=(14,8))

y_pred_train = regressor_SLR_OLS.predict(X_train)

y_pred_test = regressor_SLR_OLS.predict(X_test)



plt.scatter(y_train, y_train-y_pred_train, s= 2, c= 'R', alpha=0.8)

plt.scatter(y_test, y_test-y_pred_test, s= 2, c= 'B', alpha=0.8)

plt.plot([0, 18000], [0,0], '-k', linewidth = 3, alpha = 0.3)

p = plt.xlim((0, 18000))

p = plt.ylim((-20000, 20000))

p = plt.legend(['Reference Line','Training Data Residual', 'Testing Data Residual'])

p = plt.title('Residual Graph')
def Calculate_Error(original_values, predicted_values):

    assert len(original_values) == len(predicted_values), 'Both list should have same length'

    temp = 0

    error = 0

    n = len(original_values)

    for o, p in zip(original_values, predicted_values):

        temp = temp + ((o-p)**2)

        

    temp = temp/n

    error = np.sqrt(temp)

    return error

        
train_data_error = Calculate_Error(original_values=y_train, predicted_values=y_pred_train)

test_data_error = Calculate_Error(original_values=y_test, predicted_values=y_pred_test)



print('MSE for training data is {}'.format(np.round(train_data_error, 4)))

print('MSE for testing data is {}'.format(np.round(test_data_error, 4)))
random.seed(22)

plt.figure(figsize=(15, 10))

rand_nums1 = random.sample(range(0, len(y_train)), 50)

plt.subplot(2,1,1)

# plt.figure(figsize=(15,10))

p = plt.plot(range(0, 50), y_train[rand_nums1, ], color = 'red')

p = plt.plot(range(0, 50), y_pred_train[rand_nums1, ], color = 'blue')

plt.title(r'$ \mathrm{\mathsf{Training Data Output}}$')

plt.legend(['Original Output', 'Predicted Output'])

plt.xlabel(r'$Observation  Number \longrightarrow$')

plt.ylabel(r'$Output \longrightarrow$')

plt.ylim(-5000, 25000)





rand_nums2 = random.sample(range(0, len(y_test)), 50)

plt.subplot(2,1,2)

p = plt.plot(range(0, 50), y_test[rand_nums2, ], color = 'red')

p = plt.plot(range(0, 50), y_pred_test[rand_nums2, ], color = 'blue')

plt.title(r'$ \mathrm{\mathsf{Testing Data Output}}$')

plt.legend(['Original Output', 'Predicted Output'])

plt.xlabel(r'$Observation  Number \longrightarrow$')

plt.ylabel(r'$Output \longrightarrow$')

plt.ylim(-5000, 25000)



plt.subplots_adjust(hspace=0.4)
training_residual = []

for o, p in zip(y_train[rand_nums1, ], y_pred_train[rand_nums1, ]):

    training_residual.append(o-p)

    

testing_residual = []

for o, p in zip(y_test[rand_nums2, ], y_pred_test[rand_nums2, ]):

    testing_residual.append(o-p)
plt.figure(figsize=(15,10))



plt.subplot(2,1,1)

# p = plt.bar(list(range(0, 50)),training_residual, align='center', width = 1, fill= False, edgecolor = 'k')

p = plt.scatter(list(range(0, 50)),training_residual)

p = plt.plot([0, 50], [0, 0], '-k')

plt.text(30,3500, 'Mean of residual is {}'.format(np.round(np.mean(training_residual), 4)))

plt.ylim(-5000, 5000)

plt.xlim(0,50)

plt.title('Training Residuals')

plt.xlabel(r'$Observation  Number \longrightarrow$')

plt.ylabel(r'$Residual \longrightarrow$')





plt.subplot(2,1,2)

# p = plt.bar(list(range(0, 50)),testing_residual, align='center', width = 1, fill= False, edgecolor = 'k')

p = plt.scatter(list(range(0, 50)),testing_residual)

p = plt.plot([0, 50], [0, 0], '-k')

plt.text(30,3500, 'Mean of residual is {}'.format(np.round(np.mean(testing_residual), 4)))

plt.ylim(-5000, 5000)

plt.xlim(0,50)

plt.title('Testing Residuals')

plt.xlabel(r'$Observation  Number \longrightarrow$')

plt.ylabel(r'$Residual \longrightarrow$')



plt.subplots_adjust(hspace= 0.4)
print('Mean of \'price\' for training data is {}.'.format(np.round(np.mean(y_train),4)))

print('Mean of residual of \'price\' for training data is {}.'.format(np.round(np.mean(training_residual),4)))



print('Mean of \'price\' for testing data is {}.'.format(np.round(np.mean(y_test),4)))

print('Mean of residual of \'price\' for training data is {}.'.format(np.round(np.mean(testing_residual),4)))
# Now it's time to do backward elimination and check wether it improves performance of our regression model



regressor_list, r2, r2adjusted, list_of_changes = DoBackwardElimination(the_regressor=regressor_SLR_OLS, 

                                                                        X= X_train, y= y_train, minP2eliminate = 0.05)
new_list = list_of_changes[1]

new_regressor = regressor_list[0]

print(new_regressor.summary())
y_pred_train_new = new_regressor.predict(X_train[:, new_list])

y_pred_test_new = new_regressor.predict(X_test[:, new_list])



training_residual_new = []

for o, p in zip(y_train[rand_nums1, ], y_pred_train_new[rand_nums1, ]):

    training_residual_new.append(o-p)

    

testing_residual_new = []

for o, p in zip(y_test[rand_nums2, ], y_pred_test_new[rand_nums2, ]):

    testing_residual_new.append(o-p)
print('Mean of \'price\' for training data is {}.'.format(np.round(np.mean(y_train),4)))

print('Mean of residual of \'price\' for training data is {}.'.format(np.round(np.mean(training_residual_new),4)))



print('Mean of \'price\' for testing data is {}.'.format(np.round(np.mean(y_test),4)))

print('Mean of residual of \'price\' for training data is {}.'.format(np.round(np.mean(testing_residual_new),4)))