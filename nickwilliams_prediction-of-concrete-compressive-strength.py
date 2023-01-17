import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# This code cell reads the concrete compressive strenght .csv into a Pandas data frame.

concrete = pd.read_csv('../input/concrete-compressive-strength-uci/Concrete_Data.csv')

concrete.head()
# As seen above, the column names in this dataset are quite long. 

# The code below will rename the columns so they are easier to work with.

columns = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseagg', 'fineagg', 'age', 'strength']

concrete.columns = columns

concrete.head()
# Check column data types.

concrete.dtypes
# Check for missing values in the dataset.

concrete.isnull().sum()
# There is no missing data. Thus our data cleaning efforts will be minimal.

# This cell provides summary information for the dataframe.

concrete.describe()
# The code below allows us to gain a first pass look at the correlation between different feature and our target feature, (compressive) strength.

concrete.corr()
# This cell plots key features vs. compressive strength. 

fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



ax1.scatter(concrete.cement, concrete.strength, color='gray')

ax2.scatter(concrete.age, concrete.strength, color='gray')

ax3.scatter(concrete.superplasticizer, concrete.strength, color='gray')

ax4.scatter(concrete.water, concrete.strength, color='gray')



ax1.title.set_text('Strength vs. Cement')

ax2.title.set_text('Strength vs. Age')

ax3.title.set_text('Strength vs. Superplasticizer')

ax4.title.set_text('Strength vs. Water')
# This cell copies the original DataFrame in order to engineer new features without disturbing the original data.

concrete_eng = concrete.copy()

concrete_eng.head()
# This code cell creates new features for the DataFrame following the procedure of Yeh (1998). 

concrete_eng['total'] = concrete_eng.cement + concrete_eng.slag + concrete_eng.flyash + concrete_eng.water + concrete_eng.superplasticizer + concrete_eng.coarseagg + concrete_eng.fineagg

concrete_eng['binder'] = concrete_eng.cement + concrete_eng.flyash + concrete_eng.slag

concrete_eng['wcratio'] = concrete_eng.water/concrete_eng.cement

concrete_eng['wbratio'] = concrete_eng.water/concrete_eng.binder

concrete_eng['spbratio'] = concrete_eng.superplasticizer/concrete_eng.binder

concrete_eng['fabratio'] = concrete_eng.flyash/concrete_eng.binder

concrete_eng['sbratio'] = concrete_eng.slag/concrete_eng.binder

concrete_eng['fasbratio'] = (concrete_eng.flyash + concrete_eng.slag)/concrete_eng.binder

concrete_eng.describe()
# Initial look at correlation between engineered features and compressive strength, which helps select key features for visualization.

concrete_eng.corr()
# Plots of key engineered features vs. compressive strength.

engfig = plt.figure(figsize=(15, 10))

ax1 = engfig.add_subplot(221)

ax2 = engfig.add_subplot(222)

ax3 = engfig.add_subplot(223)

ax4 = engfig.add_subplot(224)



ax1.scatter(concrete_eng.wcratio, concrete_eng.strength, color='gray')

ax2.scatter(concrete_eng.wbratio, concrete_eng.strength, color='gray')

ax3.scatter(concrete_eng.spbratio, concrete_eng.strength, color='gray')

ax4.scatter(concrete_eng.binder, concrete_eng.strength, color='gray')



ax1.title.set_text('Strength vs. W/C Ratio')

ax2.title.set_text('Strength vs. W/B Ratio')

ax3.title.set_text('Strength vs. Superplasticizer to Binder Ratio')

ax4.title.set_text('Strength vs. Binder')
# Create the final DataFrame for model development based on observations from the previous sections.

final_columns = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseagg', 'fineagg', 'age', 'strength', 'binder', 'wcratio', 'wbratio']

concrete_final = concrete_eng[final_columns]

concrete_final.head(10)
# The code below sets up the variables for prediction and splits the data into training and testing sets.

from sklearn.model_selection import train_test_split



indep_var = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseagg', 'fineagg', 'age', 'binder', 'wcratio', 'wbratio']

dep_var = ['strength']

X = concrete_final[indep_var]

y = concrete_final[dep_var]

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size = 0.8, random_state=1)
# Here four models are set up. The predictions of various models will be compared later in this analysis.

from sklearn import linear_model

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



linear = linear_model.LinearRegression()

ridge = linear_model.BayesianRidge()

tree = DecisionTreeRegressor(random_state=1)

forest = RandomForestRegressor(random_state=1)
# Each model is fit using the validation data.

linear.fit(X_train, y_train.values.ravel())

ridge.fit(X_train, y_train.values.ravel())

tree.fit(X_train, y_train.values.ravel())

forest.fit(X_train, y_train.values.ravel())
# This code cell uses the models created above to predict concrete strength with the test data.

linear_preds = linear.predict(X_test)

ridge_preds = ridge.predict(X_test)

tree_preds = tree.predict(X_test)

forest_preds = forest.predict(X_test)
# Here I visualize the predicted values vs. the known values.

engfig = plt.figure(figsize=(15, 10))

ax1 = engfig.add_subplot(221)

ax2 = engfig.add_subplot(222)

ax3 = engfig.add_subplot(223)

ax4 = engfig.add_subplot(224)



ax1.scatter(linear_preds, y_test, color='gray')

ax2.scatter(ridge_preds, y_test, color='gray')

ax3.scatter(tree_preds, y_test, color='gray')

ax4.scatter(forest_preds, y_test, color='gray')



ax1.title.set_text('Linear Model')

ax2.title.set_text('Ridge Model')

ax3.title.set_text('Decision Tree Model')

ax4.title.set_text('Random Forest Model')
# To compare models beyond visualizations, a measure of accuracy is needed. For this analysis, I use...



from sklearn.metrics import mean_squared_error

from sklearn.metrics import max_error

from sklearn.metrics import r2_score



linear_mse = mean_squared_error(y_test, linear_preds)

ridge_mse = mean_squared_error(y_test, ridge_preds)

tree_mse = mean_squared_error(y_test, tree_preds)

forest_mse = mean_squared_error(y_test, forest_preds)



linear_max = max_error(y_test, linear_preds)

ridge_max = max_error(y_test, ridge_preds)

tree_max = max_error(y_test, tree_preds)

forest_max = max_error(y_test, forest_preds)



linear_r2 = r2_score(y_test, linear_preds)

ridge_r2 = r2_score(y_test, ridge_preds)

tree_r2 = r2_score(y_test, tree_preds)

forest_r2 = r2_score(y_test, forest_preds)



print('Linear model MSE, max error, R2 =', linear_mse, ',', linear_max, ',', linear_r2)

print('Ridge model MSE, max error, R2 =', ridge_mse, ',', ridge_max, ',', ridge_r2)

print('Decision tree model MSE, max error, R2 =', tree_mse, ',', tree_max, ',', tree_r2)

print('Random forest model MSE, max error, R2 =', forest_mse, ',', forest_max, ',', forest_r2)
# This cell imports and establishes the permutation importance function.

import eli5

from eli5.sklearn import PermutationImportance



linear_perm = PermutationImportance(linear, random_state=1).fit(X_train, y_train)

ridge_perm = PermutationImportance(ridge, random_state=1).fit(X_train, y_train)

tree_perm = PermutationImportance(tree, random_state=1).fit(X_train, y_train)

forest_perm = PermutationImportance(forest, random_state=1).fit(X_train, y_train)
# The code below from the eli5 library creates a visualization of permutation importance for the linear model.

eli5.show_weights(linear_perm, feature_names = X_train.columns.tolist())
# The code below from the eli5 library creates a visualization of permutation importance for the ridge model.

eli5.show_weights(ridge_perm, feature_names = X_train.columns.tolist())
# The code below from the eli5 library creates a visualization of permutation importance for the decision tree model.

eli5.show_weights(tree_perm, feature_names = X_train.columns.tolist())
# The code below from the eli5 library creates a visualization of permutation importance for the random forest model.

eli5.show_weights(forest_perm, feature_names = X_train.columns.tolist())