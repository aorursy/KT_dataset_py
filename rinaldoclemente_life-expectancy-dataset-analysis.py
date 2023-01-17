import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import graphviz

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree
# Load the dataset.

le = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv', sep=',')

le.dataframeName = 'Life Expectancy Data.csv'

le.head()
# Modify the original names of the features using a standard format for all the features.

orig_cols = list(le.columns) 

new_cols = [] 

for col in orig_cols:     

    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower()) 



le.columns = new_cols



# Compute a summary of statistics only for the numerical features.

le.describe()
# Discard the metadata (country and year).

le = le.drop(['country','year'], axis=1)
# For each feature count all rows with NULL values.

le.isnull().sum()
# For each feature delete all rows with NULL values.

le.dropna(inplace=True)

le.isnull().sum()
#Change column order to better perform splits

new_order = [1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

le = le[le.columns[new_order]]

le.head()
# Get a concise summary of the dataset.

le.info()
# Create a dictionary of columns representing the features of the dataset.

col_dict = {'life_expectancy':1,'adult_mortality':2,'infant_deaths':3,'alcohol':4,'percentage_expenditure':5,'hepatitis_b':6,'measles':7,'bmi':8,

            'under-five_deaths':9,'polio':10,'total_expenditure':11,'diphtheria':12,'hiv/aids':13,'gdp':14,'population':15,'thinness_1-19_years':16,

            'thinness_5-9_years':17,'income_composition_of_resources':18,'schooling':19}



# Visualize the data for each feature using box plots.

plt.figure(figsize=(18,30))



for variable,i in col_dict.items():

                     plt.subplot(5,4,i)

                     plt.boxplot(le[variable],whis=1.5)

                     plt.title(variable)



plt.show()

le.shape
# Remove the outliers using the interquartile range (IQR).

Q1 = le.quantile(0.25)

Q3 = le.quantile(0.75)

IQR = Q3 - Q1



le = le[~((le < (Q1 - 1.5 * IQR)) |(le > (Q3 + 1.5 * IQR))).any(axis=1)]



#Replace Status into boolean variables

le["status"].replace({"Developing": 1, "Developed": 0}, inplace=True)



# Print the dimensions of the cleaned dataset.

le.shape
# Visualize the cleaned data for each feature using box plots.

plt.figure(figsize=(18,30))



for variable,i in col_dict.items():

                     plt.subplot(5,4,i)

                     plt.boxplot(le[variable],whis=1.5)

                     plt.title(variable)

plt.show()
# Plot heatmap to visualize the correlations.

plt.figure(figsize = (14, 12))

sns.heatmap(le.corr(), annot = True)

plt.title('Correlation between different features');
# Create a vector containing all the features of the dataset.

all_col = ['adult_mortality','infant_deaths','alcohol','percentage_expenditure','hepatitis_b','measles','bmi',

         'under-five_deaths','polio','total_expenditure','diphtheria','hiv/aids','gdp','population','thinness_1-19_years',

         'thinness_5-9_years','income_composition_of_resources','schooling']



plt.figure(figsize=(15,30))



# Plot each feature in function of the target variable (life_expectancy) using scatter plots.

for i in range(len(all_col)):

    plt.subplot(7,3,i+1)

    plt.scatter(le[all_col[i]], le['life_expectancy'])

    plt.xlabel(all_col[i])

    plt.ylabel('Life Expectancy')



plt.show()
# Separate the features from the labels.

X = le.iloc[:,1:].values

y = le.iloc[:,0].values #Life Expectancy
# Normalize the data.

X_std= StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)



# Compute covariance matrix.

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
# Compute eigenvalues and eigenvectors.

eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
# Compute the variance for every eigenvalue.

tot = sum(eig_vals)



var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]



var_exp
# Plot the principal components.

plt.figure(figsize=(10,4))

plt.bar(range(19), var_exp, alpha=0.7, align='center', label='Individual Variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.xticks(np.arange(0, 19, 1.0))

plt.tight_layout()
# Plot the cumulative variance.

pca = PCA(n_components=19).fit(X_std)

plt.figure(figsize=(12, 4))

plt.plot(np.cumsum(pca.explained_variance_ratio_), label='Cumulative Variance')

plt.xlim(0,18,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.legend(loc='best')

plt.grid(color='#E3E3E3')

plt.xticks(np.arange(0, 19, 1.0));
# Take the the values of the target variable and of the most correlated feature with the target variable.

le_features = le['income_composition_of_resources'].values.reshape(-1,1)

le_labels = le['life_expectancy'].values.reshape(-1,1)



# Normalize the data.

min_max_scaler = MinMaxScaler()

le_features = min_max_scaler.fit_transform(le_features)



# Split the dataset in training and test set.

le_features_train, le_features_test, le_labels_train, le_labels_test = train_test_split(le_features, le_labels, train_size = 0.7, test_size = 0.3)
linear_model = LinearRegression()



# Train the model.

linear_model.fit(le_features_train, le_labels_train);
# Test the model.

linear_model_score = linear_model.predict(le_features_test)



# Plot the result.

plt.figure(figsize=(10, 6))

plt.scatter(le_features_test, le_labels_test,  color='black')

plt.plot(le_features_test, linear_model_score, color='blue', linewidth=2)

plt.xlabel('Income Composition of Resources')

plt.ylabel('Life Expectancy')

plt.show()



print('Coefficients: \n', linear_model.coef_)

print("Mean squared error: %.2f" % mean_squared_error(le_labels_test,linear_model_score))

print("R^2 score : %.2f" % r2_score(le_labels_test,linear_model_score))
# Separate the features from the labels.

le_features = le.iloc[:, 1:].values

le_labels = le.iloc[:,0] #Life Expectancy



# Normalize the data.

min_max_scaler = MinMaxScaler()

le_features = min_max_scaler.fit_transform(le_features)



# Split the dataset in training and test set.

le_features_train, le_features_test, le_labels_train, le_labels_test = train_test_split(le_features, le_labels, train_size = 0.7, test_size = 0.3)
# Train the model.

linear_model.fit(le_features_train, le_labels_train);



# Test the model.

linear_model_score = linear_model.predict(le_features_test)



print('Coefficients: \n', linear_model.coef_)

print("Mean squared error: %.2f" % mean_squared_error(le_labels_test,linear_model_score))

print("R^2 score : %.2f" % r2_score(le_labels_test,linear_model_score))
# Calculate the life expeectancy average

le_avg = le['life_expectancy'].mean()

le_lr = le.copy()



# Replace 1 if life expectancy > avg, 0 otherwise

le_lr['life_expectancy'] = (le_lr['life_expectancy'] > le_avg).astype(int)



# Separate the features from the labels.

le_features_lr = le_lr.iloc[:, 1:].values

le_labels_lr = le_lr.iloc[:,0] #Life Expectancy



# Normalize the data.

min_max_scaler = MinMaxScaler()

le_features_lr = min_max_scaler.fit_transform(le_features_lr)



# Split the dataset in training and test set.

le_features_train_lr, le_features_test_lr, le_labels_train_lr, le_labels_test_lr = train_test_split(le_features_lr, le_labels_lr, train_size = 0.7, test_size = 0.3)
logistic_model = LogisticRegression(solver='liblinear')



#Train The Model

logistic_model.fit(le_features_train_lr, le_labels_train_lr);
logistic_score = logistic_model.predict(le_features_test_lr)



#Perform confusion matrix

confusion_matrix = confusion_matrix(le_labels_test_lr, logistic_score)



#Print confusion matrix as heatmap

class_names=[0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(confusion_matrix),cmap='YlGnBu',annot=True,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.xlabel('Predicted label')

plt.ylabel('Actual label')

plt.show()
print("Score on the train data: %.2f" % logistic_model.score(le_features_train_lr, le_labels_train_lr))

print("Score on the test data: %.2f" % logistic_model.score(le_features_test_lr, le_labels_test_lr))
# Perform DecisionTreeRegresson with three different depths

decision_tree_model3 = DecisionTreeRegressor(max_depth=3)

decision_tree_model5 = DecisionTreeRegressor(max_depth=5)

decision_tree_model7 = DecisionTreeRegressor(max_depth=7)



# Train the model.

decision_tree_model3 = decision_tree_model3.fit(le_features_train, le_labels_train)

decision_tree_model5 = decision_tree_model5.fit(le_features_train, le_labels_train)

decision_tree_model7 = decision_tree_model7.fit(le_features_train, le_labels_train)



print("Score on the train data with depth 3: %.2f" % decision_tree_model3.score(le_features_train, le_labels_train))

print("Score on the test data with depth 3: %.2f" % decision_tree_model3.score(le_features_test, le_labels_test))

print("Score on the train data with depth 5: %.2f" % decision_tree_model5.score(le_features_train, le_labels_train))

print("Score on the test data with depth 5: %.2f" % decision_tree_model5.score(le_features_test, le_labels_test))

print("Score on the train data with depth 7: %.2f" % decision_tree_model7.score(le_features_train, le_labels_train))

print("Score on the test data with depth 7: %.2f" % decision_tree_model7.score(le_features_test, le_labels_test))
# Plot the result.

dot_data = tree.export_graphviz(decision_tree_model3, 

                                filled=True, 

                                rounded=True, 

                                out_file=None, 

                                feature_names=le.iloc[:, 1:].columns)

graph = graphviz.Source(dot_data)

graph
random_forest_model = RandomForestRegressor(n_estimators=100,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)



# Train the model.

random_forest_model.fit(le_features_train, le_labels_train);
df_ = pd.DataFrame(le.iloc[:, 1:].columns, columns = ['feature'])

df_['fscore'] = random_forest_model.feature_importances_[:, ]



# Plot the relative importance of the top 10 features.

df_['fscore'] = df_['fscore'] / df_['fscore'].max()

df_.sort_values('fscore', ascending = False, inplace = True)

df_ = df_[0:19]

df_.sort_values('fscore', ascending = True, inplace = True)

ax = df_.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 10))



# Plot the result.

plt.title('Random forest feature importance')

plt.xlabel('')

plt.ylabel('')

plt.xticks([], [])

plt.yticks()



# Create a list to collect the plt.patches data.

totals = []



# Find the values and append to list.

for i in ax.patches:

    totals.append(i.get_width())



# Set individual bar lables using above list.

total = sum(totals)



# Set individual bar lables using above list.

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down.

    ax.text(i.get_width(), i.get_y()+.13, \

            str(round((i.get_width()/total)*100, 2))+'%', fontsize=10,

color='#505050')



plt.show()
# Test the model.

random_forest_score = random_forest_model.predict(le_features_test)



print("Score on the train data: %.2f" % random_forest_model.score(le_features_train, le_labels_train))

print("Score on the test data: %.2f" % random_forest_model.score(le_features_test, le_labels_test))