import pandas as pd   # package for data analysis

import numpy as np    # package for numerical computations



# libraries for visualization

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings

import warnings

warnings.filterwarnings('ignore')



# For Preprocessing, ML models and Evaluation

from sklearn.model_selection import train_test_split   # To split the dataset into train and test set



from sklearn.linear_model import LogisticRegression     # Logistic regression model



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder    # for converting categorical to numerical



from sklearn.metrics import f1_score    # for model evaluation
data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Training_set_advc.csv')
# Take a look at the first five observations

data.head()
data.Treated_with_drugs.value_counts()
data['Treated_with_drugs'] = data['Treated_with_drugs'].str.upper()
data.Patient_Smoker.value_counts()
def smoker(r):

  if (r == "NO") or (r == "NO "):

    return 'NO'

  elif (r == "YES") or (r == "YES ") or (r == "YESS") or (r == "YESS "):

    return 'YES'

  else:

    return 'Cannot say'



data.Patient_Smoker = data.Patient_Smoker.apply(smoker)  # Applying the function to all the entries of Patient_Smoker column
data.Patient_Rural_Urban.value_counts()
data.Patient_mental_condition.value_counts()
# A concise summary of the data

data.info()
sns.countplot(x='Survived_1_year', data=data)

plt.show()
# getting only the numerical features

numeric_features = data.select_dtypes(include=[np.number])    # select_dtypes helps you to select data of particular types 

numeric_features.columns
numeric_data=data[['Diagnosed_Condition', 'Patient_Age', 'Patient_Body_Mass_Index', 'Number_of_prev_cond', 'Survived_1_year']]  #keeping in the target varibale for analysis purposes

numeric_data.head()



# ID_Patient_Care_Situation and Patient_ID are just an ID we can ignore them for data analysis.

# Number_of_prev_cond is dependent on 7 columns - A, B, C, D, E, F, Z
# Checking the null values in numerical columns

numeric_data.isnull().sum()
data['Number_of_prev_cond'] = data['Number_of_prev_cond'].fillna(data['Number_of_prev_cond'].mode()[0])  # filling the missing value of 'Number_of_prev_cond'



numeric_data['Number_of_prev_cond']=data['Number_of_prev_cond']

numeric_data.isnull().sum()



# The returned object by using mode() is a series so we are filling the null value with the value at 0th index ( which gives us the mode of the data)
# Taking a look at the basic statistical description of the numerical columns

numeric_data.describe()


for feature in numeric_data.drop('Survived_1_year', axis = 1).columns:

  sns.boxplot(x='Survived_1_year', y=feature, data=numeric_data)

  plt.show()
numeric_data=numeric_data.drop(['Survived_1_year'], axis=1)

colormap = sns.diverging_palette(10, 220, as_cmap = True)

sns.heatmap(numeric_data.corr(),

            cmap = colormap,

            square = True,

            annot = True)

plt.show()
data.isnull().sum()
data['Treated_with_drugs']=data['Treated_with_drugs'].fillna(data['Treated_with_drugs'].mode()[0])
data['A'].fillna(data['A'].mode()[0], inplace = True)

data['B'].fillna(data['B'].mode()[0], inplace = True)

data['C'].fillna(data['C'].mode()[0], inplace = True)

data['D'].fillna(data['D'].mode()[0], inplace = True)

data['E'].fillna(data['E'].mode()[0], inplace = True)

data['F'].fillna(data['F'].mode()[0], inplace = True)

data['Z'].fillna(data['Z'].mode()[0], inplace = True)
data.isnull().sum()
categorical_data = data.drop(numeric_data.columns, axis=1)    # dropping the numerical columns from the dataframe 'data'

categorical_data.drop(['Patient_ID', 'ID_Patient_Care_Situation'], axis=1, inplace = True)    # dropping the id columns form the dataframe 'categorical data'

categorical_data.head()    # Now we are left with categorical columns only. take a look at first five observaitons
categorical_data.nunique()   # nunique() return you the number of unique values in each column/feature
# Visualization of categorical columns

for feature in ['Patient_Smoker', 'Patient_Rural_Urban', 'Patient_mental_condition']:

  sns.countplot(x=feature,  hue='Survived_1_year', data=categorical_data)

  plt.show()





plt.figure(figsize=(15,5))

sns.countplot(x='Treated_with_drugs',  hue='Survived_1_year', data=categorical_data)

plt.xticks(rotation=90)

plt.show()

drugs = data['Treated_with_drugs'].str.get_dummies(sep=' ') # split all the entries separated by space and create dummy variable

drugs.head()
data = pd.concat([data, drugs], axis=1)     # concat the two dataframes 'drugs' and 'data'

data = data.drop('Treated_with_drugs', axis=1)    # dropping the column 'Treated_with_drugs' as its values are now splitted into different columns



data.head()
data.Patient_Smoker.value_counts()
data.Patient_Smoker[data['Patient_Smoker'] == "Cannot say"] = 'NO'    # we already know 'NO' is the mode so directly changing the values 'Cannot say' to 'NO'
data.drop('Patient_mental_condition', axis = 1, inplace=True)
data = pd.get_dummies(data, columns=['Patient_Smoker', 'Patient_Rural_Urban'])
data.head()
data.info()
print(data.ID_Patient_Care_Situation.nunique())     # nunique() gives you the count of unique values in the column

print(data.Patient_ID.nunique())
data.drop(['ID_Patient_Care_Situation'], axis =1, inplace=True)
X = data.drop('Survived_1_year',axis = 1)

y = data['Survived_1_year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LogisticRegression(max_iter = 1000)     # The maximum number of iterations will be 1000. This will help you prevent from convergence warning.

model.fit(X_train,y_train)
pred = model.predict(X_test)
print(f1_score(y_test,pred))
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)

 

forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)



fscore = f1_score(y_test ,y_pred)

fscore
!pip install Boruta
from boruta import BorutaPy
boruta_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)   # initialize the boruta selector

boruta_selector.fit(np.array(X_train), np.array(y_train))       # fitting the boruta selector to get all relavent features. NOTE: BorutaPy accepts numpy arrays only.
print("Selected Features: ", boruta_selector.support_)    # check selected features

 



print("Ranking: ",boruta_selector.ranking_)               # check ranking of features



print("No. of significant features: ", boruta_selector.n_features_)
selected_rfe_features = pd.DataFrame({'Feature':list(X_train.columns),

                                      'Ranking':boruta_selector.ranking_})

selected_rfe_features.sort_values(by='Ranking')
X_important_train = boruta_selector.transform(np.array(X_train))

X_important_test = boruta_selector.transform(np.array(X_test))
# Create a new random forest classifier for the most important features

rf_important = RandomForestClassifier(random_state=1, n_estimators=1000, n_jobs = -1)



# Train the new classifier on the new dataset containing the most important features

rf_important.fit(X_important_train, y_train)
y_important_pred = rf_important.predict(X_important_test)

rf_imp_fscore = f1_score(y_test, y_important_pred)
print(rf_imp_fscore)
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True, False],

    'max_depth': [5, 10, 15],

    'n_estimators': [500, 1000]}
rf = RandomForestClassifier(random_state = 1)



# Grid search cv

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 2, n_jobs = -1, verbose = 2)
grid_search.fit(X_important_train, y_train)
grid_search.best_params_
pred = grid_search.predict(X_important_test)
f1_score(y_test, pred)
import pandas as pd
# Load the data

test_new_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Testing_set_advc.csv")
test_new_data.info()
# take a look how the new test data look like

test_new_data.head()
test_new_data.Treated_with_drugs.value_counts()
test_new_data['Treated_with_drugs'] = test_new_data['Treated_with_drugs'].str.upper()
test_new_data.Patient_Smoker.value_counts()
def smoker(r):

  if (r == "NO") or (r == "NO "):

    return 'NO'

  elif (r == "YES") or (r == "YES ") or (r == "YESS") or (r == "YESS "):

    return 'YES'

  else:

    return 'Cannot say'



test_new_data.Patient_Smoker = test_new_data.Patient_Smoker.apply(smoker)
test_new_data.Patient_Rural_Urban.value_counts()
test_new_data.Patient_mental_condition.value_counts()
test_new_data.isnull().sum()
drugs = test_new_data['Treated_with_drugs'].str.get_dummies(sep=' ') # split all the entries

drugs.head()
test_new_data = pd.concat([test_new_data, drugs], axis=1)     # concat the two dataframes 'drugs' and 'data'

test_new_data = test_new_data.drop('Treated_with_drugs', axis=1)    # dropping the column 'Treated_with_drugs' as its values are splitted into different columns



test_new_data.head()
test_new_data.Patient_Smoker.value_counts()
test_new_data.drop('Patient_mental_condition', axis = 1, inplace=True)
test_new_data = pd.get_dummies(test_new_data, columns=['Patient_Smoker', 'Patient_Rural_Urban'])
test_new_data.head()
test_new_data.info()
test_new_data.drop(['ID_Patient_Care_Situation'], axis =1, inplace=True)
test_new_data.info()
imp_test_features = boruta_selector.transform(np.array(test_new_data))
prediction = grid_search.predict(imp_test_features)