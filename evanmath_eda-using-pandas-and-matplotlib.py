# import libraries we need for EDA
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import libraries we need for predictions
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import metrics
# importing our file
students_df = pd.read_csv('../input/StudentsPerformance.csv')
# Look at the top 5 rows, uncomment the line below to run the code
#students_df.head()
# Look at the last 5 rows, uncomment the line below to run the code
#students_df.tail()
# Let's take a look on columns, shape and descriptive information of our data set
# uncomment the line below to run the code
#students_df.columns
# Shape of our dataset
# uncomment the line below to run the code
#students_df.shape
students_df.info()
# Summary statistics of our numeric columns of entire dataset
students_df.describe()
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
# Counts on categorical columns
for feature in categorical_features:
    print(feature,':')
    print(students_df[feature].value_counts())
    print('----------------------------')
fig, axes = plt.subplots(3,2, figsize=(12,12))

def get_x_labels(column):
    # helper function to get all xlabels for all axes
    col_dict = dict(students_df[column].value_counts())
    return col_dict.keys()

x_labels = [list(get_x_labels(feature)) for feature in categorical_features]

def get_y_ticks(column):
    # helper function to get all heights for all axes
    return students_df[column].value_counts()

y_ticks = [list(get_y_ticks(feature)) for feature in categorical_features]

for i in range(3):
    for j in range(2):
        if i==1:
            axes[i,j].bar(x_labels[i+j+1], y_ticks[i+j+1])
            axes[i,j].set_frame_on(False)
            axes[i,j].set_xticklabels(x_labels[i+j+1], rotation=45)
            axes[i,j].set_title('{} Counts'.format(categorical_features[i+j+1].capitalize()))
            axes[i,j].minorticks_off()
        elif i==2:
            axes[i,j].bar(x_labels[i+j+1], y_ticks[i+j+1])
            axes[i,j].set_frame_on(False)
            axes[i,j].set_xticklabels(x_labels[i+j+1], rotation=45)
            axes[i,j].set_title('{} Counts'.format(categorical_features[i+j+1].capitalize()))
        else:
            axes[i,j].bar(x_labels[i+j], y_ticks[i+j])
            axes[i,j].set_frame_on(False)
            axes[i,j].set_xticklabels(x_labels[i+j], rotation=45)
            axes[i,j].set_title('{} Counts'.format(categorical_features[i+j].capitalize()))
plt.tight_layout()
plt.show()
#To help you understand better, what we mean by figure and by axes
#below I use a very nice way to clarify those. (I saw it first time 
#in LinkedIn by Ted Petrou).

fig_new, ax = plt.subplots()

fig_new.set_facecolor('tab:cyan') #our paper has cyan color, our axes in white color
ax.set_facecolor('tab:green') #our axes now has green color
fig_new
numeric_features = ['math score', 'reading score', 'writing score']
# First of all let us take a look on 
# the distribution of each numeric column

for feature in numeric_features:
    students_df[feature].plot(kind='hist', bins=20)
    plt.title('{} Distribution'.format(feature))
    plt.show()
# We print all the minimum values for each numeric feature

print('The minimum score for Maths is: {}'.format(students_df['math score'].min()))
print('The minimum score for Reading is: {}'.format(students_df['reading score'].min()))
print('The minimum score for Writing is: {}'.format(students_df['writing score'].min()))
students_df.boxplot(column=numeric_features, by='gender', rot=45, figsize=(15,6), layout=(1,3));
students_df.boxplot(column=numeric_features, by='test preparation course', rot=45, figsize=(15,6), layout=(1,3));
students_df.boxplot(column=numeric_features, by='parental level of education', rot=90, figsize=(15,6), layout=(1,3));
students_df.boxplot(column=numeric_features, by='race/ethnicity', rot=45, figsize=(15,6), layout=(1,3));
students_df.boxplot(column=numeric_features, by='lunch', rot=45, figsize=(15,6), layout=(1,3));
# We are going to split our dataset to smaller,
# one for each category, and compare their statistics
# with the overall statistics.

df_compl = students_df[students_df['test preparation course'] == 'completed']
df_notcompl = students_df[students_df['test preparation course'] == 'none']
# A good way to decide if and how the test preparation course helped,
# is to compare the mean values of our two subsets to the entire dataset
print(students_df.mean() - df_compl.mean())
print(students_df.mean() - df_notcompl.mean())
print(students_df.std() - df_compl.std())
print('--------------')
print(students_df.std() - df_notcompl.std())
df_BD = students_df[students_df['parental level of education'] == "bachelor's degree"]
df_MD = students_df[students_df['parental level of education'] == "master's degree"]
df_sc = students_df[students_df['parental level of education'] == 'some college']
df_AD = students_df[students_df['parental level of education'] == "associate's degree"]
df_hs = students_df[students_df['parental level of education'] == 'high school']
df_shs = students_df[students_df['parental level of education'] == 'some high school']
print(students_df.mean() - df_BD.mean())
print('--------------')
print(students_df.mean() - df_MD.mean())
print('--------------')
print(students_df.mean() - df_sc.mean())
print('--------------')
print(students_df.mean() - df_shs.mean())
print('--------------')
print(students_df.mean() - df_hs.mean())
print('--------------')
print(students_df.mean() - df_AD.mean())
print(students_df.std() - df_BD.std())
print('--------------')
print(students_df.std() - df_MD.std())
print('--------------')
print(students_df.std() - df_sc.std())
print('--------------')
print(students_df.std() - df_shs.std())
print('--------------')
print(students_df.std() - df_hs.std())
print('--------------')
print(students_df.std() - df_AD.std())
df_A = students_df[students_df['race/ethnicity'] == 'group A']
df_B = students_df[students_df['race/ethnicity'] == 'group B']
df_C = students_df[students_df['race/ethnicity'] == 'group C']
df_D = students_df[students_df['race/ethnicity'] == 'group D']
df_E = students_df[students_df['race/ethnicity'] == 'group E']
print(students_df.mean() - df_A.mean())
print('--------------')
print(students_df.mean() - df_B.mean())
print('--------------')
print(students_df.mean() - df_C.mean())
print('--------------')
print(students_df.mean() - df_D.mean())
print('--------------')
print(students_df.mean() - df_E.mean())
print(students_df.std() - df_A.std())
print('--------------')
print(students_df.std() - df_B.std())
print('--------------')
print(students_df.std() - df_C.std())
print('--------------')
print(students_df.std() - df_D.std())
print('--------------')
print(students_df.std() - df_E.std())
students_dummies = pd.get_dummies(students_df, drop_first=True, columns=categorical_features)
students_dummies.head()
features = ['reading score', 'writing score', 'gender_male',
       'race/ethnicity_group B', 'race/ethnicity_group C',
       'race/ethnicity_group D', 'race/ethnicity_group E',
       'parental level of education_bachelor\'s degree',
       'parental level of education_high school',
       'parental level of education_master\'s degree',
       'parental level of education_some college',
       'parental level of education_some high school', 'lunch_standard',
       'test preparation course_none']
target = ['math score']
class ColumnLinearRegression(BaseEstimator, RegressorMixin):
     # columns are a "Hyperparameter" for our estimator, 
     # so we have to pass it in, in the __inti__ method,
     # we need to keep track of the columns, so we have to save them
        
    def __init__(self, columns):
        if not isinstance(columns, list):
            raise ValueError("columns must be a list")
        self.columns= columns
        self.lr = LinearRegression()
        
    def _select(self, X):
        return X[self.columns]
        
    def fit(self, X, y):
        self.lr.fit(self._select(X), y)
        return self
    
    def predict(self, X):
        return self.lr.predict(self._select(X))
def r2_adj(y_t, X_t, feat, pred):
    #this function calculates the r^2 adjusted
    r2 = metrics.r2_score(y_t, pred)
    n = len(X_t)
    p = len(feat)
    return 1-((1-r2)*(n-1)/(n-p-1))
feat_list = []
new_dict = {}
i=0
r2_adj_max = 0
best_model = None
for feature in features:
    feat_list.append(feature)
    clr = ColumnLinearRegression(feat_list)
    X_train, X_test, y_train, y_test = train_test_split(students_dummies[feat_list], students_dummies[target], 
                                                    test_size=0.3, random_state=42)
    clr.fit(X_train, y_train)
    predictions = clr.predict(X_test)
    variables = len(clr.columns)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)
    r2_adjusted = r2_adj(y_test, X_test, clr.columns, predictions)
    if r2_adjusted > r2_adj_max:
        best_model = clr
        r2_adj_max = r2_adjusted
        new_dict[i] = {'var': variables, 
                       'MSE': mse,
                       'R^2': r2, 
                       'R^2_adjusted': r2_adj_max}
    else:
        feat_list.remove(feat_list[-1])
    i+=1
print(best_model.columns)    
df = pd.DataFrame(new_dict)
df.head()