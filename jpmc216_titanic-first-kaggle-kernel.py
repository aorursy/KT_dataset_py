# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
train_data.info()
# PassengerId Seems to be unique ID 

# Survived is the target variable

# That leaves us with Pclass, Age, SibSp, Parch, Fare

num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    print(train_data[col].value_counts())
    print('\n\n')
# Pclass seem to be more of categorical variable indicated in number field. That leaves us with 'Age', 'SibSp', 'Parch', 'Fare' numerical fields. 
from scipy import stats
from collections import Counter

def detect_outliers_numerical_features(df, num_features):
    """
    Take list of numerical features and populate the list of indices corresponding 
    to rows containing more than 2 outliers for each row from the given dataframe.
    """
    
    outlier_indices = []
    
    for feature in num_features:
        ser = df[feature]
        
        # First quartile (Q1)
        Q1 = np.nanpercentile(ser, 25) 

        # Third quartile (Q3) 
        Q3 = np.nanpercentile(ser, 75) 

        # Interquaritle range (IQR) 
        IQR = Q3 - Q1
        
        print('Feature is :{} and IQR is :{}'.format(feature,IQR))
        # The default is the true IQR: (25, 75)
        # IQR = stats.iqr(ser, nan_policy='omit') 
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_col_indices = df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step )].index
        
        print(outlier_col_indices)
        print('\n\n')
        # Extend will keep adding the current list to the end of original list
        outlier_indices.extend(outlier_col_indices)
     
    # select observations containing more than 2 outliers ( v > 2, means more than twice the occurence)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
    
    return multiple_outliers   
        

train_data_outlier_indices = detect_outliers_numerical_features(train_data, ['Age', 'SibSp', 'Parch', 'Fare'])

train_data.loc[train_data_outlier_indices] 
# One way to handle outliers is ignore the records

# Drop columns and Reset Index - Use the drop parameter to avoid the old index being added as a column
train_data = train_data.drop(train_data_outlier_indices, axis=0).reset_index(drop=True)
# Combine Train and Test data sets

train_data['type'] = 'train'
test_data['type'] = 'test'

total_df =  pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)
# Look for train data metrics 
train_data.info()
train_data.isnull().sum()
# Fill empty/missing values with nan
total_df = total_df.fillna(np.nan)

# Look for counts around nan entries
total_df.isnull().sum()
# Summary of the training data (The below will only cover Numerical fields)

train_data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')


# Correlation Matrix between different Numerical fields 

corr = sns.heatmap(train_data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = plt.cm.RdBu)
bar_sibsp = sns.catplot(x="SibSp",y="Survived", data=train_data, kind="bar")
bar_sibsp = bar_sibsp.set_ylabels("Survived")
bar_parch = sns.factorplot(x="Parch",y="Survived", data=train_data, kind="bar")
bar_parch = bar_parch.set_ylabels("Survived")
# Age has diverse values and hence we may have to draw a distribution plot

dist_age = sns.FacetGrid(train_data, col='Survived', height=5, xlim=(0, 80))
dist_age = dist_age.map(sns.distplot, "Age")
# Lets plot kde for Age vs Survived/Non-Surived

non_survived = train_data["Age"][(train_data["Survived"] == 0) & (train_data["Age"].notnull())]
survived = train_data["Age"][(train_data["Survived"] == 1) & (train_data["Age"].notnull())]
                                 
kde_age = sns.kdeplot(non_survived, color="Red", shade = True)
kde_age = sns.kdeplot(survived, ax = kde_age, color="Blue", shade= True)
kde_age.set_xlabel("Age")
kde_age.set_ylabel("Frequency")
kde_age = kde_age.legend(["Not Survived","Survived"])
# Distribution plot for Fare

dist_fare = sns.distplot(train_data["Fare"], color="r", label="Skewness : %.2f"%(train_data["Fare"].skew()))
dist_fare = dist_fare.legend(loc="best")
# Log function will tranform Fare values to reduce the skewness distribution

total_df["Fare"] = total_df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# If we plot the distribution plot now to check for skewness. 

dist_fare = sns.distplot(total_df["Fare"], color="r", label="Skewness : %.2f"%(total_df["Fare"].skew()))
dist_fare = dist_fare.legend(loc="best")
train_data.select_dtypes(exclude=['int64','float64']).info()
bar_s = sns.catplot(x="Sex",y="Survived", data=train_data, kind="bar")
bar_s = bar_s.set_ylabels("Survived")
bar_pclass = sns.catplot(x="Pclass",y="Survived", data=train_data, kind="bar")
bar_pclass = bar_pclass.set_ylabels("Survived")
bar_pclass = sns.catplot(x="Pclass",y="Survived", hue='Sex', data=train_data, kind="bar")
bar_pclass = bar_pclass.set_ylabels("Survived")
bar_embarked = sns.catplot(x="Embarked",y="Survived", data=train_data, kind="bar")
bar_embarked = bar_embarked.set_ylabels("Survived")
bar_em_p = sns.catplot(x="Embarked", y="Survived", hue="Pclass", data=train_data, kind="bar")
bar_em_p = bar_em_p.set_ylabels("Survived")