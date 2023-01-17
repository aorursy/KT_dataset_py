# Data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from sklearn.impute import SimpleImputer

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Machine learning
import sklearn
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test = test.fillna(np.nan)
test.head()
# Plot graphic of missing values
test.isnull().sum()
# List number of missing values by feature
train.isnull().sum()
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp, Parch, and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop]
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]

# One-hot encode data (turn categorical features into seperate binary features)
X_train = pd.get_dummies(train[features])
y_train = train["Survived"]
X_test = pd.get_dummies(test[features])


my_imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))

# Imputation removed column names; put them back
X_train_imputed.columns = X_train.columns
X_test_imputed.columns = X_test.columns


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train_imputed, y_train)
predictions = model.predict(X_test_imputed)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission2.csv', index=False)
print("Your submission was successfully saved!")