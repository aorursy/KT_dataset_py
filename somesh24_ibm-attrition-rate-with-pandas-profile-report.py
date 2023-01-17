#import libraries



import os

import numpy as np

import pandas as pd

from collections import Counter

from sklearn.pipeline import Pipeline



#import visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt



#import preprocessing libraries

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score



#import dimensionality reduction libraries

from sklearn.decomposition import PCA



#import algorithm libraries

from sklearn.linear_model import LogisticRegression, SGDClassifier

import xgboost as xgb

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



#import error metrics

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, accuracy_score

#get the path of the dataset



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(data.shape)

data.head()
data.describe()
data.info()
from pandas_profiling import ProfileReport



profile = ProfileReport(data, title='Pandas Profiling Report')
profile.to_widgets()
# Create a function to detect outliers

def detect_outliers(df,n,features):

    # create a array to store the values of outliers

    outliers_index = []

    

    # calculate the Q1 and Q3. Then we can calculate IQR

    for col in features:

        Q1 = df[col].quantile(0.25)

        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        

        #outlier step

        outlier_step = 1.5 * IQR

        

        # Determine the indices of outliers in a particular column

        outlier_list_col = df[(df[col] < (Q1 - outlier_step))|(df[col] > (Q3 + outlier_step))].index

        

        # append the found outlier indices for col to the list of outlier indices

        outliers_index.extend(outlier_list_col)

        

        # select observations containing more than 2 outliers

        outliers_index = Counter(outliers_index)

        multiple_outliers = list( k for k, v in outliers_index.items() if v > n )

        

        return multiple_outliers
detect_outliers(data,2,['DailyRate', 'DistanceFromHome','YearsAtCompany','YearsWithCurrManager',

                        'EmployeeNumber','JobInvolvement','PercentSalaryHike'])
# since our output is categorical, we will try with chi-squared (chi²) statistical test first



from sklearn.feature_selection import SelectKBest, chi2,f_classif



# Get list of categorical variables

s = (data.dtypes == 'object')

object_cols = list(s[s].index)



# apply label encoder to all the categorical columns

data_copy = data.copy()

le = LabelEncoder()

for col in object_cols:

    data_copy[col] = le.fit_transform(data_copy[col].astype(str))

   

 # get the x and y value

xl,yl = data_copy.drop(columns = 'Attrition'), data_copy['Attrition']

    

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

dfit = bestfeatures.fit(xl,yl)

df_scores = pd.DataFrame(dfit.scores_)

df_cols = pd.DataFrame(xl.columns)

# now we will concatenate the 2 dataframes

feature_importance = pd.concat([df_cols, df_scores],axis=1)

feature_importance.columns = ['Feature','Score']

print(feature_importance.nlargest(20,'Score'))
# now we will check with ANOVA test. If the features are quantitative, compute the ANOVA F-value 

# between each feature and the target vector.



#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=f_classif, k=10)

dfit = bestfeatures.fit(xl,yl)

df_scores = pd.DataFrame(dfit.scores_)

df_cols = pd.DataFrame(xl.columns)

# now we will concatenate the 2 dataframes

feature_importance = pd.concat([df_cols, df_scores],axis=1)

feature_importance.columns = ['Feature','Score']

print(feature_importance.nlargest(10,'Score'))
# Get list of categorical columns

s = (data.dtypes == 'object')

object_cols = list(s[s].index)



# apply label encoder to all the categorical columns

data_copy = data.copy()

le = LabelEncoder()

for col in object_cols:

    data_copy[col] = le.fit_transform(data_copy[col].astype(str))

    

 # get the x and y value

x,y = data_copy.drop(columns = 'Attrition'), data_copy['Attrition']    
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state= 142)

for train_index, val_index in skf.split(x,y):

    train_x, val_x = x.iloc[train_index], x.iloc[val_index]

    train_y, val_y = y.iloc[train_index], y.iloc[val_index]

    

train_x.shape, val_x.shape    
dtc_model = DecisionTreeClassifier(criterion="entropy", max_depth=67)



# Train Decision Tree Classifer

dtc_model.fit(train_x,train_y)



#Predict the response for validation dataset

y_pred = dtc_model.predict(val_x)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(val_y, y_pred))
#create a function to get the importance of features



def feat_imp(model, data, text):

    importances = model.feature_importances_

    

    #sort features in decreasing order of their importance

    indices = np.argsort(importances)[::-1]

    

    # Rearrange feature names so they match the sorted feature importances

    names = [data.columns[i] for i in indices]

    

    # Create plot

    plt.figure(figsize=(14,6))



    # Create plot title

    plt.title(text)



    # Add bars

    plt.bar(range(train_x.shape[1]), importances[indices])



    # Add feature names as x-axis labels

    plt.xticks(range(train_x.shape[1]), names, rotation=90)



    # Show plot

    plt.show()
feat_imp(dtc_model,data, "Feature Importance for Decision Tree")
rfc_model = RandomForestClassifier()



rfc_model.fit(train_x,train_y)



#Predict the response for validation dataset

y_pred = rfc_model.predict(val_x)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(val_y, y_pred))
feat_imp(rfc_model,data, "Feature Importance for Random Forest Classifier")
svc_model = SVC(kernel='linear')



svc_model.fit(train_x,train_y)



#Predict the response for validation dataset

y_pred = svc_model.predict(val_x)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(val_y, y_pred))
knn_model = KNeighborsClassifier()



knn_model.fit(train_x,train_y)



#Predict the response for validation dataset

y_pred = knn_model.predict(val_x)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(val_y, y_pred))
nb_model = GaussianNB()



nb_model.fit(train_x,train_y)



#Predict the response for validation dataset

y_pred = nb_model.predict(val_x)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(val_y, y_pred))