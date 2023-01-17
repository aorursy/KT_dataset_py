# Import the necessary libraries
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns



# Model development libraries
from sklearn.linear_model import BayesianRidge
from fancyimpute import IterativeImputer as MICE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

%matplotlib inline
# Load the data
heart = pd.read_csv('../input/heart-disease-uci/heart.csv')
heart.head()
heart.describe()
heart.info()
# There are no missing values in the data

# Pandas Profiling to show us an analysis of the data
ProfileReport(heart)
# From the above let's investigate the Slope and the Cp columns in order to deal with the zeros
heart['slope'].value_counts()
heart['cp'].value_counts()
# Drop the duplicate column
heart.drop_duplicates(inplace = True)
# The columns are ok. Let's plot a histogram for all the features
heart.hist(figsize=(20,20))
# Correlation of the data

x = heart.iloc[:,:]
y = heart.iloc[:, 0]
corrmat = heart.corr()
top_features = corrmat.index
plt.figure(figsize=(10,10))
matrix =sns.heatmap(heart[top_features].corr(),annot=True,cmap="RdYlGn")
heart.drop('fbs', axis = 1 , inplace = True)
# Let's check for anomalies

sns.boxplot(data = heart)


#For thalach, we delete the outlier

#For cholestrol, we group the data

heart.loc[heart['chol']<220, 'chol']=0

heart.loc[heart['chol']>220, 'chol']=1

# We do the same for trestbps

heart.loc[heart['trestbps']<=90, 'trestbps']=0
heart.loc[(heart['trestbps']>90) & (heart['trestbps']<=120), 'trestbps']=1
heart.loc[(heart['trestbps']<=140) & (heart['trestbps']>120) , 'trestbps']=2
heart.loc[heart['trestbps']>140 , 'trestbps']=3


sns.boxplot(data = heart)
# Thalach
# heart = heart[(z < 3).all(axis=1)]
# Thalach

#ex = heart['thalach'].values
#ex =ex.reshape((-1,1))
#from sklearn import preprocessing
#mm_scaler = preprocessing.MinMaxScaler()

#ex_scaled = mm_scaler.fit_transform(ex)
#heart['thalach'] = pd.DataFrame(ex_scaled)

#Age

#why = heart['age'].values
#why = why.reshape((-1,1))
#why_scaled = mm_scaler.fit_transform(why)
#heart['age'] = pd.DataFrame(why_scaled)

#thalach

normalized_th=(heart['thalach']-heart['thalach'].mean())/heart['thalach'].std()
heart['thalach']=normalized_th

#Age
normalized_age=(heart['age']-heart['age'].mean())/heart['age'].std()
heart['age']=normalized_age
heart.head()
heart.isnull().sum()
#Missing value in thalach
heart['thalach'].replace(np.nan,heart['thalach'].median())

#Missing value in age
heart['age'].fillna(heart['age'].median())
heart.isnull().sum()
x = heart.drop('target', axis = 1)
y = heart['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)
#Logistic Regression
#Call the model
lr = LogisticRegression()

#Train
lr.fit(x_train,y_train)

#Predict
lr_pred = lr.predict(x_test)

# Check accuracy

print(classification_report(y_test,lr_pred))

#random Forest
#Call the model
rf = RandomForestClassifier()

#Train
rf.fit(x_train,y_train)

#Predict
rf_pred = rf.predict(x_test)

# Check accuracy

print(classification_report(y_test,rf_pred))
