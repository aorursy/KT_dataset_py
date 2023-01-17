# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading the data
fugadf = pd.read_csv("../input/fuga_train.csv")
fugadf.head()
fugadf.info()
# Checking the percentage of missing values
round(100*(fugadf.isnull().sum()/len(fugadf.index)),2)
### Data Preparation
#converting yes to 1 and no to 0
fugadf['Int\'l Plan'] = fugadf['Int\'l Plan'].map({'yes': 1, 'no':0})
fugadf['VMail Plan'] = fugadf['VMail Plan'].map({'yes': 1, 'no':0})
fugadf.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
fugadf['State']=le.fit_transform(fugadf['State'])
fugadf.head()
fugadf = fugadf.drop(['Phone'],1)
fugadf.head()
fugadf.info()
# Checking outliers at 25%,50%,75%,90%,95% and 99%
fugadf.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Adding up the missing values (column-wise)
fugadf.isnull().sum()
# Normalising continuous features
df = fugadf[['Day Charge','Eve Charge','Night Charge','Intl Charge']]
normalized_df=(df-df.mean())/df.std()
normalized_df.head()
fugadf = fugadf.drop(['Day Charge','Eve Charge','Night Charge','Intl Charge'],1)
fugadf = pd.concat([fugadf,normalized_df],axis=1)
###Checking the Churn Rate
churn = (sum(fugadf['fuga'])/len(fugadf['fuga'].index))*100
churn
###Splitting Data into Training and Test Sets
from sklearn.model_selection import train_test_split
#Putting feature varaible to X
X = fugadf.drop(['fuga'],axis=1)
#Putting response varaiable to y
y = fugadf['fuga']
y.head()
#Splitting the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
np.asarray(X_train)
X_train.head()
###Running Your First Training Model
import statsmodels.api as sm
#Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
###Correlation Matrix
#Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Let's see the correlation matrix
plt.figure(figsize = (20,10))  #Size of the figure
sns.heatmap(fugadf.corr(),annot = True)
X_train = X_train.drop(['VMail Plan','VMail Message','Day Mins','Day Calls','Eve Mins','Eve Calls','Night Mins','Night Calls','Intl Mins','Intl Calls'],1)
X_test = X_test.drop(['VMail Plan','VMail Message','Day Mins','Day Calls','Eve Mins','Eve Calls','Night Mins','Night Calls','Intl Mins','Intl Calls'],1)
import statsmodels.api as sm
logm2 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm2.fit().summary()
X_train = X_train.drop(['Account Length','Area Code','Int\'l Plan'],1)
X_test = X_test.drop(['Account Length','Area Code','Int\'l Plan'],1)
X_train.info()
###Feature Selection using RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 7) #running RFE with 7 variables as output
rfe = rfe.fit(X,y)
print(rfe.support_)  #Printing the boolean results
print(rfe.ranking_)
plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
#Re-Running the Model
import statsmodels.api as sm
logm2 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm2.fit().summary()
X_train = X_train.drop(['State'],1)
X_test = X_test.drop(['State'],1)
#Re-Running the Model
import statsmodels.api as sm
logm3 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm3.fit().summary()
X_train = X_train.drop(['Night Charge'],1)
X_test = X_test.drop(['Night Charge'],1)
#Re-Running the Model
import statsmodels.api as sm
logm4 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm4.fit().summary()
#UDF for calculatig vif value
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
            y=x_vars[xvar_names[i]]
            x=x_vars[xvar_names.drop(xvar_names[i])]
            rsq=sm.OLS(y,x).fit().rsquared
            vif=round(1/(1-rsq),2)
            vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)   
fugadf.columns
X_train.columns
fugadf = fugadf.drop(['State', 'Account Length', 'Area Code', 'Int\'l Plan',
       'VMail Plan', 'VMail Message', 'Day Mins', 'Day Calls', 'Eve Mins',
       'Eve Calls', 'Night Mins', 'Night Calls', 'Intl Mins', 'Intl Calls','Intl Charge'
 ],1)
#Calculating Vif value,let's do later
#vif_cal(input_data=fugadf,dependent_col='fuga')
#Let's run the model using the selected variables
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logsk = LogisticRegression()
logsk.fit(X_train, y_train)
### Making Predictions
# Predicted probabilities
y_pred = logsk.predict_proba(X_test)
# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_pred)
# Converting to column dataframe
y_pred_1 = y_pred_df.iloc[:,[1]]
y_pred_1.head()
#Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
#Putting Index to index
y_test_df['index'] = y_test_df.index
# Removing index for both dataframes to append them side by side
y_pred_1.reset_index(drop=True, inplace=True)
y_pred_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)
# Renaming the column
y_pred_final= y_pred_final.rename(columns={ 1 : 'Fuga_Prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex_axis(['Index','fuga','Fuga_Prob'], axis=1)
y_pred_final.head()
# Creating new column 'predicted' with 1 if Churn_Prob>0.5 else 0
y_pred_final['predicted'] = y_pred_final.Fuga_Prob.map( lambda x: 1 if x > 0.5 else 0)
y_pred_final.head()