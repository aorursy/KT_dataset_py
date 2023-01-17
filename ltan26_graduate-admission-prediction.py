# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/Admission_Predict.csv")
df.columns=[x.strip() for x in df.columns]
df.head()
df.info()
def boxplot(y_column):
    columns=['University Rating','SOP','LOR','Research']
    fig = plt.figure(figsize=(30, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1,len(columns)+1):
        ax=fig.add_subplot(1, len(columns), i)
        ax=sns.boxplot(x=columns[i-1],y=y_column,data=df)
boxplot('Chance of Admit')
boxplot('GRE Score')
boxplot('TOEFL Score')
boxplot('CGPA')
sns.pairplot(df.drop('Serial No.',axis=1)) #remove 'Serial No' column
data=df.drop('Serial No.',axis=1) #remove 'Serial No.'column

X=data.drop(['Chance of Admit'],axis=1).values
y=data['Chance of Admit'].values

#Select features to keep based on percentile of the highest scores
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
Selector_f = SelectPercentile(f_regression, percentile=25)
Selector_f.fit(X,y)

#get the scores of all the features
name_score=list(zip(data.drop(['Chance of Admit'],axis=1).columns.tolist(),Selector_f.scores_))
name_score_df=pd.DataFrame(data=name_score,columns=['Feat_names','F_scores'])
name_score_df.sort_values('F_scores',ascending=False)
X=data[['CGPA','GRE Score','TOEFL Score']]
y=data['Chance of Admit'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
print("Prediction\n",y_pred)
print("Actual\n",y_test)

print("R_squared Score:",regressor.score(X_test,y_test))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
print("MAE:",mae)

from sklearn.metrics import mean_squared_error
print("RMSE:",mean_squared_error(y_test,y_pred)**0.5)