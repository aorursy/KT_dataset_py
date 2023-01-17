#importing all the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder



warnings.filterwarnings("ignore")

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
#importing the data

df = pd.read_csv("/kaggle/input/student-grade-prediction/student-mat.csv")

df.info()
#creating a correlation matrix to understand the correlation between the features

df.corr()
#using seaborn heatmap to analyse the correlation graphically

import seaborn as sns

sns.heatmap(df.corr(),xticklabels=df.corr().columns,yticklabels=df.corr().columns)
#checking 

df.describe()
#checking data types of column values and presence of null values(if any)

df.info()
#distribution of female to male in the given data

total_count=df['sex'].value_counts()

plt.pie(x=total_count,colors=['lightskyblue','red'], labels=['Male','Female'],autopct='%1.1f%%',shadow=True)

plt.show()
#plot on traveltime vs G3 based on sex 

g=sns.boxplot(x='traveltime',y='G3',data=df,hue='sex')

g.set(xticklabels=['near','moderate','far','very far'])

plt.show()
#Father's education vs final score

sns.boxplot(x='Fedu',y='G3',data=df)
#countplot of age based on sex

sns.countplot('age',hue='sex',data=df)
#failures vs final score

g=sns.swarmplot(x='failures',y='G3',data=df)

g.set(xticklabels=['very low','low','moderate','high','very high'])

plt.show()
#freetime vs final Score

g=sns.boxplot(x='freetime',y='G3',data=df)

g.set(xticklabels=['very low','low','moderate','high','very high'])

plt.show()
#outing vs final score

g=sns.swarmplot(x='goout',y='G3',data=df)

g.set(xticklabels=['very low','low','moderate','high','very high'])

plt.show()
sns.barplot(x='age',y='absences',data=df)
#mother's job plot

sns.countplot('Mjob',data=df)
#plot of age vs final score based on pursuing higher education

sns.barplot(x='age',y='G3',data=df,hue='higher')
#health vs final score barplot

g=sns.boxplot(x='health',y='G3',data=df)

g.set(xticklabels=['worst','low','moderate','good','excellent'])

plt.show()
#countplot of age vs final score based on being paid

sns.barplot(x='age',y='G3',data=df,hue='paid')
#boxplot of studytime vs final score based on internet usage

g=sns.boxplot(x='studytime',y='G3',hue='internet',data=df)

g.set(xticklabels=['very low','low','high','very high'])

plt.show()
g=sns.boxplot(x='famrel',y='Walc',data=df)

g.set(xticklabels=['very low','low','moderate','high','very high'])

plt.show
#age vs final score based on romantic life

sns.barplot(x='age',y='G3',hue='romantic',data=df)
g=sns.countplot(x='goout',hue='romantic',data=df)

g.set(xticklabels=['very low','low','moderate','high','very high'])

plt.show()
#extracting major features only

df_features=df[['G1','G2','Medu','Fedu','studytime']]

df_features.head()

df_label=df[['G3']]
#getting values as numpy arrays for splitting

X=df_features.values

y=df_label.values
#splitting the X and y values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#performing the regression on various models and storing the scores

scores={}

def classifier():

    dict_models={

        'Linear Regression':LinearRegression(),

        'Support Vector Machine':SVR(kernel='linear',degree=1),

        'Decision Tree':DecisionTreeRegressor(criterion='mae'),

        'Random Forest':RandomForestRegressor(n_estimators=150,criterion='mse',verbose=0)

    }

    X_train.shape

    y_train.shape

    

    for key,value in dict_models.items():

        regression=value.fit(X_train,y_train)

        score=cross_val_score(regression,X,y,scoring='neg_mean_squared_error')

        score=np.sqrt(-score.mean())

        scores[key]=score

        print(

            f'Model Name: {key},RMSE score: {(score.mean())}')
classifier()
#scaling the values(although it doesn't change the rmse)

from sklearn.preprocessing import MinMaxScaler

sc_s=MinMaxScaler()

X_train=sc_s.fit_transform(X_train)

X_test=sc_s.transform(X_test)
classifier()
#labelling the categorical column values of the dataframe

for column in df.columns:

    if df[column].dtype=='object':

        df[column]=LabelEncoder().fit_transform(df[column])
#extracting all the features this time for evaluation

#Only Random Forest and Decision Tree are used because others require one-hot-encoding, which we will cover

#in future notebooks

X=df.iloc[:,:-1].values

y=df.iloc[:,-1].values



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#Using Random Forest Regressor

new=RandomForestRegressor()

model=new.fit(X_train,y_train)

score=cross_val_score(model,X,y,scoring='neg_mean_squared_error')

score=np.sqrt(-score.mean())

scores['Random Forest Labled']=score
#Using Decision Tree Regressor

test=DecisionTreeRegressor()

model=new.fit(X_train,y_train)

score=cross_val_score(model,X,y,scoring='neg_mean_squared_error')

score=np.sqrt(-score.mean())

scores['Decision Tree Labled']=score
#Converting scores to datafram

scores=(pd.Series(scores)).to_frame()
#renaming the column names

scores=scores.rename(columns={0:'RMSE Error'})

scores
#plotting the scores of each model for better comparison

scores.plot(kind='bar')