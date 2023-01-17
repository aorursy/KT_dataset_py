import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor





%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataframe = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

dataframe.head()
from pandas_profiling import ProfileReport

profile = ProfileReport(dataframe);
profile.to_widgets()
df = pd.DataFrame.drop(dataframe,columns=["sl_no","ssc_b","hsc_b"])

df_new = df.groupby(by  = 'status').mean()

df_new
matrix = dataframe.corr()

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(matrix,annot=True,cmap="YlGn_r")
plt.figure(figsize=(12,8))

plt.ylim([200000,450000])

sns.regplot(x="ssc_p",y="salary",data=dataframe)

sns.regplot(x="hsc_p",y="salary",data=dataframe)

sns.regplot(x="mba_p",y="salary",data=dataframe)

sns.regplot(x="etest_p",y="salary",data=dataframe)

plt.legend(["ssc percentage", "hsc percentage", "MBA", "E-test"])

plt.ylabel("mba percentage")

plt.xlabel("Percentage %")

plt.show()
sns.catplot(x="ssc_b",hue="gender",data=dataframe, kind="count",);

plt.ylabel("No. of students");

plt.xlabel("senior secondary");

sns.catplot(x="hsc_b",hue="gender",data=dataframe, kind="count");

plt.ylabel("No. of students");

plt.xlabel("higher senior secondary");

sns.catplot(x="hsc_s",hue="gender",data=dataframe, kind="count");

plt.ylabel("No. of students");

plt.xlabel("prefered subjects");
sns.catplot(x="ssc_b",y="ssc_p",hue="gender",data=dataframe,kind="boxen");

plt.ylabel("percentage");

plt.xlabel("boards");
sns.catplot(x="workex",hue="degree_t",data=dataframe, kind="count");

plt.ylabel("No. of students");

plt.xlabel("work exp in different degrees");

sns.catplot(x="degree_t",hue="workex",data=dataframe, kind="count");

plt.ylabel("No. of students");

plt.xlabel("work exp in different degrees");
sns.catplot(y="salary",x="gender",data=dataframe, kind="box", hue="specialisation" );
df1 = pd.DataFrame(dataframe['degree_t'].value_counts(normalize=True))

plot = df1.plot.pie(y='degree_t', autopct='%1.1f%%', figsize=(5, 5))
df2 = pd.DataFrame(dataframe['specialisation'].value_counts(normalize=True))

plot = df2.plot.pie(y='specialisation', autopct='%1.1f%%', figsize=(5, 5))
df3 = pd.DataFrame(dataframe['status'].value_counts(normalize=True))

plot = df3.plot.pie(y='status', autopct='%1.1f%%', figsize=(5, 5))
dataframe.hist(bins = 30, figsize=(10,10), color= 'orange');
import plotly.express as px

dfc=pd.DataFrame(dataframe.groupby(['gender','specialisation','status'])['sl_no'].count()).rename(columns={'sl_no': 'no. of students'}).reset_index()



fig = px.sunburst(dfc, path=['gender','status','specialisation'], values='no. of students')

fig.update_layout(title="Placement % of mba in each specialisation by gender ",title_x=0.5)

fig.show()
df["degree_t"] = df["degree_t"].astype('category')

df["workex"] = df["workex"].astype('category')

df["specialisation"] = df["specialisation"].astype('category')

df["status"] = df["status"].astype('category')

df["gender"] = df["gender"].astype('category')

df["hsc_s"] = df["hsc_s"].astype('category')

df.dtypes

df["workex"] = df["workex"].cat.codes

df["gender"] = df["gender"].cat.codes

df["degree_t"] = df["degree_t"].cat.codes

df["specialisation"] = df["specialisation"].cat.codes

df["status"] = df["status"].cat.codes

df["hsc_s"] = df["hsc_s"].cat.codes

df.tail()
df_class = df.copy()

X = df_class.iloc[:,0:-2].values

y = df_class.iloc[:,-2].values
#Split the dataset for training

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=0)
#Train the model

#from sklearn.linear_model import LogisticRegression

lg_classifier = LogisticRegression(random_state=0,max_iter=1000)

lg_classifier.fit(X_train, y_train)



#Predict the test cases

y_pred_lgclass = lg_classifier.predict(X_test)
#Train the model

#from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=1000,criterion="entropy")

rf_classifier.fit(X_train, y_train)



#Predict the test cases

y_pred_rfclass = rf_classifier.predict(X_test)
#from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred_rfclass)

print(cm)

print("random forest accuracy: {:2.2f}%" .format(accuracy_score(y_test, y_pred_rfclass) * 100) )
cm = confusion_matrix(y_test, y_pred_lgclass)

print(cm)

print("Logistic regressor accuracy: {:2.2f}%" .format(accuracy_score(y_test, y_pred_lgclass)*100) )
df_reg = df.copy()
df_reg.dropna(inplace=True)

df_reg = df_reg[df_reg["salary"]<350000.0]
#PDF of Salary

sns.kdeplot(df["salary"])

plt.legend(["before"])

plt.show()
sns.kdeplot(df_reg["salary"])

plt.legend(["after"])
#select the features of regression model

X = df_reg.iloc[:,:-2].values

y = df_reg.iloc[:,-1].values



#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
from statsmodels.api import OLS

summ=OLS(y_train,X_train).fit()

summ.summary()
df_reg = pd.DataFrame.drop(df_reg,columns=["degree_p","ssc_p","specialisation","workex"])



#select the features of regression model

X = df_reg.iloc[:,:-2].values

y = df_reg.iloc[:,-1].values



#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)



summ=OLS(y_train,X_train).fit()

summ.summary()
#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#Predict the salary

y_pred_m = regressor.predict(X_test)
#from sklearn.ensemble import RandomForestRegressor

rfregressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

rfregressor.fit(X_train, y_train)



#Predict the salary

y_pred_r = rfregressor.predict(X_test)
from sklearn.metrics import r2_score, accuracy_score

print("R2 score")

print("multiple regressor " + str(r2_score(y_test, y_pred_m)))

print("random forest "+ str(r2_score(y_test, y_pred_r)))
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from math import sqrt

print("Mean Absolute error")

MAE = mean_absolute_error(y_test, y_pred_m)

print("Multiple linear regressor "+str(MAE))

MAE = mean_absolute_error(y_test, y_pred_r)

print("Random forest regressor "+ str(MAE))
print("regression coeff:" + str(regressor.coef_))

print("regression intercept: " + str(regressor.intercept_))