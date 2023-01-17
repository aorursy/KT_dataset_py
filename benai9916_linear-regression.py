import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

placement = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
placement.head()
placement.shape
from pandas_profiling import ProfileReport
profile = ProfileReport(placement);
profile.to_widgets()
df = pd.DataFrame.drop(placement,columns=["sl_no","ssc_b","hsc_b"])
df_new = df.groupby(by  = 'status').mean()
df_new
matrix = placement.corr()
plt.figure(figsize=(9,7))

#plot heat map
g=sns.heatmap(matrix,annot=True)
plt.figure(figsize=(12,8))
plt.ylim([200000,450000])
sns.regplot(x="ssc_p",y="salary",data=placement)
sns.regplot(x="hsc_p",y="salary",data=placement)
sns.regplot(x="mba_p",y="salary",data=placement)
sns.regplot(x="etest_p",y="salary",data=placement)
plt.legend(["ssc percentage", "hsc percentage", "MBA", "E-test"])
plt.ylabel("mba percentage")
plt.xlabel("Percentage %")
plt.show()
for cols in ["hsc_s","ssc_b","hsc_b"]:
    sns.countplot(x="ssc_b",hue="gender",data=placement);
    plt.ylabel("No. of students");
    plt.xlabel(cols);
    plt.show()
sns.catplot(x="ssc_b",y="ssc_p", hue='gender', data=placement, kind='boxen')
plt.ylabel("percentage")
plt.xlabel("boards")
sns.catplot(x="workex",hue="degree_t",data=placement, kind="count")
plt.ylabel("No. of students")
plt.xlabel("work exp in different degrees");
sns.catplot(x="degree_t",hue="workex",data=placement, kind="count")
plt.ylabel("No. of students")
plt.xlabel("work exp in different degrees")
sns.catplot(y="salary",x="gender",data=placement, kind="box", hue="specialisation" );
placement['degree_t'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
placement['status'].value_counts().plot.pie(autopct='%1.1f%%')
placement.hist(bins = 20, figsize=(10,10), color= 'green');
import plotly.express as px
dfc=pd.DataFrame(placement.groupby(['gender','specialisation','status'])['sl_no'].count()).rename(columns={'sl_no': 'no. of students'}).reset_index()

fig = px.sunburst(dfc, path=['gender','status','specialisation'], values='no. of students')
fig.update_layout(title="Placement % of mba in each specialisation by gender ",title_x=0.5)
fig.show()

category =  [cols for cols in df.columns if placement[cols].dtype == 'O']
df.loc[:, category] = df.loc[:, category].apply(LabelEncoder().fit_transform)

df.head()
placement.isnull().sum()
placement.fillna(0, inplace=True)
df_reg = df.copy()
df_reg.dropna(inplace=True)
df_reg = df_reg[df_reg["salary"]<350000.0]
df_reg.info()
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
#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the salary
y_pred_m = regressor.predict(X_test)
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
print("R2 score " + str(r2_score(y_test, y_pred_m)))
MAE = mean_absolute_error(y_test, y_pred_m)
print("MAE "+str(MAE))
