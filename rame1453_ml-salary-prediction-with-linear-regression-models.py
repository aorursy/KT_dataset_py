# First of all import all libraries I'll need.



import warnings

warnings.simplefilter(action='ignore')

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.preprocessing import RobustScaler
# I'll use Hitter dataset for this project.



df = pd.read_csv("../input/hitters/Hitters.csv")
df.head()
# You can see all columns detail above.



df.info()
# Let's look statistic info



df.describe().T
#Any NaN data?



df.isnull().sum()
#Correlation values more than 0.5 between features



correlation_matrix = df.corr().round(2)

threshold=0.75

filtre=np.abs(correlation_matrix['Salary'])>0.50

corr_features=correlation_matrix.columns[filtre].tolist()

sns.clustermap(df[corr_features].corr(),annot=True,fmt=".2f")

plt.title('Correlation btw features')

plt.show()
# And look detail of "Salary" data



df[["Salary"]].describe()
# Dropped NA named by "df2"

df1=df.copy()

df1=df1.dropna()

df1.shape
# Transformation 

df1=pd.get_dummies(df1,columns = ['League', 'Division', 'NewLeague'], drop_first = True)

df1.head()
# Outlier Detection



clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)

clf.fit_predict(df1)

df1_scores=clf.negative_outlier_factor_

df1_scores= np.sort(df1_scores)

df1_scores[0:20]
sns.boxplot(df1_scores);
threshold=np.sort(df1_scores)[10]

print(threshold)

df1=df1.loc[df1_scores > threshold]

df1=df1.reset_index(drop=True)
df1.shape
# Standardization
df1_X=df1.drop(["Salary","League_N","Division_W","NewLeague_N"], axis=1)

df1_X.head()
from sklearn.preprocessing import StandardScaler

scaled_cols=StandardScaler().fit_transform(df1_X)







scaled_cols=pd.DataFrame(scaled_cols, columns=df1_X.columns)

scaled_cols.head()
cat_df1=df1.loc[:, "League_N":"NewLeague_N"]

cat_df1.head()
Salary=pd.DataFrame(df1["Salary"])
df2=pd.concat([Salary,scaled_cols, cat_df1], axis=1)

df2.head()
df2.head()
# Filled NA with mean, normalized and drop high correlation columns.



df3=df.copy()

df3.corr()
# Mean of the Columns



df3['Year_lab'] = pd.cut(x=df['Years'], bins=[0, 3, 6, 10, 15, 19, 24])

df3.groupby(['League','Division', 'Year_lab']).agg({'Salary':'mean'})
df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'E') & (df3['Years'] <= 3), "Salary"] = 112

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'E') & (df3['Years'] > 3) & (df3['Years'] <= 6), "Salary"] = 656

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'E') & (df3['Years'] > 6) & (df3['Years'] <= 10), "Salary"] = 853

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'E') & (df3['Years'] > 10) & (df3['Years'] <= 15), "Salary"] = 816

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'E') & (df3['Years'] > 15) & (df3['Years'] <= 19), "Salary"] = 665



df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'W') & (df3['Years'] <= 3), "Salary"] = 154

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'W') & (df3['Years'] > 3) & (df3['Years'] <= 6), "Salary"] = 401

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'W') & (df3['Years'] > 6) & (df3['Years'] <= 10), "Salary"] = 634

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'W') & (df3['Years'] > 10) & (df3['Years'] <= 15), "Salary"] = 835

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'W') & (df3['Years'] > 15) & (df3['Years'] <= 19), "Salary"] = 479

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "A") & (df3['Division'] == 'W') & (df3['Years'] > 19), "Salary"] = 487



df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'E') & (df3['Years'] <= 3), "Salary"] = 248

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'E') & (df3['Years'] > 3) & (df3['Years'] <= 6), "Salary"] = 501

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'E') & (df3['Years'] > 6) & (df3['Years'] <= 10), "Salary"] = 824

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'E') & (df3['Years'] > 10) & (df3['Years'] <= 15), "Salary"] = 894

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'E') & (df3['Years'] > 15) & (df3['Years'] <= 19), "Salary"] = 662



df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'W') & (df3['Years'] <= 3), "Salary"] = 192

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'W') & (df3['Years'] > 3) & (df3['Years'] <= 6), "Salary"] = 458

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'W') & (df3['Years'] > 6) & (df3['Years'] <= 10), "Salary"] = 563

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'W') & (df3['Years'] > 10) & (df3['Years'] <= 15), "Salary"] = 722

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'W') & (df3['Years'] > 15) & (df3['Years'] <= 19), "Salary"] = 761

df3.loc[(df["Salary"].isnull()) & (df3["League"] == "N") & (df3['Division'] == 'W') & (df3['Years'] > 19), "Salary"] = 475
# Transformation 



le = LabelEncoder()

df3['League'] = le.fit_transform(df3['League'])

df3['Division'] = le.fit_transform(df3['Division'])

df3['NewLeague'] = le.fit_transform(df3['NewLeague'])

df3.head()
df3['Year_lab'] = le.fit_transform(df3['Year_lab'])

df3.head()
df3.info()
# Normalization

df3_X= df3.drop(["Salary","League","Division","NewLeague"], axis=1)



scaled_cols3=preprocessing.normalize(df3_X)





scaled_cols3=pd.DataFrame(scaled_cols3, columns=df3_X.columns)

scaled_cols3.head()
cat_df3=pd.concat([df3.loc[:,"League":"Division"],df3.loc[:,"NewLeague":"Year_lab"]], axis=1)

cat_df3.head()
df4= pd.concat([scaled_cols3,cat_df3,df3["Salary"]], axis=1)

df4
del df4["CHits"]

del df4["CAtBat"]
df4.head()
# Filled NA with mean, normalized



df5=df.copy()
df5['Year_lab'] = pd.cut(x=df['Years'], bins=[0, 3, 6, 10, 15, 19, 24])

df5.groupby(['League','Division', 'Year_lab']).agg({'Salary':'mean'})
df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'E') & (df5['Years'] <= 3), "Salary"] = 112

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'E') & (df5['Years'] > 3) & (df5['Years'] <= 6), "Salary"] = 656

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'E') & (df5['Years'] > 6) & (df5['Years'] <= 10), "Salary"] = 853

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'E') & (df5['Years'] > 10) & (df5['Years'] <= 15), "Salary"] = 816

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'E') & (df5['Years'] > 15) & (df5['Years'] <= 19), "Salary"] = 665



df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'W') & (df5['Years'] <= 3), "Salary"] = 154

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'W') & (df5['Years'] > 3) & (df5['Years'] <= 6), "Salary"] = 401

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'W') & (df5['Years'] > 6) & (df5['Years'] <= 10), "Salary"] = 634

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'W') & (df5['Years'] > 10) & (df5['Years'] <= 15), "Salary"] = 835

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'W') & (df5['Years'] > 15) & (df5['Years'] <= 19), "Salary"] = 479

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "A") & (df5['Division'] == 'W') & (df5['Years'] > 19), "Salary"] = 487



df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'E') & (df5['Years'] <= 3), "Salary"] = 248

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'E') & (df5['Years'] > 3) & (df5['Years'] <= 6), "Salary"] = 501

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'E') & (df5['Years'] > 6) & (df5['Years'] <= 10), "Salary"] = 824

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'E') & (df5['Years'] > 10) & (df5['Years'] <= 15), "Salary"] = 894

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'E') & (df5['Years'] > 15) & (df5['Years'] <= 19), "Salary"] = 662



df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'W') & (df5['Years'] <= 3), "Salary"] = 192

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'W') & (df5['Years'] > 3) & (df5['Years'] <= 6), "Salary"] = 458

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'W') & (df5['Years'] > 6) & (df5['Years'] <= 10), "Salary"] = 563

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'W') & (df5['Years'] > 10) & (df5['Years'] <= 15), "Salary"] = 722

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'W') & (df5['Years'] > 15) & (df5['Years'] <= 19), "Salary"] = 761

df5.loc[(df["Salary"].isnull()) & (df5["League"] == "N") & (df5['Division'] == 'W') & (df5['Years'] > 19), "Salary"] = 475
# Transformation 



le = LabelEncoder()

df5['League'] = le.fit_transform(df5['League'])

df5['Division'] = le.fit_transform(df5['Division'])

df5['NewLeague'] = le.fit_transform(df5['NewLeague'])

df5.head()
df5['Year_lab'] = le.fit_transform(df5['Year_lab'])
# Normalization

df5_X= df5.drop(["Salary","League","Division","NewLeague"], axis=1)



scaled_cols5=preprocessing.normalize(df5_X)





scaled_cols5=pd.DataFrame(scaled_cols5, columns=df5_X.columns)

scaled_cols5.head()
cat_df5=pd.concat([df5.loc[:,"League":"Division"],df5.loc[:,"NewLeague":"Year_lab"]], axis=1)

cat_df5.head()
df6= pd.concat([scaled_cols5,cat_df5,df5["Salary"]], axis=1)

df6.head()
# Filled NA with medyan, normalized



df7=df.copy()
df7['Year_lab'] = pd.cut(x=df['Years'], bins=[0, 3, 6, 10, 15, 19, 24])

df7.groupby(['League','Division', 'Year_lab']).agg({'Salary':'median'})
df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'E') & (df7['Years'] <= 3), "Salary"] = 90

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'E') & (df7['Years'] > 3) & (df7['Years'] <= 6), "Salary"] = 562

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'E') & (df7['Years'] > 6) & (df7['Years'] <= 10), "Salary"] = 673

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'E') & (df7['Years'] > 10) & (df7['Years'] <= 15), "Salary"] = 700

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'E') & (df7['Years'] > 15) & (df7['Years'] <= 19), "Salary"] = 655



df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W') & (df7['Years'] <= 3), "Salary"] = 127

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W') & (df7['Years'] > 3) & (df7['Years'] <= 6), "Salary"] = 350

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W') & (df7['Years'] > 6) & (df7['Years'] <= 10), "Salary"] = 600

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W') & (df7['Years'] > 10) & (df7['Years'] <= 15), "Salary"] = 787

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W') & (df7['Years'] > 15) & (df7['Years'] <= 19), "Salary"] = 325

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W') & (df7['Years'] > 19), "Salary"] = 487



df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'E') & (df7['Years'] <= 3), "Salary"] = 132

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'E') & (df7['Years'] > 3) & (df7['Years'] <= 6), "Salary"] = 348

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'E') & (df7['Years'] > 6) & (df7['Years'] <= 10), "Salary"] = 750

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'E') & (df7['Years'] > 10) & (df7['Years'] <= 15), "Salary"] = 600

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'E') & (df7['Years'] > 15) & (df7['Years'] <= 19), "Salary"] = 662



df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W') & (df7['Years'] <= 3), "Salary"] = 120

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W') & (df7['Years'] > 3) & (df7['Years'] <= 6), "Salary"] = 415

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W') & (df7['Years'] > 6) & (df7['Years'] <= 10), "Salary"] = 617

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W') & (df7['Years'] > 10) & (df7['Years'] <= 15), "Salary"] = 595

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W') & (df7['Years'] > 15) & (df7['Years'] <= 19), "Salary"] = 636

df7.loc[(df["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W') & (df7['Years'] > 19), "Salary"] = 475
# Transformation 



le = LabelEncoder()

df7['League'] = le.fit_transform(df7['League'])

df7['Division'] = le.fit_transform(df7['Division'])

df7['NewLeague'] = le.fit_transform(df7['NewLeague'])

df7.head()
df7['Year_lab'] = le.fit_transform(df7['Year_lab'])
# Normalization

df7_X= df7.drop(["Salary","League","Division","NewLeague"], axis=1)



scaled_cols7=preprocessing.normalize(df7_X)





scaled_cols7=pd.DataFrame(scaled_cols7, columns=df7_X.columns)

scaled_cols7.head()
cat_df7=pd.concat([df7.loc[:,"League":"Division"],df7.loc[:,"NewLeague":"Year_lab"]], axis=1)

cat_df7.head()
df8= pd.concat([scaled_cols7,cat_df7,df7["Salary"]], axis=1)

df8.head()
# Drop NA and Outliers, log transformed

# log transformation of the features which have multicorrelation above 0.8 between each other

# named by "df10" 
df9= df1.copy()

print(df9.shape)

df9.head()
# log transform the variables

df9['CRuns'] = np.log(df9['CRuns'])

df9['CHits'] = np.log(df9['CHits'])

df9['CAtBat'] = np.log(df9['CAtBat'])

df9['Years'] = np.log(df9['Years'])

df9['CRBI'] = np.log(df9['CRBI'])

df9['CWalks'] = np.log(df9['CWalks'])
df9_X=df9.drop(["Salary","League_N","Division_W","NewLeague_N"], axis=1)

df9_X.head()
Rscaler = RobustScaler().fit(df9_X)

scaled_cols9=Rscaler.transform(df9_X)

scaled_cols9=pd.DataFrame(scaled_cols9, columns=df9_X.columns)

scaled_cols9.head()
df10=pd.concat([df9_X,df9.loc[:, "League_N": "NewLeague_N"], df9["Salary"]], axis=1)

df10.head()
cat_df9=df9.loc[:, "League_N":"NewLeague_N"]

cat_df9.head()
df10.head()
# Filled NA with mean,normalized



df11=df.copy()
df11['Year_lab'] = pd.cut(x=df['Years'], bins=[0,24])

df11.groupby(['League','Division', 'Year_lab']).agg({'Salary':'mean'})
# Filled NA values with mean



df11.loc[(df11["Salary"].isnull()) & (df11["League"] == "A") & (df11['Division'] == 'E'),"Salary"] = 670.849559

df11.loc[(df11["Salary"].isnull()) & (df11["League"] == "A") & (df11['Division'] == 'W'),"Salary"] = 418.593901

df11.loc[(df11["Salary"].isnull()) & (df11["League"] == "N") & (df11['Division'] == 'E'),"Salary"] = 572.348131

df11.loc[(df11["Salary"].isnull()) & (df11["League"] == "N") & (df11['Division'] == 'W'),"Salary"] = 487.259270
#Transformation



le = LabelEncoder()

df11['League'] = le.fit_transform(df11['League'])

df11['Division'] = le.fit_transform(df11['Division'])

df11['NewLeague'] = le.fit_transform(df11['NewLeague'])
# Normalization

df7_X= df7.drop(["Salary","League","Division","NewLeague"], axis=1)



scaled_cols7=preprocessing.normalize(df7_X)





scaled_cols7=pd.DataFrame(scaled_cols7, columns=df7_X.columns)
# Concatenate



cat_df11=pd.concat([df11.loc[:,"League":"Division"],df11["NewLeague"]], axis=1)

cat_df11.head()
df12= pd.concat([scaled_cols7,cat_df7,df7["Salary"]], axis=1)

df12.head()
df12.shape
#Regression
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





linreg = LinearRegression()

model = linreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df2_linreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df2_linreg_rmse
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





linreg = LinearRegression()

model = linreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df4_linreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df4_linreg_rmse
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





linreg = LinearRegression()

model = linreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df6_linreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df6_linreg_rmse
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





linreg = LinearRegression()

model = linreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df8_linreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df8_linreg_rmse
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





linreg = LinearRegression()

model = linreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df10_linreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df10_linreg_rmse
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





linreg = LinearRegression()

model = linreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df12_linreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df12_linreg_rmse
#Ridge Regression
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





ridreg = Ridge()

model = ridreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

df2_ridreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df2_ridreg_rmse 
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





ridreg = Ridge()

model = ridreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

df4_ridreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df4_ridreg_rmse 
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





ridreg = Ridge()

model = ridreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

df6_ridreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df6_ridreg_rmse 
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





ridreg = Ridge()

model = ridreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

df8_ridreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df8_ridreg_rmse 
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





ridreg = Ridge()

model = ridreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

df10_ridreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df10_ridreg_rmse 
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





ridreg = Ridge()

model = ridreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

df12_ridreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df12_ridreg_rmse 
# Lasso Regression
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





lasreg = Lasso()

model = lasreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df2_lasreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df2_lasreg_rmse
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





lasreg = Lasso()

model = lasreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df4_lasreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df4_lasreg_rmse
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





lasreg = Lasso()

model = lasreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df6_lasreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df6_lasreg_rmse
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





lasreg = Lasso()

model = lasreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df8_lasreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df8_lasreg_rmse
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





lasreg = Lasso()

model = lasreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df10_lasreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df10_lasreg_rmse
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





lasreg = Lasso()

model = lasreg.fit(X_train,y_train)

y_pred = model.predict(X_test)

df12_lasreg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df12_lasreg_rmse
#Elastic Net Regression
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





enet = ElasticNet()

model = enet.fit(X_train,y_train)

y_pred = model.predict(X_test)

df2_enet_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df2_enet_rmse
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





enet = ElasticNet()

model = enet.fit(X_train,y_train)

y_pred = model.predict(X_test)

df4_enet_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df4_enet_rmse
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





enet = ElasticNet()

model = enet.fit(X_train,y_train)

y_pred = model.predict(X_test)

df6_enet_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df6_enet_rmse
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





enet = ElasticNet()

model = enet.fit(X_train,y_train)

y_pred = model.predict(X_test)

df8_enet_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df8_enet_rmse
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





enet = ElasticNet()

model = enet.fit(X_train,y_train)

y_pred = model.predict(X_test)

df10_enet_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df10_enet_rmse
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





enet = ElasticNet()

model = enet.fit(X_train,y_train)

y_pred = model.predict(X_test)

df12_enet_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df12_enet_rmse
# Ridge Regression Model Tuning
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

ridreg_cv = RidgeCV(alphas = alpha, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridreg_cv.fit(X_train, y_train)

ridreg_cv.alpha_



#Final Model 

ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(X_train,y_train)

y_pred = ridreg_tuned.predict(X_test)

df2_ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df2_ridge_tuned_rmse
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

ridreg_cv = RidgeCV(alphas = alpha, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridreg_cv.fit(X_train, y_train)

ridreg_cv.alpha_



#Final Model 

ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(X_train,y_train)

y_pred = ridreg_tuned.predict(X_test)

df4_ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df4_ridge_tuned_rmse
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

ridreg_cv = RidgeCV(alphas = alpha, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridreg_cv.fit(X_train, y_train)

ridreg_cv.alpha_



#Final Model 

ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(X_train,y_train)

y_pred = ridreg_tuned.predict(X_test)

df6_ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df6_ridge_tuned_rmse
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

ridreg_cv = RidgeCV(alphas = alpha, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridreg_cv.fit(X_train, y_train)

ridreg_cv.alpha_



#Final Model 

ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(X_train,y_train)

y_pred = ridreg_tuned.predict(X_test)

df8_ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df8_ridge_tuned_rmse
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

ridreg_cv = RidgeCV(alphas = alpha, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridreg_cv.fit(X_train, y_train)

ridreg_cv.alpha_



#Final Model 

ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(X_train,y_train)

y_pred = ridreg_tuned.predict(X_test)

df10_ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df10_ridge_tuned_rmse
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)





alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

ridreg_cv = RidgeCV(alphas = alpha, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridreg_cv.fit(X_train, y_train)

ridreg_cv.alpha_



#Final Model 

ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(X_train,y_train)

y_pred = ridreg_tuned.predict(X_test)

df12_ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df12_ridge_tuned_rmse
#Lasso Regression Model Tuning
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

lasso_cv = LassoCV(alphas = alpha, cv = 10, normalize = True)

lasso_cv.fit(X_train, y_train)

lasso_cv.alpha_



#Final Model 

lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

df2_lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

df2_lasso_tuned_rmse
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

lasso_cv = LassoCV(alphas = alpha, cv = 10, normalize = True)

lasso_cv.fit(X_train, y_train)

lasso_cv.alpha_



#Final Model 

lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

df4_lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

df4_lasso_tuned_rmse
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

lasso_cv = LassoCV(alphas = alpha, cv = 10, normalize = True)

lasso_cv.fit(X_train, y_train)

lasso_cv.alpha_



#Final Model 

lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

df6_lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

df6_lasso_tuned_rmse
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

lasso_cv = LassoCV(alphas = alpha, cv = 10, normalize = True)

lasso_cv.fit(X_train, y_train)

lasso_cv.alpha_



#Final Model 

lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

df8_lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

df8_lasso_tuned_rmse
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

lasso_cv = LassoCV(alphas = alpha, cv = 10, normalize = True)

lasso_cv.fit(X_train, y_train)

lasso_cv.alpha_



#Final Model 

lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

df10_lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

df10_lasso_tuned_rmse
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



alpha = [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]

lasso_cv = LassoCV(alphas = alpha, cv = 10, normalize = True)

lasso_cv.fit(X_train, y_train)

lasso_cv.alpha_



#Final Model 

lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

df12_lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

df12_lasso_tuned_rmse
#Elastic Net Regression Model Tuning
y=df2["Salary"]

X=df2.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}



enet_model = ElasticNet().fit(X_train,y_train)

enet_cv = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

enet_cv.best_params_



#Final Model 

enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

df2_enet_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df2_enet_tuned_rmse 
y=df4["Salary"]

X=df4.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}



enet_model = ElasticNet().fit(X_train,y_train)

enet_cv = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

enet_cv.best_params_



#Final Model 

enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

df4_enet_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df4_enet_tuned_rmse 
y=df6["Salary"]

X=df6.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}



enet_model = ElasticNet().fit(X_train,y_train)

enet_cv = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

enet_cv.best_params_



#Final Model 

enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

df6_enet_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df6_enet_tuned_rmse 
y=df8["Salary"]

X=df8.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}



enet_model = ElasticNet().fit(X_train,y_train)

enet_cv = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

enet_cv.best_params_



#Final Model 

enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

df8_enet_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df8_enet_tuned_rmse 
y=df10["Salary"]

X=df10.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}



enet_model = ElasticNet().fit(X_train,y_train)

enet_cv = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

enet_cv.best_params_



#Final Model 

enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

df10_enet_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df10_enet_tuned_rmse 
y=df12["Salary"]

X=df12.drop("Salary", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}



enet_model = ElasticNet().fit(X_train,y_train)

enet_cv = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

enet_cv.best_params_



#Final Model 

enet_tuned = ElasticNet(**enet_cv.best_params_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

df12_enet_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

df12_enet_tuned_rmse 
basicresult_df = pd.DataFrame({"CONDITIONS":["df2: Drop NA and Outliers, normalized","df4: Filled NA with mean(large group), normalized and drop high correlation datas.","df6: Filled NA with mean(large group), normalized","df8: Filled NA with median, normalized","df10: Drop NA and Outliers, log transformed","df12:Filled NA with mean(small group),normalized"],

                              "LINEAR":[df2_linreg_rmse,df4_linreg_rmse,df6_linreg_rmse,df8_linreg_rmse,df10_linreg_rmse,df12_linreg_rmse],

                               "RIDGE":[df2_ridreg_rmse,df4_ridreg_rmse,df6_ridreg_rmse,df8_ridreg_rmse,df10_ridreg_rmse,df12_ridreg_rmse],

                              "RIDGE TUNED":[df2_ridge_tuned_rmse,df4_ridge_tuned_rmse,df6_ridge_tuned_rmse,df8_ridge_tuned_rmse,df10_ridge_tuned_rmse,df12_ridge_tuned_rmse],

                              "LASSO":[df2_lasreg_rmse,df4_lasreg_rmse,df6_lasreg_rmse,df8_lasreg_rmse,df10_lasreg_rmse,df12_lasreg_rmse],

                              "LASSO TUNED":[df2_lasso_tuned_rmse,df4_lasso_tuned_rmse,df6_lasso_tuned_rmse,df8_lasso_tuned_rmse,df10_lasso_tuned_rmse,df12_lasso_tuned_rmse],                              

                              "ELASTIC NET":[df2_enet_rmse,df4_enet_rmse,df6_enet_rmse,df8_enet_rmse,df10_enet_rmse,df12_enet_rmse],

                              "ELASTIC NET TUNED":[df2_enet_tuned_rmse,df4_enet_tuned_rmse,df6_enet_tuned_rmse,df8_enet_tuned_rmse,df10_enet_tuned_rmse,df12_enet_tuned_rmse]

                              })



basicresult_df