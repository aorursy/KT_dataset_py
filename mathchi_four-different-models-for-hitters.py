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

Hitters=pd.read_csv("../input/hitters-baseball-data/Hitters.csv")
df=Hitters.copy()
df.info()
df.describe().T
df[df.isnull().any(axis=1)].head(3)
df.isnull().sum().sum()
correlation_matrix = df.corr().round(2)
threshold=0.75
filtre=np.abs(correlation_matrix['Salary']) > 0.50
corr_features=correlation_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(),annot=True,fmt=".2f")
plt.title('Correlation btw features')
plt.show()
import missingno as msno
msno.bar(df);
df1=df.copy()
df1=df1.dropna()
df1.shape
df1=pd.get_dummies(df1,columns = ['League', 'Division', 'NewLeague'], drop_first = True)
df1.head()
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
df1_X=df1.drop(["Salary","League_N","Division_W","NewLeague_N"], axis=1)
df1_X.head(2)
from sklearn.preprocessing import StandardScaler
scaled_cols=StandardScaler().fit_transform(df1_X)



scaled_cols=pd.DataFrame(scaled_cols, columns=df1_X.columns)
scaled_cols.head()
cat_df1=df1.loc[:, "League_N":"NewLeague_N"]
cat_df1.head()
Salary=pd.DataFrame(df1["Salary"])
Salary.head()
df2=pd.concat([Salary,scaled_cols, cat_df1], axis=1)
df2.head(2)
df2.head()
df5=df.copy()
df5.corr()
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
df5.shape
le = LabelEncoder()
df5['League'] = le.fit_transform(df5['League'])
df5['Division'] = le.fit_transform(df5['Division'])
df5['NewLeague'] = le.fit_transform(df5['NewLeague'])
df5.head()
df5['Year_lab'] = le.fit_transform(df5['Year_lab'])
df5.head(2)
df5.info()
df5_X= df5.drop(["Salary","League","Division","NewLeague"], axis=1)

scaled_cols5=preprocessing.normalize(df5_X)


scaled_cols5=pd.DataFrame(scaled_cols5, columns=df5_X.columns)
scaled_cols5.head()
cat_df5=pd.concat([df5.loc[:,"League":"Division"],df5.loc[:,"NewLeague":"Year_lab"]], axis=1)
cat_df5.head()
df6= pd.concat([scaled_cols5,cat_df5,df5["Salary"]], axis=1)
df6
df6.shape
df3= df1.copy()
print(df3.shape)
df3.head(2)
# log transform the variables
df3['CRuns'] = np.log(df3['CRuns'])
df3['CHits'] = np.log(df3['CHits'])
df3['CAtBat'] = np.log(df3['CAtBat'])
df3['Years'] = np.log(df3['Years'])
df3['CRBI'] = np.log(df3['CRBI'])
df3['CWalks'] = np.log(df3['CWalks'])
df3_X=df3.drop(["Salary","League_N","Division_W","NewLeague_N"], axis=1)
df3_X.head(2)
df3_X.shape
Rscaler = RobustScaler().fit(df3_X)
scaled_cols3=Rscaler.transform(df3_X)
scaled_cols3=pd.DataFrame(scaled_cols3, columns=df3_X.columns)
scaled_cols3.head()
df4=pd.concat([df3_X,df3.loc[:, "League_N": "NewLeague_N"], df3["Salary"]], axis=1)
df4.head()
scaled_cols3.shape
cat_df3=df3.loc[:, "League_N":"NewLeague_N"]
cat_df3.head()
df4.head()
df7=df.copy()
# Filled NaN values with mean

df7.loc[(df7["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'E'),"Salary"] = 670.849559
df7.loc[(df7["Salary"].isnull()) & (df7["League"] == "A") & (df7['Division'] == 'W'),"Salary"] = 418.593901
df7.loc[(df7["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'E'),"Salary"] = 572.348131
df7.loc[(df7["Salary"].isnull()) & (df7["League"] == "N") & (df7['Division'] == 'W'),"Salary"] = 487.259270
le = LabelEncoder()
df7['League'] = le.fit_transform(df7['League'])
df7['Division'] = le.fit_transform(df7['Division'])
df7['NewLeague'] = le.fit_transform(df7['NewLeague'])
df7_X= df7.drop(["Salary","League","Division","NewLeague"], axis=1)

scaled_cols7=preprocessing.normalize(df7_X)


scaled_cols7=pd.DataFrame(scaled_cols7, columns=df7_X.columns)
cat_df7=pd.concat([df7.loc[:,"League":"Division"],df7["NewLeague"]], axis=1)
cat_df7.head()
df8= pd.concat([scaled_cols7,cat_df7,df7["Salary"]], axis=1)
df8.head()
df8.shape
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
basicsonuc_df = pd.DataFrame({"CONDITIONS":["df2: drop NA and Outliers, normalized","df6: filled with mean, normalized","df4: drop NA and Outliers, log transformed","df8: filled with mean,normalized"],
                              "LINEAR":[df2_linreg_rmse,df6_linreg_rmse,df4_linreg_rmse,df8_linreg_rmse],
                               "RIDGE":[df2_ridreg_rmse,df6_ridreg_rmse,df4_ridreg_rmse,df8_ridreg_rmse],
                              "RIDGE TUNED":[df2_ridge_tuned_rmse,df6_ridge_tuned_rmse,df4_ridge_tuned_rmse,df8_ridge_tuned_rmse],
                              "LASSO":[df2_lasreg_rmse,df6_lasreg_rmse,df4_lasreg_rmse,df8_lasreg_rmse],
                              "LASSO TUNED":[df2_lasso_tuned_rmse,df6_lasso_tuned_rmse,df4_lasso_tuned_rmse,df8_lasso_tuned_rmse],                              
                              "ELASTIC NET":[df2_enet_rmse,df6_enet_rmse,df4_enet_rmse,df8_enet_rmse],
                              "ELASTIC NET TUNED":[df2_enet_tuned_rmse,df6_enet_tuned_rmse,df4_enet_tuned_rmse,df8_enet_tuned_rmse]
                              })

basicsonuc_df
