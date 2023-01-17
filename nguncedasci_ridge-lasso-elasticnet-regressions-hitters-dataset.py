from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
Hitters=pd.read_csv("../input/hitters/Hitters.csv")
df=Hitters.copy()
df.head()
df.info()
df.isnull().sum()
df.describe().T
import seaborn as sns
sns.boxplot(x=df["Salary"]);
NAdf= df[df.isnull().any(axis=1)]
NAdf.describe().T
notNAdf=df[df.notnull().all(axis=1)]
notNAdf.describe().T
notNAdf.corr()
print("New League= A" ,notNAdf[notNAdf["NewLeague"]=="A"].agg({"Salary":"mean"}))
print("New League= N" ,notNAdf[notNAdf["NewLeague"]=="N"].agg({"Salary":"mean"}))
print("League= A" ,notNAdf[notNAdf["League"]=="A"].agg({"Salary":"mean"}))
print("League= N" ,notNAdf[notNAdf["League"]=="N"].agg({"Salary":"mean"}))
print("Division= E" ,notNAdf[notNAdf["Division"]=="E"].agg({"Salary":"mean"}))
print("Division= W" ,notNAdf[notNAdf["Division"]=="W"].agg({"Salary":"mean"}))
df.head(2)
df.loc[(df["Salary"].isnull())& (df['Division'] == 'E'),"Salary"]=624.27
df.head(2)
df.loc[(df["Salary"].isnull())& (df['Division'] == 'W'),"Salary"]=450.87
df.isnull().sum().sum()
df[df["Salary"]<0]    
dff = pd.get_dummies(df, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
dff.head()
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.05)
clf.fit_predict(dff)
dff_scores = clf.negative_outlier_factor_
np.sort(dff_scores)[0:20]
sns.boxplot(x=dff_scores);
threshold = np.sort(dff_scores)[13]
dff.loc[dff_scores < threshold]
dff.loc[(dff_scores < threshold)&(dff["Salary"]>1500)]
df["Salary"].describe([0.75,0.90,0.95,0.99]).T
dff.loc[(dff_scores < threshold)&(dff["Salary"]>1500),"Salary"]=1967
dff.loc[dff_scores < threshold]
dff.loc[dff_scores < threshold]
df1=dff.loc[df1_scores > esik_deger]
df1.shape
import seaborn as sns
sns.pairplot(df1)
plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
CHits_cor=abs(cor["CHits"])
CHits_relevant_features = CHits_cor[CHits_cor>0.9]
CHits_relevant_features
import matplotlib.pyplot as plt
plt.scatter(df["CHits"],df["CAtBat"], alpha=0.3,
            cmap='viridis');
df1=df1.drop("CHits", axis=1)
df1=df1.drop("CAtBat", axis=1)
plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
df1=df1.drop("CRBI", axis=1)
df1=df1.drop("CWalks", axis=1)
plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
df1.describe().T
df1.shape
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
SS=StandardScaler()
col=df1.columns
df2=SS.fit_transform(df1)
df3=pd.DataFrame(df2, columns=col)
df3.head(2)



df6=preprocessing.normalize(df1, axis=0)
col=df1.columns
df7=pd.DataFrame(df6, columns=col)
df7.head(2)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

print("intercept(b0):",reg_model.intercept_)
print("coefficients(b1,b2..):","\n",reg_model.coef_)
reg_model.predict(X)[0:10]
y.head(10)
df1[y<0]
# df3 (Standar Scaler)

y=df3["Salary"]
X=df3.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#df7 (Normalize)

y=df7["Salary"]
X=df7.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
ridge_model=Ridge().fit(X_train,y_train)
y_pred= ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas1, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train,y_train)
ridge_cv.alpha_
ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                   random_state=46)
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
lasso_model = Lasso().fit(X_train, y_train)
y_pred=lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
lasso_cv_model = LassoCV(alphas = alphas1, cv = 10).fit(X_train, y_train)

print(lasso_cv_model.alpha_)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV


y=df1["Salary"]
X=df1.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                   random_state=46)
enet_model = ElasticNet().fit(X_train, y_train)
y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
enet_cv_model = ElasticNetCV(alphas = alphas1, cv = 10).fit(X_train, y_train)

enet_cv_model.alpha_
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
enet_params = {"l1_ratio": [0,0.01,0.05,0.1,0.2,0.4,0.5,0.6,0.8,1],
              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1,2,5,7,10,13,20,45,99,100]}
enet_model = ElasticNet().fit(X, y)
from sklearn.model_selection import GridSearchCV
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)

gs_cv_enet.best_params_
enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Note for me

dff=dff.drop("CRBI", axis=1)
dff=dff.drop("CWalks", axis=1)
dff=dff.drop("CHits", axis=1)
dff=dff.drop("CAtBat", axis=1)
dff=dff.loc[df1_scores > esik_deger]
y=dff["Salary"]
X=dff.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
