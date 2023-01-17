!pip install pyforest
from pyforest import *
df=pd.read_csv("../input/water-quality-data/waterquality.csv", sep=',', engine='python')
df
df.isnull().sum()
df= df.fillna(df.mean())
df.head()
df.isnull().sum()
sns.heatmap(df.corr(),annot=True)
df= df.drop("STATION CODE", axis=1)
df.head()
sns.pairplot(df)
ph= df["pH"]
ph.value_counts()
PH= pd.DataFrame(ph, index=None)
PH.head()
PH.pH.value_counts()
PH["QI"]=PH.replace(to_replace =6.4,  
                            value =54)
PH.head()
PH["QI"]=PH["QI"].replace(to_replace =[6.5,6.7,6.8,6.9],  
                            value =75)

PH["QI"]=PH["QI"].replace(to_replace =[7.0,7.1,7.2,7.3,7.4],  
                            value =80)

PH["QI"]=PH["QI"].replace(to_replace =[7.5,7.6,7.7,7.8,7.9],  
                            value =95)

PH["QI"]=PH["QI"].replace(to_replace =[8.0,8.1,8.2,8.3,8.4],  
                            value =85)

PH["QI"]=PH["QI"].replace(to_replace =[8.5,8.6,8.7,8.8,8.9],  
                            value =65)

PH["QI"]=PH["QI"].replace(to_replace =[9.0,9.1,9.2,9.3,9.4],  
                            value =48)

PH["QI"]=PH["QI"].replace(to_replace =[9.5,9.6,9.7,9.8,9.9],  
                            value =30)

PH["QI"]=PH["QI"].replace(to_replace =[10.0,10.1,10.2,10.3,10.4],  
                            value =20)

PH["QI"]=PH["QI"].replace(to_replace =[10.5,10.6,10.7,10.8,10.9],  
                            value =12)

PH["QI"]=PH["QI"].replace(to_replace =[11.0,11.1,11.2,11.3,11.4],  
                            value =8)

PH["QI"]=PH["QI"].replace(to_replace =[11.5,11.6,11.7,11.8,11.9],  
                            value =4)

PH["QI"]=PH["QI"].replace(to_replace =[12.0,12.1,12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13.0,13.1,13.2,13.3,13.4,13.5,13.6,13.7,13.8],  
                            value =75)


PH["QI"].value_counts()
PH.head(), PH.tail()
sns.jointplot(x="pH", y="QI", data=PH, kind="reg")
sns.distplot(PH["pH"], kde=True, bins=10)
ls= df[["LOCATIONS","STATE"]]
ls.head()
df_col_merged = pd.concat([ls, PH], axis=1)

df_col_merged.head()
ax=sns.barplot(x="STATE", y= "QI", data=df_col_merged)
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right')
X= df_col_merged.drop(["LOCATIONS","STATE","QI"], axis=1)
X.head()
X.shape
y= df_col_merged["pH"]
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
ridge= Ridge()
parameters= {"alpha":[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}

ridge_regressor= GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error", cv=10)

ridge_regressor.fit(X,y)
ridge_regressor.best_params_
ridge_regressor.best_score_, ridge_regressor.score
ridge_regressor.cv_results_
df2= pd.DataFrame(ridge_regressor.cv_results_)
df2