import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data2019["Year"] = 2019

data2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

data2018["Year"] = 2018

data = pd.concat((data2019,data2018))

data.reset_index(inplace = True,drop=True)

data
data[data.isna().any(axis=1)]
data[data["Country or region"]=="United Arab Emirates"]
!pip install ycimpute

from ycimpute.imputer import knnimput
#we need numerical values for prediction

num_data = data.select_dtypes(include=["float64","int64"])



#we kept column names for create new similar dataframe 

var_names = list(num_data)

var_names
#to take values as np.array

var_values = num_data.values

var_values
#after the prediction

completed_values = knnimput.KNN(k=4).complete(var_values)

completed_values
#new not-null dataframe

new_num_data = pd.DataFrame(completed_values,columns = var_names)

# and the value we want to find

new_num_data.iloc[175]
#check value of data (consistency)

data.iloc[170:180]
#change values

data["Perceptions of corruption"] = new_num_data["Perceptions of corruption"]
#control

data.iloc[170:180]
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)

data.boxplot("GDP per capita")

plt.subplot(2,3,2)

data.boxplot("Social support")

plt.subplot(2,3,3)

data.boxplot("Healthy life expectancy")

plt.subplot(2,3,4)

data.boxplot("Freedom to make life choices")

plt.subplot(2,3,5)

data.boxplot("Generosity")

plt.subplot(2,3,6)

data.boxplot("Perceptions of corruption")
#Rwanda has srange "Perceptions of corruption" value!

data[data["Perceptions of corruption"]>.35]
#we looked to heatmap and we can see here,too

sns.jointplot(x="Score",y="Generosity",data=data,kind="reg")
### multiple outliers

from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(new_num_data)
n_scores = clf.negative_outlier_factor_
plt.plot(np.sort(n_scores))
#Outlier values

#These cities are best and worst cities and they are out of standart.

new_num_data[n_scores<-1.01]
diff = data.groupby("Country or region")["Overall rank"].diff()[156:]
country_diff = pd.DataFrame({"country":data["Country or region"][:156].values,"diff":diff.values})

country_diff
# Countries which have big score change.

treshold = 10



country_diff = country_diff[(country_diff["diff"] > treshold) | (country_diff["diff"] < -treshold)]
plt.figure(figsize=(18,10))

sns.barplot(x="diff",y="country",data=country_diff)
# Lastly Our Overall Rank

data[data["Country or region"]=="Turkey"]
### Prepare train data



df = data[["Score","GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]]
### There are high relation between Score and GDP per capita, Social support, Healthy life expectancy



### There isn't a certain relation between Score and Generosity, we can see this



sns.heatmap(df.corr())
X = df.drop(["Score"],axis = 1)

y = df["Score"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMRegressor



lgbm_model = LGBMRegressor()



lgbm_params = {"learning_rate":[0.002,0.005,0.01,0.04,0.1],

            "max_depth":[3,5,8,12,15,18],

            "n_estimators":[5,20,50,100,300]}



from sklearn.model_selection import GridSearchCV



lgbm_cv_model = GridSearchCV(lgbm_model,lgbm_params,cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train)
lgbm_cv_model.best_params_
model = LGBMRegressor(learning_rate=0.1,

                       max_depth=3,

                       n_estimators=50).fit(X_train,y_train)
y_pred = model.predict(X_test)
import sklearn.metrics as sm



print("R2 score =", round(sm.r2_score(y_test, y_pred), 5))
feature_imp = pd.Series(model.feature_importances_,

                index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(20,10))



sns.barplot(feature_imp,feature_imp.index)
from pdpbox import pdp



def plot_partial_dep(model,data,feature):

    pdp_dist = pdp.pdp_isolate(model=model, dataset=data

                               , model_features=data.columns

                               , feature=feature)

    return pdp.pdp_plot(pdp_isolate_out=pdp_dist, feature_name=feature);
fig,ax = plot_partial_dep(model,X_train,"Freedom to make life choices")

ax["pdp_ax"].set_ylabel('Happiness Score Effect');
fig,ax = plot_partial_dep(model,X_train,"Social support")

ax["pdp_ax"].set_ylabel('Happiness Score Effect');