import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df
df.isnull().sum()
print(df["status"].value_counts())

print(df["status"].value_counts(normalize = True))
df["salary"].fillna(0,inplace = True)

df.isnull().sum()
fig,ax = plt.subplots(2,3,figsize = (12,8))

axes = ax.flatten()

sns.set()

df_percent= df[["ssc_p","hsc_p","degree_p","etest_p","mba_p","status"]]

placed = df_percent[df_percent["status"] == "Placed"]

not_placed = df_percent[df_percent["status"] == "Not Placed"]

df_scores = df_percent.drop(["status"],axis = 1)

name_percentage = df_scores.columns.tolist()

i = 0 

for i in np.arange(len(name_percentage)):

    feature = name_percentage[i]

    sns.kdeplot(data= placed[feature], ax = axes[i], shade = True,color = "r",legend = False)

    sns.kdeplot(data= not_placed[feature], ax= axes[i],shade = True,legend = False)

    axes[i].set_title(feature)

    axes[i].legend(["placed","not_placed"],loc = "upper right")

    



    
fig,ax = plt.subplots(2,3,figsize = (12,8))

axes = ax.flatten()

cat = ["degree_t", "specialisation","gender","workex","hsc_s","hsc_b"]

plt.tight_layout(3.0)

i = 0 

for i in np.arange(len(cat)):

    feature = cat[i]

    sns.violinplot(y = "degree_p", x = feature, hue = "status",data = df, split = True, ax = axes[i])

fig,ax = plt.subplots(2,3,figsize = (12,8))

axes = ax.flatten()

cat = ["degree_t", "specialisation","gender","workex","hsc_s","hsc_b"]

plt.tight_layout(3.0)

i = 0 

for i in np.arange(len(cat)):

    feature = cat[i]

    sns.violinplot(y = "mba_p", x = feature, hue = "status",data = df, split = True, ax = axes[i])
fig,ax = plt.subplots(3,3,figsize = (12,12))

axes = ax.flatten()

# getting all the categorical features in a list 

cat = ["degree_t", "specialisation","gender","workex","hsc_s","hsc_b","ssc_b"]

plt.tight_layout(3.0)

i = 0 

gender_split= df[["gender","status","workex"]]

# calculating the denominator 

denom = gender_split.groupby(["status"]).count().gender



for i in np.arange(len(cat)):

    feature = cat[i]

    df_concat = pd.concat([df[["status","etest_p"]],df[feature]],axis =1 )

    df_concat_group = df_concat.groupby(["status",feature]).count()

    # getting the percentage score within each status group 

    # check denom to understand more 

    df_concat_group["etest_p"]= df_concat_group["etest_p"]/denom

    a = df_concat_group.reset_index()

    sns.barplot(hue = feature,x = "status", y = "etest_p",data= a, ax = axes[i])

    axes[i].set_ylabel("E_test Percentage within status group")

    axes[i].legend(loc = "upper right")

    axes[i].set_title(f'{feature}')
df_placed = df[df["status"] == "Placed"]

df_placed[df_placed["salary"] == df_placed["salary"].max()]

quali = [i for i in list(df.columns) if df[i].dtypes == "object"]

quanti=  [i for i in list(df.columns) if df[i].dtypes != "object"]

quali.remove("status")

# because salary is directly correlated with status, we will drop it. We only get salary info after we know about status, so it makes no sense

# to use it to predict status. 

quanti.remove("salary")

quanti.remove("sl_no")

print("Quali features: ", quali)

print("="*40)

print("Quanti features: ", quanti)

cmap = sns.diverging_palette(220,10,as_cmap = True)

sns.heatmap(df[quanti].corr(),annot = True,cmap = cmap)
new_df= pd.concat([df[quanti],df[quali]], axis = 1)

new_df
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

X = new_df

y = df["status"]

lb=  LabelBinarizer()

y_trans = lb.fit_transform(y)

y_transform = np.ravel(y_trans)



X_train,X_test,y_train,y_test = train_test_split(X,y_transform,random_state = 42,test_size = 0.2)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



quali_pipe = Pipeline(steps = [("encoder", OneHotEncoder(handle_unknown = "ignore"))])

quanti_pipe = Pipeline(steps =[("scaler",StandardScaler())])

transformer = ColumnTransformer([("quali_pipe",quali_pipe,quali),

                           ("quanti_pipe",quanti_pipe,quanti)])



pipe = Pipeline(steps = [("transformer",transformer),("logreg",LogisticRegression(penalty = "l2",

                                                                                  C = 0.23,

                                                                                  class_weight = "balance",

                                                                                  max_iter = 200))])

from sklearn.model_selection import GridSearchCV

param_grid = {"logreg__C":[0.2,0.21,0.22,0.23]}



search = GridSearchCV(pipe,

                      param_grid, 

                      cv = 5,

                      scoring = "recall")

best_model = search.fit(X_train,y_train)
best_model.best_estimator_.get_params()["logreg"]
from sklearn.metrics import classification_report

y_train_predict = best_model.predict(X_train)

y_test_predict = best_model.predict(X_test)

print(classification_report(y_train,y_train_predict))

print(classification_report(y_test,y_test_predict))
# getting the feature_names of one hot encoded qualitative variables



enc= best_model.best_estimator_["transformer"].transformers[0][1]["encoder"]

enc.fit_transform(X_train[quali])

quali_features_transformed= enc.get_feature_names(quali).tolist()

# combining the features name together with quanti feature names 

feature_names_in_model= quanti.copy()

feature_names_in_model.extend(quali_features_transformed)

coef= best_model.best_estimator_["logreg"].coef_

coef_list = coef.flatten().tolist()
# Creating a dataframe of feature importances 

a = pd.DataFrame(dict(zip(feature_names_in_model,coef_list),index = [0]))

a
# sorting the dataframe in decending order 

descending_list = a.iloc[0].sort_values(ascending = False).index.tolist()

a_ordered_df = a[descending_list]

sns.barplot(y = a_ordered_df.columns, 

           x = a_ordered_df.iloc[0],

            orient = "h")