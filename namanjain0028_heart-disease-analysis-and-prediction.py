from numpy import *

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pandas_profiling import ProfileReport

import plotly.express as px

sns.set(style = "darkgrid")
df = pd.read_csv("../input/heart-disease-dataset/heart.csv")

df.head()
df.info()
df.describe()
report = ProfileReport(df)

report


print(df.isna().sum())

sns.heatmap(df.isnull() , cmap = "gray" , yticklabels = False)

plt.hist(df["age"] , bins = 10)
# relationship between chest pain type and age

df["cp_type"] = df.cp.map({0 : "Typical Angina" , 1 : "Atypical Angina" , 2 : "Non-Anginal" , 3 : "Asymptomatic"})



sns.swarmplot(data = df , x = "cp_type" , y = "age")
dis = df[df["target"] == 1]

ndis = df[df["target"] == 0]





plt.scatter(dis["age"] , dis["chol"])

plt.scatter(ndis["age"] , ndis["chol"])
sns.jointplot(data = dis , x = "age" , y = "chol" , kind = "hex")

sns.jointplot(data = ndis , x = "age" , y = "chol" , kind = "hex")
# adding a column to the dataframe that describes the chest pain type and plotting a pie chart representing percentage of people having that pain



cp_type = df[["cp_type"]]

cp_type["count"] = df.groupby(cp_type.cp_type)["cp_type"].transform("count")

cp_type = cp_type.drop_duplicates()

fig = px.pie(data_frame = cp_type , values = "count" , names = "cp_type" , template = "seaborn")

fig.update_traces(rotation = 90 , pull = 0.05 , textinfo = "percent+label")

fig.show()

fig = sns.countplot(data = df , x = "sex" , hue = "cp_type")

for p in fig.patches:

    fig.annotate(p.get_height(), (p.get_x()+0.05, p.get_height()+1))
plt.scatter(dis["trestbps"] , dis["chol"])

plt.scatter(ndis["trestbps"] , ndis["chol"])
plt.scatter(dis["trestbps"] , dis["thalach"])

plt.scatter(ndis["trestbps"] , ndis["thalach"])
df["target_group"] = df.target.map({0 : "No Disease" , 1 : "Disease"})

fig = sns.countplot(data = df , x = "sex" , hue = "target_group")

for p in fig.patches:

    fig.annotate(p.get_height(), (p.get_x()+0.05, p.get_height()+1))
df.head()
df["fbs_group"] = df.fbs.map({0 : "Fasting Blood Sugar < 120 mg/dl" , 1 : "Fasting Blood Sugar > 120 mg/dl"})



fbs = df[['fbs_group']]

fbs["count"] = fbs.groupby(fbs.fbs_group)["fbs_group"].transform("count")

fbs = fbs.drop_duplicates()

fig = px.pie(data_frame = fbs , names = "fbs_group" , values = "count" , template = "seaborn")

fig.update_traces (textinfo = "percent+label" , pull = 0.05 , rotation = 90)

fig.show()
# relationship of cholestrol with target

sns.violinplot(data= df , x = "target" , y = "chol")
df["age_group"] = pd.cut(df["age"] , (0 , 30 , 50 , 80) , labels = ["0-30" , "30-50" , "50-80"])

df.head()
fig = sns.countplot(data= df , x = "age_group" , hue = "target")

for p in fig.patches:

    fig.annotate(p.get_height(), (p.get_x()+0.05, p.get_height()+1))
# importing necessary ML libraries

from sklearn.model_selection import GridSearchCV , train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
df = df.drop(["cp_type" , "target_group" , "fbs_group"  , "age_group"] , axis = 1)

X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']]

y = df["target"]

X.shape, y.shape
# scaling the features 

scaler = StandardScaler()

scaled_x = scaler.fit_transform(X)

scaled_x_df = pd.DataFrame(scaled_x  , columns = X.columns)

scaled_x_df.head()
x_train , x_test , y_train , y_test = train_test_split(scaled_x_df , y , test_size = 0.2)

x_train.shape , x_test.shape , y_train.shape , y_test.shape
scaled_x_df.describe()

# mean = 0 and std = 1 (approx)
models = [RandomForestClassifier() , SVC() , KNeighborsClassifier()]

param_grid = [{

    "n_estimators" : [50 , 70 , 100 , 150 , 200]

} , {

    "kernel" : ["rbf" , "poly"],

    "C" : [0.1 , 0.3 , 1 , 1.3 , 2 , 5, 10]

} , {

    "n_neighbors" : [5 , 10 , 15 , 20]

}]
scores = []

for i in range(len(models)):

    if i == 0:

        m = "Random Forest"

    elif i == 1 :

        m = "SVC"

    elif i == 2 :

        m = "KNN"

        

    grid = GridSearchCV(models[i] , param_grid = param_grid[i] , cv = 5 , scoring = "accuracy")

    grid.fit(x_train , y_train)

    scores.append({"model" : m , "best parameters" : grid.best_params_ , "best score" : grid.best_score_})

    

scores = pd.DataFrame(scores)

scores
model= RandomForestClassifier(n_estimators = 50)

model.fit(x_train , y_train)

model.score(x_test , y_test)
yp = model.predict(x_test)

cm = confusion_matrix(yp , y_test)

sns.heatmap(cm , annot = True)