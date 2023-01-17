## Importing tools

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline
import xgboost as xg
# Importing data from .csv file

data_temp = pd.read_csv("mushrooms.csv")

data_temp.head()
data_temp.info
data_temp.info()
data_temp.isna().sum()
data_temp.head()
# See number of poisonous and edible mushroom

data_temp["class"].replace(['p','e'],[0,1],inplace = True)
data_temp.head()
data_temp.info()
data_temp["class"].value_counts().plot.bar(color = 'indigo');
data_temp["cap-shape"].unique()
data_temp.columns.dtype
# Convert all object features into integer data

from sklearn.preprocessing import LabelEncoder

def to_int_label_encoder(data):

    for column in data.columns:

        le = LabelEncoder()

        data[column] = le.fit_transform(data[column])

        data[column] = data[column].astype(np.int64)

    return data
to_int_label_encoder(data_temp)
data_temp.dtypes
data_temp.head()
# Relation between class and cap-shape

fig,ax = plt.subplots(tight_layout = True,figsize = (5,5))

ax.scatter(data_temp["cap-shape"],data_temp["class"],c = data_temp["class"],cmap = "winter")

ax.set(title = "Relation between class of mushroom and cap-shape",

      xlabel = "Cap Shape",

      ylabel = "Class");
data_temp["cap-shape"].value_counts()
data_temp["cap-shape"].plot.hist(figsize = (10,10));
cross = pd.crosstab(data_temp["class"],data_temp["cap-shape"])
cross
cross.plot.bar(figsize = (8,8),color = 'rgbkmyc')

plt.ylabel("Frequency")

plt.xticks([0,1],('Poisonous','Edible'))

plt.xlabel("Class")

plt.title("Cap shapes and Class of mushrooms");
data_temp.head()
# Relation between habitat and class of mushroom

fig,ax = plt.subplots()

ax.scatter(data_temp["class"],data_temp["habitat"],c = data_temp["class"],cmap = "viridis")

ax.set(title = " Relation between Class and habitat",

      xlabel = "Class",

      ylabel = "Habitat");
# Using bar graph to plot relation

pd.crosstab(data_temp["class"],data_temp["habitat"]).plot.bar()

plt.xticks([0,1],("Poisonous","Edible"));
data_temp.head()
# Relation between odor and class

pd.crosstab(data_temp["class"],data_temp["odor"]).T.plot.bar(stacked = True,figsize  =(10,10));
# Drop veil-type column

data_temp.drop("veil-type",axis = 1, inplace = True)
# Correlation matrix

f = plt.figure(figsize = (15,15))

plt.matshow(data_temp.corr(), fignum=f.number)

plt.xticks(range(data_temp.shape[1]),data_temp.columns,fontsize = 8, rotation = 45)

plt.yticks(range(data_temp.shape[1]),data_temp.columns,fontsize = 8)

cb = plt.colorbar()

cb.ax.tick_params(labelsize = 14)

plt.title("Correlation matrix",fontsize = 17);
data_temp.shape
# Separating data into features and labels

X = data_temp.drop("class",axis = 1)

y = data_temp["class"]
X.head(),y.head()
# Splitting into training and test set

np.random.seed(42)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
X_train.shape,y_train.shape
X_test.shape
# Using linear SVC model

np.random.seed(42)

from sklearn.svm import SVC

model1 = SVC()

model1.fit(X_train,y_train)
model1.score(X_test,y_test)
# Using K-nearest Neighbors

np.random.seed(42)

from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier()

model2.fit(X_train,y_train)

model2.score(X_train,y_train)
model2.score(X_test,y_test)
# Using Random Forest Classifier

np.random.seed(42)

from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier()

model3.fit(X_train,y_train)
model3.score(X_test,y_test)
from sklearn.model_selection import cross_val_score

# for SVC()

print(cross_val_score(model1,X,y,cv = 5))
# for KNearestNeighbors()

print(cross_val_score(model2,X,y,cv=5))
# for RandomForestClassifier()

print(cross_val_score(model3,X,y,cv=5))
s1 = cross_val_score(model1,X,y,cv=5).mean()

s2 = cross_val_score(model2,X,y,cv=5).mean()

s3 = cross_val_score(model3,X,y,cv=5).mean()
scores_df = {

    "SVC()": s1,

    "KNearestNeighbors()":s2,

    "RandomForestClassifier()":s3

}

scores_df = pd.DataFrame(scores_df,index = [0])
scores_df.T.plot.bar();
ypreds1 = model2.predict(X)

ypreds2 = model3.predict(X)

ypreds1.shape,ypreds2.shape
# Evaluating other metrics like precision,recall and f1-score

from sklearn.metrics import precision_score,recall_score,f1_score

def metrics_dict(ytrue,ypreds):

    m1 = precision_score(ytrue,ypreds)

    m2 = recall_score(ytrue,ypreds)

    m3 = f1_score(ytrue,ypreds)

    metrics = {

        "Precision" : m1,

        "Recall": m2,

        "F1 Score": m3

    }

    return metrics
m1 = metrics_dict(y,ypreds1)

m2 = metrics_dict(y,ypreds2)
m1,m2
data_temp.corr()
# ROC curve for two models

from sklearn.metrics import plot_roc_curve

np.random.seed(42)

plot_roc_curve(model2,X_test,y_test,linestyle = '-.')

plt.show()
np.random.seed(42)

plot_roc_curve(model3,X_test,y_test,linestyle = ':')

plt.show()
# PCA on data

from sklearn.decomposition import PCA

pca = PCA(n_components=14)

pca_data = pca.fit(X)
pca_data = pca.transform(X)
pca_data = pd.DataFrame(pca_data)
pca_data
X_train.shape,X_test.shape
np.random.seed(42)

X_pca_train,X_pca_test,y_pca_train,y_pca_test = train_test_split(pca_data,y,test_size = 0.2)
X_pca_train.shape,X_pca_test.shape
# Modelling on KNeighborsClassifier()

np.random.seed(42)

pca_model_1 = KNeighborsClassifier()

pca_model_1.fit(X_pca_train,y_pca_train)
pca_model_1.score(X_pca_test,y_pca_test)
# Evaluating other metrics on pca model

y_pca_preds1 = pca_model_1.predict(pca_data)

m1_pca = metrics_dict(y,y_pca_preds1)
m1_pca
m1_pca = pd.DataFrame(m1_pca,index = [0])
m1_pca.T.plot.bar();
# ROC curve on PCA model

plot_roc_curve(pca_model_1,X_pca_test,y_pca_test,linestyle = '-.')

plt.show();
# Confusion matrix

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(pca_model_1,X_pca_test,y_pca_test)

plt.show();
# On RandomForestClassifier

np.random.seed(42)

pca_model_2 = RandomForestClassifier()

pca_model_2.fit(X_pca_train,y_pca_train)

pca_model_2.score(X_pca_test,y_pca_test)

y_pca_preds2 = pca_model_2.predict(pca_data)

m2_pca = metrics_dict(y,y_pca_preds2)

m2_pca = pd.DataFrame(m2_pca,index = [0])

print(m2_pca)

m2_pca.T.plot.bar()

plot_roc_curve(pca_model_2,X_pca_test,y_pca_test,linestyle = '--')

plt.show()

plot_confusion_matrix(pca_model_2,X_pca_test,y_pca_test)

plt.show()
# Reducing more features using PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=10)

pca_data_2 = pca.fit_transform(X)

pca_data_2 = pd.DataFrame(pca_data_2)

pca_data_2.head()
# Splitting data into training and test set

np.random.seed(42)

X_pca_2_train,X_pca_2_test,y_pca_2_train,y_pca_2_test = train_test_split(pca_data_2,y,test_size = 0.2)

X_pca_2_train.shape,X_pca_2_test.shape
# Function to train,test and show the metrics for a model

def show_model(model,X_train,y_train,X_test,y_test,X,y):

    np.random.seed(42)

    Model = model

    Model.fit(X_train,y_train)

    Model.score(X_test,y_test)

    y_preds = Model.predict(X)

    m = metrics_dict(y,y_preds)

    m = pd.DataFrame(m,index = [0])

    print(m)

    m.T.plot.bar()

    plot_roc_curve(Model,X_test,y_test,linestyle = '--')

    plt.show()

    plot_confusion_matrix(Model,X_test,y_test)

    plt.show()

    return Model
model_x = show_model(KNeighborsClassifier(),X_pca_2_train,y_pca_2_train,X_pca_2_test,y_pca_2_test,pca_data_2,y)
model_y = show_model(RandomForestClassifier(),X_pca_2_train,y_pca_2_train,X_pca_2_test,y_pca_2_test,pca_data_2,y)
X.head()
# One hot encoding the data

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="error",drop = "if_binary")

X_enc = enc.fit_transform(X).toarray()

print(X_enc[:6])

enc.categories_
X_enc = pd.DataFrame(X_enc)

X_enc.head()
# Modelling and seeing results in onehotencoded data

np.random.seed(42)

X_enc_train,X_enc_test,y_enc_train,y_enc_test = train_test_split(X_enc,y,test_size = 0.2)

model_enc = show_model(KNeighborsClassifier(),X_enc_train,y_enc_train,X_enc_test,y_enc_test,X_enc,y)
model_enc_f = show_model(RandomForestClassifier(),X_enc_train,y_enc_train,X_enc_test,y_enc_test,X_enc,y)
import joblib

joblib.dump(model_x,filename="mushroom_model.joblib")