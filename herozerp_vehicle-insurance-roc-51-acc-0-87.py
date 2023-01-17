# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

plt.style.use('ggplot')



import cufflinks as cf

import plotly.express as px

import plotly.offline as py

from plotly.offline import plot

import plotly.graph_objects as go

import plotly.graph_objs as go



from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import f1_score, roc_auc_score

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from sklearn.utils.multiclass import type_of_target



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
train_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
print("We have :", train_df.shape[0], "Rows in the Train set")

train_df.head()
print("We have :", test_df.shape[0], "Rows in the Test set")

train_df.head()
train_df = train_df.drop(["id"], axis=1)

test_df = test_df.drop(["id"], axis=1)
null_train = train_df.isnull().sum().sum()

null_test = test_df.isnull().sum().sum()



print("There's", null_train, "null value in the Train set")

print("There's", null_test, "null value in the Test set")
train_df["Response"].value_counts().plot.bar(colormap="autumn")
negative_response = train_df[train_df["Response"] == 0].value_counts().sum()

positive_response = train_df[train_df["Response"] == 1].value_counts().sum()

print("The percentage of positive response is :", round(positive_response*100/negative_response), "%")
pd.crosstab(train_df['Response'], train_df['Gender']).plot(kind="bar", figsize=(8,6), colormap="autumn")



plt.title("Response by Gender")



plt.xlabel("0 : Customer Not interested, 1 : Customer Interested")

plt.ylabel("Count")



plt.legend(["Female", "Male"])



plt.xticks(rotation=0);
pd.crosstab(train_df['Response'], train_df['Previously_Insured']).plot(kind="bar", figsize=(8,6), colormap="autumn")



plt.title("Response by Previously Insured")



plt.xlabel("0 : Customer Not interested, 1 : Customer Interested")

plt.ylabel("Count")



plt.legend(["Client without Insurance", "Client with already Insurance"])



plt.xticks(rotation=0);
pd.crosstab(train_df['Response'], train_df['Driving_License']).plot(kind="bar", figsize=(8,6), colormap="autumn")



plt.title("Response by Driving License")



plt.xlabel("0 : Customer Not interested, 1 : Customer Interested")

plt.ylabel("Count")



plt.legend(["Client without Driving License", "Client with Driving License"])



plt.xticks(rotation=0);
train_df = train_df.drop(["Driving_License"], axis=1)

test_df = test_df.drop(["Driving_License"], axis=1)
pd.crosstab(train_df['Response'], train_df['Vehicle_Age']).plot(kind="bar", figsize=(10,6), colormap="autumn")



plt.title("Response by Vehicle Age")



plt.xlabel("0 : Customer Not interested, 1 : Customer Interested")

plt.ylabel("Count")



plt.legend(["1-2 Year", "< 1 Year", "> 2 Years"])



plt.xticks(rotation=0);
pd.crosstab(train_df['Response'], train_df['Vehicle_Damage']).plot(kind="bar", figsize=(10,6), colormap="autumn")



plt.title("Response by Vehicle Damage")



plt.xlabel("0 : Customer Not interested, 1 : Customer Interested")

plt.ylabel("Count")



plt.legend(["Vehicle damage", "No vehicle damage"])



plt.xticks(rotation=0);
#Graph : Age by responses

fig = px.bar(train_df["Age"].value_counts(), orientation="v", color=train_df["Age"].value_counts(), color_continuous_scale=px.colors.sequential.Plasma, 

             log_x=False, labels={'value':'Count', 

                                'index':'Ages',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Age by number of responses"

)



fig.show()
train_df.head()
def encoding_gender(item):

    if item == "Male":

        return 0

    else:

        return 1

    

train_df["Gender"] = train_df["Gender"].apply(encoding_gender)

test_df["Gender"] = test_df["Gender"].apply(encoding_gender)



train_df["Gender"].value_counts()
def encoding_vehicle_age(item):

    if item == "< 1 Year":

        return 0

    elif item == "1-2 Year":

        return 1

    else:

        return 2

    

train_df["Vehicle_Age"] = train_df["Vehicle_Age"].apply(encoding_vehicle_age)

test_df["Vehicle_Age"] = test_df["Vehicle_Age"].apply(encoding_vehicle_age)



train_df["Vehicle_Age"].value_counts()
def encoding_vehicle_dmg(item):

    if item == "No":

        return 0

    else:

        return 1

    

train_df["Vehicle_Damage"] = train_df["Vehicle_Damage"].apply(encoding_vehicle_dmg)

test_df["Vehicle_Damage"] = test_df["Vehicle_Damage"].apply(encoding_vehicle_dmg)



train_df["Vehicle_Damage"].value_counts()
train_df.head()
test_df.head()
plt.figure(figsize=(12, 8))

sns.heatmap(train_df.corr(), annot=True)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,recall_score

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer

from sklearn.utils.multiclass import type_of_target



from catboost import CatBoostClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression



from sklearn import preprocessing
#Standardization



numerical_cols = ['Age', 'Vintage', 'Policy_Sales_Channel', 'Region_Code']



scaler = StandardScaler()

train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])



scaler_2 = MinMaxScaler()

train_df[["Annual_Premium"]] = scaler_2.fit_transform(train_df[["Annual_Premium"]])



train_df.head()
X = train_df.drop(["Response"], axis=1)

y = train_df['Response']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# r = 42



# RFC = RandomForestClassifier(random_state = r)

# LGR = LogisticRegression(max_iter=10000)

# KNN = KNeighborsClassifier(n_neighbors = 10)

# SGD = SGDClassifier()



# classifiers = [RFC, ADA, KNN, XGB]

# classifiers_names = ['Random Forest',

#                      'Logistic Regreesion',

#                      'KNeighborsClassifier',

#                      'SGD Classifier']

# acc_mean = []



# for cl in classifiers:

#     acc = cross_val_score(estimator = cl, X = X_train, y  = y_train, cv = 2)

#     acc_mean.append(acc.mean()*100)

    

# acc_df = pd.DataFrame({'Classifiers': classifiers_names,

#                        'Accuracies Mean': acc_mean})



# acc_df.sort_values('Accuracies Mean',ascending=False)
final_model = KNeighborsClassifier(n_neighbors = 11)



final_model.fit(X_train, y_train)

y_pred_final_model = final_model.predict(X_test)

accuracy_score(y_test, y_pred_final_model)
#k_range = list(range(1,31))

#weight_options = ["uniform", "distance"]



#param_grid = dict(n_neighbors = k_range, weights = weight_options)

#print (param_grid)

#KNN = KNeighborsClassifier()



#grid = GridSearchCV(KNN, param_grid, cv = 10, scoring = 'accuracy')

#grid.fit(X,y)



#print(grid.best_score_)

#print(grid.best_params_)

#print(grid.best_estimator_)
KNN = KNeighborsClassifier()



final_model = KNeighborsClassifier(n_neighbors = 30, weights = "uniform")



final_model.fit(X_train, y_train)

y_pred_final_model = final_model.predict(X_test)

accuracy_score(y_test, y_pred_final_model)
roc_auc_score(y_test, y_pred_final_model, average = 'weighted')
f1_score(y_test, y_pred_final_model, average='weighted')
recall_score(y_test, y_pred_final_model, average='weighted')