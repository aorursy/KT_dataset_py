# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization Library

import matplotlib.pyplot as plt 

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from collections import Counter



import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("../input/airline-passenger-satisfaction/test.csv")

test_df=pd.read_csv("../input/airline-passenger-satisfaction/train.csv")
train_df.columns
train_df.drop(labels=["Unnamed: 0"],axis=1,inplace=True)

test_df.drop(labels=["Unnamed: 0"],axis=1,inplace=True)
train_df.head()
train_df.info()
train_df.describe()
train_df.info()
def bar_plot(variable):

    

    var=train_df[variable]

    var_Value=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(var_Value.index,var_Value.values)

    

    plt.xlabel("Passengers Score")

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,var_Value))
category1=["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",]

for c in category1:

    bar_plot(c)
category2=["Gender", "Customer Type", "Type of Travel", "Class","satisfaction"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Fruequency")

    plt.title("{} distribution with histogram".format(variable))

    plt.show()

numericVar=["id","Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

for n in numericVar:

    plot_hist(n)
train_df.columns=[each.replace(" ","_") for each in train_df.columns]
train_df.head()
train_df["satisfaction"]=[1 if each=="satisfied" else 0 for each in train_df.satisfaction]
train_df.head(10)
# Gender vs satisfaction

train_df[["Gender","satisfaction"]].groupby(["Gender"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)

# Age vs satisfaction

train_df[["Age","satisfaction"]].groupby(["Age"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Type_of_Travel vs satisfaction

train_df[["Type_of_Travel","satisfaction"]].groupby(["Type_of_Travel"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Class vs satisfaction

train_df[["Class","satisfaction"]].groupby(["Class"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Customer_Type vs satisfaction

train_df[["Customer_Type","satisfaction"]].groupby(["Customer_Type"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Cleanliness vs satisfaction

train_df[["Cleanliness","satisfaction"]].groupby(["Cleanliness"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Inflight_wifi_service vs satisfaction

train_df[["Inflight_wifi_service","satisfaction"]].groupby(["Inflight_wifi_service"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Inflight_entertainment vs satisfaction

train_df[["Inflight_entertainment","satisfaction"]].groupby(["Inflight_entertainment"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Food_and_drink vs satisfaction

train_df[["Food_and_drink","satisfaction"]].groupby(["Food_and_drink"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
# Seat_comfort vs satisfaction

train_df[["Seat_comfort","satisfaction"]].groupby(["Seat_comfort"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)
numerical_features = train_df.select_dtypes(exclude=['object']).drop(["satisfaction"],axis=1).copy()

numerical_features.columns
fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9,4,i+1)

    sns.boxplot(y=numerical_features.iloc[:,i])



plt.tight_layout()

plt.show()
def detect_outliers(df,features):

    outlier_indices=[]

    

    for c in features:

        # 1st quartile

        Q1=np.percentile(df[c],25)

        

        # 3rd quartile

        Q3=np.percentile(df[c],75)

        

        # IQR

        IQR= Q3-Q1

        

        # Outlier Step

        outlier_step= IQR * 1.5

        

        # Detect outlier and their indeces 

        outlier_list_col = df[(df[c]< Q1 - outlier_step)|( df[c] > Q3 + outlier_step)].index

        

        # Store indices 

        outlier_indices.extend(outlier_list_col)

    

    outliers_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i , v in outliers_indices.items() if v>2 )

    return multiple_outliers
train_df.loc[detect_outliers(train_df,[ 'Age', 'Flight_Distance', 'Inflight_wifi_service',

       'Departure/Arrival_time_convenient', 'Ease_of_Online_booking',

       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',

       'Inflight_entertainment', 'On-board_service', 'Leg_room_service',

       'Baggage_handling', 'Checkin_service', 'Inflight_service',

       'Cleanliness', 'Departure_Delay_in_Minutes',

       'Arrival_Delay_in_Minutes'])]
# drop outliers

train_df = train_df.drop(detect_outliers(train_df,[ 'Age', 'Flight_Distance', 'Inflight_wifi_service',

       'Departure/Arrival_time_convenient', 'Ease_of_Online_booking',

       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',

       'Inflight_entertainment', 'On-board_service', 'Leg_room_service',

       'Baggage_handling', 'Checkin_service', 'Inflight_service',

       'Cleanliness', 'Departure_Delay_in_Minutes',

       'Arrival_Delay_in_Minutes']),axis = 0).reset_index(drop = True)
test_df.columns=[each.replace(" ","_") for each in test_df.columns]

test_df["satisfaction"]=[1 if each=="satisfied" else 0 for each in test_df.satisfaction]
train_df.shape
train_df_len=len(train_df)

train_df= pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
import missingno as msno

msno.bar(train_df)

plt.title("Missing Value Graphs")

plt.show()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
np.mean(train_df["Arrival_Delay_in_Minutes"])
train_df[train_df["Arrival_Delay_in_Minutes"].isnull()]
train_df.shape
train_df["Arrival_Delay_in_Minutes"]=train_df["Arrival_Delay_in_Minutes"].fillna(np.mean(train_df["Arrival_Delay_in_Minutes"]))
train_df[train_df["Arrival_Delay_in_Minutes"].isnull()]
plt.figure(figsize=(10,10))

list1=["Age",'Inflight_wifi_service',

       'Departure/Arrival_time_convenient', 'Ease_of_Online_booking',

       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',

       'Inflight_entertainment', 'On-board_service', 'Leg_room_service',

       'Baggage_handling', 'Checkin_service', 'Inflight_service',

       'Cleanliness',"satisfaction"]

sns.heatmap(train_df[list1].corr(),annot=True,fmt=".2f")

plt.show()
g=sns.catplot(x="Gender",y="satisfaction",data=train_df,kind="bar",size=6)

g.set_ylabels("Satisfaction Probability")

plt.show()
g= sns.FacetGrid(train_df,col="satisfaction")

g.map(sns.distplot,"Age",bins=25)

plt.show()

# 0=neutral or dissatisfied, 1=satisfied 
g=sns.factorplot(x="Customer_Type",y="satisfaction",data=train_df,kind="bar",size=6)

g.set_ylabels("Satisfaction Probability")

plt.show()
g=sns.factorplot(x="Type_of_Travel",y="satisfaction",data=train_df,kind="bar",size=6)

g.set_ylabels("Satisfaction Probability")

plt.show()
g=sns.factorplot(x="Class",y="satisfaction",data=train_df,kind="bar",size=6)

g.set_ylabels("Satisfation Probability")

plt.show()
sns.swarmplot(x="Gender", y="Age",hue="satisfaction", data=train_df.head(1000))

plt.show()

# 0=neutral or dissatisfied, 1=satisfied  
personal=train_df[train_df.Type_of_Travel=="Personal Travel"]

personal.head()
def service_plot(variable):

    

    var=personal[variable]

    var_Value=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(var_Value.index,var_Value.values)

    

    plt.xlabel("Passengers Score")

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,var_Value))
service=["On-board_service", "Leg_room_service", "Checkin_service","Inflight_service"]



for c in service:

    service_plot(c)
def eat_plot(variable):

    

    var=personal[variable]

    var_Value=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(var_Value.index,var_Value.values)

    

    plt.xlabel("Passengers Score")

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,var_Value))
eat=["Food_and_drink", "Cleanliness", "Inflight_entertainment"]



for c in eat:

    eat_plot(c)
def flight_plot(variable):

    

    var=personal[variable]

    var_Value=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(var_Value.index,var_Value.values)

    

    plt.xlabel("Passengers Score")

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,var_Value))
flight=["Gate_location", "Seat_comfort", "Baggage_handling"]



for c in flight:

    flight_plot(c)
train_df.head()
sns.countplot(x="Customer_Type",data=train_df)

train_df=pd.get_dummies(train_df,columns=["Customer_Type"])

train_df.head()
train_df.Type_of_Travel.head()
sns.countplot(x="Type_of_Travel",data=train_df)
train_df=pd.get_dummies(train_df,columns=["Type_of_Travel"])

train_df.head()
sns.barplot(x=train_df.Class.value_counts().values,y=train_df.Class.value_counts().index)

plt.xlabel("Number of Passenger")

plt.show()
train_df=pd.get_dummies(train_df,columns=["Class"])

train_df.head()
sns.countplot(x="Gender",data=train_df)
train_df=pd.get_dummies(train_df,columns=["Gender"])

train_df.head()
train_df.drop(labels=["id"],axis=1,inplace=True)
train_df.head()
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
train_df_len
test=train_df[train_df_len:]

test.drop(labels=["satisfaction"],axis=1,inplace=True)
test.head()
train=train_df[:train_df_len]

X_train=train.drop(labels="satisfaction",axis=1)

y_train=train["satisfaction"]

X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.33,random_state=42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg=LogisticRegression()

logreg.fit(X_train,y_train)

acc_log_train=round(logreg.score(X_train,y_train)*100,2)

acc_log_test=round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Test Accuracy: % {}".format(acc_log_test))
y_pred=logreg.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,fmt="d") 

plt.show()
random_state=42

classifier = [DecisionTreeClassifier(random_state=random_state),

             RandomForestClassifier(random_state=random_state),

             LogisticRegression(random_state=random_state),

             KNeighborsClassifier()]

dt_param_grid={"min_samples_split":range(10,500,20),

              "max_depth":range(1,20,2)}

rf_param_grid={"max_features":[1,3,10],

              "min_samples_split":[2,3,10],

              "min_samples_leaf":[1,3,10],

              "bootstrap":[False],

              "n_estimators":[100,300],

              "criterion":["gini"]}



logreg_param_grid={"C":np.logspace(-3,3,7),

                  "penalty":["l1","l2"]}

knn_param_grid={"n_neighbors": np.linspace(1,19,10,dtype=int).tolist(),

               "weights":["uniform","distance"],

               "metric":["euclidean","manhattan"]}

classifier_param=[dt_param_grid,

                 rf_param_grid,

                 logreg_param_grid,

                 knn_param_grid]

cv_result=[]

best_estimators=[]

for i in range(len(classifier)):

    clf=GridSearchCV(classifier[i],param_grid=classifier_param[i],cv=StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs=-1,verbose=1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":[0.9426983403065574,0.9574115458084643,0.8198176386217473,0.784126066638906], "ML Models":["DecisionTreeClassifier","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[1])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_satisfaction = pd.Series(votingC.predict(test), name = "satisfaction").astype(int)

results = pd.concat([test_df.id, test_satisfaction],axis = 1)

results.to_csv("satisfaction.csv", index = False)

results.head()