#Imports for Data Science



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import train_test_split
# Storing the Dataset as an Object

df_raw = pd.read_csv("../input/Pokemon.csv")
!cd
#Data Cleaning

df_raw["Mega"] = 0

df_raw.loc[df_raw["Name"].str.contains("Mega"),"Mega"] = 1

df_raw.drop(df_raw[df_raw["Mega"] == 1].index,axis =0,inplace = True) #Cleaning to remove Mega Items

del_cols = ["#","Generation","Mega"]

df_raw = df_raw.drop(del_cols,axis = 1)

df_raw["Single Type"] = 0

df_raw.loc[df_raw["Type 2"].isnull() == True,"Single Type"] = 1

df = df_raw.copy()
df_raw.head()
#EDA - How do Stats of Legendary Pokemon compare with Normal

plt.hist(df[df["Legendary"] == False]["Total"],bins = 20,label = "Non-Legend")

plt.hist(df[df["Legendary"] == True]["Total"],bins = 20,label = "Legend")

plt.legend()
sns.pairplot(data = df_raw,hue = "Legendary")
df.head()
#First Iteration of Logistic Regression only using Total Stats

X = df[["Total","Single Type"]]

y = df["Legendary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Logistic Regression Algorithm



log = LogisticRegression()

log.fit(X_train,y_train)

predictions = log.predict(X_test)



#Data Metrics

print(confusion_matrix(y_test,predictions))

print("\n")

print(classification_report(y_test,predictions))





pred_series = pd.Series(predictions,index = y_test.index)

name_list = df.loc[pred_series.index]["Name"]

compare = pd.concat([name_list,y_test,pred_series],axis = 1)

compare.columns = ["Name","Actual","Predicted"]

wrong_ind = compare[compare.Actual != compare.Predicted].index

correct_ind = compare[compare["Predicted"] == True].index
compare[compare["Predicted"] ==True]
actual_leg_ind = compare[compare["Actual"] ==True].index
X_test.loc[wrong_ind]
X_test.loc[correct_ind]
#Second Iteration of Logistic Regression - Now using all Stats instead of just total stats



X_2 = df[["Single Type","HP","Speed","Attack","Defense","Sp. Atk","Sp. Def"]]

y_2 = df["Legendary"]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.33, random_state=42)

#Logistic Regression Algorithm



log = LogisticRegression()

log.fit(X_train,y_train)

predictions = log.predict(X_test)



#Data Metrics

print(confusion_matrix(y_test,predictions))

print("\n")

print(classification_report(y_test,predictions))



df.loc[actual_leg_ind]
df[df["Name"].str.contains("Kyogre")]
#When using a Decision Tree, it can deal with String Inputs 

#-> See if using Types will allow the Algorithm to have higher accuracy

X = df[["Total","Single Type"]]

y = df["Legendary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Different set of X-Values to include other stats

X_2 = df[["Total", "HP","Speed","Attack","Defense","Sp. Atk","Sp. Def"]]

y_2 = df["Legendary"]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.33, random_state=42)
#Importing Decision Tree ML Algorithm

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
#Fitting models to data with only Total Values 

dtc.fit(X_train,y_train)

rfc.fit(X_train,y_train)



#Running Metrics on ML Algorithms

dtc_pred= dtc.predict(X_test)



print(confusion_matrix(y_test,dtc_pred))

print("\n")

print(classification_report(y_test,dtc_pred))



rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))

print("\n")

print(classification_report(y_test,rfc_pred))
#Fitting Models to Train on all Stats instead of just Total Stats

dtc.fit(X_train_2,y_train_2)

rfc.fit(X_train_2,y_train_2)



dtc_pred_2 = dtc.predict(X_test_2)

print(confusion_matrix(y_test_2,dtc_pred_2))

print("\n")

print(classification_report(y_test_2,dtc_pred_2))



rfc_pred_2 = rfc.predict(X_test_2)

print(confusion_matrix(y_test_2,rfc_pred_2))

print("\n")

print(classification_report(y_test_2,rfc_pred_2))
wrong_class(df,y_test_2,rfc_pred_2,X_test_2)
wrong_class(df,y_test,dtc_pred,X_test)
dtc_series = pd.Series(dtc_pred,index = y_test.index)

name_list = df.loc[pred_series.index]["Name"]

compare = pd.concat([name_list,y_test,dtc_series],axis = 1)

compare.columns = ["Name","Actual","Predicted"]

compare.sort_index(inplace= True)

compare.head()
compare[(compare["Predicted"] == True) & (compare.Actual == False)]
def wrong_class(dataset,y_test,pred,X_test):

    '''

    Parameters

    ----------------

    df: Original Dataset which contains the original Index

    y_test: Actual Classifiers - Pandas Series

    pred: Predicted Classifier - Numpy Array

    '''

    print(type(y_test))

    pred_series = pd.Series(pred,index = y_test.index) #Convert predicted values from a numpy array to a Pandas Series

    name_list = dataset.loc[pred_series.index]["Name"]

    X_test_data = X_test.loc[pred_series.index]

    compare = pd.concat([name_list,y_test,pred_series,X_test_data],axis =1 )

    cols = ["Name","Actual","Predicted"]

    for i in X_train.columns:

        cols.append(i)

    compare.columns = cols

    #Note that for df.sort_values(), the axis parameter can be a little counter intuitive

    #Axis refers to the axis where rearrangement takes place, in this case we want to rearrange the rows (axis = 0) but the position of the columns (axis = 1) remains the same

    return(compare[compare["Actual"]!= compare["Predicted"]].sort_values(by = ["Predicted"],axis = 0))
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz,export_text,plot_tree







plot_tree(dtc)