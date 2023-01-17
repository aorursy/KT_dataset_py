import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
df=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges']=df['TotalCharges'].convert_objects(convert_numeric=True)
dataobject=df.select_dtypes(['object'])

for i in range(1,len(dataobject.columns)):
    df[dataobject.columns[i]] = LabelEncoder().fit_transform(df[dataobject.columns[i]])
unwantedcolumnlist=["customerID","gender","MultipleLines","PaymentMethod","tenure"]

df = df.drop(unwantedcolumnlist, axis=1)
features = df.drop(["Churn"], axis=1).columns

df_train, df_val = train_test_split(df, test_size=0.30)
df_train['TotalCharges'].fillna(-999, inplace=True)
df_val['TotalCharges'].fillna(-999, inplace=True)

clf = RandomForestClassifier(n_estimators=30 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50)
clf.fit(df_train[features], df_train["Churn"])

# Make predictions
predictions = clf.predict(df_val[features])
probs = clf.predict_proba(df_val[features])

score = clf.score(df_val[features], df_val["Churn"])
print("Accuracy: ", score)

get_ipython().magic('matplotlib inline')
cm = pd.DataFrame(
    confusion_matrix(df_val["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(cm)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_val["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(clf, random_state=1).fit(df_val[features], df_val["Churn"])
eli5.show_weights(perm, feature_names = list(features))
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(perm, threshold=-0.0001, prefit=True)
X_tr = sel.transform(df_train[features])
X_val = sel.transform(df_val[features])

clf2 = RandomForestClassifier(n_estimators=30 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50)
clf2.fit(X_tr, df_train["Churn"])

# Make predictions
predictions = clf2.predict(X_val)
probs = clf2.predict_proba(X_val)

score = clf2.score(X_val, df_val["Churn"])
print("Accuracy: ", score)

get_ipython().magic('matplotlib inline')
cm = pd.DataFrame(
    confusion_matrix(df_val["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(cm)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_val["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
features = df_val.drop(["Churn"], axis=1).columns.values
feature_to_plot = 'TotalCharges'
pdp_goals = pdp.pdp_isolate(model=clf, dataset=df_val[features], model_features=list(features), feature=feature_to_plot)

# plot it
pdp.pdp_plot(pdp_goals, feature_to_plot)
plt.show()
# Create the data that we will plot
feature_to_plot = 'Contract'
pdp_goals = pdp.pdp_isolate(model=clf, dataset=df_val[features], model_features=list(features), feature=feature_to_plot)

# plot it
pdp.pdp_plot(pdp_goals, feature_to_plot)
plt.show()
