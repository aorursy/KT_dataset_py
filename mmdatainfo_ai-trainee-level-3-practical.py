import pandas as pd
# This will control the display options of pandas within this notebook only
pd.options.display.max_columns = None
pd.options.display.max_rows = 20
# Our master table containing main information
df = pd.read_csv("../input/ai-trainee/churn_data.csv")
print("Input data-frame shape: {}".format(df.shape))
df.head()
# The cutomer-specific data
customer = pd.read_csv("../input/ai-trainee/customer_data.csv")
customer.head()
# Contract data
contract = pd.read_csv("../input/ai-trainee/internet_data.csv")
contract.head()
# Read metadata. 
# If you get an error, just inspect the file manually and remove empty spaces or tabs after each column (' ,'>'')
# This actually happens very often that the provided data set contains some strange symbols leading to errors. Get custom to it :)
meta = pd.read_csv("../input/ai-trainee/Telecom Churn Data Dictionary.csv")
# the columns in input data and meta do not match (some are in lowercase, some contain empty spaces, ...)
# Will create a new column that will unify all. Same will be done in following section for input dataframes
meta = meta.assign(name_id=meta["Variable Name"].replace({" ":"","\t":""},regex=True).str.lower()).set_index("name_id")
meta.head()
# Set customer ID (=unique ID) as index for further joining
# This is clearly a repeating task. Imagine doing this ten times or so! => use a for loop
# In addition, rename the columns to be identical with "meta"
for i in [df, customer, contract]:
    i.set_index("customerID",inplace=True)
    i.rename(columns={j:j.lower() for j in i.columns},inplace=True)
    
df.head()
# Join all three data-frames (one after another)
df = df.join(customer).join(contract)

# Make sure no 1:N relation = no duplicates, print shape again (compare number of rows with input above)
print("Joined dataframe shape: {}".format(df.shape))
df.head()
from sklearn.model_selection import train_test_split
# to achive the identical result each run (just for this AI trainee lesson), use 'random_state' option
train, test = train_test_split(df,train_size=0.75,shuffle=True,random_state=123)
print(train.shape)
train.head()
def check_stats(df):
    """
    This function will return a table (dataframe) showing main statistics and additional indicators
    """
    # We will store the data types in a separate dataframe
    dfinfo = pd.DataFrame(df.dtypes,columns=["dtypes"])
    
    # We are interested if we have any missing data (sum all). 
    # Again, join the result with the dfinfo (append new column). Consider '' or ' ' also as missing
    dfinfo = dfinfo.join((df.replace({'':None,' ':None}) if "('O')" in str(df.dtypes.values) else df).isna().sum().rename("isna"))
        

    # In the last step, add statistics (will be computed for numerical columns only)
    # We need to "T"ranspose the dataframe to same shape as df.describe() output
    return dfinfo.T.append(df.describe(),sort=False)
check_stats(train).T.query("isna != 0")
# The missing data are in the input file marked with ' '
# drop them = overwrite the dataframe
test = test[test.totalcharges!=' ']
train = train[train.totalcharges!=' ']
# for visualization purposes, we will store the results in pandas dataframe and print the result in the next jupyter notebook cell
# nr of unique: will count the number of unique entries
# first 5 unique: will show first 5 unique entries (if less, only those)
dfsummary = pd.DataFrame({"nr of unique":[],"first 5 unique":[]})

# Run loop over all columns computing length (len) of unique entries in each column and converting first 5 enties to a string
for i in train.columns:
    dfsummary = pd.concat([dfsummary,pd.DataFrame({"nr of unique":[len(train[i].unique())],
                                                   "first 5 unique":[str(train[i].unique()[0:5])]},index=[i])],sort=False)
# join the result with metadata-column description
# Will join on column name. However, the provided metadata column names are not identical (e.g, contain empty spaces)
meta = meta.assign(join_name=meta["Variable Name"].replace({" ":"","\t":""},regex=True).str.lower()).set_index("join_name")

# need to do tha same for dfsummary
dfsummary = dfsummary.assign(join_name=dfsummary.index.astype(str).str.lower()).set_index("join_name")

# now having identical indices, join tables
dfsummary = dfsummary.join(meta[["Meaning"]])
# set column with only for the next step (to see the Meaning description)
pd.options.display.max_colwidth = 100
dfsummary
# set back to normal
pd.options.display.max_colwidth = 50
train = train.replace({'No phone service':'No','No internet service':'No'})
test = test.replace({'No phone service':'No','No internet service':'No'})
df = df.replace({'No phone service':'No','No internet service':'No'})
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Tenure, seniorcitizen, and monthlycharges do not need to be converted
# We will convert 
train_e = train[["tenure","seniorcitizen","monthlycharges"]].copy()
test_e = test[["tenure","seniorcitizen","monthlycharges"]].copy()

# Convert to float (str because of missing data that was, however, dropped before)
train_e["totalcharges"] = train.totalcharges.astype("float64")
test_e["totalcharges"] = test.totalcharges.astype("float64")
# We want to ensure that 'No' is 0 and 'Yes' is 1. 
# To do that "fit" the encoder first, and apply (=transform) afterwards
# "fit" means that the object/variable "le_no_yes" will "remember" that no=0, and yes=1
le_no_yes = LabelEncoder().fit(['No','Yes'])

# now, apply to all columns where 'Yes', 'No' (or inverse order) occurs
for i in dfsummary.index:
    if "yes" in dfsummary.loc[i,"first 5 unique"].lower() and "no" in dfsummary.loc[i,"first 5 unique"].lower():
        print(i)
        train_e[i] = le_no_yes.transform(train[i])
        test_e[i] = le_no_yes.transform(test[i])
# to ensure the values are "ordered" = month-to-month = 0, year=1, 2 years = 2, fit first
le_contract = LabelEncoder().fit(['Month-to-month','One year','Two year'])
train_e["contract"] = le_contract.transform(train["contract"])
test_e["contract"] = le_contract.transform(test["contract"])
# Here is a for loop that will convert all remaining columns applying OneHotEncoder
# we will "declare" the encoder just to use its 'categories_' attribute 
ohe = OneHotEncoder() 
for i in df.columns:
    if i not in train_e.columns:
        print(i)
        # fit = get new mapping for each column
        ohe = OneHotEncoder().fit(train[i].unique().reshape(-1,1))
        # OneHotEncoder (just like ML models) expects/require a numpy matrix/array as input
        temp = pd.DataFrame(ohe.transform(df[i].to_numpy().reshape(-1,1)).toarray(),
                            index=df.index,
                            columns=[i+"_"+cat.lower().replace(" ","_") for cat in ohe.categories_[0]])
        # Check also category-encoders library for easier encoding
        train_e = train_e.join(temp)
        test_e = test_e.join(temp)
train_e.head()
check_stats(train_e)
train_e.corr().round(3).style.background_gradient(cmap="viridis")
# see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler
# It makes, of course, only sense to apply scaling only to numerical values
# Again, "fit" on train and apply/"transform" test
scaler = StandardScaler()
for i in [["tenure","monthlycharges","totalcharges"]]:
    train_e[i] = scaler.fit_transform(train_e[i])
    test_e[i] = scaler.transform(test_e[i])
check_stats(train_e)
import plotly as py
import plotly.graph_objects as go
py.offline.init_notebook_mode(connected=True)
fig = go.Figure(data=[go.Bar(
                            x=train_e.churn.value_counts().index, 
                            y=train_e.churn.value_counts(),
                            text=train_e.churn.value_counts().index,
                            name="train"
                            ),
                      go.Bar(
                            x=test_e.churn.value_counts().index, 
                            y=test_e.churn.value_counts(),
                            text=test_e.churn.value_counts().index,
                            name="test"
                            ),
                     ],
                layout = go.Layout(
                                   title="Checking target/churn imbalance",
                                   # Reduce default (rather big) margins between Figure edges and axes
                                   margin=go.layout.Margin(l=50,r=50,b=50,t=50),
                                   # Set figure size
                                   width=600,
                                   height=400,
                                   xaxis=go.layout.XAxis(
                                                        showgrid=False,
                                                        zeroline=False,
                                                        showticklabels=False
                                                        )
                                   )
               )

fig.show()
# select features & labels
X_train, X_test = train_e.drop(columns=["churn"]).to_numpy(), test_e.drop(columns=["churn"]).to_numpy()
y_train, y_test = train_e["churn"].to_numpy(), test_e["churn"].to_numpy()
# Scikit-learn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# Classification metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# cross-validation
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=50)
du = DummyClassifier(strategy="stratified")
for clf in [du,dt,rf]:
    clf.fit(X_train,y_train)
    print(f"\n{clf}")
    print(classification_report(y_test,clf.predict(X_test)))
# ## Visualize the tree: https://www.kaggle.com/willkoehrsen/visualize-a-decision-tree-w-python-scikit-learn

# from sklearn.tree import export_graphviz
# # Export as dot file
# export_graphviz(dt, out_file='tree.dot', 
#                 feature_names = list(train_e.drop(columns=["churn"]).columns),
#                 class_names = ['no','yes'],
#                 rounded = True, proportion = False, 
#                 precision = 2, filled = True)

# # Convert to png > works on Linux
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# # Display in python
# import matplotlib.pyplot as plt
# plt.figure(figsize = (14, 18))
# plt.imshow(plt.imread('tree5.png'))
# plt.axis('off');
# plt.show();

tuning_parameter = {'min_samples_leaf': [1, 3, 6],
                    'min_samples_split': [2, 10, 15],
                    'n_estimators': [100,350,600]}
cv = KFold(n_splits=5)
rf = GridSearchCV(RandomForestClassifier(),
                  param_grid=tuning_parameter, 
                  scoring="f1_weighted",
                  cv=cv,
                  n_jobs = 10,
                  ) # verbose=10
rf.fit(X_train,y_train)
rf.best_estimator_
print(classification_report(y_test,rf.best_estimator_.predict(X_test)))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
pca = PCA(n_components=0.999)# or set n_components="mle"
# As always, fit on train, transform test
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("Nr. of features after PCA = {} (input = {})".format(X_train_pca.shape[1],X_train.shape[1]))
tuning_parameter = {"n_neighbors":list(range(1,30,2)),
                    "weights":["uniform","distance"]}
knn = GridSearchCV(KNeighborsClassifier(), 
                   tuning_parameter, 
                   cv=cv,
                   scoring="f1_weighted",
                   n_jobs = 10)
knn.fit(X_train_pca,y_train)
knn.best_estimator_
print(classification_report(y_test,knn.best_estimator_.predict(X_test_pca)))
from sklearn.svm import SVC
import numpy as np
tuning_parameter = {"C":np.logspace(-3, 3, 7),
                    "gamma":np.logspace(-3, 3, 7)}
svc = GridSearchCV(SVC(kernel="rbf"), 
                      tuning_parameter, 
                      cv=cv,
                      scoring="f1_weighted",
                      n_jobs = 10,
                      return_train_score=True)
svc.fit(X_train,y_train)
svc.best_estimator_
print(classification_report(y_test,svc.best_estimator_.predict(X_test)))
# To show how the score varies on parameters, plot the results (in test set!)
#pd.DataFrame(search.cv_results_)



# ##You can try to train SVM with the reduced data set: the results should be similar
# svc.fit(X_train_pca,y_train)
# print(classification_report(y_test,svc.best_estimator_.predict(X_test_pca)))
# ##Try re-sampling to suppres the moderate imbalance
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
# tuning_parameter = {'min_samples_leaf': [1, 3, 6],
#                     'min_samples_split': [2, 10, 15],
#                     'n_estimators': [100,350,600]}
# rf = GridSearchCV(RandomForestClassifier(),
#                   param_grid=tuning_parameter, 
#                   scoring="f1_weighted",
#                   cv=cv,
#                   n_jobs = 10,
#                   )
# rf.fit(X_resampled, y_resampled)
# print(classification_report(y_test,rf.best_estimator_.predict(X_test)))
# ## Check if the result changes if using more k-folds
# search = GridSearchCV(RandomForestClassifier(),
#                       param_grid={'min_samples_leaf': [1, 3, 6],
#                                   'min_samples_split': [2, 10, 15],
#                                   'n_estimators': [100,350,600]}, 
#                       scoring="f1_weighted",
#                       cv=KFold(n_splits=10),
#                       n_jobs = 10,
#                       )
# rf.fit(X_train,y_train)
# print(classification_report(y_test,rf.best_estimator_.predict(X_test)))
