## Calling the libraries
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
# To increase the display size for rows and columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


word=pd.read_excel(r"../input/lead-scoring-dataset/Leads Data Dictionary.xlsx",index_col=0)
print(word)
#reading the Leads csv file
df = pd.read_csv('../input/lead-scoring-dataset/Lead Scoring.csv')
# reading the first 5 rows
df.head()
# shape of the data frame
df.shape
# info of the dataframe
df.info()
#stastical information of the data frame
df.describe()
## checking the object columns
ob=df.select_dtypes(include=["object"]).columns
ob
#Replacing the select with null values for all columns
df = df.replace({'Select':np.nan})
#Percentage of missing values
null_perc = pd.DataFrame(round((df.isnull().sum())*100/df.shape[0],2)).reset_index()
null_perc.columns = ['Column Name', 'Null Values Percentage']
null_value = pd.DataFrame(df.isnull().sum()).reset_index()
null_value.columns = ['Column Name', 'Null Values']
null_lead = pd.merge(null_value, null_perc, on='Column Name')
null_lead.sort_values("Null Values", ascending = False)
## removing columns greater than 45% null values
null_column =round((df.isnull().sum()/len(df))*100,4) 
null_column_45 = null_column[null_column.values > 45.0000]
null_column_45 = list(null_column_45.index)
df.drop(labels=null_column_45,axis=1,inplace=True)
# Columns contains  data type objects
ob=df.select_dtypes(include=["object"]).columns
# Checking unique values and null values for the categorical columns
def Cat_info(df, categorical_column):
    df_result = pd.DataFrame(columns=["columns","values","unique_values"])
    
    df_temp=pd.DataFrame()
    for value in categorical_column:
        df_temp["columns"] = [value]
        df_temp["values"] = [df[value].unique()]
        df_temp["unique_values"] = df[value].nunique()
        df_result = df_result.append(df_temp)

    df_result.set_index("columns", inplace=True)
    return df_result
df_cat = Cat_info(df, ob)
df_cat
def column_category_counts(data):
    return pd.DataFrame(data.value_counts(dropna=False))


for column in ob:
    print("Column Name : ",column)
    display(column_category_counts(df[column]).T)
#Dropping columns which are highly skewed
df.drop(["Newspaper Article","Do Not Email","Do Not Call","What matters most to you in choosing a course","Search","Magazine","X Education Forums","Newspaper","Digital Advertisement","Through Recommendations","Receive More Updates About Our Courses","Update me on Supply Chain Content","Get updates on DM Content","I agree to pay the amount through cheque"],axis=1,inplace=True)
df.drop(["Tags","Prospect ID","Lead Number","City"],axis=1,inplace=True)
## Dropping Lead Notable activity as this field is similar to Lead activity
df.drop(["Last Notable Activity"],axis=1,inplace=True)
# Converting uneven distribution to "OTHERS" for Lead source, Last activity , Country and Last notable activity
df.loc[(df["Lead Source"].isin(["Facebook","bing","google","Click2call","Social Media","Live Chat","Press_Release","testone","welearnblog_Home","blog","youtubechannel","NC_EDM","Pay per Click Ads","WeLearn"])),"Lead Source"]="Other_Internet_Sources"
df.loc[(df["Last Activity"].isin(["Unreachable","Unsubscribed","Had a Phone Conversation","Approached upfront","View in browser link Clicked","Email Marked Spam","Email Received","Resubscribed to emails","Visited Booth in Tradeshow"])),"Last Activity"]="All Others"
df.loc[(df["Country"].isin(["Bahrain","Hong Kong","France","Oman","unknown","Nigeria","South Africa","Canada","Kuwait","Germany","Sweden","Ghana","Italy"                      
,"Belgium","China","Uganda","Asia/Pacific Region","Philippines","Bangladesh","Netherlands","Kenya","Sri Lanka","Indonesia","Denmark","Tanzania","Malaysia","Switzerland","Russia","Liberia","Vietnam"])),"Country"]="All Others"

# impute the mode for country, city, specialization and what is your current occupation with hightest value counts
df.loc[df['Specialization'].isnull(),'Specialization']=df['Specialization'].value_counts().index[0]
df.loc[df['Country'].isnull(),'Country']=df['Country'].value_counts().index[0]
df.loc[df['What is your current occupation'].isnull(),'What is your current occupation']=df['What is your current occupation'].value_counts().index[0]
## removing the remaining null values 
df=df.dropna()
# object data types columns
ob=df.select_dtypes(include=["object"]).columns
ob
for i in ob:
    plt.figure(figsize=(15,5))
    sns.countplot(df[i])
    plt.xticks(rotation='vertical')
## checking integer and float datatypes
nu=df.select_dtypes(include=["int","float"]).columns
nu
fig=px.box(df["TotalVisits"])

fig.show()
## Outer range of ouliers are moving to .95 percentile
q4=df["TotalVisits"].quantile(q=.95)
df["TotalVisits"][df["TotalVisits"]>=q4]=q4
fig=px.box(df["Page Views Per Visit"])
fig.show()
fig=px.box(df["Total Time Spent on Website"])
fig.show()
## converting to  q4 percentile
q4=df["Page Views Per Visit"].quantile(q=.95)
df["Page Views Per Visit"][df["Page Views Per Visit"]>=q4]=q4
fig=px.box(df["Page Views Per Visit"])
fig.show()
#dummy vaiables
df = pd.get_dummies(df,drop_first=True)
## checking the shape after adding the dummy variables
df.shape
## checking the info
df.info()
# Importing the required library to perform the test_train_split
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Converted'],axis=1)

#Putting the response variable in y
y = df[['Converted']]
# Performing the train_test_split with 70% of data for training set and 30% data for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state = 42)

X_train.shape , X_test.shape
dt = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=10)


dt.fit(X_train, y_train)
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_test_pred))
from sklearn.metrics import plot_roc_curve
plot_roc_curve(dt, X_train, y_train, drop_intermediate=False)
plt.show()
from sklearn.model_selection import GridSearchCV
dt_ = DecisionTreeClassifier(random_state=42)
params = {
    "max_depth": [2,3,5,10,20],
    "min_samples_leaf": [5,10,20,50,100,500]
}
grid_search = GridSearchCV(estimator=dt_,
                           param_grid=params,
                           cv=6,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train, y_train)
dtt=grid_search.best_score_
dt_best = grid_search.best_estimator_
dt_best
plot_roc_curve(dt_best, X_train, y_train)
plt.show()
dt_best.feature_importances_
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": dt_best.feature_importances_
})
imp_df.sort_values(by="Imp", ascending=False)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,random_state=42, n_jobs=-1, max_depth=5, min_samples_leaf=10,oob_score=True)
rf.fit(X_train, y_train)
rf.oob_score_
plot_roc_curve(rf, X_train, y_train)
plt.show()
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10, 25, 50, 100]
}
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(X_train, y_train)
dtr=grid_search.best_score_
rf_best = grid_search.best_estimator_
rf_best
plot_roc_curve(rf_best, X_train, y_train)
plt.show()
rf_best.feature_importances_
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})
imp_df.sort_values(by="Imp", ascending=False)