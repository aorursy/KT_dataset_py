import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Analysis

import time

import pandas as pd

import numpy as np

import scipy.stats as stats



#Data Visualistion

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.tools import make_subplots

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

from plotly.graph_objs import *



#Data Pre Processing

from sklearn import preprocessing

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import LabelEncoder 

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce



#Modeling

import statsmodels.api as sm

from statsmodels.formula.api import ols

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



#Validation

from sklearn.model_selection import cross_val_score
train_df = pd.read_csv("/kaggle/input/competition09marketing-analytics/train.csv")

users_df = pd.read_csv("/kaggle/input/competition09marketing-analytics/users.csv")

test_df = pd.read_csv("/kaggle/input/competition09marketing-analytics/test.csv")

submission_df = pd.read_csv("/kaggle/input/competition09marketing-analytics/sample_submission_0_1.csv")
print("========== Training Dataset ===============")
train_df.head(2)
print("========== Test Dataset ===============")
test_df.head(2)
print("========== User Dataset ===============")
users_df.head(2)
print("========== Statistics of Training Dataset ===============")
train_df.describe()
print("========== Statistics of Test Dataset ===============")
test_df.describe()
print("========== Statistics of User Dataset ===============")
users_df.describe()
train_df = pd.merge(train_df, users_df, how='inner', on=["user_id"])

test_df = pd.merge(test_df, users_df, how='inner', on=["user_id"])
def epoch_converter(date_time_string):

    date_time_string

    date_time_format = "%Y-%m-%d %H:%M:%S%z"

    time_object = time.strptime(date_time_string, date_time_format)

    return time.mktime(time_object)



train_df["Epoch Time"] = train_df["grass_date"].apply(epoch_converter)

test_df["Epoch Time"] = test_df["grass_date"].apply(epoch_converter)
train_df.isnull().sum()
test_df.isnull().sum()
train_df["last_open_day"] = train_df["last_open_day"].apply(lambda x: 10**10 if x == "Never open" else x)

train_df["last_login_day"] = train_df["last_login_day"].apply(lambda x: 10**10 if x == "Never login" else x)

train_df["last_checkout_day"] = train_df["last_checkout_day"].apply(lambda x: 10**10 if x == "Never checkout" else x)





test_df["last_open_day"] = test_df["last_open_day"].apply(lambda x: 10**10 if x == "Never open" else x)

test_df["last_login_day"] = test_df["last_login_day"].apply(lambda x: 10**10 if x == "Never login" else x)

test_df["last_checkout_day"] = test_df["last_checkout_day"].apply(lambda x: 10**10 if x == "Never checkout" else x)
train_df["domain"] = train_df["domain"].astype('category')

train_df["domain_cat"] = train_df["domain"].cat.codes



test_df["domain"] = test_df["domain"].astype('category')

test_df["domain_cat"] = test_df["domain"].cat.codes
#Removing Unecessary Columns

train_df = train_df.drop(['user_id', 'domain', "row_id"], axis=1)

test_df = test_df.drop(['user_id', 'domain', "row_id"], axis=1)
train_df.head(2)
train_df.isna().sum()

test_df.isna().sum()
train_df = train_df.fillna(-1)

test_df = test_df.fillna(-1)
train_df.isnull().sum()

test_df.isnull().sum()
def time_to_categorical_series(df,type="hour"):

    if type == "hour":

        return df['date_time'].dt.hour.astype('category')

    elif type == "dayofweek":

        return df['date_time'].dt.dayofweek.astype('category')

    elif type == "month":

        return df['date_time'].dt.month.astype('category')

    else:

        return None

    

def time_to_categorical(df):

    hour_series = time_to_categorical_series(df,type='hour')

    dayofweek_series = time_to_categorical_series(df,type='dayofweek')

    month_series = time_to_categorical_series(df,type='month')



    df['hour'] = hour_series

    df['dayofweek'] = dayofweek_series

    df['month'] = month_series
train_df['date_time'] = pd.to_datetime(train_df['grass_date'])

test_df['date_time'] = pd.to_datetime(test_df['grass_date'])
time_to_categorical(train_df)

time_to_categorical(test_df)
train_df.to_csv("train_df_cleaned.csv")

test_df.to_csv("test_df_cleaned.csv")
#Plotting Histogram

def plotHistogram(variable):

    """Plots histogram and density plot of a variable."""

    

    # Create subplot object.

    fig = make_subplots(

        rows=2,

        cols=1,

        print_grid=False,

    subplot_titles=(f"Distribution of {variable.name} with Histogram", f"Distribution of {variable.name} with Density Plot"))

    

    # This is a count histogram

    fig.add_trace(

        go.Histogram(

            x = variable,

            hoverinfo="x+y",

            marker = dict(color = "chocolate")

        ),

    row=1,col=1)

    

    # This is a density histogram

    fig.add_trace(

        go.Histogram(

            x = variable,

            hoverinfo="x+y",

            histnorm = "density",

            marker = dict(color = "darkred")

        ),

    row=2,col=1)

    

    # Update layout

    fig.layout.update(

        height=400, 

        width=800,

        hovermode="closest",

        showlegend=False,

        paper_bgcolor="rgb(243, 243, 243)",

        plot_bgcolor="rgb(243, 243, 243)"

        )

    

    # Update axes

    fig.layout.yaxis1.update(title="<b>Abs Frequency</b>")

    fig.layout.yaxis2.update(title="<b>Density(%)</b>")

    fig.layout.xaxis2.update(title=f"<b>{variable.name}</b>")

    return fig.show()
train_df.columns
df_Count = train_df['country_code'].value_counts().rename_axis('country_code').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="country_code", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['subject_line_length'].value_counts().rename_axis('subject_line_length').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="subject_line_length", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['last_open_day'].value_counts().rename_axis('last_open_day').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="last_open_day", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['last_login_day'].value_counts().rename_axis('last_login_day').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="last_login_day", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['last_checkout_day'].value_counts().rename_axis('last_checkout_day').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="last_checkout_day", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['age'].value_counts().rename_axis('age').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="age", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['Epoch Time'].value_counts().rename_axis('Epoch Time').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="Epoch Time", y='Count',nbins=100)

fig.update_layout(height=300)

fig.show('notebook')
df_Count = train_df['domain_cat'].value_counts().rename_axis('domain_cat').reset_index(name='Count')

df_Count



#Plotly Style

fig = px.histogram(df_Count, x="domain_cat", y='Count',nbins=50)

fig.update_layout(height=300)

fig.show('notebook')
train_df.dtypes
xTrain = train_df.drop(columns=["open_flag"])



#Checking for collinearity

pearsoncorr = xTrain.corr(method='pearson')

#Styling

plt.figure(figsize=(30, 10))

sns.heatmap(pearsoncorr, 

            xticklabels=pearsoncorr.columns,

            yticklabels=pearsoncorr.columns,

            cmap='RdBu_r',

            annot=True,

            linewidth=0.3)
train_df.columns
continuous_variable_list = ['subject_line_length','last_open_day','last_login_day','last_checkout_day','open_count_last_10_days','open_count_last_30_days','open_count_last_60_days', 'login_count_last_10_days',

                            'login_count_last_30_days','login_count_last_60_days','checkout_count_last_10_days','checkout_count_last_30_days','checkout_count_last_60_days','attr_1','attr_2','attr_3',

                            'age','Epoch Time']
anova_dict = {}

for that_column in continuous_variable_list:

    

    numVariable = train_df[that_column]

    catVariable = train_df["open_flag"]

    #Seperating into the 2 different population dataset

    groupNumVariableByCatVariable0 = numVariable[catVariable == 0]

    groupNumVariableByCatVariable1 = numVariable[catVariable == 1]



    fValue, pValue = stats.f_oneway(groupNumVariableByCatVariable0, groupNumVariableByCatVariable1)

   

    anova_dict[that_column] = pValue
anova_dict
discrete_variable_list = ["country_code", "domain_cat"]



chi_sq_dict = {}

for that_column in discrete_variable_list:

    

    X = train_df[that_column]

    y = train_df["open_flag"]



    array_by_open_flag = pd.crosstab(index = X, columns = y)



    chi2_stat, p_val, dof, ex = stats.chi2_contingency(array_by_open_flag)

   

    chi_sq_dict[that_column] = p_val
chi_sq_dict
train_df.dtypes
train_df.head()
train_df = train_df.drop(columns=["grass_date", "open_count_last_30_days","open_count_last_60_days","login_count_last_30_days","login_count_last_60_days","checkout_count_last_30_days","checkout_count_last_60_days", "Epoch Time", "date_time"])

test_df = test_df.drop(["grass_date", "open_count_last_30_days","open_count_last_60_days","login_count_last_30_days","login_count_last_60_days","checkout_count_last_30_days","checkout_count_last_60_days", "Epoch Time", "date_time"], axis=1)
train_df.head()
train_df.age.describe()
#Create bin categories for Age.

ageGroups = [1,2,3,4,5]



#Create range for each bin categories of Age.

groupRanges = [-17,0,23,31,40,118]



#Create and view categorized Age with original Age.

train_df["Age Binned"] = pd.cut(train_df.age, groupRanges, labels = ageGroups)

test_df["Age Binned"] = pd.cut(test_df.age, groupRanges, labels = ageGroups)
train_df.head()
train_df = train_df.drop(columns=["age"])

test_df = test_df.drop(columns=["age"])
train_df.columns
train_df.dtypes
encoder = ce.BinaryEncoder(cols=["domain_cat"])

df_bin = encoder.fit_transform(train_df['domain_cat'])

train_df = pd.concat([train_df, df_bin], axis=1)



encoder = ce.BinaryEncoder(cols=["domain_cat"])

df_bin = encoder.fit_transform(test_df['domain_cat'])

test_df = pd.concat([test_df, df_bin], axis=1)
encoder = ce.BinaryEncoder(cols=["country_code"])

df_bin = encoder.fit_transform(train_df['country_code'])

train_df = pd.concat([train_df, df_bin], axis=1)



encoder = ce.BinaryEncoder(cols=["country_code"])

df_bin = encoder.fit_transform(test_df['country_code'])

test_df = pd.concat([test_df, df_bin], axis=1)
encoder = ce.BinaryEncoder(cols=["hour"])

df_bin = encoder.fit_transform(train_df['hour'])

train_df = pd.concat([train_df, df_bin], axis=1)



encoder = ce.BinaryEncoder(cols=["hour"])

df_bin = encoder.fit_transform(test_df['hour'])

test_df = pd.concat([test_df, df_bin], axis=1)
encoder = ce.BinaryEncoder(cols=["dayofweek"])

df_bin = encoder.fit_transform(train_df['dayofweek'])

train_df = pd.concat([train_df, df_bin], axis=1)



encoder = ce.BinaryEncoder(cols=["dayofweek"])

df_bin = encoder.fit_transform(test_df['dayofweek'])

test_df = pd.concat([test_df, df_bin], axis=1)
# encoder = ce.BinaryEncoder(cols=["month"])

# df_bin = encoder.fit_transform(train_df['month'])

# train_df = pd.concat([train_df, df_bin], axis=1)



# encoder = ce.BinaryEncoder(cols=["month"])

# df_bin = encoder.fit_transform(test_df['month'])

# test_df = pd.concat([test_df, df_bin], axis=1)
encoder = ce.BinaryEncoder(cols=["Age Binned"])

df_bin = encoder.fit_transform(train_df['Age Binned'])

train_df = pd.concat([train_df, df_bin], axis=1)



encoder = ce.BinaryEncoder(cols=["Age Binned"])

df_bin = encoder.fit_transform(test_df['Age Binned'])

test_df = pd.concat([test_df, df_bin], axis=1)
train_df.columns
train_df = train_df.drop(columns=["country_code","domain_cat","hour","dayofweek","Age Binned"])



test_df = test_df.drop(columns=["country_code","domain_cat","hour","dayofweek","Age Binned"])
train_df.dtypes
train_df["last_open_day"] = train_df["last_open_day"].astype('int64')

train_df["last_login_day"] = train_df["last_login_day"].astype('int64')

train_df["last_checkout_day"] = train_df["last_checkout_day"].astype('int64')

train_df["month"] = train_df["month"].astype('int64')



test_df["last_open_day"] = test_df["last_open_day"].astype('int64')

test_df["last_login_day"] = test_df["last_login_day"].astype('int64')

test_df["last_checkout_day"] = test_df["last_checkout_day"].astype('int64')

test_df["month"] = test_df["month"].astype('int64')
#final_check

train_df.isna().sum()

test_df.isna().sum()
"""Gradient Boosting Classifier"""

gbc = GradientBoostingClassifier(random_state = 43)





"""#10.Extreme Gradient Boosting"""

xgbc = XGBClassifier()



"""List of all the models with their indices."""

modelNames = ["GBC","XGBC"]

models = [gbc, xgbc]
yTrain = train_df[["open_flag"]].values.ravel()

xTrain = train_df.drop(columns=["open_flag"])
xTrain.dtypes
def calculateTrainAccuracy(model):

    """Returns training accuracy of a model."""

    model.fit(xTrain, yTrain)

    trainAccuracy = model.score(xTrain, yTrain)

    trainAccuracy = round(trainAccuracy*100, 2)

    return trainAccuracy



# Calculate train accuracy of all the models and store them in a dataframe

modelScores = list(map(calculateTrainAccuracy, models))

trainAccuracy = pd.DataFrame(modelScores, columns = ["trainAccuracy"], index=modelNames)

trainAccuracySorted = trainAccuracy.sort_values(by="trainAccuracy", ascending=False)

print("~~~~~~~ Training Accuracy of the Classifiers ~~~~~~~~~~~")

print(trainAccuracySorted)
def calculateXValScore(model):

    """Returns models' cross validation scores."""

    

    xValScore = cross_val_score(model, xTrain, yTrain, cv = 10, scoring="accuracy").mean()

    xValScore = round(xValScore*100, 2)

    return xValScore



# Calculate cross validation scores of all the models and store them in a dataframe

modelScores = list(map(calculateXValScore, models))

xValScores = pd.DataFrame(modelScores, columns = ["xValScore"], index=modelNames)

xValScoresSorted = xValScores.sort_values(by="xValScore", ascending=False)



display(xValScoresSorted)
# # Save model

# import pickle

# modelNames_list = ["LR", "SVC", "RF", "KNN", "GNB", "DT", "GBC", "ABC", "ETC", "XGBC"]

# models_list = [lr, svc, rf, knn, gnb, dt, gbc, abc, etc, xgbc]



# for model, modelName in zip(models_list, modelNames_list):

#     model_filename = modelName + ".sav"

#     pickle.dump(model, open(model_filename, "wb"))
# #Read model

# import pickle



# model = pickle.load(open("ETC.sav", "rb"))
len(xTrain.columns)
len(test_df.columns)
yPred = xgbc.predict(test_df)
yPred_df = pd.DataFrame(data=yPred, columns=["open_flag"])

yPred_df.insert(0, 'row_id', range(0, len(yPred_df)))

yPred_df.reset_index(drop=True, inplace=True)

yPred_df.head()
submission_df.head()
len(yPred_df.index)
yPred_df.to_csv("yPred_new.csv")