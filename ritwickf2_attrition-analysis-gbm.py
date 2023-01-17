#Importing the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix)

#Suppressing warnings

import warnings

warnings.filterwarnings('ignore')
#Importing  the Dataset

print('Importing the CSV file.')

df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print('File imported successfully!')



#Datatypes in the dataset

print('Imported Dataframe Structure : \n', df.dtypes.value_counts())
df.head(3)
#Checking the number of 'Yes' and 'No' in 'Attrition'

ax = sns.catplot(x="Attrition", kind="count", palette="ch:.25", data=df);

ax.set(xlabel = 'Attrition', ylabel = 'Number of Employees')

plt.show()
#Identifying columns with missing information

missing_col = df.columns[df.isnull().any()].values

print('The missing columns in the dataset are: ',missing_col)
#Extracting the Numeric and Categorical features

df_num = pd.DataFrame(data = df.select_dtypes(include = ['int64']))

df_cat = pd.DataFrame(data = df.select_dtypes(include = ['object']))

print("Shape of Numeric: ",df_num.shape)

print("Shape of Categorical: ",df_cat.shape)
#Dropping 'Attrition' from df_cat before encoding

df_cat = df_cat.drop(['Attrition'], axis=1) 



#Encoding using Pandas' get_dummies

df_cat_encoded = pd.get_dummies(df_cat)

df_cat_encoded.head(5)
#Using StandardScaler to scale the numeric features

standard_scaler = StandardScaler()

df_num_scaled = standard_scaler.fit_transform(df_num)

df_num_scaled = pd.DataFrame(data = df_num_scaled, columns = df_num.columns, index = df_num.index)

print("Shape of Numeric After Scaling: ",df_num_scaled.shape)

print("Shape of categorical after Encoding: ",df_cat_encoded.shape)
#Combining the Categorical and Numeric features

df_transformed_final = pd.concat([df_num_scaled,df_cat_encoded], axis = 1)

print("Shape of final dataframe: ",df_transformed_final.shape)
#Extracting the target variable - 'Attrition'

target = df['Attrition']



#Mapping 'Yes' to 1 and 'No' to 0

map = {'Yes':1, 'No':0}

target = target.apply(lambda x: map[x])



print("Shape of target: ",target.shape)



#Copying into commonly used fields for simplicity

X = df_transformed_final #Features

y = target #Target
#Splitting into Train and Test dataset in 90-10 ratio

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state = 0, stratify = y)

print("Shape of X Train: ",X_train.shape)

print("Shape of X Test: ",X_test.shape)

print("Shape of y Train: ",y_train.shape)

print("Shape of y Test: ",y_test.shape)
#Using Gradient Boosting to predict 'Attrition' and create the Trees to identify important features

gbm = GradientBoostingClassifier(n_estimators = 200, max_features = 0.7, learning_rate = 0.3, max_depth = 5, random_state = 0, verbose = 0)

print('Training Gradient Boosting Model')



#Fitting Model

gbm.fit(X_train, y_train)

print('Model Fitting Completed')



#Predicting

print('Starting Predictions!')

y_pred = gbm.predict(X_test)

print('Prediction Completed!')
print('Accuracy of the model is:  ',accuracy_score(y_test, y_pred))
#Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print('The confusion Matrix : \n',cm)
# Scatter plot 

trace = go.Scatter(

    y = gbm.feature_importances_,

    x = df_transformed_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 10,

        color = gbm.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = df_transformed_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Model Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter')
#Setting Seaborn font-size

sns.set(font_scale = 1)



#Attrition based on Overtime

ax = sns.catplot(x="OverTime", kind="count",hue="Attrition", palette="ch:.25", data=df);

ax.set(xlabel = 'Overtime', ylabel = 'Number of Employees', title = 'Overtime')

plt.show()
#Stock Option Level

ax = sns.catplot(x="StockOptionLevel", kind="count",hue="Attrition", palette="ch:.25", data=df);

ax.set(xlabel = 'Stock Option Level', ylabel = 'Number of Employees', title = 'Stock Option Level')

plt.show()
#Job Satisfaction

ax = sns.catplot(x="JobSatisfaction", kind="count",hue="Attrition", palette="ch:.25", data=df);

ax.set(xlabel = 'Job Satisfaction', ylabel = 'Number of Employees', title = 'Job Satisfaction')

plt.show()
#JobLevel

ax = sns.catplot(x="JobLevel", kind="count",hue="Attrition", palette="ch:.25", data=df);

ax.set(xlabel = 'Job Level', ylabel = 'Number of Employees', title = 'Job Level')

plt.show()
#EnvironmentSatisfaction

ax = sns.catplot(x="EnvironmentSatisfaction", kind="count",hue="Attrition", palette="ch:.25", data=df);

ax.set(xlabel = 'Environment Satisfaction', ylabel = 'Number of Employees', title = 'Environment Satisfaction')

plt.show()
#YearsWithCurrManager

ax = sns.catplot(x="TotalWorkingYears", kind="count",hue="Attrition", palette="ch:.25", data=df);

ax.set(xlabel = 'Total Working Years', ylabel = 'Number of Employees', title = 'Total Working Years')

plt.show()