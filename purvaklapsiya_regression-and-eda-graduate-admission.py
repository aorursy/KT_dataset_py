%matplotlib inline
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})

import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.figure_factory as ff
import cufflinks as cf

import warnings
warnings.filterwarnings('ignore')

# df = pd.read_csv('../input/Admission_Predict.csv')
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head()
#Checking for null values
df.isna().sum()
df.describe()
low = df['GRE Score'].min()-4
high = df['GRE Score'].max()
data = [go.Histogram(
        x = df['GRE Score'],
        xbins = {'start': 286, 'size': 5, 'end' :high+1}
)]
print('Average GRE score = ', np.mean(df['GRE Score']))
plotly.offline.iplot(data, filename='overall_rating_distribution')

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns= df.columns)
# x_array = np.array(df['GRE Score'])
# normalized_X = preprocessing.normalize([x_array])
# df['normalized_GRE'] = normalized_X[0]
# y_array = np.array(df['Chance of Admit '])
# normalized_Y = preprocessing.normalize([y_array])
# df['normalized_Chance'] = normalized_Y[0]
sns.lmplot(x='GRE Score',y='Chance of Admit ', 
           data=scaled_df,fit_reg = True)
# scaled_df
df['ToeflCategories'] = pd.cut(df['TOEFL Score'], [90, 95 , 100, 105, 110, 115,121], labels=['90-95','95-100', '100-105','105-110','110-115','115-120'])
number_of_students_in_category = df['ToeflCategories'].value_counts().sort_values(ascending=True)
data = [go.Pie(
        labels = number_of_students_in_category.index,
        values = number_of_students_in_category.values,
        hoverinfo = 'label+value'
    
)]
plotly.offline.iplot(data, filename='active_category')
df['ToeflCategories'] = pd.cut(df['TOEFL Score'], [90, 95 , 100, 105, 110, 115,121], labels=['90-95','95-100', '100-105','105-110','110-115','115-120'])
plt.figure(figsize=(15,8))
sns.boxplot(x="ToeflCategories", y="GRE Score", data=df)
plt.title('GRE Score vs Toefl')
plt.show()
sns.jointplot(x="GRE Score", y="TOEFL Score", data=df, height=10, ratio=3, color="r")
plt.show()
# sns.boxplot(x='University Rating', y ='GRE Score', data=df)
# plt.show()
data = [go.Box(y=df['GRE Score'],x=df['University Rating'])]
plotly.offline.iplot(data, filename='box/multiple', validate = False,)
data = []
trace = {
            "type": 'violin',
            "x": df['University Rating'],
            "y": df['TOEFL Score'],
            "name": pd.unique(df['University Rating']),
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
data.append(trace)
fig = {
    "data": data,
    "layout" : {
        "title": "TOEFL Score vs University Ranking",
        "yaxis": {
            "zeroline": False,
        }
    }
}
plotly.offline.iplot(fig, filename='violin/multiple', validate = False)
temp = df.drop(['Serial No.'],axis=1)
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(temp.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Let's build a Regression model
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

df_org = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
# Preparing Data
y= df_org.iloc[:,-1].values
x = df_org.iloc[:,:-1]
x = x.iloc[:,1:]
y = pd.Series(y)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)
yPrediction = linearRegressor.predict(xTest)
maxcoef = np.argsort(-np.abs(linearRegressor.coef_))
coef = linearRegressor.coef_[maxcoef]
for i in range(0, 7):
    print("{:.<025} {:< 010.4e}".format(x.columns[maxcoef[i]], coef[i]))
Ls = LassoCV()
# Train the model using the training sets
Ls.fit(xTrain, yTrain)
maxcoef = np.argsort(-np.abs(Ls.coef_))
coef = Ls.coef_[maxcoef]
print("Importance of Parameters: ")
for i in range(0, 7):
    print("{:.<025} {:< 010.4e}".format(x.columns[maxcoef[i]], coef[i]))
Rr = RidgeCV()
Rr.fit(xTrain, yTrain)
maxcoef = np.argsort(-np.abs(Rr.coef_))
coef = Rr.coef_[maxcoef]
print("Importance of Parameters: ")
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(x.columns[maxcoef[i]], coef[i]))
EN = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 5)) # we are essentially smashing most of the Rr model here
# Train the model using the training sets
train_EN = EN.fit(xTrain, yTrain)
maxcoef = np.argsort(-np.abs(EN.coef_))
coef = EN.coef_[maxcoef]
print("Importance of Parameters: ")
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(x.columns[maxcoef[i]], coef[i]))
model = [linearRegressor,Ls, Rr, EN]
model_name = ['Linear Regression','LassoCV','RidgeCV','ElasticNetCV']
M = len(model)
CV = 5
score = np.empty((M, CV))
print("Training Cross Validation Score:")
for i in range(0, M):
    score[i, :] = cross_val_score(model[i], xTrain, yTrain, cv=CV)
    print(model_name[i],":",score.mean(axis=1)[i])
print(score.mean(axis=1))
model = [linearRegressor,Ls, Rr, EN]
model_name = ['Linear Regression','LassoCV','RidgeCV','ElasticNetCV']
M = len(model)
CV = 5
score = np.empty((M, CV))
print("Testing Cross Validation Score:")
for i in range(0, M):
    score[i, :] = cross_val_score(model[i], xTest, yTest, cv=CV)
    print(model_name[i],":",score.mean(axis=1)[i])
print(score.mean(axis=1))    

