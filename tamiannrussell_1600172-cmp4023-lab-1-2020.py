# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #import



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Data=pd.read_csv("../input/dwdm-dirt-lab-1/DWDM Dirt Lab 1.csv", encoding="ISO-8859-1")
Data
Data["Region"].value_counts().index[2]
import plotly.graph_objects as go
Data["Region"].value_counts()
RegionVisualization = go.Figure(data=[go.Bar(x=Data["Region"], y=[p for p in Data['Region'].value_counts() if p>=8])],

                               layout=go.Layout(title=go.layout.Title(text="Regions Occuring More Than 8 Times")))

RegionVisualization.update_layout(xaxis_title="Region Names",

                 yaxis_title="Frequency of Region")

RegionVisualization.show()
import plotly.graph_objects as go
genderPie= go.Figure(data=go.Pie(labels=Data["Gender"].value_counts().index,values=Data["Gender"].value_counts()),

                    layout=go.Layout(title=go.layout.Title(text="Males & Females in the Data Set")))

genderPie.show()
for i in [Data]:

    for index, value in enumerate(i):

        if ((value=="") or (value==" ") or (value=="  ") or (value!=value)):

            i[index]= np.nan
for i in [Data["Height_cm"],Data["Footlength_cm"],Data["Armspan_cm"],Data["Reaction_time"]]:

    for index, value in enumerate(i):

        if ((value=="") or (value==" ") or (value=="  ") or (value!=value)):

            i[index]= np.nan
Data["Height_cm"]= pd.to_numeric(Data["Height_cm"])

Data["Footlength_cm"]= pd.to_numeric(Data["Footlength_cm"])

Data["Armspan_cm"]= pd.to_numeric(Data["Armspan_cm"])

Data["Reaction_time"]= pd.to_numeric(Data["Reaction_time"])
Data.isnull()
Data.isnull().sum()
missing=sum(Data.isnull().sum())

print("%d records have missing values!"%(missing))
Data.drop(0)
Data= Data.replace("NA", np.nan)

Data=Data.dropna()
newMissing=sum(Data.isnull().sum())

print("%d records have missing values!"%(newMissing))
Data[['Age-years','Height_cm','Footlength_cm','Armspan_cm','Languages_spoken','Travel_time_to_School',

      'Reaction_time','Score_in_memory_game']].corr()
bodyInfo= Data[['Height_cm','Footlength_cm','Armspan_cm','Reaction_time']]

bodyInfo
bodyInfoScatter = go.Figure (data=go.Scatter(x=Data['Height_cm'], y=Data['Armspan_cm'], mode='markers'))

bodyInfoScatter.update_layout(title='Correlation between Height & Armspan',

                 xaxis_title='Height in cm',

                 yaxis_title='Armspan in cm')



bodyInfoScatter.show()
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn import linear_model
def create_label_encoder_dict(df):

    

    label_encoder_dict = {}

    for column in df.columns:

        if not np.issubdtype(df[column].dtype, np.number):

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
Data2= Data

label_encoders = create_label_encoder_dict(Data2)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_,

                       index=['Encoded Values']  ).T)
Data3 = Data2.copy() # create copy of initial data set

for column in Data3.columns:

    if column in label_encoders:

        Data3[column] = label_encoders[column].transform(Data3[column])



print("Transformed data set")

print("="*32)

Data3
# separate our data into dependent (Y) and independent(X) variables

X_data = Data3[['Height_cm','Footlength_cm','Armspan_cm']]

Y_data = Data3['Reaction_time']
#Using linear regression

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)

model = linear_model.LinearRegression()

model.fit(X_train,y_train)
print("Coefficient")

pd.DataFrame(model.coef_,index=X_train.columns,columns=["Coefficient"])
#Testing the model

test_predicted = model.predict(X_test)

test_predicted
Data3 = X_test.copy()

Data3['Predicted_Reaction_time']=test_predicted

Data3['Reaction_time']=y_test

Data3.head()
model.score(X_test,y_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(Data3[X_train.columns])
#Test data

X_test
X_reduced = pca.transform(X_test)

X_reduced
plt.scatter(model.predict(X_train), model.predict(X_train)-y_train,c='b',s=40,alpha=0.5)

plt.scatter(model.predict(X_test),model.predict(X_test)-y_test,c='g',s=40)

plt.hlines(y=0,xmin=np.min(model.predict(X_test)),xmax=np.max(model.predict(X_test)),color='red',linewidth=3)

plt.title('Residual Plot using Training (blue) and test (green) data ')

plt.ylabel('Residuals')
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error, mean_squared_error
print('The coefficent of Determination (R^2) of my model is %.2f' % r2_score(y_test, test_predicted))
Data3[["Height_cm","Footlength_cm","Armspan_cm","Reaction_time"]].corr()
print("Formula")

print("="*32)

print("y = {} + {}*Height_cm + {}*Footlength_cm + {}*Armspan_cm + e".format(round(model.intercept_,3),

                                                                            round(model.coef_[0],3),

                                                                            round(model.coef_[1],3),

                                                                            round(model.coef_[2],3)))
model.predict([[183,30,150]])
print("Mean Absolute error: %.4f" % mean_absolute_error(y_test, test_predicted))



print("Root Mean squared error: %.4f" % np.sqrt(mean_squared_error(y_test, test_predicted)))



print("Mean squared error: %.4f" % mean_squared_error(y_test, test_predicted))