# importing the libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
# reading the dataset
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
df.head()
df.describe()
# checking if any missing data is present
df.isnull().any()
# dropping the serial number column
df = df.drop(['Serial No.'], axis=1)
df.head()
sns.distplot(df['GRE Score'])
sns.boxplot(df['GRE Score'])
sns.distplot(df['CGPA'])
sns.boxplot(df['CGPA'])
sns.distplot(df['TOEFL Score'])
sns.catplot(x="Research", kind="count", palette="coolwarm", data=df);
sns.countplot(df['SOP'], palette='coolwarm')
sns.countplot(df['LOR '], palette='coolwarm')
sns.countplot(df['University Rating'], palette='coolwarm')
df.CGPA.describe()
def grouping(x):
    if x<= 6.00:
        return '< 6'
    elif x <= 7.00:
        return '6 - 7'
    elif x <= 8.00:
        return '7 - 8'
    elif x <= 9.00:
        return '8 - 9'
    else:
        return '> 9'
    
groups = df.CGPA.apply(grouping)
values = groups.value_counts()
labels = values.index
fig = px.pie(df, values = values, names = labels)
fig.update_layout(title='CGPA distribution')

fig.show()
# printing the student with CGPA less than 7.0
print(df.loc[df['CGPA'] < 7.00 ])
fig = px.box(df, y="GRE Score", color="University Rating", points='all', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.box(df, y="TOEFL Score", color="University Rating", points='all',hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.violin(df, y="CGPA", color="University Rating", points='all', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.scatter(df, x="CGPA", y="Chance of Admit ",color='GRE Score', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.box(df, x="LOR ", y="Chance of Admit ", color='Research', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.box(df, x="SOP", y="Chance of Admit ", color='Research', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
df1 = df[(df['University Rating'] > 2) & (df['GRE Score'] <= 320) ]
fig = px.box(df1, y="GRE Score", color="University Rating", points='all', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
df5 = df[(df['University Rating'] == 5) & (df['GRE Score'] <= 320) ]
df5
fig = px.box(df5, y="CGPA", color="Research", points='all', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.box(df1, x="LOR ", y="Chance of Admit ",color='Research', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
df2 = df[(df['University Rating'] > 3) & (df['CGPA'] <= 8.00)]
df2
fig = px.scatter(df2, x="CGPA", y="Chance of Admit ",color='Research', hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
fig = px.box(df2, x="SOP", y="Chance of Admit ", hover_data=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '])
fig.show()
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('ggplot')

sns.pairplot(df, hue='Research')
plt.show()
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('ggplot')

sns.heatmap(df.corr(), cmap = 'twilight', annot=True)
plt.title('Correlation Plot', fontsize = 20)
plt.show()
!pip install pycaret
# importing the regression module from PyCaret
from pycaret.regression import *
from sklearn.model_selection import train_test_split 
# splitting the data
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
# seperating the target feature from the test dataset
X_test = test.drop(['Chance of Admit '],axis=1)
y_test = test['Chance of Admit ']
X_test.head()
# Pre-processing and setting up the data
reg = setup(data=train,
           target='Chance of Admit ',
           numeric_imputation='mean',
           categorical_features=['University Rating','Research'],
           silent=True)
# comparing different regression models (PyCaret made it easy)
compare_models()
# creating the model
r = create_model('ridge')
# tuning the model
tr = tune_model(r,  optimize = 'RMSE')
import pandas as pd    
import numpy as np   
new = np.array([[315, 110, 4, 4.5, 4.5, 8.00, 0],[320,110,5,4.5,4.5,8.90,1]])
new_df = pd.DataFrame(new)
new_df = new_df.rename(index=str, columns={0:'GRE Score', 1:'TOEFL Score',2:'University Rating', 3:'SOP',4:'LOR ',5:'CGPA',6:'Research'})
predictions = predict_model(tr, data=new_df)
predictions