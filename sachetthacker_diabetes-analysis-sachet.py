import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')
df = pd.read_csv('../input/early-stage-diabetes-risk-prediction-dataset/diabetes_data_upload.csv')
df.head(1)
df = df.rename({'class':'Result'} ,axis = 'columns')
positive = df[df.Result == 'Positive']

negative= df[df.Result == 'Negative']
positive.head(1)
slices_Gender = positive['Gender'].value_counts()

slices_Polyuria = positive['Polyuria'].value_counts()

slices_Polydipsia = positive['Polydipsia'].value_counts()

slices_wtloss = positive['sudden weight loss'].value_counts()

slices_weakness = positive['weakness'].value_counts()

slices_Polyphagia = positive['Polyphagia'].value_counts()

slices_Genital = positive['Genital thrush'].value_counts()

slices_vis = positive['visual blurring'].value_counts()

slices_itch = positive['Itching'].value_counts()

slices_delheal = positive['delayed healing'].value_counts()

slices_partpersis = positive['partial paresis'].value_counts()

slices_muscle = positive['muscle stiffness'].value_counts()

slices_Alopecia = positive['Alopecia'].value_counts()

slices_Obesity = positive['Obesity'].value_counts()





labels_gender = ['Male' , 'Female']

plt.title('Gender Data')



plt.pie(slices_Gender ,shadow = True,labels = labels_gender , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels_2 = ['Yes' , 'No']

plt.title('Polyuria')



plt.pie(slices_Polyuria ,shadow = True,labels = labels_2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels_2 = ['Yes' , 'No']

plt.title('Polydipsia')

plt.pie(slices_Polydipsia  ,shadow = True,labels = labels_2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Weight Loss')



plt.pie(slices_wtloss ,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Weakness')



plt.pie(slices_weakness ,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Polyphagia')



plt.pie(slices_Polyphagia ,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Genital thrush')



plt.pie(slices_Genital,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Vision Impairment')



plt.pie(slices_vis,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Itching')



plt.pie(slices_itch,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Delayed Healing')



plt.pie(slices_delheal,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Partial Persis')



plt.pie(slices_partpersis,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Muscle Stiffness')



plt.pie(slices_muscle,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Alopecia')



plt.pie(slices_Alopecia,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
labels2 = ['Yes' , 'No']

plt.title('Obesity')



plt.pie(slices_Obesity,shadow = True,labels = labels2 , wedgeprops = {'edgecolor':'black'} ,autopct='%1.1f%%')
finaldf = df
finaldf.head(1)
finaldf['Gender'] =finaldf['Gender'].map({'Male' : 1 , 'Female':0})

finaldf['Polyuria'] =finaldf['Polyuria'].map({'Yes' : 1 , 'No':0})

finaldf['Polydipsia'] =finaldf['Polydipsia'].map({'Yes' : 1 , 'No':0})

finaldf['sudden weight loss'] =finaldf['sudden weight loss'].map({'Yes' : 1 , 'No':0})

finaldf['weakness'] =finaldf['weakness'].map({'Yes' : 1 , 'No':0})

finaldf['Polyphagia'] =finaldf['Polyphagia'].map({'Yes' : 1 , 'No':0})

finaldf['Genital thrush'] =finaldf['Genital thrush'].map({'Yes' : 1 , 'No':0})

finaldf['visual blurring'] =finaldf['visual blurring'].map({'Yes' : 1 , 'No':0})

finaldf['Itching'] =finaldf['Itching'].map({'Yes' : 1 , 'No':0})

finaldf['Irritability'] =finaldf['Irritability'].map({'Yes' : 1 , 'No':0})

finaldf['delayed healing'] =finaldf['delayed healing'].map({'Yes' : 1 , 'No':0})

finaldf['partial paresis'] =finaldf['partial paresis'].map({'Yes' : 1 , 'No':0})

finaldf['muscle stiffness'] =finaldf['muscle stiffness'].map({'Yes' : 1 , 'No':0})

finaldf['Alopecia'] =finaldf['Alopecia'].map({'Yes' : 1 , 'No':0})

finaldf['Obesity'] =finaldf['Obesity'].map({'Yes' : 1 , 'No':0})

finaldf['Result'] =finaldf['Result'].map({'Positive' : 1 , 'Negative':0})
X = finaldf[['Age' , 'Gender' , 'Polyuria',  'Polydipsia' , 'sudden weight loss' , 'weakness','Polyphagia' , 'Genital thrush', 'visual blurring' ,'Itching','Irritability','delayed healing', 'partial paresis', 'muscle stiffness' , 'Alopecia','Obesity','Result']]

y = finaldf['Result']
X.tail(10)
model = LogisticRegression()

X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size = 0.2)

model.fit(X_train , y_train)
model.score(X,y)