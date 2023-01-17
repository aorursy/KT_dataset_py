# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as st

from sklearn.preprocessing import StandardScaler,LabelEncoder



data=pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head()

data.isna().sum()
data.groupby('Attrition')['Attrition'].count()
data_att = data[data['Attrition']=='Yes']

plt.hist(data_att['Age'])

plt.title('Age distribution')

plt.show()

plt.hist(data_att['DailyRate'])

plt.title('Salary distribution')

plt.show()

plt.hist(data_att['NumCompaniesWorked'])

plt.title('Companies worked previously')

plt.show()

plt.hist(data_att['TotalWorkingYears'])

plt.title('Total experience')

plt.show()

plt.hist(data_att['EnvironmentSatisfaction'])

plt.title('Environment satisfaction')

plt.show()

plt.hist(data_att['PercentSalaryHike'])

plt.title('Salary Hike')

plt.show()

plt.hist(data_att['PerformanceRating'])

plt.title('Performance rating')

plt.show()



plt.hist(data_att['JobInvolvement'])

plt.title('Job involvement')

plt.show()



plt.hist(data_att['DistanceFromHome'])

plt.title('Distance from home distribution')

plt.show()

plt.hist(data_att['RelationshipSatisfaction'])

plt.title('Relationship satisfaction distribution')

plt.show()

plt.hist(data_att['TrainingTimesLastYear'])

plt.title('Training hours distribution')

plt.show()

plt.hist(data_att['WorkLifeBalance'])

plt.title('Work life balance distribution')

plt.show()

plt.hist(data_att['YearsSinceLastPromotion'])

plt.title('Promotion distribution')

plt.show()

plt.hist(data_att['YearsWithCurrManager'])

plt.title('Time spent with current manager distribution')

plt.show()

data_att.groupby('BusinessTravel')['BusinessTravel'].count()
data_att.groupby('OverTime')['OverTime'].count()
print(data_att.groupby(['Gender','MaritalStatus'])['MaritalStatus'].count())

print(data_att.groupby('Gender')['Gender'].count())
data_att.groupby('Department')['Department'].count()
data_att.groupby('EducationField')['EducationField'].count()
cont_params = ['Age','DistanceFromHome','RelationshipSatisfaction','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsSinceLastPromotion','YearsWithCurrManager','EnvironmentSatisfaction','JobSatisfaction','PercentSalaryHike','PerformanceRating']

cat_params = ['BusinessTravel','Department','EducationField','Gender','MaritalStatus']

inp = data[cont_params + cat_params]

output = data['Attrition']



scaler = StandardScaler()

inp[cont_params]=scaler.fit_transform(data[cont_params])

categories=pd.get_dummies(data[cat_params])

inp=inp.drop(cat_params,axis=1)

inp=inp.join(categories)
encoder=LabelEncoder()

output=pd.DataFrame(encoder.fit_transform(output),columns=['Attrition'])
print('Shape of input: ',inp.shape)

print('Shape of output: ',output.shape)
from sklearn.model_selection import train_test_split



inp_tr_unb,inp_te,output_tr_unb,output_te = train_test_split(inp,output,test_size=0.1,random_state=1)



print('Shape of training input: ',inp_tr_unb.shape)

print('Shape of training output: ',output_tr_unb.shape)
from imblearn.over_sampling import SMOTE



smote=SMOTE()

inp_tr,output_tr = smote.fit_sample(inp_tr_unb,output_tr_unb)
import tensorflow as tf

from tensorflow import keras



from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from tensorflow.keras.layers import Dense,Activation,Dropout

model=Sequential()

model.add(Dense(6,activation='relu',input_shape=(29,)))

model.add(Dense(4,activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(2,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(inp_tr,output_tr,batch_size=8,epochs=500,validation_split=0.3)
from sklearn.metrics import accuracy_score,classification_report

predictions=model.predict_classes(inp_te)

acc=(accuracy_score(output_te,predictions))*100

acc=round(acc,0)

print('Accuracy of model for test data = ',acc," %")

print(classification_report(output_te,predictions))
