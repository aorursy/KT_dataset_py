import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
iot_data = pd.read_csv('../input/ml-prediction-for-lightdetection-sensor-iot/iot_telemetry_data.csv')

iot_data
iot_data.info()
iot_data.replace(['00:0f:00:70:91:0a', '1c:bf:ce:15:ec:4d', 'b8:27:eb:bf:9d:51'], [1, 2, 3], inplace=True)

print(iot_data.head())
iot_data['time_stamp'] = pd.to_datetime(iot_data['ts'], unit='s')

#since in the Time column, a date isn’t specified and hence Pandas will put Some date automatically in that case.

iot_data.drop(columns=['ts'], inplace=True) 

print(iot_data.head())
sns.heatmap(iot_data.corr()) 
light = iot_data.iloc[:,3]

print(light.tail(10))

motion = iot_data.iloc[:, 5]

print(motion.head(10))
iot_data['motion'].unique
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder=LabelEncoder()

light = labelencoder.fit_transform(light)

print(light)

motion = labelencoder.fit_transform(motion)

print(motion)

onehotencoder=OneHotEncoder()

iot_data['light'] = light

iot_data['motion'] = motion

iot_data
iot_data_df = pd.DataFrame(iot_data)

iot_data_df.head()
iot_data_df.isnull().sum()
#converting the given temperature in  Fahrenheit to degree Celsius

iot_data_df['temp'] = (iot_data_df['temp'] * 1.8) + 32

iot_data_df
affect=['co', 'humidity', 'lpg', 'smoke', 'temp']

slice=[3,7,8,6,9]

color=['r', 'g', 'm', 'b', 'c']



plt.pie(slice, labels=affect, colors=color, startangle=90,shadow=True, 

       explode=(0,0,0,0.1,0), autopct='%1.2f%%')

plt.legend(bbox_to_anchor =(0.85, 1.20), ncol = 2) 

plt.show()
sns.set_style('darkgrid')

sns.countplot('device', hue='light',palette="rocket", edgecolor=sns.color_palette("dark", 3),linewidth=2, data=iot_data_df)
sns.scatterplot('device', 'time_stamp', hue= 'light' , data=iot_data_df)
iot_data_df.drop('time_stamp', axis=1, inplace=True)

iot_data_df.drop('motion', axis=1, inplace=True)

iot_data_df.head()
x = iot_data_df.drop('light', axis= 1)

y = iot_data_df['light'].values

y
x
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(X_train, y_train)
prediction = reg.predict(X_test)

prediction
from sklearn import metrics

from sklearn.metrics import confusion_matrix

cnf_matrix = metrics.confusion_matrix(y_test, prediction)

#cnf_matrix

sns.heatmap(cnf_matrix, annot=True, cmap="Spectral" ,fmt='g', linewidth = 3)

plt.tight_layout()

plt.title('Confusion matrix')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, prediction))

print("Precision:",metrics.precision_score(y_test, prediction))

print("Recall:",metrics.recall_score(y_test, prediction))
sns.barplot('device', 'light', data=iot_data_df)