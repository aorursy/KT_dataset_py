from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
accidents = pd.read_csv('../input/dft-accident-data/Accidents0515.csv',index_col='Accident_Index')
casualties = pd.read_csv('../input/dft-accident-data/Casualties0515.csv' ,index_col='Accident_Index',error_bad_lines=False
                         ,warn_bad_lines=False)
accidents.head()
missing_values_count = accidents.isnull().sum()

missing_values_count[0:50]
accidents.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR','LSOA_of_Accident_Location',
                'Junction_Control' ,'2nd_Road_Class','Time','Date'], axis=1, inplace=True)
accidents['Longitude'].fillna(999, inplace = True)
accidents['Latitude'].fillna(999, inplace = True)

plt.figure(figsize=(12,6))
accidents.Day_of_Week.hist(bins=7,rwidth=0.7,color= 'orange')
plt.title('Accidents on the day of a week' , fontsize= 30)
plt.grid(False)
plt.ylabel('Accident count' , fontsize = 20)
plt.xlabel('1 - Sunday ,  2 - Monday  ,3 - Tuesday , 4 - Wednesday , 5 - Thursday , 6 - Friday , 7 - Saturday' , fontsize = 13)
plt.figure(figsize=(20,5))
sns.countplot('Age_of_Casualty',data=casualties)
plt.title('CASUALITY DISTRIBUTION BASED ON AGE', fontsize=25)
plt.xticks(rotation=90)
plt.grid(alpha=0.5)
plt.show()
plt.figure(figsize=(40,10))
sns.countplot('Age_of_Casualty',hue='Sex_of_Casualty',data=casualties)
plt.xticks(fontsize=15,rotation=90)
plt.legend(['Missing data','Male','Female'], prop={'size': 40})
plt.grid(alpha=0.4)
plt.xlabel('AGE_OF_CASUALITIES', fontsize=25)
plt.ylabel('COUNT', fontsize=25)
plt.show()
import matplotlib.image as mpimg
plt.figure(figsize=(18,8))
img=mpimg.imread('../input/images/traffic.png')
imgplot = plt.imshow(img)
plt.title('Traffic heat map of UK Roads', fontsize=25)
plt.show()
import matplotlib.image as mpimg
plt.figure(figsize=(18,8))
img=mpimg.imread('../input/images/accidents.png')
imgplot = plt.imshow(img)
plt.title('Accident prone area', fontsize=25)
plt.show()
X = casualties.drop('Casualty_Severity', axis = 1)
y = casualties['Casualty_Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))