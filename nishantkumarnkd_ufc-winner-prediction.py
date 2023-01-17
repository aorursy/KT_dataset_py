import pandas as pd

fights = pd.read_csv('../input/ufcdata/data.csv')

fights

#to find missing values

fights.isnull().sum()

#retain only  those columns or features having at least 4800 non na values 

fights.dropna(thresh=4800,inplace=True,axis=1)

#filling missing values

fights.fillna({'B_Height_cms':fights['B_Height_cms'].median(),

               'B_Weight_lbs':fights['B_Weight_lbs'].median(),

                'R_Reach_cms':fights['R_Reach_cms'].median(),

                'R_Weight_lbs':fights['R_Weight_lbs'].median(),

                 'B_age':fights['B_age'].median(),

                'R_age':fights['R_age'].median(),

                 'R_Height_cms':fights['R_Height_cms'].median()},inplace=True)

import scipy.stats

crosstab = pd.crosstab(fights['Winner'],fights['R_Stance'])

chi = scipy.stats.chi2_contingency(crosstab)

chi

#R_stance should be dropped as it has no correlation with the output

fights.drop(columns=['B_Stance','R_Stance'],inplace=True)



# as herb dean is the most popular and more frequently occur in the referee column

import numpy as np

fights['Referee']= fights['Referee'].replace(np.NaN , 'Herb Dean')

fights



from sklearn.model_selection import train_test_split

y = fights['Winner']

fights.drop(columns='Winner',inplace=True)



X_train,X_test,y_train,y_test = train_test_split(fights,y,test_size=0.3,random_state=42)



#label encoding the categorical columns present in our train data

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



for col in X_train.columns.values:



       if X_train[col].dtypes=='object':

            data=X_train[col].append(X_train[col])

            le.fit(data.values)

            X_train[col]=le.fit_transform(X_train[col])

            

#label encoding the categorical columns present in our test data

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



for col in X_test.columns.values:



       if X_test[col].dtypes=='object':

            data=X_test[col].append(X_test[col])

            le.fit(data.values)

            X_test[col]=le.transform(X_test[col])



# scaling our data to normalize the influence of all the features with respect to the output

from sklearn.preprocessing import MinMaxScaler

minmaxscaling = MinMaxScaler()

X_train_scaled = minmaxscaling.fit_transform(X_train)

X_test_scaled = minmaxscaling.transform(X_test)



from sklearn.metrics import confusion_matrix,balanced_accuracy_score,accuracy_score,classification_report





#using adaptive boosting technique

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=130,learning_rate=.65)

ada.fit(X_train_scaled,y_train)

predicted_ada = ada.predict(X_test_scaled)

predicted_ada



#checking score

accuracy_score(y_test,predicted_ada)*100
