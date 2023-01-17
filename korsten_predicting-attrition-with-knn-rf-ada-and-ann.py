import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder



#Get data

data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv', encoding='utf-8-sig')
#Drop columns irrelevant to our analysis

drop_columns = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']

data = data.drop(drop_columns, axis=1)



#Investigate relationship between incomes

mnth_inc = data['MonthlyIncome']

mnth_rte = data['MonthlyRate']

dly_rte = data['DailyRate']

hly_rte = data['HourlyRate']



#Normalize the large income data for HourlyRate, DailyRate, MonthlyRate and MonthlyIncome

norm_col = ['HourlyRate','DailyRate','MonthlyRate','MonthlyIncome']

for col in norm_col:

    data[col] = (data[col] - data[col].mean())/ data[col].std()
#Convert categorical features to numerical features

le = LabelEncoder()

attrition = data['Attrition']

attrition = le.fit_transform(attrition)

data.drop('Attrition', axis=1, inplace=True)



cat_col = data.select_dtypes(include=['object']).columns.values

data_col = data[cat_col]

data.drop(cat_col, axis=1, inplace=True)



#One hot encode categorical data and combine dataframes

data_col = pd.get_dummies(data_col)

data = pd.concat([data, data_col], axis=1).as_matrix() 
#Visiualize the 'questionable' income correlations

f, ((p1, p2), (p3, p4)) = plt.subplots(2,2,figsize=(12,10))

p1.hist(dly_rte / hly_rte, bins=12, edgecolor='k')

p1.set_xlabel('Hours Needed for Daily Rate')

p2.hist(mnth_rte / dly_rte, edgecolor='k')

p2.set_xlabel('Days Needed for Monthly Rate')

p3.hist((mnth_rte * 12 / 26) / (hly_rte * 80), bins=30, edgecolor='k')

p3.set_xlabel('Monthly rate normalized against 80 hour pay period')

p4.hist(mnth_inc/mnth_rte, bins=30, edgecolor='k')

p4.set_xlabel('Ratio of Monthly Income to Monthly Rate')

f.show()
#Choose our classifiers

rfc = RandomForestClassifier(n_estimators=1000, random_state=0, max_features=.1, max_depth=15)

ada = AdaBoostClassifier(random_state=0)

knn = KNeighborsClassifier()



rfc_accuracy = cross_val_score(rfc, data, attrition, cv=5)

ada_accuracy = cross_val_score(ada, data, attrition, cv=5)

knn_accuracy = cross_val_score(knn, data, attrition, cv=5)



#Create an ANN using Keras and Tensorflow backend

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam



def ANN_Classifier():

    cl = Sequential()

    cl.add(Dense(20, activation='relu', kernel_initializer='random_normal',  input_shape=(data.shape[1],)))

    cl.add(Dropout(0.1))

    cl.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    

    adm = Adam()

    

    cl.compile(optimizer=adm, loss='binary_crossentropy', metrics=['accuracy'])

    return cl



ann = KerasClassifier(build_fn=ANN_Classifier, batch_size=100, epochs=150, verbose=0)

ann_accuracy = cross_val_score(ann, data, attrition, cv=20)
print('Max RandomForest Accuracy: {:.4f}'.format(rfc_accuracy.max()))

print('Max AdaBoost Accuracy: {:.4f}'.format(ada_accuracy.max()))

print('Max KNN Accuracy: {:.4f}'.format(knn_accuracy.max()))

print('Max ANN Accuracy: {:.4f}'.format(ann_accuracy.max()))