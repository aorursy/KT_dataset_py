import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
loan_data = pd.read_csv("../input/Loan payments data.csv")
# Data exploration

print('Shape of data ', loan_data.shape)

loan_data.head()
# Exploring loan status information

import matplotlib.pyplot as plt



plt.figure(figsize=(10,6))

plt.bar(x=loan_data['loan_status'].unique(),height=loan_data['loan_status'].value_counts(),width=0.25,color='#00BDBD')

plt.xlabel("Loan status")

plt.ylabel("Count")

plt.title("Loan status distributon")
#Custom function to visualise loan status for different features



def proportionBarChart(i_column,i_width):



    values = sorted(loan_data[i_column].unique())



    ind = [x for x, _ in enumerate(values)]



    #creating dataframe datastructure like

    #           PAIDOFF COLLECTION COLLECTION_PAIDOFF SUM

    #  index1    ....    ....       ....             ....

    #  index2    ....    ....       ....             ....

    #  .....................



    chart_data = pd.DataFrame(columns=['PAIDOFF','COLLECTION','COLLECTION_PAIDOFF','SUM'])

    



    for k in range(len(values)):

    

        count_paidoff = loan_data[(loan_data[i_column] == values[k]) & (loan_data.loan_status == 'PAIDOFF')]

        count_collection_paidoff = loan_data[(loan_data[i_column] == values[k]) & (loan_data.loan_status == 'COLLECTION_PAIDOFF')]

        count_collection = loan_data[(loan_data[i_column] == values[k]) & (loan_data.loan_status == 'COLLECTION')]

    

        chart_data.loc[k] = [count_paidoff.shape[0],count_collection_paidoff.shape[0],count_collection.shape[0],

                        count_paidoff.shape[0]+count_collection_paidoff.shape[0]+count_collection.shape[0]]

    



    c_index = values

    paid_off = chart_data['PAIDOFF']

    c_paid_off= chart_data['COLLECTION']

    c_collection = chart_data['COLLECTION_PAIDOFF']

    row_sum = chart_data['SUM']





    proportion_paidoff = np.true_divide(paid_off,row_sum)*100

    proportion_collection_paidoff = np.true_divide(c_paid_off,row_sum)*100

    proportion_collection = np.true_divide(c_collection,row_sum)*100



    plt.figure(figsize=(10,6))

    plt.bar(ind, proportion_paidoff, width=i_width, label='Loan Paidoff', color='#43C97E', bottom=proportion_collection+proportion_collection_paidoff)

    plt.bar(ind, proportion_collection_paidoff, width=i_width, label='Collection Paidoff', color='#FFDD99',bottom = proportion_collection)

    plt.bar(ind, proportion_collection, width=i_width, label='Collection', color='red')



    plt.xticks(ind, values)

    plt.legend()

    plt.ylabel("Loan status")

    plt.xlabel(i_column)

    plt.title("Loan terms' status by proportion of " + i_column)

    plt.ylim=1.0
# Exploring loan Pricipal information



y_values = loan_data.groupby(['Principal']).size()

x_values = sorted(loan_data['Principal'].unique())



plt.figure(figsize=(10,6))

plt.bar(x_values,height=y_values,width=25,color='#00BDBD')

plt.xlabel("Principal")

plt.ylabel("Count")

plt.title("Loan Principal distributon")



proportionBarChart('Principal',0.8)
# Exploring overall past due date



mean_of_due_date = loan_data['past_due_days'].mean()

mean_due_date_label = 'Mean       =' + str(mean_of_due_date) + ' days'



plt.figure(figsize=(10,6))

plt.hist(loan_data['past_due_days'],color='#00BDBD')

plt.axvline(mean_of_due_date, color='#801500', linewidth=2,label = mean_due_date_label)

plt.xlabel("past_due_days")

plt.ylabel("Count")

plt.title("Overall Loan past_due_days distributon")

plt.legend()
# Past due date for'COLLECTION_PAIDOFF'



df_temp = loan_data[loan_data.loan_status.str.contains('COLLECTION_PAIDOFF')]



mean_of_due_date_cp = df_temp['past_due_days'].mean()

mean_due_date_label_cp = 'Mean       =' + str(mean_of_due_date_cp) + ' days'



plt.figure(figsize=(10,6))

plt.hist(df_temp['past_due_days'],color='#00BDBD')

plt.axvline(mean_of_due_date_cp, color='#801500', linewidth=2,label=mean_due_date_label_cp)

plt.xlabel("past_due_days")

plt.ylabel("Count")

plt.legend()

plt.title("Loan past_due_days distributon for COLLECTION_PAIDOFF")
#Loan distribution by term



mean_term = loan_data['terms'].mean()

mean_term_label = 'Mean     =' + str(mean_term) + 'months'



plt.figure(figsize=(10,6))

plt.hist(loan_data['terms'],color='#00BDBD')

plt.axvline(mean_term, color='#801500', linewidth=2,label=mean_term_label)

plt.xlabel("Loan term")

plt.ylabel("Count")

plt.legend()

plt.title("Loan term distributon")



proportionBarChart('terms',0.2)
#Loan distribution by age



mean_age = loan_data['age'].mean()

mean_age_label = 'Mean       =' + str(mean_age) + ' years'



plt.figure(figsize=(10,6))

plt.hist(loan_data['age'],color='#00BDBD')

plt.axvline(mean_age, color='#801500', linewidth=2,label=mean_age_label)

plt.xlabel("Age")

plt.ylabel("Count")

plt.legend()

plt.title("Loan age distributon")



proportionBarChart('age',0.8)
#Loan distribution by education



y_values = loan_data.groupby(['education']).size()

x_values = sorted(loan_data['education'].unique())



plt.figure(figsize=(10,6))

plt.bar(x_values,height=y_values,width=0.25,color='#00BDBD')

plt.xlabel("Education")

plt.ylabel("Count")

plt.title("Loan education distributon")



proportionBarChart('education',0.4)
# Loan distribution by Gender



y_values = loan_data.groupby(['Gender']).size()

x_values = sorted(loan_data['Gender'].unique())



plt.figure(figsize=(10,6))

plt.bar(x_values,height=y_values,width=0.1,color='#00BDBD')

plt.xlabel("Gender")

plt.ylabel("Count")

plt.title("Loan distributon by Gender")



proportionBarChart('Gender',0.1)
features = ['Principal','terms','age','education','Gender','past_due_days']
#Filling zero instead of NAN in past_due_days column

loan_data['past_due_days'] = loan_data['past_due_days'].fillna(0).astype(int)
X = loan_data[features]

y = loan_data['loan_status']
X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
#Creating baseline or dummy classifier

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score



clf_d = DummyClassifier(strategy='most_frequent', random_state=0)

clf_d.fit(X_train, y_train)



print('Train Accurancy = ',accuracy_score(y_train,clf_d.predict(X_train)))

print('Test Accurancy  = ',accuracy_score(y_test,clf_d.predict(X_test)))
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.svm import SVC 

import sklearn.metrics as skm



clf = SVC(kernel='linear', C = 1.0)



svm = clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)



print('Train Accurancy = ',accuracy_score(y_train,clf.predict(X_train)))

print('Test Accurancy  = ',accuracy_score(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier



dt2 = RandomForestClassifier(n_estimators=20)

dt2.fit(X_train, y_train)



y_pred_dt2 = dt2.predict(X_test)



print('Train Accurancy = ',accuracy_score(y_train,dt2.predict(X_train)))

print('Test Accurancy  = ',accuracy_score(y_test,y_pred_dt2))
from sklearn.neighbors import KNeighborsClassifier



kn_clf = KNeighborsClassifier(n_neighbors=50)

kn_clf.fit(X_train, y_train)



print('Train Accurancy = ',accuracy_score(y_train,kn_clf.predict(X_train)))

print('Test Accurancy  = ',accuracy_score(y_test,kn_clf.predict(X_test)))