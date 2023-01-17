import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt

dataset=pd.read_csv('/kaggle/input/titanic/train.csv')
print(dataset)
y=dataset.iloc[:,1].values
x_names=dataset[['Name']]
names_in_x=x_names.iloc[:,0].values

#Extracting titles from the name column

titles=['Col.','Dr.','Lady','Master','Miss','Mr.','Mrs.','Ms.','Rev.','Sir']
passenger_titles=[]
for i in range(len(x_names)):
    has_title=False
    for j in range(len(titles)):
        if(titles[j] in names_in_x[i]):
            passenger_titles.append(titles[j])
            has_title=True
            break
    if(has_title==False):
        passenger_titles.append('None')
print(passenger_titles)
#Storing passenger data in a numeric form

numeric_passenger_titles=[]
for i in range(len(passenger_titles)):
    if(passenger_titles[i] in titles):
        numeric_passenger_titles.append(titles.index(passenger_titles[i]))
    else:
        numeric_passenger_titles.append(len(titles))
print(numeric_passenger_titles)
#Calculating the total family size per passenger
        
x_sibsp=dataset[['SibSp']]
x_parch=dataset[['Parch']]

x_s=x_sibsp.iloc[:,0].values
x_p=x_parch.iloc[:,0].values

numeric_family_size=[]

for i in range(len(x_s)):
    numeric_family_size.append(x_s[i]+x_p[i]+1)
    
print(numeric_family_size)
#Extracting the ticket code

x_ticket=dataset[['Ticket']]
x_t=x_ticket.iloc[:,0].values
ticket_code=[]
for i in range(len(x_t)):
    if(x_t[i][0]>='A' and x_t[i][0]<='Z'):
        ticket_code.append(x_t[i][0])
    else:
        ticket_code.append('N')

#Assigning a numeric value to each ticket code.

ticket_code_labels=[]
for i in ticket_code: 
    if i not in ticket_code_labels: 
        ticket_code_labels.append(i)
numeric_ticket_code=[]
for i in range(len(ticket_code)):
    numeric_ticket_code.append(ticket_code_labels.index(ticket_code[i]))

print(numeric_ticket_code)
#Extracting the cabin code

x_cabin=dataset[['Cabin']]
x_cabin=x_cabin.replace(np.nan, 'N', regex=True)
x_cb=x_cabin.iloc[:,0].values
x_cb=np.array(x_cb)
x_cabin_code=[]
for i in range(len(x_cb)):
    x_cabin_code.append(x_cb[i])
cabin_code=[]
for i in range(len(x_cabin_code)):
    cabin_code.append(x_cabin_code[i][0])
    
#Assigning a numeric value to each cabin code.
    
cabin_code_labels=[]
for i in cabin_code: 
    if i not in cabin_code_labels: 
        cabin_code_labels.append(i)
numeric_cabin_code=[]
for i in range(len(cabin_code)):
    numeric_cabin_code.append(cabin_code_labels.index(cabin_code[i]))

print(numeric_cabin_code)
#Assigning a numeric value to gender

x_gender=dataset[['Sex']]
x_g=x_gender.iloc[:,0].values
numeric_gender=[]
for value in x_g:
    if(value=='male'):
        numeric_gender.append(1)
    else:
        numeric_gender.append(0)

print(numeric_gender)
#Extracting the place from where the passenger embarked
x_embarked=dataset[['Embarked']]
x_embarked=x_cabin.replace(np.nan, 'N', regex=True)
x_em=x_embarked.iloc[:,0].values
x_em=np.array(x_cb)
x_embarked_code=[]
for i in range(len(x_cb)):
    x_embarked_code.append(x_em[i])
embarked_code=[]
for i in range(len(x_embarked_code)):
    embarked_code.append(x_embarked_code[i][0])
    
#Assigning a numeric value to each embarked code.
    
embarked_code_labels=[]
for i in embarked_code: 
    if i not in embarked_code_labels: 
        embarked_code_labels.append(i)
numeric_embarked_code=[]
for i in range(len(embarked_code)):
    numeric_embarked_code.append(embarked_code_labels.index(embarked_code[i]))
    
print(numeric_embarked_code)
#Filling the missing values of fare
    
x_fare=dataset[['Fare']]
x_f=x_fare.iloc[:,:].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(x_f)
x_f = imputer.transform(x_f)

numeric_fare=[]
for i in range(len(x_f)):
    numeric_fare.append(x_f[i][0])
#print(x_f)

x_pclass=dataset[['Pclass']]
x_pc=x_pclass.iloc[:,:].values
numeric_pclass=[]
for i in range(len(x_pc)):
    numeric_pclass.append(x_pc[i])

x_age_data=[]
x_age_data.append(numeric_ticket_code)
x_age_data.append(numeric_cabin_code)
x_age_data.append(numeric_passenger_titles)
x_age_data.append(numeric_gender)
x_age_data.append(numeric_embarked_code)
x_age_data.append(numeric_family_size)
x_age_data.append(numeric_pclass)
x_age_data.append(numeric_fare)

x_age=dataset[['Age']]
x_numeric_age=x_age.iloc[:,0].values
numeric_age=[]
for i in range(len(x_numeric_age)):
    numeric_age.append(x_numeric_age[i])

x_age_data_transpose=np.transpose(x_age_data)

age_trainer=[]
numeric_age_train=[]
age_tester=[]
numeric_age_tester=[]
for i in range(len(numeric_age)):
    if(mt.isnan(numeric_age[i])):
        age_tester.append(x_age_data_transpose[i])
        numeric_age_tester.append(numeric_age[i])
    else:
        age_trainer.append(x_age_data_transpose[i])
        numeric_age_train.append(numeric_age[i])
        
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(age_trainer, numeric_age_train)
y_pred=regressor.predict(age_tester)   

iterator_value=0
for i in range(len(numeric_age)):
    if(mt.isnan(numeric_age[i])):
        numeric_age[i]=y_pred[iterator_value]
        iterator_value=iterator_value+1
x_age_data.append(numeric_age)
x_age_data_transpose=np.transpose(x_age_data)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_age_data_transpose, y, test_size = 1/3, random_state = 0)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

num_correct=0
for i in range(len(Y_pred)):
    if(Y_pred[i]==Y_test[i]):
        num_correct=num_correct+1
    
print(num_correct/len(Y_pred))