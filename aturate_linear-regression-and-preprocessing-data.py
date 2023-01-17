# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import matplotlib.pyplot as pt

# %matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from IPython.display import IFrame



# Any results you write to the current directory are saved as output.

#for sorting dictionary

import operator
#Loading Data

train_data = pd.read_csv('../input/Final_Train_Dataset.csv')

test_data = pd.read_csv('../input/Final_Test_Dataset.csv')
#Training Data

print(train_data.head())

print(train_data.info())
print('Distinct job types - ' + str(train_data['job_type'].unique()))
print('Distinct salary - ' + str(train_data['salary'].unique()))
new_columns = []

length_all = []

#Handling List to columns

def column_add(data, col, typ):

    #Handling nulls

    df = data.groupby(col).count().reset_index()

    most_count = df.sort_values('company_name_encoded',ascending=False)[col].head(1).tolist()

    if data[col].isnull().values.any():

        #Replacing NAN values with the most frequent value of column

        data[col] = data[col].fillna(value=str(most_count))

    

    #Code execution only for training data

    if typ == 'train' :

        dict_all = {}

        for id,row in data.iterrows():

            ite = list(row[col].split(","))

            for items in ite:

                if items in dict_all :

                    dict_all[items] += 1

                else :

                    dict_all[items] = 1

        dict_all_sorted = sorted(dict_all.items(), key=operator.itemgetter(1),reverse=True)

        i = 1

        

        for item in dict_all_sorted:

            if i == 8 :

                data[col + '_others'] = 0

                new_columns.append(col + '_others')

                break

            elif item[0] != '...' :

                data[item[0].strip()] = 0

                new_columns.append(item[0].strip())

                

                length_all.append(len(item[0].strip()))

                

                i += 1

    #Inserting Same Columns as in Training dataset

    else :

        for i in new_columns :

            #Preventing ReInsert of already present columns

            if i not in data.columns:

                data[i] = 0

     

    for id,row in data.iterrows():

        ite = list(row[col].split(","))

        for items in ite:

            flag = 0

            for lent in set(length_all):

                #Handling different cases such as 'Mumbai', 'Mumbai(sub)'

                if items[:lent] in new_columns:

                    data.at[id,items[:lent]] = 1

                    flag = 1

            if flag == 0:

                data.at[id,col + '_others'] = 1

                

#Splitting Experience column          

def experience_split(data, col):

    data['Min exp.yrs'] = 0

    data['Max exp.yrs'] = 0

    for id,row in data.iterrows():

        ite = list(row[col].replace('yrs','').split("-"))

        data.at[id,'Min exp.yrs'] = ite[0]

        data.at[id,'Max exp.yrs'] = ite[1].strip()

        

#Handling Salary Column

salary = {'0to3': 1, '3to6': 2, '6to10' : 3, '10to15':4, '15to25' :5, '25to50':6}

key_list = list(salary.keys()) 

val_list = list(salary.values()) 

def salary_ind(data):

    data['salary_ind'] = data['salary'].replace(salary)
#Passing training data

column_add(train_data,'location','train')

column_add(train_data,'key_skills','train')

experience_split(train_data, 'experience')

salary_ind(train_data)



#Passing test data

column_add(test_data,'location','test')

column_add(test_data,'key_skills','test')

experience_split(test_data, 'experience')
print('Columns of Training Data - ' + str(train_data.columns))
train_data.head()
train_data_y = train_data['salary_ind']

train_data_x = train_data.drop(['job_description','job_desig','job_type','experience','Unnamed: 0','job_type'\

                                ,'key_skills'\

                                ,'location','salary','salary_ind'], axis=1)
#Splitting training data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_data_x,train_data_y,test_size=0.25, random_state=42)
#Importing LinearRegression module from sklearn

from sklearn.linear_model import LinearRegression

#Instanciate

model=LinearRegression()

#Fitting Model

model.fit(x_train,y_train)

#Predicting Salaries

prediction=model.predict(x_test)
#Calculating Mean Squared Error

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,(np.rint(prediction)).astype(int))
#Predicting salaries for Test data

test_data_x = test_data.drop(['job_description','job_desig','job_type','experience','job_type','key_skills'\

                                ,'location'], axis=1)

pred_test=model.predict(test_data_x)

prediction_linear = (np.rint(pred_test)).astype(int)



pred_lin = []

for i in prediction_linear:

    if i in salary.values() :

        pred_lin.append(key_list[val_list.index(i)] )

    else:

        pred_lin.append('25to50')



#Adding the predicted score to test dataframe

test_data_x['Predicted_salary'] = pred_lin
test_data_x.head()
#saving test dataframe to output file

test_data_x.to_excel('test_data_output.xlsx')
IFrame('https://public.tableau.com/views/Indiandatascientistsalary/Dashboard1?:embed=y&:display_count=yes', width=800, height=800)