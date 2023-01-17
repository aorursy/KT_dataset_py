import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df=pd.read_csv("../input/titanic-cleaned-data/train_clean.csv")

df_test=pd.read_csv("../input/titanic-cleaned-data/test_clean.csv")



df.info() # Print a concise summary of a DataFrame.

df.head() # Return the first 5 rows.

df.columns # Return the column labels of the DataFrame.

df.describe() # Generate descriptive statistics.



# Since majority of cabin values are missing -> remove the column



# * PassengerId is unique -> drop column

# * Name is unique -> drop column

# * TicketId is unique-> drop column

# * They do not contribute to the survival probability.



df.drop(["Cabin","Name","PassengerId","Ticket"],axis=1,inplace=True)

df_test.drop(["Cabin","Name","PassengerId","Ticket"],axis=1,inplace=True)



df=df[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Title', 'Family_Size','Survived']]

df_test=df_test[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Title', 'Family_Size','Survived']]



# Age and fare are in float -> convert it into integer and then into categories.



# Age is grouped into 7 categories

data=[df,df_test]

for d in data:

    d['Age'] = d['Age'].astype(int)

    d.loc[ d['Age'] <= 10, 'Age'] = 0

    d.loc[(d['Age'] > 10) & (d['Age'] <= 18), 'Age'] = 1

    d.loc[(d['Age'] > 18) & (d['Age'] <= 25), 'Age'] = 2

    d.loc[(d['Age'] > 25) & (d['Age'] <= 30), 'Age'] = 3

    d.loc[(d['Age'] > 30) & (d['Age'] <= 35), 'Age'] = 4

    d.loc[(d['Age'] > 35) & (d['Age'] <= 40), 'Age'] = 5

    d.loc[(d['Age'] > 40) & (d['Age'] <= 65), 'Age'] = 6

    d.loc[ d['Age'] > 65, 'Age'] = 6



# Fare is grouped into 6 categories

data = [df,df_test]

for d in data:

    d.loc[ d['Fare'] <= 8, 'Fare'] = 0

    d.loc[(d['Fare'] > 8) & (d['Fare'] <= 15), 'Fare'] = 1

    d.loc[(d['Fare'] > 15) & (d['Fare'] <= 31), 'Fare']   = 2

    d.loc[(d['Fare'] > 31) & (d['Fare'] <= 99), 'Fare']   = 3

    d.loc[(d['Fare'] > 99) & (d['Fare'] <= 250), 'Fare']   = 4

    d.loc[ d['Fare'] > 250, 'Fare'] = 5

    d['Fare'] = d['Fare'].astype(int)



# Convert Survived from float-> int

df.Survived=df.Survived.astype(int)



# Creating test and training samples. Splitting the dataframe into two random samples(80% and 20%) for traing and testing.



train, test = train_test_split(df, test_size=0.2)



survived_yes=train.loc[train.Survived==1]

P_yes=len(survived_yes)/len(train)

P_yes # Probability of Survival in training data





survived_no=train.loc[train.Survived==0]

P_no=len(survived_no)/len(train)

P_no # Probability of not Survival in training data





# value counts of each category of an attribute.



for col in train.columns:

    count=train[col].value_counts() 

    print(count)

    

atr=list(df.columns.values)

output_dataframe= pd.DataFrame(columns = ['Actual', 'Predicted']) 



for i in test.itertuples():

    test1=list(i)

    test1.pop(0) # removing Index (unwanted)

    ans=test1.pop() # removing actual value

    py=1

    for i in range(9):

        val = train[(train[atr[i]] == test1[i]) & (train.Survived == 1)].count().values.item(0)

        py = py * (val) / len(survived_yes)

        total_yes = py * P_yes

    pn=1

    for i in range(9):

        val = train[(train[atr[i]] == test1[i]) & (train.Survived == 0)].count().values.item(0)

        pn = pn * (val) / len(survived_no)

        total_no = pn * P_no

    if total_yes>total_no:

        list1=[[ans,1]] #Survived

        output_dataframe=output_dataframe.append(pd.DataFrame(list1,columns=['Actual','Predicted']),ignore_index=True)

    else:

        list0=[[ans,0]] #NotSurvived

        output_dataframe=output_dataframe.append(pd.DataFrame(list0,columns=['Actual','Predicted']),ignore_index=True)





# Evaluation metrics



TP=0

TN=0

FP=0

FN=0

for index,row in output_dataframe.iterrows():

    if row['Predicted']== row['Actual'] and row['Predicted']==1:

        TP += 1

    elif row['Predicted']== row['Actual'] and row['Predicted']==0:

        TN +=1

    elif row['Predicted']==1:

        FP +=1

    else: 

        FN +=1

        

# Accuracy = [TP + TN] / Total Population

accuracy= (TP+TN)/len(output_dataframe)

print("The accuracy for the test set is ",accuracy *100,"%")



# Precision = TP / [TP + FP]

# tells us about the success probability of making a correct positive class classification.

precision = TP / (TP+FP)

print("The precision for the test set is ",precision *100,"%")



# Recall = TP / [TP + FN]

# explains how sensitive the model is towards identifying the positive class.

recall =  TP / (TP+FN)

print("The recall for the test set is ",recall *100,"%")