import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/haberman.csv')

df.head()
#Change column names

df.columns = ['age', 'year_of_treatment', 'positive_nodes', 'survival_status']

df.head()
#get data statistics

df.mean()

df.std()

df.median()

df.survival_status.unique()

#1 = the patient survived 5 years or longer 

#2 = the patient died within 5 year
df.describe()

#gives all the information about the data 

#percentile values mean , std  etc....
df.survival_status.value_counts()

#out of all 224 patients survived 5 years or longer and 81 died within 5 years
print("number of rows: " + str(df.shape[0]))

print("number of columns: " + str(df.shape[1]))

print("columns: " + ", ".join(df.columns))
#what we know so far is that dataset has no missing values 

#our target variable has (survival status after 5 years) to be converted as a categorical datatype as{Yes:1 No:2}

#it will not be easily to classify and visualize
(df['survival_status'].value_counts()/len(df)).plot.bar(color=['green', 'red'])

(df['survival_status'].value_counts()/len(df))

#of the data given more then 73% survived and

#26% did not 
sns.set_style('darkgrid')

sns.FacetGrid(df, hue='survival_status', height=7).map(plt.scatter, "age", 'positive_nodes').add_legend();

plt.show();

#what we can see is both of classes are mixed 

#so classification would be difficult

#it appears that people with less positive nodes survived the most
from sklearn.model_selection import train_test_split

features = ["age", "year_of_treatment", "positive_nodes"]

target = ["survival_status"]

df_train,df_test = train_test_split(df,test_size=0.25,random_state=100)
from sklearn.linear_model import LogisticRegression

#Using LogisticRegression to predict

model = LogisticRegression()

model.fit(df_train[features],df_train[target])

print("Intercept:",model.intercept_,"\nCoefficients:", model.coef_)
train_accuracy = model.score(df_train[features],df_train[target])

test_accuracy = model.score(df_test[features],df_test[target])

print(train_accuracy,test_accuracy)
from sklearn.metrics import confusion_matrix

train_predictions = model.predict(df_train[features])

test_predictions = model.predict(df_test[features])

train_conf_matrix = confusion_matrix(df_train[target],train_predictions)

test_conf_matrix = confusion_matrix(df_test[target],test_predictions)
pd.DataFrame(train_conf_matrix,columns=model.classes_,index=model.classes_)
pd.DataFrame(test_conf_matrix,columns=model.classes_,index=model.classes_)
train_correct_predictions = train_conf_matrix[0][0]+train_conf_matrix[1][1]

train_total_predictions = train_conf_matrix.sum()

train_accuracy = train_correct_predictions/train_total_predictions

print(train_accuracy)
from sklearn.metrics import classification_report

print(classification_report(df_train[target],train_predictions))
print(classification_report(df_test[target],test_predictions))
#enter as age, year_of_treatment, positive_nodes

input_data = [[30, 64 , 20],

                      [31, 52, 35], [24, 43, 4], [24, 43, 4], [24, 23, 0], [100, 100, 100]]

result = model.predict(input_data)

for i in result:

    if i == 1:

        print("More than 5 years")

    elif i == 2:

        print("Less than 5 years")