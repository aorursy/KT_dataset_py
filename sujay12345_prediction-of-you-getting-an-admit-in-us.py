import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
Reading = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")

Reading.head() #printing the first five rows 
Reading.describe()
Null=Reading.isnull()

Null.sum()
Reading = Reading.rename(columns={'GRE Score': 'GRE Score', 'TOEFL Score': 'TOEFL Score', 'LOR ': 'LOR', 'Chance of Admit ': 'Admit Possibilty'})

Reading.head()
Reading.drop('Serial No.', axis='columns', inplace=True)

Reading.head()


gre_score = Reading[["GRE Score"]] #selecting only the required coloumn

toefl_score = Reading[["TOEFL Score"]] 

uni_rating=  Reading[["University Rating"]]

fig=sns.distplot(gre_score,color='black',kde=False)

plt.title("GRE SCORES")

plt.show()



fig=sns.distplot(toefl_score,color='r',kde=False)

plt.title("TOEFL SCORES")

plt.show()



fig=sns.distplot(uni_rating,color='r',kde=False)

plt.title("UNIVERSITY RATING")

plt.show()





fig=sns.lmplot(x='GRE Score',y='CGPA',data=Reading)

plt.title("CGPA VS GRE SCORE")

plt.show()





fig=sns.jointplot(x='CGPA',y='Admit Possibilty',data=Reading,kind='kde')

plt.show()
fig=sns.lmplot(x='CGPA',y='TOEFL Score',data=Reading)

plt.title("CGPA VS TOEFL")

plt.show()
sns.pairplot(data=Reading,vars=["GRE Score","Admit Possibilty"])

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
x=Reading.drop('Admit Possibilty',axis='columns')

y=Reading['Admit Possibilty']

x_train,x_test,y_train,y_test=train_test_split(x, y)







                      

                                                                

x_train.shape

x_test.shape
y_train.shape
y_test.shape
linear_regression = LinearRegression()

linear_regression = linear_regression.fit(x_train,y_train)
def get_cv_scores(linear_regression):

    scores = cross_val_score(linear_regression,

                             x_train,

                             y_train,

                             cv=5,

                             scoring='r2')

    

    print('CV Mean: ', np.mean(scores))

    print('STD: ', np.std(scores))

    print('\n')



# get cross val scores

get_cv_scores(linear_regression)

model = LinearRegression(normalize=True)

model.fit(x_test, y_test)

model.score(x_test, y_test)
print('The chance of you getting an admit in the US is {}%'.format(round(model.predict([[305, 108, 4, 4.5, 4.5, 8.35, 0]])[0]*100, 1)))