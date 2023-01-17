import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import SelectKBest,chi2,f_regression

import seaborn as sns
dataset = pd.read_csv("../input/insurance/insurance.csv")

dataset.shape
dataset.head()
dataset.info()
X_train,X_test,y_train,y_test = train_test_split(dataset.drop(['charges'],axis=1),dataset['charges'],test_size=0.25,random_state=1000)
X_train.head()
encoder = LabelEncoder()



X_train['sex'] = encoder.fit_transform(X_train['sex'])

X_test['sex'] = encoder.fit_transform(X_test['sex'])

X_train['smoker'] = encoder.fit_transform(X_train['smoker'])

X_test['smoker'] = encoder.fit_transform(X_test['smoker'])

X_train.region.unique()


new_cols_train = pd.get_dummies(X_train['region'])

new_cols_test = pd.get_dummies(X_test['region'])

X_train.drop(columns=['region'],axis=1,inplace=True)

X_test.drop(columns=['region'],axis=1,inplace=True)

X_train = pd.concat([X_train,new_cols_train],axis=1)

X_test = pd.concat([X_test,new_cols_test],axis=1)

X_train.head()
l_reg = LinearRegression()

l_reg.fit(X_train,y_train)
l_reg.predict(X_test).shape
l_reg.score(X_test,y_test)
sns.heatmap(pd.concat([X_train,y_train],axis=1).corr())

plt.show()
k_best = SelectKBest(f_regression,k=5)

k_best_transformed = k_best.fit_transform(X_train,y_train)

k_best.scores_
X_train.head()
sns.pairplot(X_train,hue='sex',vars=['age','bmi','children'])

cols_to_drop = ['northeast','northwest','southwest']

X_train.drop(columns=cols_to_drop,axis=1,inplace=True)

X_test.drop(cols_to_drop,axis=1,inplace=True)
l_reg.fit(X_train,y_train)

l_reg.score(X_test,y_test)
def age_transform(ages):

    transformed_list = []

    #Here 1 means 'Young', 2 means 'Middle Aged' and 3 means 'Old Age'

    for age in ages:

        if age <= 30:

            transformed_list.append(1)

        elif age < 60:

            transformed_list.append(2)

        else:

            transformed_list.append(3)

    

    return transformed_list

#Adding a new feature 'life_stage' based on persons age

X_train['life_stage'] = age_transform(X_train.age.values)

X_test['life_stage'] = age_transform(X_test.age.values)
def bmi_category(bmi):

    transformed_list = []

    #Here 1 means 'Under weight', 2 means 'Normal' , 3 means 'Over Weight' and 4 means 'Obese'

    for index in bmi:

        if index < 18.5:

            transformed_list.append(1)

        elif index >= 18.5 and index <= 24.9:

            transformed_list.append(2)

        elif index >= 25 and index <= 29.9:

            transformed_list.append(3)

        else:

            transformed_list.append(4)

    

    return transformed_list

#We'll shift the bmi values to it's corresponding category

X_train['bmi'] = bmi_category(X_train.bmi.values)

X_test['bmi'] = bmi_category(X_test.bmi.values)
def calculate_risk(life_stage,smoker,bmi):

    transformed_list = []

    #Here from 1 till 6 we've increasing risk based on life stage, smoker and bmi

    counter = 0

    if len(life_stage) == len(smoker):

        for stage,smoke in zip(life_stage,smoker):

            if (stage == 1) and (smoke == 1) and (bmi[counter] == 2):

                transformed_list.append(1)

            elif (stage == 1) and (smoke == 1) and (bmi[counter] == 3):

                transformed_list.append(2)

            elif (stage == 2) and (smoke == 1) and (bmi[counter] == 2):

                transformed_list.append(3)

            elif (stage == 2) and (smoke == 1) and (bmi[counter] == 3):

                transformed_list.append(4)

            elif (stage == 3) and (smoke == 1) and (bmi[counter] == 2):

                transformed_list.append(5)

            elif (stage == 3) and (smoke == 1) and (bmi[counter] == 3):

                transformed_list.append(6)

            else:

                transformed_list.append(0)

            counter=counter+1

    

    return transformed_list
X_train['life_risk'] = calculate_risk(X_train.life_stage.values,X_train.smoker.values,X_train.bmi.values)

X_test['life_risk'] = calculate_risk(X_test.life_stage.values,X_test.smoker.values,X_test.bmi.values)
l_reg.fit(X_train,y_train)

l_reg.score(X_test,y_test)