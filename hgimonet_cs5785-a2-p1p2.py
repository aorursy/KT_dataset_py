##### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json



# read file

with open('//kaggle/input/whats-cooking/train.json', 'r') as myfile:

    train_data=myfile.read()

with open('//kaggle/input/whats-cooking/test.json', 'r') as myfile:

    test_data=myfile.read()

    

# parse file

train_obj = json.loads(train_data)

test_obj = json.loads(test_data)
ingredients=[]

categories=[]

recipeCount=0

for recipe in train_obj:

    recipeCount+=1

    category=recipe['cuisine']

    ingredientList=recipe['ingredients']

    if category not in categories:

        categories.append(category)

    for ingredient in ingredientList:

        if ingredient not in ingredients:

            ingredients.append(ingredient)
print(recipeCount,'recipes, ',len(categories),'categories, ', len(ingredients),' ingredients')
xtrain=np.zeros((recipeCount,len(ingredients)))

ytrain=np.empty(recipeCount,dtype=object)



for i in range(recipeCount):

    ytrain[i]=train_obj[i]['cuisine']

    for ingredient in train_obj[i]['ingredients']:

        index=ingredients.index(ingredient)

        xtrain[i][index]=1
# from sklearn.model_selection import KFold

# from sklearn.naive_bayes import GaussianNB,BernoulliNB



# numFold=3



# kf = KFold(n_splits=numFold)

# gnb = GaussianNB()

# bnb = BernoulliNB()



# gr=0

# br=0



# for trainIndex, testIndex in kf.split(xtrain):

#     yGNBpred = gnb.fit(xtrain[trainIndex], ytrain[trainIndex]).predict(xtrain[testIndex])

#     yBNBpred = bnb.fit(xtrain[trainIndex], ytrain[trainIndex]).predict(xtrain[testIndex])

#     gr+=np.mean(yGNBpred==ytrain[testIndex])/numFold

#     br+=np.mean(yBNBpred==ytrain[testIndex])/numFold

# print('Gaussian Accuracy:',gr,', Bernoulli Accuracy:',br)
# from sklearn.linear_model import LogisticRegression



# # lr=LogisticRegression()

# lr=LogisticRegression(random_state=0, solver='newton-cg', multi_class='auto')



# lrAR=0



# for trainIndex, testIndex in kf.split(xtrain):

#     yLRpred = lr.fit(xtrain[trainIndex], ytrain[trainIndex]).predict(xtrain[testIndex])

#     lrAR+=np.mean(yLRpred==ytrain[testIndex])/numFold

    

# print('Logistic Regression Accuracy:',lrAR)
from sklearn.linear_model import LogisticRegression



l=len(test_obj)

xtest=np.zeros((l,len(ingredients)))

testIndex=np.zeros(l,dtype=int)



for i in range(l):

    testIndex[i]=test_obj[i]['id']

    for ingredient in test_obj[i]['ingredients']:

        if ingredient in ingredients:

            index=ingredients.index(ingredient)

            xtest[i][index]=1
lr=LogisticRegression(random_state=0, solver='sag',multi_class='auto')

ypred=lr.fit(xtrain,ytrain).predict(xtest)
assert(testIndex.size==ypred.size)



# intialise data of lists. 

data = {'id':testIndex, 'cuisine':ypred} 

  

# Create DataFrame 

submit = pd.DataFrame(data) 



# Save submission

submit.to_csv('submission.csv',index = None, header=True)