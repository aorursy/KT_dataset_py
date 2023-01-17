import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics

from sklearn.linear_model import LogisticRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# get training data and testing data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# define useful functions

import matplotlib.pyplot as plt



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    fig, ax = plt.subplots()

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')



def hist_Label(Label):

    Clean_data = train_data[Label].fillna(0)

    Clean_data = Clean_data.drop_duplicates(keep="first")

    Portion_Label = []

    for i in range(0,len(Clean_data)):

        sublist = train_data[train_data[Label] == Clean_data.iloc[i]]["Survived"]

        Portion_Label.append(sum(sublist)/len(sublist))

    return Clean_data,Portion_Label



def hist_img(Label,Clean_data,Portion_Label):

    fig, ax = plt.subplots()

    rect = plt.bar(Clean_data, np.round(Portion_Label,2)*100, width = 0.5)

    ax.set_ylabel('Precentage of people survived (%)')

    ax.set_title(Label +' vs. Survive precentage')

    ax.set_xticks(Clean_data)

    #autolabel(rect)

    fig.tight_layout()

    plt.show()

    

def mapping(train_data):

# mapping one column with numbers

    Input_data = train_data.fillna(0)

    Distinct_features = Input_data.drop_duplicates(keep="first")

    mapping = {}

    for i in range (0,len(Distinct_features)):

        mapping[Distinct_features.iloc[i]] = i+1

    #print(mapping)

    A = Input_data.replace(mapping)

    return A



print(list(train_data))

print('All functions are loaded')
# for feature in ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Embarked']:

#     print(feature)

#     C,P = hist_Label(feature)

#     hist_img(feature,C,P)



# clean data

# clean train data: replace Nan with 0, map sex and embarked with numbers

list(train_data)

Clean_train_data = train_data[['Pclass','Sex','Age','SibSp','Parch','Embarked','Survived']]

Clean_train_data = Clean_train_data.fillna(0)

Clean_train_data['Sex'] = mapping(Clean_train_data['Sex'])

Clean_train_data['Embarked'] = mapping(Clean_train_data['Embarked'])

Clean_train_data = Clean_train_data.fillna(0)

#print(Clean_train_data)

train_data_x = Clean_train_data[['Pclass','Sex','Age','SibSp','Parch','Embarked']]

train_data_y = Clean_train_data[['Survived']]

# clean test data

Clean_test_data = test_data[['Pclass','Sex','Age','SibSp','Parch','Embarked']]

Clean_test_data = Clean_test_data.fillna(0)

Clean_test_data['Sex'] = mapping(Clean_test_data['Sex'])

Clean_test_data['Embarked'] = mapping(Clean_test_data['Embarked'])

#print(Clean_test_data)

X_test = Clean_test_data

# Feature selection & data clean 

# clean train data: replace Nan with 0, remove sex and embarked

list(train_data)

features = ['Pclass','Age','SibSp','Parch']

Clean_train_data = pd.get_dummies(train_data[features])

Clean_train_data = Clean_train_data.fillna(0)

train_data_x = Clean_train_data[features]

train_data_y = train_data[['Survived']]

# clean test data

Clean_test_data = pd.get_dummies(test_data[features])

Clean_test_data = Clean_test_data.fillna(0)

#print(Clean_test_data)

X_test = Clean_test_data
# split training data to test set and dev set

test_set_x = train_data_x[:50]

test_set_y = train_data_y[:50]

train_set_x = train_data_x[51:]

train_set_y= train_data_y[51:]
# Cross validation, return cost_error

def cross_valid(y_pred,data_set_y):

    cost_error = np.mean((y_pred!=data_set_y.values.ravel()))

    target_names = ['Survived', 'Not Survived']

    print('Cost_error is '+str(cost_error) +'\n Classification_report:\n' + classification_report(data_set_y, y_pred, target_names=target_names))
# Output to .csv

def to_csv(data_set_PId,y_pred,Algorithm):

    output = pd.DataFrame({'PassengerId':  data_set_PId.PassengerId, 'Survived': y_pred})

    output.to_csv('Submission_'+ str(Algorithm) + '.csv', index=False)

    print('File name is: ' + 'Submission_'+ str(Algorithm) + '.csv')

    print("Your submission was successfully executed and saved!")
# Using Support Vector Machine 'rbf' on test set

from sklearn.svm import SVC

from sklearn.metrics import classification_report

mykernel = 'rbf'

mygamma = 0.01

myC = 30

svclassifier = SVC(kernel= mykernel, gamma = mygamma, C = myC)

svclassifier.fit(train_set_x, train_set_y.values.ravel())

y_pred_svm_rbf = svclassifier.predict(test_set_x)

#print('gamma:' + str(i) + ',C: '+str(j))

#to_csv(train_data[:150],y_pred,'SVM_rbf')

#cross_valid(y_pred_svm_rbf,test_set_y)

print("Accuracy:",metrics.accuracy_score(test_set_y, y_pred_svm_rbf))
# training accuracy

mykernel = 'rbf'

mygamma = 0.1

myC = 30

svclassifier = SVC(kernel= mykernel, gamma = mygamma, C = myC)

svclassifier.fit(train_set_x, train_set_y.values.ravel())

y_pred1 = svclassifier.predict(train_set_x)

print('gamma:' + str(mygamma) + ',C: '+ str(myC))

cross_valid(y_pred1,train_set_y)

print("Accuracy:",metrics.accuracy_score(train_set_y, y_pred1))
# SVM - kernel linear

from sklearn.svm import SVC

from sklearn.metrics import classification_report

mykernel = 'linear'

svclassifier = SVC(kernel= mykernel)

svclassifier.fit(train_set_x, train_set_y.values.ravel())

y_pred_svm_linear = svclassifier.predict(test_set_x)

#to_csv(train_data[:50],y_pred,mykernel)

#cross_valid(y_pred_svm_linear,test_set_y)

print("Accuracy:",metrics.accuracy_score(test_set_y, y_pred_svm_linear))
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(train_set_x, train_set_y)

#predictions = model.predict(X_test)

#print(predictions)

y_pred_random = model.predict(test_set_x)

print(y_pred_random)

print("Accuracy:",metrics.accuracy_score(test_set_y, y_pred_random))



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('Submission_RFC.csv', index=False)

# print("Your submission was successfully saved!")

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

model = GaussianNB()



model.fit(train_set_x, train_set_y.values.ravel())



#Predict the response for test dataset

y_pred_naive = model.predict(test_set_x)



print(y_pred_naive)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(test_set_y, y_pred_naive))
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics



logisticRegr = LogisticRegression()

logisticRegr.fit(train_set_x, train_set_y.values.ravel())

y_pred_logistic = logisticRegr.predict(test_set_x)



cm = metrics.confusion_matrix(test_set_y, y_pred_logistic)

#print(cm)



score = logisticRegr.score(test_set_x, test_set_y)



print(score)



plt.figure(figsize=(2,2))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15);
# Blending method 1 (most frequent)

from scipy import stats

Final = np.array([y_pred_logistic,y_pred_svm_rbf,y_pred_svm_linear,y_pred_random,y_pred_naive])

Final_pred_Y = stats.mode(Final)

print(y_pred_logistic)

print(y_pred_svm_rbf)

print(y_pred_svm_linear)

print(y_pred_random)

print(y_pred_naive)

#print(Final_pred_Y[0])

#print(test_set_y)

#print(Final_pred_Y[0][0])

print(metrics.accuracy_score(test_set_y, Final_pred_Y[0][0]))

# Blending method 2 (logistic regression)