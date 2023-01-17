#importing all the dependencies

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn import svm

from sklearn import neighbors

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



from sklearn.metrics import classification_report
#importing data and storing into dataframe

path = "../input/weight-height.csv"

df = pd.read_csv(path)

df.head()
#Checking if there are any null values in dataframe at all

df.isnull().sum().sum()
print(df.describe())
#Checking if data is balanced or not using pie chart for our target variable - gender

labels = df.Gender.unique()

sizes = [(df.Gender == labels[0]).sum(), (df.Gender == labels[1]).sum()]

plt.pie(sizes, labels = labels, autopct='%1.1f%%', startangle = 90)

plt.title("Checking if data is balanced or not with a pie chart")

plt.show()
#Outliers

plt.boxplot(df.Weight)

plt.title('Box-plot for weight')

plt.show()



plt.boxplot(df.Height)

plt.title('Box-plot for height')

plt.show()
#finding relationship between gender and height

print(df.groupby('Gender')['Height'].describe())

#finding relationship between gender and weight

print(df.groupby('Gender')['Weight'].describe())

#scatter plot to analyse height vs weight

weight = df['Weight']

height = df['Height']

plt.scatter(height, weight)

plt.xlabel('Height')

plt.ylabel('Weight')

plt.title("Height vs Weight")

plt.show()
#histogram for height

plt.hist(height, bins = 10)

plt.xlabel('Bins')

plt.ylabel('Frequency')

plt.title('Histogram for height')

plt.show()



#histogram for weight

plt.hist(weight, bins = 10)

plt.xlabel('Bins')

plt.ylabel('Frequency')

plt.title('Histogram for weight')

plt.show()
#model to split data

train, test = train_test_split(df, test_size=0.2)

print("Test data set")

print(test.head())

print()

print("Train data set")

print(train.head())

print('')

print("No. of data in test:" +str(len(test)))

print("No. of data in train:" +str(len(train)))



print(train.groupby('Gender').count())

#creating x and y variables

feature_names = ['Height', 'Weight']

x_train = train[feature_names].values.tolist()

y_train = train['Gender']

x_test = test[feature_names].values.tolist()

y_test = test['Gender']
#defining classifiers

clf1 = tree.DecisionTreeClassifier()

clf2 = svm.SVC(gamma='auto')

clf3 = neighbors.KNeighborsClassifier()

clf4 = GaussianNB()



#fitting data

clf1 = clf1.fit(x_train,y_train)

clf2 = clf2.fit(x_train,y_train)

clf3 = clf3.fit(x_train, y_train)

clf4 = clf4.fit(x_train, y_train)



#making predictions

prediction1 = clf1.predict(x_test)

prediction2 = clf2.predict(x_test)

prediction3 = clf3.predict(x_test)

prediction4 = clf4.predict(x_test)
#checking accuracy

r1 = accuracy_score(y_test, prediction1)

r2 = accuracy_score(y_test, prediction2)

r3 = accuracy_score(y_test, prediction3)

r4 = accuracy_score(y_test, prediction4)



print("Accuracy score of Model 1: DecisionTreeClassifier is "+str(r1))

print("Accuracy score of Model 2: SupportVectorMachine is "+str(r2))

print("Accuracy score of Model 3: KNN is "+str(r3))

print("Accuracy score of Model 4: GaussianNB is "+str(r4))

#finding misclassification rate

mr1 = (1-metrics.accuracy_score(y_test, prediction1))

mr2 = (1-metrics.accuracy_score(y_test, prediction2))

mr3 = (1-metrics.accuracy_score(y_test, prediction3))

mr4 = (1-metrics.accuracy_score(y_test, prediction4))

print("Misclassification Rate of Decision Tree: "+ str(mr1))

print("Misclassification Rate of SVM: "+ str(mr2))

print("Misclassification Rate of KNN: "+ str(mr3))

print("Misclassification Rate of Naive Bayes: "+ str(mr4))
#confusion matrix



#decision tree

cm1 = confusion_matrix(y_test, prediction1)

TP1 = cm1[1,1]

TN1 = cm1[0,0]

FP1 = cm1[0,1]

FN1 = cm1[1,0]



#svm

cm2 = confusion_matrix(y_test, prediction2)

TP2 = cm2[1,1]

TN2 = cm2[0,0]

FP2 = cm2[0,1]

FN2 = cm2[1,0]



#knn

cm3 = confusion_matrix(y_test, prediction3)

TP3 = cm3[1,1]

TN3 = cm3[0,0]

FP3 = cm3[0,1]

FN3 = cm3[1,0]



#naive-bayes

cm4 = confusion_matrix(y_test, prediction4)

TP4 = cm4[1,1]

TN4 = cm4[0,0]

FP4 = cm4[0,1]

FN4 = cm4[1,0]



#sensitivity

sen1 = TP1 / float(FN1 + TP1)

sen2 = TP2 / float(FN2 + TP2)

sen3 = TP3 / float(FN3 + TP3)

sen4 = TP4 / float(FN4 + TP4)



#specificity

spec1 = TN1 / (TN1 + FP1)

spec2 = TN2 / (TN2 + FP2)

spec3 = TN3 / (TN3 + FP3)

spec4 = TN4 / (TN4 + FP4)



#printing

print("Sensitivity Rate of Decision Tree: "+ str(sen1))

print("Sensitivity Rate of SVM: "+ str(sen2))

print("Sensitivity Rate of KNN: "+ str(sen3))

print("Sensitivity Rate of Naive Bayes: "+ str(sen4))

print()

print("Specificity Rate of Decision Tree: "+ str(spec1))

print("Specificity Rate of SVM: "+ str(spec2))

print("Specificity Rate of KNN: "+ str(spec3))

print("Specificity Rate of Naive Bayes: "+ str(spec4))

print()

print("Classification Report: Decision Tree")

print(classification_report(y_test, prediction1))

print("Classification Report: SVM")

print(classification_report(y_test, prediction2))

print("Classification Report: KNN")

print(classification_report(y_test, prediction3))

print("Classification Report: Naive-Bayes")

print(classification_report(y_test, prediction4))
#Making a prediction for a user

height = 71

weight = 176

prediction = clf2.predict([[height, weight]])

print("The classifier predicts that you could be " +str(prediction[0]))