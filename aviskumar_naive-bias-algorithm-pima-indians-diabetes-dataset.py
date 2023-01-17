import pandas

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv('../input/pima-indians-dataset/pima-indians-diabetes.csv')

dataframe.head()
array = dataframe.values

array

X = array[:,0:8] # select all rows and first 7 columns which are the attributes

Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties

test_size = 0.30 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = GaussianNB()

model.fit(X_train, Y_train)



# make predictions

expected = Y_test

predicted = model.predict(X_test)



# summarize the fit of the model

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))



#Since the count of the two categories are not same, we need to verify the 

#Recall value of the 2 catgeories
#Here fp and fn are almost same , so the recall and f1 score are almost same.

#if they are different use f1 score to calculate
#Precision: Within a given set of positively-labeled results, the fraction that were true positives = tp/(tp + fp)

#Recall: Given a set of positively-labeled results, the fraction of all positives that were retrieved = tp/(tp + fn)

#Accuracy: tp + tn / (tp + tn + fp +fn) But this measure can be dominated by larger class. Suppose 10, 90 and 80 of 90 

        #is correctly predicted while only 2 of 0 is predicted correctly. Accuracy is 80+2 / 100 i.e. 82%



#TO over come the dominance of the majority class, use weighted measure (not shown)



#F is harmonic mean of precision and recal given by ((B^2 +1) PR) / (B^2P +R)

#When B is set to 1 we get F1 = 2PR / (P+R)