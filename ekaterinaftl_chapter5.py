import pickle

ofname = open('../input/chapter/files/ch05/dataset_small.pkl','rb')

#x stores input data and y target values

(x, y)=pickle.load(ofname, encoding='iso8859')

#Check on the test set

yhat = knn.predict(X_test)

print ('TESTING STATS:')

print ('classification accuracy:', metrics. accuracy_score(yhat , y_test))

print ('confusion matrix: \n' + str(metrics. confusion_matrix(yhat , y_test)))