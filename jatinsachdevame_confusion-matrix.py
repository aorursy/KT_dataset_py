#Confusion matrix example.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
actual = [1, 0, 0, 1, 1, 0, 1, 1, 1, 0] 
predicted = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] 
results = confusion_matrix(actual, predicted)
print ('Confusion Matrix :')

results
print ('Accuracy Score :',accuracy_score(actual, predicted) )