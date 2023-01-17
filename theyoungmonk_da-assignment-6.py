import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("../input/Absenteeism_at_work.csv") 
print(data)
data.head()
q75, q25 = np.percentile(data['Absenteeism time in hours'], [75 ,25])
iqr = q75 - q25
print("Lower outlier bound:",q25 - (1.5*iqr))
print("Upper outlier bound:",q75 + (1.5*iqr))
#setting the lower and upper bounds for outliers
sns.boxplot(data['Absenteeism time in hours'])
data= data[data['Absenteeism time in hours']<=17]
data= data[data['Absenteeism time in hours']>=-7]

y = data['Absenteeism time in hours']
del data['Absenteeism time in hours']

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2 , shuffle = False)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print("The top 10 scores for different values of k")
print(sorted(scores,reverse = True)[0:10])
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

print(metrics.classification_report(y_test, y_pred))
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  
y_pred = 0
y_pred = svclassifier.predict(X_test) 
print("Accuracy using SVM :",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
