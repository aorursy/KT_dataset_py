import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost
import seaborn as sns
import os
import sys
import pickle
sys.path.append("/kaggle/input/enrondata/enron-project")
from feature_format import featureFormat
from feature_format import targetFeatureSplit
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

original = "/kaggle/input/enrondata/enron-project/final_project_dataset.pkl"
destination = "final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
 content = infile.read()
with open(destination, 'wb') as output:
 for line in content.splitlines():
  outsize += len(line) + 1
  output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
import pickle
main_data=pickle.load(open("final_project_dataset_unix.pkl","rb")) 
#for counting no of poi in the data 
print("The number of people in the dataset:",len(main_data))
print('data for any random person for counting no of features available for example 3rd person:',list(main_data.keys())[2],"\n",main_data[list(main_data.keys())[2]])
#for counting no of poi in the data 
no_of_poi=1
for name, record in main_data.items():
	if record['poi'] == 1:
		no_of_poi += 1
print('There are {} Person of intrest in the dataset'.format(no_of_poi))
#visualizing the data and then removing the outlier by analysis

df = pd.DataFrame.from_dict(main_data,orient='index')

import matplotlib.pyplot as plt
df.plot(kind = 'scatter', x = 'salary', y = 'bonus',figsize=(10,10))
#remove the key 'total'
main_data.pop('TOTAL')
from feature_format import featureFormat as ft
from feature_format import targetFeatureSplit as tfs
plot = ft(main_data, ['poi','salary',"bonus"])

for j in range(len(plot)):
        if plot[j][0]==True:
            plt.scatter(plot[j][1],plot[j][2],color = 'b')
        else:
            plt.scatter(plot[j][1],plot[j][2],color = 'g')

plt.ylabel('bonus')
plt.xlabel('salary')   
plt.show()

plot = ft(main_data, ['poi','salary',"total_payments"])

for j in range(len(plot)):
        if plot[j][0]==True:
            plt.scatter(plot[j][1],plot[j][2],color = 'b')
        else:
            plt.scatter(plot[j][1],plot[j][2],color = 'g')

plt.ylabel('total_payments')
plt.xlabel('salary')   
plt.show()
plot = ft(main_data, ['poi',"total_payments","loan_advances"])

for j in range(len(plot)):
        if plot[j][0]==True:
            plt.scatter(plot[j][1],plot[j][2],color = 'b')
        else:
            plt.scatter(plot[j][1],plot[j][2],color = 'g')

plt.ylabel('total_payments')
plt.xlabel('loan_advances')   
plt.show()
plot = ft(main_data, ['poi',"from_poi_to_this_person", "from_this_person_to_poi"])

for j in range(len(plot)):
        if plot[j][0]==True:
            plt.scatter(plot[j][1],plot[j][2],color = 'b')
        else:
            plt.scatter(plot[j][1],plot[j][2],color = 'g')

plt.ylabel('from_poi_to_this_person')
plt.xlabel('from_this_person_to_poi')   
plt.show()
#SELECTING FEATURES WE WILL USE FURTHER IN THE PROECT and splitting training and teationg data 
from sklearn.model_selection import train_test_split
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'other', 'shared_receipt_with_poi', 
                 'restricted_stock', 'director_fees', 'to_messages','from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
my_data= featureFormat(main_data, features_list, sort_keys = True)
label, feature = targetFeatureSplit(my_data)
X_train, X_test, y_train, y_test = train_test_split(feature, label,test_size=0.15)
                                                   
#final training and testing various algorithms k nearest given best accuracy on further adjusting accuracy improved


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
classifier1 = RandomForestClassifier()
classifier1.fit(X_train, y_train)
pred1=classifier1.predict(X_test)
acc1=accuracy_score(pred1,y_test)
print('accuracy of random forest',acc1)
classifier2 = AdaBoostClassifier(n_estimators=100)
classifier2.fit(X_train, y_train)
pred2=classifier2.predict(X_test)
acc2=accuracy_score(pred2,y_test)
print("accuracy of AdaBoost",acc2)
classifier3 = KNeighborsClassifier(n_neighbors=20)
classifier3.fit(X_train, y_train)
pred3=classifier3.predict(X_test)
acc3=accuracy_score(pred3,y_test)
print("accuracy of k nearest neighbors",acc3)
classifier4 = GaussianNB()
classifier4.fit(X_train, y_train)
pred4 = classifier4.predict(X_test)
acc4=accuracy_score(pred4,y_test)
print("accuracy of gaussian nb",acc4)
classifier5 =SVC()
classifier5.fit(X_train, y_train)
pred5 = classifier5.predict(X_test)
acc5=accuracy_score(pred5,y_test)
print("accuracy of svc",acc5)
#outputing data as pickle file 
pickle.dump(main_data, open("my_dataset.pkl", "wb") )
pickle.dump(features_list, open("my_feature_list.pkl", "wb") )
pickle.dump(acc3, open("my_classifier.pkl", "wb") )

