from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#Dict of accuracies
d = {}

p = pd.read_csv("../input/phishing.csv")
#p.info()
#print(type(p))

p.columns = ['having_IPhaving_IP_Address',
'URLURL_Length',
'Shortining_Service',
'having_At_Symbol',
'double_slash_redirecting',
'Prefix_Suffix',
'having_Sub_Domain',
'SSLfinal_State',
'Domain_registeration_length',
'Favicon',
'port',
'HTTPS_token',
'Request_URL',
'URL_of_Anchor',
'Links_in_tags',
'SFH',
'Submitting_to_email',
'Abnormal_URL',
'Redirect',
'on_mouseover',
'RightClick',
'popUpWidnow',
'Iframe',
'age_of_domain',
'DNSRecord',
'web_traffic',
'Page_Rank',
'Google_Index',
'Links_pointing_to_page',
'Statistical_report',
'Result']
target = p["Result"].T

data = p.drop('Result', axis = 1)
#print(data)
#print(target)

#KNN classification

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(data, target, test_size=0.3,random_state=21, stratify=target)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print("accuracy of KNN: ",knn.score(X_test, y_test))

d['accuracy of KNN'] = knn.score(X_test, y_test)

#len(X_test)

y_test = np.asarray(y_test)
X_test = np.asarray(X_test)

misclassified = np.where(y_test != y_pred)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")
##Decision Trees

from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=25, random_state=1)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred_dt = dt.predict(X_test)

from sklearn.metrics import accuracy_score


# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred_dt)
print("accuracy of DT: {}".format(acc))

d['accuracy of DT'] = acc

misclassified = np.where(y_test != y_pred_dt)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")
## Ensemble bagging

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt_b = DecisionTreeClassifier(random_state=1, max_depth = 25)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt_b, 
                       n_estimators=50,
                       oob_score=True,
                       random_state=1)

# Fit bc to the training set 
bc.fit(X_train, y_train)

# Predict test set labels
y_pred_dt_b = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred_dt_b)

d['accuracy of Bagging DT'] = acc_test


# Evaluate OOB accuracy
acc_oob = bc.oob_score_

d['oob accuracy of Bagging DT'] = acc_oob

# Print acc_test and acc_oob
print('accuracy of DT Bagging: {}, OOB accuracy: {}'.format(acc_test, acc_oob))

misclassified = np.where(y_test != y_pred_dt_b)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")


##Ensemble Voting

from sklearn.linear_model import LogisticRegression
SEED = 1


# Instantiate knn
knn = KNeighborsClassifier(n_neighbors=5)

# Instantiate dt
dt = DecisionTreeClassifier(random_state=SEED, max_depth = 25)


classifiers = [ ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
for clf_name, clf in classifiers:  
    # Fit clf to the training set
    clf.fit(X_train, y_train)  
    # Predict y_pred
    y_pred = clf.predict(X_test)    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)  
    d['accuracy of'+ ' ' + clf_name +' in Voting'] = accuracy
    # Evaluate clf's accuracy on the test set
    print('{} : {}'.format(clf_name, accuracy))
    

# Import VotingCLassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc 
vc = VotingClassifier(estimators=classifiers)     

# Fit vc to the training set
vc.fit(X_train, y_train)   

# Evaluate the test set predictions
y_pred_vc = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {}'.format(accuracy))

d['accuracy of Voting classifiers collectively'] = accuracy

misclassified = np.where(y_test != y_pred_vc)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")

##Logistic regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C = 100, solver='newton-cg', max_iter = 25)

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred_LoR = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_LoR)
print('Logistic Regression: {}'.format(accuracy))

d['accuracy of Logistic Regression'] = accuracy

misclassified = np.where(y_test != y_pred_LoR)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")

##Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 30, random_state = 12)

# Fit the classifier to the training data
RFC.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred_RFC = RFC.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_RFC)
print('Random Forest Classifier Accuracy: {}'.format(accuracy))

d['accuracy of Random Forest Classifier'] = accuracy

misclassified = np.where(y_test != y_pred_RFC)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")


#svm
from sklearn.svm import SVC

svc = SVC(kernel ='rbf' ,C = 1.0, random_state = 21, gamma = 'scale', max_iter = -1)

svc.fit(X_train, y_train)

y_svc = svc.predict(X_test)

accuracy = accuracy_score(y_pred, y_svc)
print('SVM Classifier Accuracy: {}'.format(accuracy))

d['accuracy of SVC'] = accuracy

misclassified = np.where(y_test != y_svc)
print("misclassfied elements")
print(misclassified)
print("misclassfied elements count: ", len(misclassified[0]) )
print("######################################")



print('\n')
print('\n')
for i in d:
    print(i,' : ' ,d[i])
    



