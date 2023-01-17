import pandas as pd

from sklearn.ensemble import RandomForestClassifier



#Load training features

features_train = pd.read_csv('../input/train.csv', sep=',', index_col=0, usecols=[0,2])#,3,4,5,6,7,8,9,10,11])



#Load training labels

labels_train = pd.read_csv('../input/train.csv', sep=',', index_col=0, usecols=[0,1])



#Load test features

features_test = pd.read_csv('../input/test.csv', sep=',', index_col=0, usecols=[0,1])#,3,4,5,6,7,8,9,10,11])



#Create classifier

clf = RandomForestClassifier()



#Fit classifier

clf.fit(features_train,labels_train)



#Predict

pred = clf.predict(features_test)



#Create submission output

test=pd.read_csv('../input/test.csv')

pred =pd.DataFrame(pred,columns=['Survived']) #aus der Prediction nehme ich die Vorhersagespalte und nenne die ‘Survived’

submission=pd.concat([test['PassengerId'],pred],1)



#Write output

submission.to_csv('submission.csv',index=False)