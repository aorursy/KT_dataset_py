# This Python 3 environment comes with many helpful analytics libraries installed

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
Train=pd.read_csv('/kaggle/input/Train.csv')

Test=pd.read_csv("/kaggle/input/Test.csv")

Train_1=pd.read_csv('/kaggle/input/Train.csv')

Test_1=pd.read_csv("/kaggle/input/Test.csv")
Train.loc[Train['Client Retention Flag']=='Yes','Client Retention Flag']=1

Train.loc[Train['Client Retention Flag']=='No','Client Retention Flag']=0



Train=Train.drop(['Company ID','Flag 1'],axis=1)

Train=pd.get_dummies(Train,columns=['Flag 6','Flag 2','Flag 3','Flag 4','Flag 5','Client Contract Starting Month'])



Test=Test.drop(['Company ID','Flag 1'],axis=1)

Test=pd.get_dummies(Test,columns=['Flag 6','Flag 2','Flag 3','Flag 4','Flag 5','Client Contract Starting Month'])

#feature_generation based on Activities

Train['Activity 1 end']=Train['Activity 1 Time Period 11']+Train['Activity 1 Time Period 10']+Train['Activity 1 Time Period 9']+Train['Activity 1 Time Period 8']

Train['Activity 1 Frequency']=Train['Activity 1 Time Period 11']+Train['Activity 1 Time Period 10']+Train['Activity 1 Time Period 9']+Train['Activity 1 Time Period 8']+Train['Activity 1 Time Period 7']+Train['Activity 1 Time Period 6']+Train['Activity 1 Time Period 5']+Train['Activity 1 Time Period 4']+Train['Activity 1 Time Period 3']+Train['Activity 1 Time Period 2']+Train['Activity 1 Time Period 1']+Train['Activity 1 Time Period 0']



Train['Activity 2 end']=Train['Activity 2 Time Period 11']+Train['Activity 2 Time Period 10']+Train['Activity 2 Time Period 9']+Train['Activity 2 Time Period 8']

Train['Activity 2 Frequency']=Train['Activity 2 Time Period 11']+Train['Activity 2 Time Period 10']+Train['Activity 2 Time Period 9']+Train['Activity 2 Time Period 8']+Train['Activity 2 Time Period 7']+Train['Activity 2 Time Period 6']+Train['Activity 2 Time Period 5']+Train['Activity 2 Time Period 4']+Train['Activity 2 Time Period 3']+Train['Activity 2 Time Period 2']+Train['Activity 2 Time Period 1']+Train['Activity 2 Time Period 0']



Train['Activity 3 end']=Train['Activity 3 Time Period 11']+Train['Activity 3 Time Period 10']+Train['Activity 3 Time Period 9']+Train['Activity 3 Time Period 8']

Train['Activity 3 Frequency']=Train['Activity 3 Time Period 11']+Train['Activity 3 Time Period 10']+Train['Activity 3 Time Period 9']+Train['Activity 3 Time Period 8']+Train['Activity 3 Time Period 7']+Train['Activity 3 Time Period 6']+Train['Activity 3 Time Period 5']+Train['Activity 3 Time Period 4']+Train['Activity 3 Time Period 3']+Train['Activity 3 Time Period 2']+Train['Activity 3 Time Period 1']+Train['Activity 3 Time Period 0']



Train['Activity 4 end']=Train['Activity 4 Time Period 11']+Train['Activity 4 Time Period 10']+Train['Activity 4 Time Period 9']+Train['Activity 4 Time Period 8']

Train['Activity 4 Frequency']=Train['Activity 4 Time Period 11']+Train['Activity 4 Time Period 10']+Train['Activity 4 Time Period 9']+Train['Activity 4 Time Period 8']+Train['Activity 4 Time Period 7']+Train['Activity 4 Time Period 6']+Train['Activity 4 Time Period 5']+Train['Activity 4 Time Period 4']+Train['Activity 4 Time Period 3']+Train['Activity 4 Time Period 2']+Train['Activity 4 Time Period 1']+Train['Activity 4 Time Period 0']



Train['Activity 5 end']=Train['Activity 5 Time Period  11']+Train['Activity 5 Time Period  10']+Train['Activity 5 Time Period  9']+Train['Activity 5 Time Period  8']

Train['Activity 5 Frequency']=Train['Activity 5 Time Period  11']+Train['Activity 5 Time Period  10']+Train['Activity 5 Time Period  9']+Train['Activity 5 Time Period  8']+Train['Activity 5 Time Period  7']+Train['Activity 5 Time Period  6']+Train['Activity 5 Time Period  5']+Train['Activity 5 Time Period  4']+Train['Activity 5 Time Period  3']+Train['Activity 5 Time Period  2']+Train['Activity 5 Time Period  1']+Train['Activity 5 Time Period  0']



Train['Activity 6 end']=Train['Activity 6 Time Period  11']+Train['Activity 6 Time Period  10']+Train['Activity 6 Time Period  9']+Train['Activity 6 Time Period  8']

Train['Activity 6 Frequency']=Train['Activity 6 Time Period  11']+Train['Activity 6 Time Period  10']+Train['Activity 6 Time Period  9']+Train['Activity 6 Time Period  8']+Train['Activity 6 Time Period  7']+Train['Activity 6 Time Period  6']+Train['Activity 6 Time Period  5']+Train['Activity 6 Time Period  4']+Train['Activity 6 Time Period  3']+Train['Activity 6 Time Period  2']+Train['Activity 6 Time Period  1']+Train['Activity 6 Time Period  0']



Train['Activity 7 end']=Train['Activity 7 Time Period  11']+Train['Activity 7 Time Period  10']+Train['Activity 7 Time Period  9']+Train['Activity 7 Time Period  8']

Train['Activity 7 Frequency']=Train['Activity 7 Time Period  11']+Train['Activity 7 Time Period  10']+Train['Activity 7 Time Period  9']+Train['Activity 7 Time Period  8']+Train['Activity 7 Time Period  7']+Train['Activity 7 Time Period  6']+Train['Activity 7 Time Period  5']+Train['Activity 7 Time Period  4']+Train['Activity 7 Time Period  3']+Train['Activity 7 Time Period  2']+Train['Activity 7 Time Period  1']+Train['Activity 7 Time Period  0']



Train['Activity 8 end']=Train['Activity 8 Time Period 11']+Train['Activity 8 Time Period 10']+Train['Activity 8 Time Period 9']+Train['Activity 8 Time Period 8']

Train['Activity 8 Frequency']=Train['Activity 8 Time Period 11']+Train['Activity 8 Time Period 10']+Train['Activity 8 Time Period 9']+Train['Activity 8 Time Period 8']+Train['Activity 8 Time Period 7']+Train['Activity 8 Time Period 6']+Train['Activity 8 Time Period 5']+Train['Activity 8 Time Period 4']+Train['Activity 8 Time Period 3']+Train['Activity 8 Time Period 2']+Train['Activity 8 Time Period 1']+Train['Activity 8 Time Period 0']



Train['Activity end']=Train['Activity 1 end']+Train['Activity 2 end']+Train['Activity 3 end']+Train['Activity 4 end']+Train['Activity 5 end']+Train['Activity 6 end']+Train['Activity 7 end']+Train['Activity 8 end']

Train['Time Period 0']=Train['Activity 1 Time Period 0']+Train['Activity 2 Time Period 0']+Train['Activity 3 Time Period 0']+Train['Activity 6 Time Period  0']+Train['Activity 8 Time Period 0']

Train['Time Period 1']=Train['Activity 1 Time Period 1']+Train['Activity 2 Time Period 1']+Train['Activity 3 Time Period 1']+Train['Activity 6 Time Period  1']+Train['Activity 8 Time Period 1']

Train['Time Period 2']=Train['Activity 1 Time Period 2']+Train['Activity 2 Time Period 2']+Train['Activity 3 Time Period 2']+Train['Activity 6 Time Period  2']+Train['Activity 8 Time Period 2']

Train['Time Period 3']=Train['Activity 1 Time Period 3']+Train['Activity 2 Time Period 3']+Train['Activity 3 Time Period 3']+Train['Activity 6 Time Period  3']+Train['Activity 8 Time Period 3']

Train['Time Period 4']=Train['Activity 1 Time Period 4']+Train['Activity 2 Time Period 4']+Train['Activity 3 Time Period 4']+Train['Activity 6 Time Period  4']+Train['Activity 8 Time Period 4']

Train['Time Period 5']=Train['Activity 1 Time Period 5']+Train['Activity 2 Time Period 5']+Train['Activity 3 Time Period 5']+Train['Activity 6 Time Period  5']+Train['Activity 8 Time Period 5']

Train['Time Period 6']=Train['Activity 1 Time Period 6']+Train['Activity 2 Time Period 6']+Train['Activity 3 Time Period 6']+Train['Activity 6 Time Period  6']+Train['Activity 8 Time Period 6']

Train['Time Period 7']=Train['Activity 1 Time Period 7']+Train['Activity 2 Time Period 7']+Train['Activity 3 Time Period 7']+Train['Activity 6 Time Period  7']+Train['Activity 8 Time Period 7']

Train['Time Period 8']=Train['Activity 1 Time Period 8']+Train['Activity 2 Time Period 8']+Train['Activity 3 Time Period 8']+Train['Activity 6 Time Period  8']+Train['Activity 8 Time Period 8']

Train['Time Period 9']=Train['Activity 1 Time Period 9']+Train['Activity 2 Time Period 9']+Train['Activity 3 Time Period 9']+Train['Activity 6 Time Period  9']+Train['Activity 8 Time Period 9']

Train['Time Period 10']=Train['Activity 1 Time Period 10']+Train['Activity 2 Time Period 10']+Train['Activity 3 Time Period 10']+Train['Activity 6 Time Period  10']+Train['Activity 8 Time Period 10']

Train['Time Period 11']=Train['Activity 1 Time Period 11']+Train['Activity 2 Time Period 11']+Train['Activity 3 Time Period 11']+Train['Activity 6 Time Period  11']+Train['Activity 8 Time Period 11']
Test['Activity 1 end']=Test['Activity 1 Time Period 11']+Test['Activity 1 Time Period 10']+Test['Activity 1 Time Period 9']+Test['Activity 1 Time Period 8']

Test['Activity 1 Frequency']=Test['Activity 1 Time Period 11']+Test['Activity 1 Time Period 10']+Test['Activity 1 Time Period 9']+Test['Activity 1 Time Period 8']+Test['Activity 1 Time Period 7']+Test['Activity 1 Time Period 6']+Test['Activity 1 Time Period 5']+Test['Activity 1 Time Period 4']+Test['Activity 1 Time Period 3']+Test['Activity 1 Time Period 2']+Test['Activity 1 Time Period 1']+Test['Activity 1 Time Period 0']



Test['Activity 2 end']=Test['Activity 2 Time Period 11']+Test['Activity 2 Time Period 10']+Test['Activity 2 Time Period 9']+Test['Activity 2 Time Period 8']

Test['Activity 2 Frequency']=Test['Activity 2 Time Period 11']+Test['Activity 2 Time Period 10']+Test['Activity 2 Time Period 9']+Test['Activity 2 Time Period 8']+Test['Activity 2 Time Period 7']+Test['Activity 2 Time Period 6']+Test['Activity 2 Time Period 5']+Test['Activity 2 Time Period 4']+Test['Activity 2 Time Period 3']+Test['Activity 2 Time Period 2']+Test['Activity 2 Time Period 1']+Test['Activity 2 Time Period 0']



Test['Activity 3 end']=Test['Activity 3 Time Period 11']+Test['Activity 3 Time Period 10']+Test['Activity 3 Time Period 9']+Test['Activity 3 Time Period 8']

Test['Activity 3 Frequency']=Test['Activity 3 Time Period 11']+Test['Activity 3 Time Period 10']+Test['Activity 3 Time Period 9']+Test['Activity 3 Time Period 8']+Test['Activity 3 Time Period 7']+Test['Activity 3 Time Period 6']+Test['Activity 3 Time Period 5']+Test['Activity 3 Time Period 4']+Test['Activity 3 Time Period 3']+Test['Activity 3 Time Period 2']+Test['Activity 3 Time Period 1']+Test['Activity 3 Time Period 0']



Test['Activity 4 end']=Test['Activity 4 Time Period 11']+Test['Activity 4 Time Period 10']+Test['Activity 4 Time Period 9']+Test['Activity 4 Time Period 8']

Test['Activity 4 Frequency']=Test['Activity 4 Time Period 11']+Test['Activity 4 Time Period 10']+Test['Activity 4 Time Period 9']+Test['Activity 4 Time Period 8']+Test['Activity 4 Time Period 7']+Test['Activity 4 Time Period 6']+Test['Activity 4 Time Period 5']+Test['Activity 4 Time Period 4']+Test['Activity 4 Time Period 3']+Test['Activity 4 Time Period 2']+Test['Activity 4 Time Period 1']+Test['Activity 4 Time Period 0']



Test['Activity 5 end']=Test['Activity 5 Time Period  11']+Test['Activity 5 Time Period  10']+Test['Activity 5 Time Period  9']+Test['Activity 5 Time Period  8']

Test['Activity 5 Frequency']=Test['Activity 5 Time Period  11']+Test['Activity 5 Time Period  10']+Test['Activity 5 Time Period  9']+Test['Activity 5 Time Period  8']+Test['Activity 5 Time Period  7']+Test['Activity 5 Time Period  6']+Test['Activity 5 Time Period  5']+Test['Activity 5 Time Period  4']+Test['Activity 5 Time Period  3']+Test['Activity 5 Time Period  2']+Test['Activity 5 Time Period  1']+Test['Activity 5 Time Period  0']



Test['Activity 6 end']=Test['Activity 6 Time Period  11']+Test['Activity 6 Time Period  10']+Test['Activity 6 Time Period  9']+Test['Activity 6 Time Period  8']

Test['Activity 6 Frequency']=Test['Activity 6 Time Period  11']+Test['Activity 6 Time Period  10']+Test['Activity 6 Time Period  9']+Test['Activity 6 Time Period  8']+Test['Activity 6 Time Period  7']+Test['Activity 6 Time Period  6']+Test['Activity 6 Time Period  5']+Test['Activity 6 Time Period  4']+Test['Activity 6 Time Period  3']+Test['Activity 6 Time Period  2']+Test['Activity 6 Time Period  1']+Test['Activity 6 Time Period  0']



Test['Activity 7 end']=Test['Activity 7 Time Period  11']+Test['Activity 7 Time Period  10']+Test['Activity 7 Time Period  9']+Test['Activity 7 Time Period  8']

Test['Activity 7 Frequency']=Test['Activity 7 Time Period  11']+Test['Activity 7 Time Period  10']+Test['Activity 7 Time Period  9']+Test['Activity 7 Time Period  8']+Test['Activity 7 Time Period  7']+Test['Activity 7 Time Period  6']+Test['Activity 7 Time Period  5']+Test['Activity 7 Time Period  4']+Test['Activity 7 Time Period  3']+Test['Activity 7 Time Period  2']+Test['Activity 7 Time Period  1']+Test['Activity 7 Time Period  0']



Test['Activity 8 end']=Test['Activity 8 Time Period 11']+Test['Activity 8 Time Period 10']+Test['Activity 8 Time Period 9']+Test['Activity 8 Time Period 8']

Test['Activity 8 Frequency']=Test['Activity 8 Time Period 11']+Test['Activity 8 Time Period 10']+Test['Activity 8 Time Period 9']+Test['Activity 8 Time Period 8']+Test['Activity 8 Time Period 7']+Test['Activity 8 Time Period 6']+Test['Activity 8 Time Period 5']+Test['Activity 8 Time Period 4']+Test['Activity 8 Time Period 3']+Test['Activity 8 Time Period 2']+Test['Activity 8 Time Period 1']+Test['Activity 8 Time Period 0']



Test['Activity end']=Test['Activity 1 end']+Test['Activity 2 end']+Test['Activity 3 end']+Test['Activity 4 end']+Test['Activity 5 end']+Test['Activity 6 end']+Test['Activity 7 end']+Test['Activity 8 end']
Test['Time Period 0']=Test['Activity 1 Time Period 0']+Test['Activity 2 Time Period 0']+Test['Activity 3 Time Period 0']+Test['Activity 6 Time Period  0']+Test['Activity 8 Time Period 0']

Test['Time Period 1']=Test['Activity 1 Time Period 1']+Test['Activity 2 Time Period 1']+Test['Activity 3 Time Period 1']+Test['Activity 6 Time Period  1']+Test['Activity 8 Time Period 1']

Test['Time Period 2']=Test['Activity 1 Time Period 2']+Test['Activity 2 Time Period 2']+Test['Activity 3 Time Period 2']+Test['Activity 6 Time Period  2']+Test['Activity 8 Time Period 2']

Test['Time Period 3']=Test['Activity 1 Time Period 3']+Test['Activity 2 Time Period 3']+Test['Activity 3 Time Period 3']+Test['Activity 6 Time Period  3']+Test['Activity 8 Time Period 3']

Test['Time Period 4']=Test['Activity 1 Time Period 4']+Test['Activity 2 Time Period 4']+Test['Activity 3 Time Period 4']+Test['Activity 6 Time Period  4']+Test['Activity 8 Time Period 4']

Test['Time Period 5']=Test['Activity 1 Time Period 5']+Test['Activity 2 Time Period 5']+Test['Activity 3 Time Period 5']+Test['Activity 6 Time Period  5']+Test['Activity 8 Time Period 5']

Test['Time Period 6']=Test['Activity 1 Time Period 6']+Test['Activity 2 Time Period 6']+Test['Activity 3 Time Period 6']+Test['Activity 6 Time Period  6']+Test['Activity 8 Time Period 6']

Test['Time Period 7']=Test['Activity 1 Time Period 7']+Test['Activity 2 Time Period 7']+Test['Activity 3 Time Period 7']+Test['Activity 6 Time Period  7']+Test['Activity 8 Time Period 7']

Test['Time Period 8']=Test['Activity 1 Time Period 8']+Test['Activity 2 Time Period 8']+Test['Activity 3 Time Period 8']+Test['Activity 6 Time Period  8']+Test['Activity 8 Time Period 8']

Test['Time Period 9']=Test['Activity 1 Time Period 9']+Test['Activity 2 Time Period 9']+Test['Activity 3 Time Period 9']+Test['Activity 6 Time Period  9']+Test['Activity 8 Time Period 9']

Test['Time Period 10']=Test['Activity 1 Time Period 10']+Test['Activity 2 Time Period 10']+Test['Activity 3 Time Period 10']+Test['Activity 6 Time Period  10']+Test['Activity 8 Time Period 10']

Test['Time Period 11']=Test['Activity 1 Time Period 11']+Test['Activity 2 Time Period 11']+Test['Activity 3 Time Period 11']+Test['Activity 6 Time Period  11']+Test['Activity 8 Time Period 11']



list(Train.columns)
X=Train.drop(['Client ID','Client Retention Flag','Flag 2_1','Flag 2_9','Flag 3_C','Flag 6_Unknown','Flag 4_Unknown'],axis=1)

y=Train['Client Retention Flag']

from sklearn.model_selection import cross_val_score, train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import f1_score

model_gb=GradientBoostingClassifier(n_estimators=1000)

model_gb.fit(X_train,y_train)

pred_gb=model_gb.predict(X_test)

print(f1_score(y_test,pred_gb))
import matplotlib.pyplot as plt

features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = model_gb.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25,25))

plt.show()

from sklearn.metrics import confusion_matrix

cm1=confusion_matrix(pred_gb,y_test)

total1=sum(sum(cm1))

#####from confusion matrix calculate accuracy

accuracy1=(cm1[0,0]+cm1[1,1])/total1

print ('Accuracy : ', accuracy1)



sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])

print('Sensitivity : ', sensitivity1 )



specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

print('Specificity : ', specificity1)
XTest=Test.drop(['Client ID','Flag 2_8','Flag 2_11','Flag 2_12','Flag 3_C','Flag 6_Unknown','Flag 4_Unknown'],axis=1)

ClientId=pd.DataFrame(Test['Client ID'],columns=['Client ID'])
finalpredarray=model_gb.predict(XTest)

finalpred=pd.DataFrame(finalpredarray,columns=['Client Retention Flag'])

sub1=pd.concat([ClientId,finalpred],axis=1)

sub1.to_csv('submission_1.csv')