#Importing Library

import numpy as np

import pandas as pd

from sklearn.cross_validation import train_test_split
DataFrame=pd.read_csv('../input/creditcard.csv')
#Dropping Time 

newDframe = DataFrame.drop('Time',axis=1)

#Features shortlisted using Feature Importance of LDA

cols_of_intrest=['V10','V7','V11','V18','V3','V14','V9','V2','V5','Class']

finalDframe=newDframe[cols_of_intrest]
#Randome Spliting for train and test

X, Xt, Y, Yt = train_test_split(finalDframe.drop('Class', axis=1), finalDframe['Class'], test_size=0.20, random_state=10)
#Removing NA's if any

X = X.fillna(method='ffill')

Xt = Xt.fillna(method='ffill')

#Importing Decison

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(max_depth=5)

dtc.fit(X,Y)
#Tree generated using Soft Classification of Decision Tree 

def tree(V10, V7, V11, V18, V3, V14, V9, V2, V5):

    if V14 <= -5.87109088898:

        if V7 <= 0.384688615799:

            if V11 <= 7.6252784729:

                if V3 <= -26.7404098511:

#                     1.0 0.0 1.0 0.0

                    return 0

                else:  # if V3 > -26.7404098511

                    if V10 <= 4.49727106094:

#                         12.0 191.0 203.0 0.940886699507

                        return 1

                    else:  # if V10 > 4.49727106094

#                         1.0 0.0 1.0 0.0

                        return 0

            else:  # if V11 > 7.6252784729

                if V9 <= -3.40974831581:

#                     0.0 28.0 28.0 1.0

                    return 1

                else:  # if V9 > -3.40974831581

                    if V3 <= -6.86412239075:

#                         17.0 0.0 17.0 0.0

                        return 0

                    else:  # if V3 > -6.86412239075

#                         0.0 1.0 1.0 1.0

                        return 1

        else:  # if V7 > 0.384688615799

            if V10 <= -1.84730434418:

                if V18 <= 3.08181285858:

                    if V3 <= -26.3150920868:

#                         1.0 0.0 1.0 0.0

                        return 0

                    else:  # if V3 > -26.3150920868

#                         0.0 7.0 7.0 1.0

                        return 1

                else:  # if V18 > 3.08181285858

#                     6.0 0.0 6.0 0.0

                    return 0

            else:  # if V10 > -1.84730434418

                if V18 <= 3.78000307083:

#                     51.0 0.0 51.0 0.0

                    return 0

                else:  # if V18 > 3.78000307083

                    if V18 <= 3.79609036446:

#                         0.0 1.0 1.0 1.0

                        return 1

                    else:  # if V18 > 3.79609036446

#                         5.0 0.0 5.0 0.0

                        return 0

    else:  # if V14 > -5.87109088898

        if V18 <= -4.60471248627:

#             0.0 19.0 19.0 1.0

            return 1

        else:  # if V18 > -4.60471248627

            if V14 <= -4.28678417206:

                if V10 <= -1.85863828659:

                    if V18 <= 1.90742075443:

#                         4.0 54.0 58.0 0.931034482759

                        return 1

                    else:  # if V18 > 1.90742075443

#                         7.0 2.0 9.0 0.222222222222

                        return 1

                else:  # if V10 > -1.85863828659

                    if V7 <= -1.98475766182:

#                         0.0 2.0 2.0 1.0

                        return 1

                    else:  # if V7 > -1.98475766182

#                         200.0 3.0 203.0 0.0147783251232

                        return 0

            else:  # if V14 > -4.28678417206

                if V10 <= -3.36159563065:

                    if V14 <= -3.03404426575:

#                         7.0 13.0 20.0 0.65

                        return 1

                    else:  # if V14 > -3.03404426575

#                         196.0 2.0 198.0 0.010101010101

                        return 0

                else:  # if V10 > -3.36159563065

                    if V11 <= 4.52177429199:

#                         226939.0 74.0 227013.0 0.000325972521397

                        return 0

                    else:  # if V11 > 4.52177429199

#                         0.0 1.0 1.0 1.0

                        return 1

#Storing Index values of Test data into A list

B=[]

for i in Xt.index:

    B.append(i)
#Predicting for test data

predict=[]

for i in range(0,len(Xt)):

    predict.append(tree(Xt.V10[B[i]],Xt.V7[B[i]],Xt.V11[B[i]],Xt.V18[B[i]],Xt.V3[B[i]],Xt.V14[B[i]],Xt.V9[B[i]],Xt.V2[B[i]],Xt.V5[B[i]]))
from sklearn import metrics

Accuracy=metrics.accuracy_score(predict,Yt, normalize=True, sample_weight=None)

Accuracy
tn, fp, fn,tp = metrics.confusion_matrix(predict,Yt).ravel() 

Sensitivity=tp/float((tp+fn))#Sensitivity 

Sensitivity
Specificity=tn/float((tn+fp))#Specificity 

Specificity