import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df= pd.read_csv("../input/HR_comma_sep.csv")

print(df.columns)



X_promotion_left=df.iloc[:, [4,6]].values

ps = pd.Series([tuple(i) for i in X_promotion_left])

counts = ps.value_counts()

print(len(counts))

list1_X=[]

list1_Y=[]

list2_X=[]

list2_Y=[]

for key, value in counts.items():

    if key[1]==0:

        

        list1_X.append(key[0])

        list1_Y.append(value)

        

    else:

        

        list2_X.append(key[0])

        list2_Y.append(value)

        



fig=plt.figure()



ax1=fig.add_subplot(211)

ax1.bar(list2_X, list2_Y)

ax1.set_title("Those Who Left\n")

#plt.subplots_adjust(hspace = .001)

ax2=fig.add_subplot(212)

ax2.bar(list1_X,list1_Y)

ax2.set_title("\n\n Those Who Stayed")

plt.show()

print(" Average Years Spent For the Ones Who Left=",np.mean(list2_X),"  Average Years Spent For the Ones Who Stayed=" ,np.mean(list1_X))



#So Far I did the analysis by a trial and error method. Let us try to get the feature importances. 

#I will use the ensemble of random forests to find the feature importance.



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



df1=df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years','sales', 'salary'] ]

df1=pd.get_dummies(df1[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years','sales', 'salary']])

main_labels=df1.columns[1:]



X=df1.iloc[1:,].values

y=df.iloc[1:,6].values

X_train, X_test, y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=1)







forest=RandomForestClassifier(n_estimators=100,random_state=0)



forest.fit(X_train,y_train)

y_predict=forest.predict(X_test)

print("Accuracy with random forest=", accuracy_score(y_predict,y_test))

importances=forest.feature_importances_

for i in range(len(importances)):

    if importances[i]>0.1:

          print(main_labels[i])

          print(importances[i])