# import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# import the data
df = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
df.head()
# drop the object id columns, they are of no use in the analysis
df.drop(['objid','specobjid'],axis=1, inplace=True)
df.head()
sns.countplot(x='class',data=df)
def change_category_to_number(classCat):
    if classCat =='STAR':
        return 1
    elif classCat =='GALAXY':
        return 2
    else:
        return 3
# assign a numerical value to the categorical field of class, by using the above function
df['classCat'] = df['class'].apply(change_category_to_number)
sns.pairplot(df[['u', 'g', 'r', 'i']])
plt.figure(figsize=(12,6))
plt.scatter(x=df['u'],y=df['i'],c=df['classCat'])
fig, axs = plt.subplots(2, 2)


axs[0,0].scatter(x=df['u'],y=df['i'],c=df['classCat'])
axs[0,1].scatter(x=df['u'],y=df['g'],c=df['classCat'])
axs[1,0].scatter(x=df['u'],y=df['r'],c=df['classCat'])
axs[1,1].scatter(x=df['u'],y=df['i'],c=df['classCat'])

plt.show()
# drop the not required columns
df.drop(['class','run','rerun','camcol','field'],axis=1,inplace=True)
df.head()
# import the train_test_split
from sklearn.model_selection import train_test_split
X = df.drop('classCat',axis=1)
y = df['classCat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtc =  DecisionTreeClassifier()
dtc.fit(X_train,y_train)
predictions = dtc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predictions))
conf_matrix = confusion_matrix(y_test,predictions)
confusion_matrix_df = pd.DataFrame(data=conf_matrix,columns=['STAR','GALAXY','QSO'],index=['STAR','GALAXY','QSO'])
confusion_matrix_df
