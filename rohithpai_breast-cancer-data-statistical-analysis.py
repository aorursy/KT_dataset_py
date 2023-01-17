# importing the modules for analysis

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/data.csv')
df.info()
df.head()
cancer_df =df.drop(['id','Unnamed: 32'], axis = 1)
cancer_df= pd.get_dummies(cancer_df,'diagnosis',drop_first=True) # dropping the column called diagnosis and having a columns of 0 and 1

cancer_df.head() 
sns.countplot(x='diagnosis',data = df,palette='BrBG')
colors = np.array('b g r c m y k'.split()) #Different colors for plotting



fig,axes = plt.subplots(nrows =15,ncols=2, sharey=True,figsize = (15,50))

plt.tight_layout()

row = 0

iteration = 0

for j in range(0,len(cancer_df.columns[:-1])):

    iteration+=1

    if(j%2==0):

        k = 0

    else:

        k = 1

    sns.distplot(cancer_df[cancer_df.columns[j]],kde=False,hist_kws=dict(edgecolor="w", linewidth=2),color = np.random.choice(colors) ,ax=axes[row][k])

    if(iteration%2==0):

        row+=1

        plt.ylim(0,200)
cancer_df.std()
plt.figure(figsize =(20,6))

sns.barplot(x='radius_mean',y='texture_mean',data =df, hue= 'diagnosis',palette='viridis')

plt.xlabel('Mean Radius of the lump')

plt.ylabel('Texture of the lump')
plt.figure(figsize =(20,6))

sns.barplot(x='perimeter_worst',y='area_worst',data =df, hue= 'diagnosis')

plt.figure(figsize= (10,10), dpi=100)

sns.heatmap(cancer_df.corr()) # plotting the correlation matrix of the dataset
from sklearn.model_selection import train_test_split #Importing module

X = cancer_df.drop('diagnosis_M',axis=1)

y = cancer_df['diagnosis_M']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3) # splitting the data for training and testing
from sklearn.linear_model import LogisticRegression #Logistical Regression Module

lm = LogisticRegression()

lm.fit(X_train,y_train)

prediction = lm.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print("Confusion Matrix")

print(confusion_matrix(y_test,prediction))

CM = confusion_matrix(y_test,prediction)

accuracy = (CM[0,0]+CM[1,1])/CM.sum()*100

error = (CM[0,1]+CM[1,0])/CM.sum()*100

print('Accuracy of the model: {0:.2f}%'.format(accuracy))

print('Error/ Misclassification rate: {0:.2f}%'.format(error))



print('\n\n')

print('Classification Report')

print(classification_report(y_test,prediction))
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scalar.fit(cancer_df.drop('diagnosis_M',axis=1))

scalar_features = scalar.transform(cancer_df.drop('diagnosis_M',axis=1))



# Converting these features into a Data frame

df_feat = pd.DataFrame(scalar_features,columns=cancer_df.columns[:-1])
from sklearn.neighbors import KNeighborsClassifier # importing KNN module

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_test,y_test)# fitting the test data to the model



# Predicting the outcome for the test data

prediction = knn.predict(X_test)
# Confusion matrix and Classsification report for K = 1

from sklearn.metrics import classification_report,confusion_matrix

print('Confusion matrix and Classsification report for K = 1')

print('\n')



print("Confusion Matrix")

print(confusion_matrix(y_test,prediction))

CM = confusion_matrix(y_test,prediction)

accuracy = (CM[0,0]+CM[1,1])/CM.sum()*100

error = (CM[0,1]+CM[1,0])/CM.sum()*100

print('Accuracy of the model: {0:.2f}%'.format(accuracy))

print('Error/ Misclassification rate: {0:.2f}%'.format(error))



print('\n\n')

print('Classification Report')

print(classification_report(y_test,prediction))
error_rate = []

for k in range(1,41):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_test,y_test)# fitting the test data to the model

    predi = knn.predict(X_test)

    error_rate.append(np.mean(predi!=y_test))



    
plt.figure(figsize = (10,10))

plt.plot(range(1,41),error_rate,ls = '--',color = 'blue',marker = 'o',markerfacecolor = 'red')

plt.xlabel("K- value")

plt.ylabel("Error Rate")

plt.xlim((0,40))

plt.title("Error Rate vs K- value")

from sklearn.model_selection import cross_val_score

CV_scores = cross_val_score(knn,X_test,y_test,cv =5)

CV_scores
print("Accuracy: %0.2f (+/- %0.2f)" % (CV_scores.mean(),CV_scores.std() * 2))
