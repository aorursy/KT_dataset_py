import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('../input/social/Social_Network_Ads.csv')

print('Dataset :',data.shape)

data.info()

data[0:10]
cnt_pro = data['Purchased'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Kelas', fontsize=12)

plt.xticks(rotation=90)

plt.show();
sns.set_style("whitegrid")

sns.pairplot(data,hue="Purchased",size=3);

plt.show()
data = data[['User ID','Gender','Age','EstimatedSalary','Purchased']] #Subsetting the data

cor = data.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
#Convert sting to numeric

Gender  = {'Male': 1,'Female': 0} 

  

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

data.Gender = [Gender[item] for item in data.Gender] 

print(data)
from sklearn.model_selection import train_test_split

Y = data['Purchased']

X = data.drop(columns=['Purchased'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
# We define the number of trees in the forest in 100. 



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)



# We train model

rfcla.fit(X_train, Y_train)



# We predict target values

Y_predict5 = rfcla.predict(X_test)
test_acc_rfcla = round(rfcla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_rfcla = round(rfcla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
# The confusion matrix

rfcla_cm = confusion_matrix(Y_test, Y_predict5)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(rfcla_cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")

plt.title('Random Forest Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
model1 = pd.DataFrame({

    'Model': ['Random Forest'],

    'Train Score': [train_acc_rfcla],

    'Test Score': [test_acc_rfcla]

})

model1.sort_values(by='Test Score', ascending=False)
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict5)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(rfcla,X_train, Y_train)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))