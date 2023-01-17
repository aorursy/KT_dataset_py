import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier  
import operator
import random
import matplotlib.pyplot as plt

random.seed(3) # To get the same train-test datasets
input_file = '../input/Absenteeism_at_work.csv'
df = pd.read_csv(input_file)
print(df.head())
print((np.unique(np.array(df.iloc[:,14]))))
print(df.isna().sum())

l = []
l.append(df.index[df['Absenteeism time in hours'] == 7].tolist()[0])
l.append(df.index[df['Absenteeism time in hours'] == 48].tolist()[0])
l.append(df.index[df['Absenteeism time in hours'] == 104].tolist()[0])
df_temp = df.iloc[df.index[l]]
df.drop(df.index[l], inplace = True)
df_train, df_test = train_test_split(df.iloc[:, 0:15],  stratify = df.iloc[:,14]) # 25% test and 75% train
df_train = df_train.append(df_temp)
cat_indices = [1, 2, 3, 4, 11, 12]
for i in cat_indices:
    labelEncoder = LabelEncoder()
    labelEncoder.fit(np.unique(np.array(df.iloc[:,i])))
    df_train.iloc[:,i] = labelEncoder.transform(df_train.iloc[:,i])
    df_test.iloc[:,i] = labelEncoder.transform(df_test.iloc[:,i])
    
#     Alternative:
#     labelEncoder = LabelEncoder()
#     df_train.iloc[:,i] = labelEncoder.fit_transform(df_train.iloc[:,i])
#     df_test.iloc[:,i] = labelEncoder.fit_transform(df_test.iloc[:,i])
#     oneHotEncoder = OneHotEncoder()
#     df_train.iloc[:,i] = oneHotEncoder.fit_transform(df_train.iloc[:,i].values.reshape(-1,1)).toarray()
#     df_test.iloc[:,i] = oneHotEncoder.fit_transform(df_test.iloc[:,i].values.reshape(-1,1)).toarray()

X_train = np.array(df_train.drop(['Absenteeism time in hours'], 1).astype(float))
y_train = np.array(df_train['Absenteeism time in hours'])

X_test = np.array(df_test.drop(['Absenteeism time in hours'], 1).astype(float))
y_test = np.array(df_test['Absenteeism time in hours'])




scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

scaler = MinMaxScaler()
X_test_scaled = scaler.fit_transform(X_test)
k_list = [i for i in range(2,50)]                 #Trying for values of k from 2 to 49
train_acc = {}
test_acc = {}
sse = {}


for no_clusters in k_list:
    kmeans = KMeans(n_clusters=no_clusters, random_state = 3)       #Using sklearn function for k-Means clustering, calling k-Means class
    kmeans.fit(X_train_scaled)                    #Fitting model with the scaled given data
    
    l = []
    for i in range(no_clusters):                  #Finding the number of occurences of each class for each cluster
        temp = {}
        for k in np.unique(np.array(df_train.iloc[:,14])):
            temp[k] = 0
        for j in range(len(kmeans.labels_)):
            if(kmeans.labels_[j] == i):
               temp[df_train.iloc[j, 14]] += 1
        l.append(temp)
    
    op = []
    for i in range(no_clusters):                  #Choosing max no. of occurences of the class as the label for the cluster
        op.append(max(l[i].items(), key=operator.itemgetter(1))[0])
        
    correct = 0
    for i in range(len(X_train_scaled)):          #Training Data Accuracy
        indiv_record = np.array(X_train_scaled[i].astype(float))
        indiv_record = indiv_record.reshape(-1, len(indiv_record))
        prediction = kmeans.predict(indiv_record)
        prediction = op[prediction[0]]
        if prediction == y_train[i]:
            correct += 1
    train_acc[no_clusters] = (correct/len(X_train_scaled))
    
    
    correct = 0
    sse[no_clusters] = 0
    for i in range(len(X_test_scaled)):                       #Testing Data Accuracy
        indiv_record = np.array(X_test_scaled[i].astype(float))
        indiv_record = indiv_record.reshape(-1, len(indiv_record))
        prediction = kmeans.predict(indiv_record)
        prediction = op[prediction[0]]
        if prediction == y_test[i]:
            correct += 1
        sse[no_clusters] += (prediction - y_test[i])**2          #SSE Calc
    test_acc[no_clusters] = (correct/len(X_test_scaled))
    

lists = sorted(train_acc.items()) 
x1, y1 = zip(*lists)
h1, = plt.plot(x1,y1,label='Training')

lists = sorted(test_acc.items()) 
x2, y2 = zip(*lists)
h2, = plt.plot(x2,y2,label='Testing')
plt.title("Train and Test Accuracies for Varying K")
plt.xlabel("K")
plt.ylabel("Accuracy Percentage")
plt.legend(handles = [h1, h2])

lists = sorted(sse.items()) 
x, y = zip(*lists)
plt.plot(np.array(x),np.array(y))
plt.title("SSE for Varying K")
plt.xlabel("K")
plt.ylabel("SSE Error")


plt.plot(np.array(x2), np.array(y2) - np.array(y1))
plt.title("Difference in Predicted values and ground truth for Varying K")
plt.xlabel("K")
plt.ylabel("Difference")
#Hence we use k = 24, minimum value for SSE, and the train and test accuracies are amongst the highest
no_clusters = 24
print("Train Accuracy: ", train_acc[no_clusters])
print("Test Accuracy: ", test_acc[no_clusters])
kmeans = KMeans(n_clusters=no_clusters, random_state = 3)       #Using sklearn function for k-Means clustering, calling k-Means class
kmeans.fit(X_train_scaled)                    #Fitting model with the scaled given data

l = []
results = []
for i in range(no_clusters):                  #Finding the number of occurences of each class for each cluster
    temp = {}
    for k in np.unique(np.array(df_train.iloc[:,14])):
        temp[k] = 0
    for j in range(len(kmeans.labels_)):
        if(kmeans.labels_[j] == i):
           temp[df_train.iloc[j, 14]] += 1
    l.append(temp)

op = []
for i in range(no_clusters):                  #Choosing max no. of occurences of the class as the label for the cluster
    op.append(max(l[i].items(), key=operator.itemgetter(1))[0])


for i in range(len(X_test_scaled)):                       #Testing Data Accuracy
    indiv_record = np.array(X_test_scaled[i].astype(float))
    indiv_record = indiv_record.reshape(-1, len(indiv_record))
    prediction = kmeans.predict(indiv_record)
    results.append(op[prediction[0]])
    

print(confusion_matrix(y_test, results))
print(classification_report(y_test, results))
mean_error = []
test_acc = []

k = [x for x in range(1,41)]

for num_neighbors in k:     # Evaluating metrics for different values of k from 1 to 39
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    mean_error.append(np.mean(pred_i != y_test))
    test_acc.append(np.mean(pred_i == y_test))
# Plotting the graph for mean-error for varying k
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 41), mean_error, color='orange', linestyle='dashed', marker='o',  
         markerfacecolor='red', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
# Plotting the graph for Testing accuracy for varying k
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 41), test_acc, color='green', linestyle='dashed', marker='o',  
         markerfacecolor='yellow', markersize=10)
plt.title('Testing accuarcy  K Value')  
plt.xlabel('K Value')  
plt.ylabel('Testing accuracy')
# Finding the value of k with the lowest mean error or highest test accuracy from 1 to 40
num_neighbors = np.argmin(mean_error)
print("The value of K the gives the lowest error is: ", num_neighbors+1)
print("The value of error is: ", mean_error[num_neighbors])
print("The value of testing accuracy is: ", test_acc[num_neighbors])
# Classification report and confusion matrix for k
knn = KNeighborsClassifier(n_neighbors=num_neighbors+1)
knn.fit(X_train_scaled, y_train)
predict = knn.predict(X_test_scaled)

print("The confusion matrix for the optimum value of k")
print(confusion_matrix(y_test, predict))  
print("The classification metrics when using the optimum value of k found")
print(classification_report(y_test, predict))  

