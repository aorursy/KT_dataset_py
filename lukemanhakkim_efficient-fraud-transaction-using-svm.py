import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

#from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

import itertools
data = pd.read_csv('../input/creditcardfraud/creditcard.csv') 

df = pd.DataFrame(data) 

df.describe() 
df_fraud = df[df['Class'] == 1] # Recovery of fraud data

plt.figure(figsize=(15,10))

plt.scatter(df_fraud['Time'], df_fraud['Amount'], color='red') # Display fraud amounts according to their time

plt.title('Scatter plot')

plt.xlabel('Time')

plt.ylabel('Amount')

plt.xlim([0,175000])

plt.ylim([0,2500])

plt.show()



df_fraud = df[df['Class'] == 0] # Recovery of non-fraud data

plt.figure(figsize=(15,10))

plt.scatter(df_fraud['Time'], df_fraud['Amount'], color='red') # Display fraud amounts according to their time

plt.title('Scatter plot')

plt.xlabel('Time')

plt.ylabel('Amount')

plt.xlim([0,180000])

plt.ylim([0,2500])

plt.show()
count_Class=pd.value_counts(data["Class"], sort= True)

count_Class.plot(kind= 'bar', color = 'red')

print(count_Class)


No_of_frauds= len(data[data["Class"]==1])

No_of_normals = len(data[data["Class"]==0])

fraud_index= np.array(data[data["Class"]==1].index)

normal_index= data[data["Class"]==0].index

random_normal_indices= np.random.choice(normal_index, No_of_frauds, replace= False)

#print(random_normal_indices)

random_normal_indices= np.array(random_normal_indices)

#print(random_normal_indices)

undersampled_indices= np.concatenate([fraud_index, random_normal_indices])

undersampled_data= data.iloc[undersampled_indices, :]

#print(undersampled_data.head())
No_of_frauds_sampled= len(undersampled_data[undersampled_data["Class"]== 1])

No_of_normals_sampled = len(undersampled_data[undersampled_data["Class"]== 0])

print("fraudulent transactions( Class 1) : ", No_of_frauds_sampled)

print("normal transactions( Class 0) : ", No_of_normals_sampled)

total_sampled= No_of_frauds_sampled + No_of_normals_sampled

print("The total number of rows: ", total_sampled)

Fraud_percent_sampled= (No_of_frauds_sampled / total_sampled)*100

Normal_percent_sampled= (No_of_normals_sampled / total_sampled)*100

print("Class 0 = ", Normal_percent_sampled)

print("Class 1 = ", Fraud_percent_sampled)

#Check the data count now

count_sampled=pd.value_counts(undersampled_data["Class"], sort= True)

count_sampled.plot(kind= 'bar', color = "red")
#We have to scale the Amount feature before fitting our model to our dataset



sc= StandardScaler()

undersampled_data["scaled_Amount"]=  sc.fit_transform(undersampled_data.iloc[:,29].values.reshape(-1,1))



#dropping time and old amount column

undersampled_data= undersampled_data.drop(["Time","Amount"], axis= 1)



print(undersampled_data.head())
X= undersampled_data.iloc[:, undersampled_data.columns != "Class"].values



y= undersampled_data.iloc[:, undersampled_data.columns == "Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

print("The split of the under_sampled data is as follows")

print("X_train: ", len(X_train))

print("X_test: ", len(X_test))

print("y_train: ", len(y_train))

print("y_test: ", len(y_test))
#Using the gaussian kernel to build the initail model. Let us see if this is the best parameter later

classifier= SVC(C= 1, kernel= 'rbf', random_state= 0, probability=True)

classifier.fit(X_train, y_train.ravel())
#Predict the class using X_test

y_pred = classifier.predict(X_test)
class_names=np.array(['0','1']) # Binary label, Class = 1 (fraud) and Class = 0 (no fraud)

# Function to plot the confusion Matrix

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = 'd' 

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    print("The accuracy is "+str((cm[1,1]+cm[0,0])/(cm[0,0] + cm[0,1]+cm[1,0] + cm[1,1])*100) + " %")

    print("The recall from the confusion matrix is "+ str(cm[1,1]/(cm[1,0] + cm[1,1])*100) +" %")



#cm1 is the confusion matrix 1 which uses the undersampled dataset

cm1 = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm1, class_names)
accuracies = cross_val_score(estimator = classifier, X=X_train, y = y_train.ravel(), cv = 10)

mean_accuracy= accuracies.mean()*100

std_accuracy= accuracies.std()*100

print("The mean accuracy in %: ", accuracies.mean()*100)

print("The standard deviation in % ", accuracies.std()*100)

print("The accuracy of our model in % is betweeen {} and {}".format(mean_accuracy-std_accuracy, mean_accuracy+std_accuracy))
#creating a new dataset to test our model

datanew= data.copy()



#Now to test the model with the whole dataset

datanew["scaled_Amount"]=  sc.fit_transform(datanew["Amount"].values.reshape(-1,1))



#dropping time and old amount column

datanew= datanew.drop(["Time","Amount"], axis= 1)



#separating the x and y variables to fit our model

X_full= datanew.iloc[:, undersampled_data.columns != "Class"].values



y_full= datanew.iloc[:, undersampled_data.columns == "Class"].values

#splitting the full dataset into training and test set

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size= 0.25, random_state= 0)



print("The split of the full dataset is as follows")

print("X_train_full: ", len(X_train_full))

print("X_test_full: ", len(X_test_full))

print("y_train_full: ", len(y_train_full))

print("y_test_full: ", len(y_test_full))

#predicting y_pred_full_dataset

y_pred_full_dataset= classifier.predict(X_test_full)



#confusion matrix usign y_test_full and ypred_full

cm3 = confusion_matrix(y_test_full, y_pred_full_dataset)
plot_confusion_matrix(cm3, class_names)
from sklearn.metrics import classification_report



print(classification_report(y_test_full, y_pred_full_dataset))
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

%matplotlib inline



y_pred_prob = classifier.predict_proba(X_test_full)[:,1]



fpr, tpr, thresholds = roc_curve(y_test_full, y_pred_prob)



# create plot

plt.plot(fpr, tpr, label='ROC curve')

plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

_ = plt.xlabel('False Positive Rate')

_ = plt.ylabel('True Positive Rate')

_ = plt.title('ROC Curve')

_ = plt.xlim([-0.02, 1])

_ = plt.ylim([0, 1.02])

_ = plt.legend(loc="lower right")



# save figure

plt.savefig('roc_curve.png', dpi=200)


from sklearn.metrics import roc_auc_score



roc_auc_score(y_test_full, y_pred_prob)