import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 


%matplotlib inline
data = pd.read_csv("../input/creditcard.csv")
data.head()
number_of_fraud = len(data[data.Class == 1])
number_of_normal= len(data[data.Class == 0])

print ("Fraud:", number_of_fraud)
print ("Normal:",number_of_normal)
sns.countplot("Class",data=data)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
bins = 50

ax1.hist(data.Time[data.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(data.Time[data.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time')
plt.ylabel('Number of Transactions')
plt.show()
print ("Fraud")
print (data.Amount[data.Class == 1].describe())
print ()
print ("Normal")
print (data.Amount[data.Class == 0].describe())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
bins = 10

ax1.hist(data.Amount[data.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(data.Amount[data.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount')
plt.ylabel('Number of Transactions')
plt.show()
PCA_features = data.iloc[:,1:29].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(data[PCA_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class == 1], bins=50)
    sns.distplot(data[cn][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()
data = data.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
data.head()
data['Normalized_Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data.head()
data = data.drop(['Time','Amount'],axis=1)
data.head()
#indices of normal class
indices_of_normal = data[data.Class==0].index
#randomly choose same amount of samples as the fraud, and return their indices
random_indices_of_normal = np.array(np.random.choice(indices_of_normal, number_of_fraud, replace=False))
#indices of fraud class
indices_of_fraud = np.array(data[data.Class == 1].index)
#indices of undersampled dataset
indices_of_undersampled = np.concatenate([random_indices_of_normal, indices_of_fraud])
#undersampled dataset)
data_of_undersampled = data.iloc[indices_of_undersampled,:]

print(len(data_of_undersampled))
#whole dataset
X = data.loc[:,data.columns!='Class']
y = data.loc[:,data.columns=='Class']

#train and test dataset splitted from whole dataset, with 70/30 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))
#undersampled dataset
X_undersampled = data_of_undersampled.loc[:,data_of_undersampled.columns!='Class']
y_undersampled = data_of_undersampled.loc[:,data_of_undersampled.columns=='Class']

#train and test dataset splitted from undersampled dataset, with 70/30 ratio
X_train_undersampled, X_test_undersampled, y_train_undersampled, y_test_undersampled = train_test_split(X_undersampled,y_undersampled,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train_undersampled))
print("Number transactions test dataset: ", len(X_test_undersampled))
print("Total number of transactions: ", len(X_train_undersampled)+len(X_test_undersampled))
def train(model,X,y):
    
    # Call the model
    clf = model
    
    # Different C parameters for regularization
    C_param = [0.01,0.1,1,10,100]

    # K-Fold Cross-validation
    kf = KFold(n_splits=5)
    
    # Initialization
    scores     =[]
    best_score = 0
    best_C     = 0
    
    for C in C_param:
        
        clf.C = C

        score = []
        for train_index, test_index in kf.split(X): 

            # Use the splitted training data to fit the model. 
            clf.fit(X.iloc[train_index,:].values,y.iloc[train_index,:].values.ravel())

            # Predict values using the splitted test data
            y_pred = clf.predict(X.iloc[test_index,:].values)
            
            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            rec = recall_score(y.iloc[test_index,:].values.ravel(),y_pred)
            
            # Append recall score of each iteration to score
            score.append(rec)

        # Calculate average reall score for all iterations and compare it with the best score.
        mean_score = np.mean(score)
        if mean_score > best_score:
            best_score = mean_score
            best_C     = C
        
        # Append mean_score for each C to scores
        scores.append(np.mean(score))
        
    # Create a DataFrame to show the mean_score for each C parameter    
    lr_results = pd.DataFrame({'score':scores, 'C':C_param}) 
    print(lr_results)
    
    print("Best recall score is: ", best_score)
    print("Best C parameter is: ", best_C)
    
    return best_score, best_C
def predict(model, X_train, y_train, X_test, y_test):
    # Call the model
    clf = model
    #clf = LogisticRegression(C=C, penalty = 'l1')
    # Use the whole undersampled train dataset to fit the model. 
    clf.fit(X_train.values,y_train.values.ravel())
    # Predict on undersampled test dataset
    y_pred = clf.predict(X_test.values)

    # Confusion matrix
    CM = confusion_matrix(y_test.values, y_pred)
    # Get true positives(tp), false positives(fp), false negatives(fn)
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

    # Prediction report
    sns.heatmap(CM,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(y_test.values, y_pred))
clf = LogisticRegression(penalty = 'l2', solver ='lbfgs')
best_score, best_C = train(clf, X_train_undersampled,y_train_undersampled)
clf = LogisticRegression(C=best_C, penalty = 'l2', solver ='lbfgs')
predict(clf, X_train_undersampled,y_train_undersampled,X_test_undersampled,y_test_undersampled)
predict(clf,X_train_undersampled,y_train_undersampled,X_test,y_test)
clf = LogisticRegression(penalty = 'l2',solver ='lbfgs')
best_score_whole, best_C_whole = train(clf,X_train,y_train)
clf = LogisticRegression(C=best_C_whole,penalty = 'l2',solver='lbfgs')
predict(clf,X_train,y_train,X_test,y_test)
clf = SVC(gamma='auto')
best_score, best_C = train(clf, X_train_undersampled,y_train_undersampled)
clf = SVC(C=best_C,gamma='auto')
predict(clf, X_train_undersampled,y_train_undersampled,X_test_undersampled,y_test_undersampled)
predict(clf,X_train_undersampled,y_train_undersampled,X_test,y_test)
predict(clf,X_train,y_train,X_test,y_test)