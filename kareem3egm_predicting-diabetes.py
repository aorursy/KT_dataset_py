import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
diabetes_data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')



#Print the first 5 rows of the dataframe.

diabetes_data.head()
diabetes_data.shape
diabetes_data.describe()

diabetes_data.info(verbose=True)
sns.countplot(x='Outcome',data=diabetes_data)

plt.show()
diabetes_data_copy = diabetes_data.copy(deep = True)

diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



## showing the count of Nans

print(diabetes_data_copy.isnull().sum())
plt.style.use('classic')

plot = diabetes_data.hist(figsize = (20,20))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)

diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)

diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)

diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)

diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] =diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



## showing the count of Nans

print(diabetes_data_copy.isnull().sum())
plot = diabetes_data_copy.hist(figsize = (20,20))



sns.pairplot(diabetes_data )
sns.pairplot(data=diabetes_data_copy,hue='Outcome',diag_kind='kde', kind="reg")

plt.show()
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

ax = sns.heatmap(diabetes_data.corr(), xticklabels=2, annot=True ,yticklabels=False)
from pandas_profiling import ProfileReport 



profile = ProfileReport(diabetes_data.corr(), title='Pandas profiling report ' , html={'style':{'full_width':True}})



profile.to_notebook_iframe()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()
y = diabetes_data_copy.Outcome
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
# Import Libraries

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

#----------------------------------------------------



#----------------------------------------------------

#Applying VotingClassifier Model 



'''

#ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)

'''



#loading models for Voting Classifier

LRModel_ = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=33)

RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=5, random_state=33)

KNNModel_ = KNeighborsClassifier(n_neighbors= 10, weights ='uniform', algorithm='auto')

NNModel_ = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(1000, 20),learning_rate='constant',activation='relu', power_t=0.4, max_iter=250)



#loading Voting Classifier

VotingClassifierModel = VotingClassifier(estimators=[('LRModel',LRModel_),('RFModel',RFModel_),('KNNModel',KNNModel_),('NNModel',NNModel_)], voting= 'soft')

VotingClassifierModel.fit(X_train, y_train)



#Calculating Details

print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))

print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))

print('----------------------------------------------------')

y_pred = VotingClassifierModel.predict(X_test)

print('Predicted Value for VotingClassifierModel is : ' , y_pred[:10])


#Calculating Confusion Matrix



from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="BuPu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#Import Libraries

from sklearn.metrics import accuracy_score

#----------------------------------------------------



#----------------------------------------------------

#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

AccScore = accuracy_score(y_test, y_pred, normalize=False)

print('Accuracy Score is : ', AccScore)
#Import Libraries

from sklearn.metrics import f1_score

#----------------------------------------------------



#----------------------------------------------------

#Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)

# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)



F1Score = f1_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('F1 Score is : ', F1Score)
from sklearn.metrics import roc_curve

y_pred_proba = VotingClassifierModel.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('VotingClassifierModel ROC curve')

plt.show()