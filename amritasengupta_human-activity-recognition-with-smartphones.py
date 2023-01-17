import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)
df= pd.read_csv("../input/human-activity-recognition-with-smartphones/test.csv")
df.head()
df.shape
df.info()
df= df.drop_duplicates()
df.shape
df.isnull().sum()[df.isnull().sum()>0]
X= df.drop(columns=['Activity'])
X=X.values
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

# fitting the data
pca_fit=pca.fit(X)

# calculating the principal components
reduced_X = pca_fit.transform(X)
#561 Columns present in X are now represented by 3-Principal components present in reduced_X
df2= pd.DataFrame(reduced_X, columns=['PC1','PC2','PC3'])
df2['activity']=df['Activity']
df2.head()
df2.hist(['PC1','PC2','PC3'],figsize=(20,5))
def bar_graph(data,predictor):
    grouped=data.groupby(predictor)
    chart=grouped.size().plot.bar(rot=0, title='Bar Chart showing the total frequency of different '+str(predictor), figsize=(15,4))
    chart.set_xlabel(predictor)
bar_graph(df2,'activity')
df2.activity.value_counts()
df2.boxplot(column=['PC1'], by='activity', figsize=(15,10),grid=False, layout=(2,1))
df2.boxplot(column=['PC2'], by='activity', figsize=(15,5),grid=False)
df2.boxplot(column=['PC3'], by='activity', figsize=(15,5),grid=False)
def anova_test(data,target,predictor):
    data1=data.groupby(target)[predictor].apply(list)
    from scipy.stats import f_oneway
    AnovaResults = f_oneway(*data1)
    if AnovaResults[1]<0.05:
        print(str(predictor)+' is related with the target variable : ', AnovaResults[1])
    else:
        print(str(predictor)+' is NOT related with the target variable : ', AnovaResults[1])
anova_test(df2,'activity','PC1')
anova_test(df2,'activity','PC2')
anova_test(df2,'activity','PC3')
df2.activity.unique()
activity_mapping = {'STANDING': 1,
                'SITTING': 2,
                'LAYING': 3,
              'WALKING': 4,
               'WALKING_DOWNSTAIRS': 5,
               'WALKING_UPSTAIRS':6
              }
# encoding the Ordinal variable cut
df['Activity'] = df['Activity'].map(activity_mapping)

# Checking the encoded columns
df['Activity'].unique()
df.head()
TargetVariable='Activity'
df2=df.drop(columns=['Activity','subject'])
predictor = df2.columns
x=df[predictor].values
y =df[TargetVariable].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
clf = LogisticRegression(C=1)

# Creating the model on Training Data
LOG=clf.fit(x_train,y_train)
prediction=LOG.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = KNeighborsClassifier(n_neighbors=3)

# Creating the model on Training Data
KNN=clf.fit(x_train,y_train)
prediction=KNN.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

clf = DecisionTreeClassifier(max_depth=3,criterion='entropy')

# Creating the model on Training Data
DTree=clf.fit(x_train,y_train)
prediction=DTree.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(DTree.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = RandomForestClassifier(max_depth=4, n_estimators=600,criterion='entropy')

# Creating the model on Training Data
RF=clf.fit(x_train,y_train)
prediction=RF.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(RF.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = SVC(C=100, gamma=0.001, kernel='rbf')

# Creating the model on Training Data
SVM=clf.fit(x_train,y_train)
prediction=SVM.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
from imblearn.over_sampling import SMOTE
smk=SMOTE(random_state=42)
x_smote,y_smote=smk.fit_sample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_smote))
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=42)
x_over,y_over= ros.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_over))
from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=42)
x_under,y_under= rus.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_under))
clf = LogisticRegression(C=1)

# Creating the model on Training Data
LOG=clf.fit(x_smote,y_smote)
prediction=LOG.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = LogisticRegression(C=1)

# Creating the model on Training Data
LOG=clf.fit(x_over,y_over)
prediction=LOG.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = LogisticRegression(C=1)

# Creating the model on Training Data
LOG=clf.fit(x_under,y_under)
prediction=LOG.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = KNeighborsClassifier(n_neighbors=3)

# Creating the model on Training Data
KNN=clf.fit(x_smote,y_smote)
prediction=KNN.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = KNeighborsClassifier(n_neighbors=3)

# Creating the model on Training Data
KNN=clf.fit(x_over,y_over)
prediction=KNN.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

clf = KNeighborsClassifier(n_neighbors=3)

# Creating the model on Training Data
KNN=clf.fit(x_under,y_under)
prediction=KNN.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

clf = DecisionTreeClassifier(max_depth=3,criterion='entropy')

# Creating the model on Training Data
DTree=clf.fit(x_smote,y_smote)
prediction=DTree.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(DTree.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = DecisionTreeClassifier(max_depth=3,criterion='entropy')

# Creating the model on Training Data
DTree=clf.fit(x_over,y_over)
prediction=DTree.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(DTree.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = DecisionTreeClassifier(max_depth=3,criterion='entropy')

# Creating the model on Training Data
DTree=clf.fit(x_under,y_under)
prediction=DTree.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(DTree.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = RandomForestClassifier(max_depth=4, n_estimators=600,criterion='entropy')

# Creating the model on Training Data
RF=clf.fit(x_smote,y_smote)
prediction=RF.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(RF.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = RandomForestClassifier(max_depth=4, n_estimators=600,criterion='entropy')

# Creating the model on Training Data
RF=clf.fit(x_over,y_over)
prediction=RF.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(RF.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = RandomForestClassifier(max_depth=4, n_estimators=600,criterion='entropy')

# Creating the model on Training Data
RF=clf.fit(x_under,y_under)
prediction=RF.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

# Plotting the feature importance for Top 10 most important columns
%matplotlib inline
feature_importances = pd.Series(RF.feature_importances_, index=predictor)
feature_importances.nlargest(10).plot(kind='barh')
clf = SVC(C=100, gamma=0.001, kernel='rbf')

# Creating the model on Training Data
SVM_smote=clf.fit(x_smote,y_smote)
prediction=SVM_smote.predict(x_test)

# Measuring accuracy on Testing Datam
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = SVC(C=100, gamma=0.001, kernel='rbf')

# Creating the model on Training Data
SVM_over=clf.fit(x_over,y_over)
prediction=SVM_over.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = SVC(C=100, gamma=0.001, kernel='rbf')

# Creating the model on Training Data
SVM_under =clf.fit(x_under,y_under)
prediction=SVM_under.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
accuracy_values= cross_val_score(SVM_under, x, y, cv=10, scoring='f1_weighted')
print(accuracy_values)
print('Final Average Accuracy of the Model:',accuracy_values.mean())
final_svm= SVM.fit(x,y)
test= pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv')
test.drop(columns=['subject'],inplace=True)
test=test.drop_duplicates()
df.isnull().sum()[df.isnull().sum()>0]
activity_mapping = {'STANDING': 1,
                'SITTING': 2,
                'LAYING': 3,
              'WALKING': 4,
               'WALKING_DOWNSTAIRS': 5,
               'WALKING_UPSTAIRS':6
              }
# encoding the Ordinal variable cut
test['Activity'] = test['Activity'].map(activity_mapping)

# Checking the encoded columns
test['Activity'].unique()
test.head()
TargetVariable='Activity'
test2= test.drop(columns=['Activity'])
predictor = test2.columns
x_test= test[predictor].values
y_test = test[TargetVariable].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x_test)

prediction= final_svm.predict(x)
test['Activity_Predictions']=prediction
test.head()
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)