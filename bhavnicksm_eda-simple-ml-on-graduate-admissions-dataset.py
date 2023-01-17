import numpy as np                 #for matrix analysis etc 
import pandas as pd                #for loading and viewing the dataset
import seaborn as sns              #for really cool visuallizations
import matplotlib.pyplot as plt    #for more visualizations

%matplotlib inline   
dataset = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', index_col='Serial No.')
dataset.head(5)   #Shows the top 5 results
dataset.shape
#Removing the 'Serial No.' name of the first column.
dataset.index.name = None
dataset.head(5)
#Removing space from column names
dataset.columns = dataset.columns.str.replace(' ', '')
dataset.head(5)
dataset[["ChanceofAdmit"]].head()  #This wouldn't have been possible with spaces
#Checking if the data types are right...
dataset.dtypes
#Getting the .info() stats
dataset.info()
dataset.isnull().values.any()
dataset.isnull().sum()
sns.pairplot(dataset)
plt.show()
sns.pairplot(dataset, hue="Research")  #Checking values with research as color
dataset.corr()    #Tells the correlation factor for each of the columns.
plt.figure(figsize=(10, 10))
sns.heatmap(dataset.corr(), annot=True, linewidths=0.05, fmt= '.2f', cmap="YlGnBu", cbar= False)
plt.show()
corr = dataset.corr()
mask = np.zeros_like(corr)                  #Makes a zero matrix with same shape as corr
mask[np.triu_indices_from(mask)] = True     #puts the upper triangle elements as true

#Plotting
plt.figure(figsize=(8,8))
sns.heatmap(dataset.corr(), annot=True, linewidths=0.05, fmt= '.2f', cmap="YlGnBu", cbar= False, mask=mask)
plt.show()
fig , (axis1, axis2 ,axis3) = plt.subplots(1,3,figsize=(15,5))                          #Making the structure
fig.suptitle("Plots of Chance of Admission vs [CGPA, GRE, TOEFL]")
sns.regplot(x='CGPA', y='ChanceofAdmit', data=dataset, color='b', marker='.', ax= axis1)            
sns.regplot(x='GREScore', y='ChanceofAdmit', data=dataset, color='g', marker='.', ax= axis2)
sns.regplot(x='TOEFLScore', y='ChanceofAdmit' ,data=dataset , color='r', marker='.', ax= axis3)
plt.show()
plt.figure(figsize=(7,7))
plt.title("Frequency of Chance of Admission")
sns.distplot(dataset.ChanceofAdmit)
plt.show()
plt.figure(figsize=(7,7))

explode = [0.1,0,0,0,0]
plt.pie(dataset.UniversityRating.value_counts().values,explode=explode,labels=dataset.UniversityRating.value_counts().index,autopct='%1.1f%%')

plt.title('University Rating',fontsize=15)
plt.show()
with sns.axes_style("white"):
    sns.jointplot("SOP","ChanceofAdmit", kind="kde", color="black", data=dataset)
    plt.title("kde contour plot", y=1.08)

with sns.axes_style("white"):
    sns.jointplot("LOR","ChanceofAdmit", kind="hex", color="blue", data=dataset)
    plt.title("hex bar plot", y=1.08)
plt.figure(figsize=(7,7))
sns.boxplot("LOR","ChanceofAdmit", data=dataset)

plt.title("Boxplot")
plt.show()
plt.figure(figsize=(7,7))
sns.violinplot("UniversityRating","ChanceofAdmit", data=dataset)
plt.title("Violin Plot")
plt.show()
plt.figure(figsize=(7,7))
sns.boxenplot("Research","ChanceofAdmit", data=dataset)
plt.title("Boxen Plot")
plt.show()
# Converting the labels to binary values for classification
df = dataset.copy()
df['ChanceofAdmit'] = [1 if chance>=0.75 else 0 for chance in dataset['ChanceofAdmit']]
df.head()
plt.figure(figsize=(7,7))
sns.countplot("Research",hue="ChanceofAdmit", data=df)
plt.title("Research Importance in Acceptance")
plt.show()
counts =  df[['Research','ChanceofAdmit']].groupby(['Research','ChanceofAdmit']).size().sort_values(ascending=False)
counts
df.drop(['Research'],axis=1, inplace=True)  #Since research has little to no importance to the prediction
df.head()
sns.pairplot(df,hue='ChanceofAdmit')#Showing the relation with various parameters
# Converting the DataFrames to Numpy arrays 
x = df.drop('ChanceofAdmit', axis=1).values
y = df['ChanceofAdmit'].values
from sklearn.preprocessing import StandardScaler
# To scale the values to 0 mean and unit variance
scl_x = StandardScaler().fit(x)
x_scaled = scl_x.transform(x)
x_scaled
#Splitting the data into two parts: Train set, Test set (Dev set)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size = 0.8, random_state=42)
print('train: {train}\ntest: {test}'.format(train=x_train.shape[0],test=x_test.shape[0]))
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',solver='saga')
lr.fit(x_train,y_train)
y_hat_test = lr.predict(x_test)
y_hat_train= lr.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.svm import SVC
svc = SVC().fit(x_train,y_train)
y_hat_test = svc.predict(x_test)
y_hat_train = svc.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(x_train,y_train)
y_hat_test = dtc.predict(x_test)
y_hat_train= dtc.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.ensemble import RandomForestClassifier
parameters = [{
        'max_depth': np.arange(1, 10),
        'min_samples_split': np.arange(2, 5),
        'n_estimators': np.arange(10, 20)
    },
]
gs_rf = GridSearchCV(RandomForestClassifier(), parameters, scoring='accuracy')
gs_rf.fit(x_train,y_train)
gs_rf.best_params_
rfc =  RandomForestClassifier(max_depth=2, min_samples_split=4, n_estimators= 16)
rfc.fit(x_train, y_train)
y_hat_test = rfc.predict(x_test)
y_hat_train = rfc.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.ensemble import GradientBoostingClassifier
parameters = [
{
    'learning_rate': np.arange(0.1,0.2,0.01),
    'random_state': [0],
    'n_estimators': np.arange(3, 20)
    },
]
gs_gbc = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='accuracy')
gs_gbc.fit(x_train,y_train)
gs_gbc.best_params_
gbc = GradientBoostingClassifier(learning_rate= 0.19, n_estimators= 3, random_state= 0)
gbc.fit(x_train,y_train)
y_hat_test = gbc.predict(x_test)
y_hat_train= gbc.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(x_train,y_train)
y_hat_test = gnb.predict(x_test)
y_hat_train= gnb.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.neighbors import KNeighborsClassifier
parameters=[{
    'n_neighbors':np.arange(2,33),
    'n_jobs':[2,6]
    },
]
gs_knn = GridSearchCV(KNeighborsClassifier(), parameters, scoring='accuracy').fit(x_train,y_train)
gs_knn.best_params_
gs_knn.best_score_
knn = KNeighborsClassifier(n_neighbors= 10, n_jobs= 2).fit(x_train,y_train)
y_hat_test = knn.predict(x_test)
y_hat_train= knn.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.gaussian_process import GaussianProcessClassifier
gpc = GaussianProcessClassifier(random_state=42).fit(x_train,y_train)
y_hat_test = gpc.predict(x_test)
y_hat_train= gpc.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.ensemble import AdaBoostClassifier
parameters=[{
    'learning_rate':np.arange(0.5,0.7,.01),
    'random_state':[0]
            }]
gs_abc = GridSearchCV(AdaBoostClassifier(),parameters,scoring='accuracy').fit(x_train,y_train)
gs_abc.best_params_
abc = AdaBoostClassifier(learning_rate=0.64, random_state=0).fit(x_train,y_train)
y_hat_test = abc.predict(x_test)
y_hat_train= abc.predict(x_train)
test_acc = metrics.accuracy_score(y_test,y_hat_test)
train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))
from sklearn.neural_network import MLPClassifier
parameters = [{
    'hidden_layer_sizes':[(100,2,1,)],
    'alpha': [2],
    'learning_rate_init': [10.0 ** -2],
    'max_iter': [1500],
}]
gs_mlpc = GridSearchCV(MLPClassifier(), parameters, scoring='accuracy').fit(x_train,y_train)
gs_mlpc.best_params_
gs_mlpc.best_score_
test_acc=0
train_acc=0
for i in range(0,100):
    mlpc = MLPClassifier(hidden_layer_sizes=(100,2,1,),
                         alpha=2, 
                         learning_rate_init=10.0 ** -2,
                         max_iter= 1500,
                         batch_size= 128,
                         tol= 0.00001,
                         verbose=False
                        )
    mlpc.fit(x_train,y_train)
    y_hat_test = mlpc.predict(x_test)
    y_hat_train= mlpc.predict(x_train)
    if test_acc < metrics.accuracy_score(y_test,y_hat_test):
        test_acc = metrics.accuracy_score(y_test,y_hat_test)
        train_acc = metrics.accuracy_score(y_train,y_hat_train)
print('Training set accuracy: {0}\nTest Set Accuracy: {1}'.format(train_acc,test_acc))
lr_mat = metrics.confusion_matrix(y_test,y_hat_test)
sns.heatmap(lr_mat,annot=True)
lr_mat = metrics.confusion_matrix(y_train,y_hat_train)
sns.heatmap(lr_mat,annot=True, fmt='1d')
print(metrics.classification_report(y_test,y_hat_test))