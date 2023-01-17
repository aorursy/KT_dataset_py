from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # Import train_test_split from sklearn library


from sklearn import model_selection #models selection importing
from sklearn import metrics #selection metrics
from sklearn.metrics import confusion_matrix #confusion matrix
from sklearn.linear_model import LogisticRegression #Logistic regression
from sklearn.model_selection import cross_val_score #cross validarion import
from sklearn.metrics import classification_report #classification report import
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score #import the ROC curve 
from sklearn.metrics import roc_curve
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file
# Churn_Modelling.csv has 10000 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/Churn_Modelling.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Churn_Modelling.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
print(os.listdir('../input'))
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
df1.head()
#As we are predicting the exit status of the customer we need to see how some important features are collerated with the predictive feature
#plt.bar(df1.Balance,df1.Exited)
#checking the correlation between age and balance!
np.corrcoef(df1.Balance,df1.Age)
#copying the original dataset
df2=df1.copy()
df2['Zero_Balance']=np.where(df2['Balance']==0, 1, 0)
gender1=np.array(df2.groupby(['Gender'])['HasCrCard','IsActiveMember','Exited','Zero_Balance'].mean().reset_index())
Geography1=np.array(df2.groupby(['Geography'])['HasCrCard','IsActiveMember','Exited','Zero_Balance'].mean().reset_index())
#This method plot the the statistics of given entry
def plotData(data,row,label,subplt,fig):
    ax = fig.add_subplot(subplt)
    N = 4
    gen = data[row,1:]
    ind = np.array(['HasCard','IsActive','Exited','Zero_Balance'])    # the x locations for the groups
    p2 = ax.bar(ind, gen, 0.6, color=(0.2588,0.4433,1.0))
    p1 = ax.bar(ind, 1-gen, 0.6,color=(1.0,0.5,0) ,bottom=gen)
    plt.ylabel("Level")
    plt.title("%s statistics" %label)
#plotting the gender statistics
fig = plt.figure(figsize=(10,6))
plotData(gender1,0,"Female",121,fig)
plotData(gender1,1,"Male",122,fig)
#plotting the Geography statistics
fig2 = plt.figure(figsize=(15,6))
plotData(Geography1,0,"France",131,fig2)
plotData(Geography1,1,"Germany",132,fig2)
plotData(Geography1,2,"Spain",133,fig2)
df=df1.copy()
df.columns
gend=pd.get_dummies(df.Gender)
geo=pd.get_dummies(df.Geography)
#combining the object so we can concatenate with the bigger dataframe
obj=[df,gend,geo]
df=pd.concat(obj, axis=1)
#creating the dataset we will be using 
#this contain only helpfull feature by ignoring the Gender and geography columns
df3 = df[['CreditScore','Female','Male','France','Germany','Spain','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember','Balance','EstimatedSalary','Exited']]
X=df3[['CreditScore','Female','Male','France','Germany','Spain','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Balance']]
Y=df3[['Exited']]
# random_state below is a metric that is used by the function to shuffle datas while splitting. This is chosen randomly.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 2) # 0.2 test_size means 20% for testing

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
kfold = model_selection.KFold(n_splits=5, random_state=2)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train.values.ravel(), cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
#getting the Logistic Regression model
logreg = LogisticRegression()
#fitting the data
logreg.fit(X_train, y_train.values.ravel())
#predicting 
y_pred = logreg.predict(X_test)
#printing the accuracy
print('Accuracy of logistic regression classifier on test set:'+str(format(logreg.score(X_test, y_test))))
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix
print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
#printing the confusion matrix graphic
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test, y_pred))
#USING THE TREE CLASSIFIER
clf = ExtraTreesClassifier()
#fitting the features 
clf = clf.fit(X_train, y_train.values.ravel())
#
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
#extracting the selected features
feature_idx=model.get_support()
feature_name=X.columns[feature_idx]
#Getting the new features extracted 
X_test_new=X_test.loc[:,feature_idx]
X_train_new=X_train.loc[:,feature_idx]

#Those are more important features then I will reuse them again in the model
feature_name
logre = LogisticRegression()
logre.fit(X_train_new ,y_train.values.ravel())

y_pred = logre.predict(X_test_new)
print('Accuracy of logistic regression classifier on test set:'+str(format(logre.score(X_test_new, y_test))))
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix
print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier IMPROVED')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
y= (conf_matrix[0,0]+conf_matrix[1,1])/(conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[1,0]+conf_matrix[1,1])
print('Accurracy= '+str(y))
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC 
model = SVC(probability=True)  
# Fitting the model
model = model.fit(X_train_new, y_train)  
# Predictions/probs on the test dataset
predicted = pd.DataFrame(model.predict(X_test_new))  
# Store metrics
accuracy = metrics.accuracy_score(y_test, predicted)
print("Support vector machine acuuracy is ",accuracy)

# The  number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
#fitting the classifier
rfecv = rfecv.fit(X_train_new, y_train.values.ravel())

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train_new.columns[rfecv.support_])

# Plot number of features VS. cross-validation scores

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(X_train_new,y_train.values.ravel())
#printing the score 
ypred_new=clf_rf.predict(X_test_new)
ac = accuracy_score(y_test,ypred_new)
print('Accuracy is: ',ac)