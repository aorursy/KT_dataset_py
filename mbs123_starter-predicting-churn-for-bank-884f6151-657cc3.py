import os #accessing directory structure
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #plotting
import matplotlib.pyplot as plt # plotting
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn import metrics #metrics selection
from sklearn.metrics import confusion_matrix #confusion matrix
from sklearn.metrics import classification_report
from sklearn import model_selection #model selection
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.tree import DecisionTreeClassifier #Decision Trees
from sklearn.ensemble import RandomForestClassifier #random forests
from sklearn.model_selection import cross_val_score #Cross validarion import
from sklearn.metrics import accuracy_score #Accuracy of models
from sklearn.model_selection import train_test_split #Import train_test_split from sklearn library

print(os.listdir('../input'))
#Importing data
churn_data = pd.read_csv('../input/Churn_Modelling.csv', delimiter=',')

#Giving the dataframe a name
churn_data.dataframeName = 'Churn_Modelling.csv'

#Computing the number of rows and columns
nRow, nCol = churn_data.shape

#Displays the number of rows(10 000) and columns (14)
print(f'The dataset has {nRow} rows and {nCol} columns')
churn_data.head()
churn_data.info()
churn_data.describe()
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

slices=  [churn_data.Exited[churn_data.Exited ==1].count(), 
          churn_data.Exited[churn_data.Exited == 0].count()]
slice_labels = ['Churned', 'Stayed']
colours = ['b', 'g']
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(slices, labels = slice_labels, colors = colours, startangle= 90, autopct='%.1f%%')
plt.title("The number of churned cutomers and  the ones that stayed", size = 20)
plt.show()
plotPerColumnDistribution(churn_data, 10, 5)
# using seaborn library for visualization
plot, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

# extracting features with unique values that are between 2 and 49
nunique = churn_data.nunique()

# array of categorical features
cat_features = [col for col in churn_data.columns if nunique[col] >= 1 and nunique[col] < 50]

# array of non-categorical non_categorical
num_features = [col for col in churn_data.columns if nunique[col] > 50]

# looping through the array of categorical features and plotting their counts with the target variable
for axis, catplot in zip(axes, churn_data.dtypes[cat_features].index):
    sns.countplot(x=catplot, hue = 'Exited', data=churn_data, ax=axis)   
plt.tight_layout()  
plt.show()  
plotCorrelationMatrix(churn_data, 8)
# Dropping RowNumber CustomerId Surname   
churn_data1 = churn_data.drop(["RowNumber", "CustomerId","Surname"], axis=1)

# Creating dummy variables from categorical variables
churn_data2 = pd.get_dummies(churn_data1, columns=['Gender','Geography'])

#New dataset after creating dummy variables
churn_data2.head()
#Renaming dummy variables created from Gender ang Geography columns
churn_data2.rename(columns={"Gender_Female": "Female", "Gender_Female": "Male","Geography_France":"France",
                                      "Geography_Spain": "Spain","Geography_Germany":"Germany"}, inplace=True)
churn_data2.head()
churn_data2.Exited.value_counts()
# upsampling the minority class so that it has the same number of records as the mijority
churn_balanced = resample(churn_data2[churn_data2.Exited ==1],replace=True,n_samples= 7963,random_state=1) #set the seed for random resampling
churn_balanced = pd.concat([churn_data2[churn_data2.Exited ==0], churn_balanced])
churn_balanced.Exited.value_counts()
# Explanatory variable from unbalanced dataset
# Explanatory variable
X = churn_data2.loc[:,churn_data2.columns != 'Exited']

# Target variable
Y = churn_data2.loc[:,churn_data2.columns == 'Exited']
# Explanatory variable from the balanced dataset
# Explanatory variable
X_balanced = churn_balanced.loc[:,churn_balanced.columns != 'Exited']

# Target variable
Y_balanced = churn_balanced.loc[:,churn_balanced.columns == 'Exited']
#splitting into training and testing datasets from both balanced and unbalanced datasets

#imbalanced
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) 

#balanced
X_trainb, X_testb, Y_trainb, Y_testb = train_test_split(X_balanced, Y_balanced, test_size=0.2) 
#fitting a the Logistic regression model on unbalanced dataset
logregit = LogisticRegression()
logregit.fit(X_train, Y_train.values.ravel())

#predicting 
Y_pred = logregit.predict(X_test)
accuracy = accuracy_score(Y_pred, Y_test)*100
print('Accuracy of LogisticRegression: {:.2f} %'.format(accuracy))
print(classification_report(Y_test, Y_pred))

#fitting a the Logistic regression model
logregitb = LogisticRegression()
logregitb.fit(X_trainb, Y_trainb.values.ravel())

#predicting 
Y_predb = logregitb.predict(X_testb)
accuracy = accuracy_score(Y_predb, Y_testb)*100
print('Accuracy of LogisticRegression: {:.2f} %'.format(accuracy))
print(classification_report(Y_testb, Y_predb))
#Decision tree classifier
dtreeb = DecisionTreeClassifier(max_depth=5)

#Training the decision tree classifier on a balanced dataset using all features
dtreeb.fit(X_trainb, Y_trainb)

#predicting 
Y_pred = dtreeb.predict(X_testb)
accuracy = accuracy_score(Y_predb, Y_testb)*100
print('Accuracy of DecisionTreeClassifier: {:.2f}'.format(accuracy))
print(classification_report(Y_testb, Y_predb))
#Decision tree classifier
dtree = DecisionTreeClassifier(max_depth=5)

#Training the decision tree classifier
dtree.fit(X_train, Y_train)

#predicting 
Y_pred = dtree.predict(X_test)
accuracy = accuracy_score(Y_pred, Y_test)*100
print('Accuracy of DecisionTreeClassifier: {:.2f}'.format(accuracy))
print(classification_report(Y_test, Y_pred))
#Random forest classifier
randForest = RandomForestClassifier()

#Training the decision tree classifier
randForest.fit(X_train, Y_train.values.ravel())

#predicting 
Y_pred = randForest.predict(X_test)
accuracy = accuracy_score(Y_pred, Y_test.values.ravel())*100
print('RandomForestClassifier: {:.2f}'.format(accuracy))
print(classification_report(Y_test, Y_pred))
#Random forest classifier
randForestb = RandomForestClassifier()

#Training the decision tree classifier
randForestb.fit(X_trainb, Y_trainb)

#predicting 
Y_predb = randForest.predict(X_testb)
accuracy = accuracy_score(Y_predb, Y_testb.values.ravel())*100
print('RandomForestClassifier: {:.2f}'.format(accuracy))
print(classification_report(Y_testb, Y_predb))
