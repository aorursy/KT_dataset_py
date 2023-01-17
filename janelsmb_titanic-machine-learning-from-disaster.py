# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
# Importing the dataset
df_Train = pd.read_csv(r"../input/titanic/train.csv")
df_Test = pd.read_csv(r"../input/titanic/test.csv")

# Dataset Information
df_Train.info()
# First DataFrame rows
df_Train.head(10)
# Describing The Data
df_Train.describe()
# Missing Values Training Set
df_Train.isnull().sum()
def autolabel(patches,ax,mode):
    if mode == 'percentage':
        """Display Percentage"""
        for j in range(len(patches)):
            rects = patches[j]
            height = rects.get_height()
            percentage = '{:.1f}%'.format(rects.get_height())       
            ax.annotate(percentage,
                        xy=(rects.get_x() + rects.get_width() / 2, height),
                        xytext=(0, 0.5),
                        textcoords="offset points",
                        ha='center', va='bottom')            
    elif mode == 'count':
        """Display Count"""
        for j in range(len(patches)):
            rects = patches[j]
            height = rects.get_height().astype('int')   
            height = height if height >= 0 else -1 # To avoid error
            ax.annotate(height,
                        xy=(rects.get_x() + rects.get_width() / 2, height),
                        xytext=(0, 0.5),
                        textcoords="offset points",
                        ha='center', va='bottom')         
               
def autoplot(X,hue,data,colors,labels):
    fig, ax = plt.subplots(1,2,figsize=(15, 10))
    
    plt.subplot(1,2,1)
    ax[0] = sns.barplot(x=X.value_counts().index,
                        y=(X.value_counts()/len(X))*100,
                        data=data,palette='Blues_d')    
    ax[0].set_xlabel(X.name,fontsize=13)
    ax[0].set_ylabel("Percentage",fontsize=13)
    autolabel(ax[0].patches,ax[0],'percentage')
    
    plt.subplot(1,2,2)
    ax[1] = sns.countplot(x=X,hue=hue,data=df_Train,palette=colors)
    ax[1].set_ylabel("Number of Occurrences",fontsize=13)
    ax[1].set_xlabel(X.name,fontsize=13)
    ax[1].legend(title=hue.name,labels=labels,fontsize=12,title_fontsize=12,loc='upper right')
    autolabel(ax[1].patches,ax[1],'count')        
# Constants that we will use later       
colors1 =['#C03028','#78C850']#Survived: No/Yes
labels1 = ['No','Yes']
colors2 = ['#6890F0','#F85888']#Sex: Male/Female
labels2 = ['Male','Female']
colors3 = ['#78C850','#34495e','#e74c3c']#Pclass: 1/2/3
labels3 = [1,2,3]
# Survived
Survived = pd.crosstab(df_Train['Survived'],df_Train['Survived']).sum()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(Survived, labels=Survived.index, autopct='%1.1f%%',colors=colors1)
plt.legend(title='Survived',labels=['0: No','1: Yes'],fontsize=10,title_fontsize=10)
# Sex - Survived
autoplot(df_Train['Sex'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Sex'], df_Train['Survived']).apply(lambda r: r/r.sum(),axis=1)
# Embarked
# Taking care the 2 missing values of the feature Embarked by filling the most common value which the port of S: Southampton 
df_Train['Embarked'].value_counts()
df_Train['Embarked'].isnull().sum()
df_Train['Embarked'].fillna('S',inplace=True)
# Embarked - Survived
autoplot(df_Train['Embarked'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Embarked'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Pclass - Survived
autoplot(df_Train['Pclass'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Pclass'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Embarked - Pclass
autoplot(df_Train['Embarked'],df_Train['Pclass'],df_Train,colors3,labels3)
# SibSp - Survived
autoplot(df_Train['SibSp'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['SibSp'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Parch
autoplot(df_Train['Parch'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Parch'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Fare
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.distplot(df_Train['Fare'], ax=ax1)
sns.boxplot(df_Train['Fare'], ax=ax2)
print('Mean Fare = %0.2f\nMedian Fare = %0.2f' % (df_Train['Fare'].mean(),df_Train['Fare'].median()))
# Fare - Survived
ax = sns.FacetGrid(df_Train, hue='Survived',palette=colors1,aspect=2,height=5)
ax = ax.map(sns.kdeplot, "Fare",shade= True)
ax.fig.legend(title='Survived',labels=['No','Yes'],fontsize=12,title_fontsize=12)
# Fare - Pclass
ax = sns.FacetGrid(df_Train, hue='Pclass',palette=colors3,aspect=2,height=5)
ax = ax.map(sns.kdeplot, "Fare",shade= True)
ax.fig.legend()

fig, ax = plt.subplots()
ax = sns.boxplot(x='Pclass', y='Fare', data=df_Train)

F_1 = df_Train['Fare'][df_Train['Pclass'] == 1].mean()
F_2 = df_Train['Fare'][df_Train['Pclass'] == 2].mean()
F_3 = df_Train['Fare'][df_Train['Pclass'] == 3].mean()
print('Mean Fare Pclass 1: %0.1f \nMean Fare Pclass 2: %0.1f \nMean Fare Pclass 3: %0.1f' % (F_1,F_2,F_3))
# Fare binned
Fare_binned = pd.Series.copy(df_Train['Fare'])
Fare_binned = pd.cut(Fare_binned,8)
labels = ['0-64','64-128','128-192','192-256','256-320','320-384','384-448','448-512']
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.barplot(x=Fare_binned.value_counts().index,y=Fare_binned.value_counts(),data=df_Train)
ax.set_xlabel('Fare',fontsize=13)
ax.set_ylabel("Number of Occurrences",fontsize=13)
ax.set_xticklabels(labels)
autolabel(ax.patches,ax,'count')
# Age
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.distplot(df_Train['Age'], ax=ax1)
sns.boxplot(df_Train['Age'], ax=ax2)
print('Mean Age = %0.2f\nMedian Age = %0.2f' % (df_Train['Age'].mean(),df_Train['Age'].median()))
print('Age Skew: %0.2f' % (df_Train['Age'].skew()))
# Age binned
Age_binned = pd.Series.copy(df_Train['Age'])
Age_binned = pd.cut(Age_binned,8)
labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.barplot(x=Age_binned.value_counts().index,y=Age_binned.value_counts(),data=df_Train)
ax.set_xlabel('Age',fontsize=13)
ax.set_ylabel("Number of Occurrences",fontsize=13)
ax.set_xticklabels(labels)
autolabel(ax.patches,ax,'count')
# Age - Survived
ax = sns.FacetGrid(df_Train, hue='Survived',palette=colors1,aspect=2,height=5)
ax = ax.map(sns.kdeplot, "Age",shade=True)
ax.fig.legend(title='Survived',labels=['No','Yes'],fontsize=12,title_fontsize=12)
# Age - Survived Boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Survived', y=df_Train['Age'], data=df_Train)
Q1 = df_Train['Age'].quantile(0.25)
Q3 = df_Train['Age'].quantile(0.75)
IQR = Q3 - Q1

th1 = Q1 - 1.5 * IQR
th2 = Q3 + 1.5 * IQR
fliers = df_Train['Age'][(df_Train['Age'] < th1) | (df_Train['Age'] > th2)]

A_0 = df_Train['Age'][df_Train['Survived'] == 0].mean()
A_1 = df_Train['Age'][df_Train['Survived'] == 1].mean()
print('Mean Age Not Survived: %0.1f \nMean Age Survived: %0.1f' % (A_0,A_1))
# Age - Sex 
fig, ax = plt.subplots()
ax = sns.boxplot(x='Sex', y=df_Train['Age'], data=df_Train)

F = df_Train['Age'][df_Train['Sex'] == 'female'].mean()
M = df_Train['Age'][df_Train['Sex'] == 'male'].mean()
print('Mean Age Men: %0.1f \nMean Age Women: %0.1f' % (M,F))
# Age - Sex - Survived
ax = sns.FacetGrid(df_Train,col='Sex', hue='Survived',palette=colors1,height=5)
ax = ax.map(sns.kdeplot, "Age",shade= True)
#ax = ax.map(sns.distplot, "Age")
ax.fig.legend(title='Survived',labels=['No','Yes'],fontsize=12,title_fontsize=12)
# Age - Pclass Boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Pclass', y='Age',hue='Survived', data=df_Train,palette=colors1)

for i in range(df_Train['Pclass'].nunique()):
    print('Mean Age Pclass %i: %0.1f' %
       (i+1,df_Train['Age'][(df_Train['Pclass'] == i+1)].mean()))
# Family Feature: Combine the features of SibSp and Parch
df_Train["Family"] = df_Train["SibSp"] + df_Train["Parch"]
df_Test["Family"] = df_Test["SibSp"] + df_Test["Parch"]

# Family size above 3 will be signed the number 4
df_Train["Family"][(df_Train["Family"] > 3)] = 4
df_Test["Family"][(df_Test["Family"] > 3)] = 4

# Family
autoplot(df_Train['Family'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Family'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Age - Family - Survived Boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Family', y='Age',hue='Survived', data=df_Train,palette=colors1)

for i in range(df_Train['Family'].nunique()):
    print('Mean Age Family %i: %0.1f' %
       (i,df_Train['Age'][(df_Train['Family'] == i)].mean()))
# Cabin feature has 1014 missing values (both Train and Test) and takes alphanumeric values e.g."C23 C25 C27"
# We create the Deck feature for both Train Set and Test Set by extracting the first letter of the Cabin feature
# and filling missing information with the letter U: Unknown.
# Deck Feature
df_Train['Deck'] = df_Train['Cabin'].str[:1]
df_Test['Deck'] = df_Test['Cabin'].str[:1]

df_Train['Deck'].isna().sum()
df_Test['Deck'].isna().sum()
df_Train['Deck'] = df_Train['Deck'].fillna('U')
df_Test['Deck'] = df_Test['Deck'].fillna('U')

# Deck
autoplot(df_Train['Deck'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Deck'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Name Feature includes the first name, last name and title information of each passenger.
# We create the Title feature which has meaning for our model by extracting the title info for each passenger
# Title takes 17 unique values, we keep the 4 most frequent ones and name all the others with the Title: "Other"
df_Train['Title'] = df_Train['Name'].str.split(', ').str[1].str.split('.').str[0]
df_Test['Title'] = df_Test['Name'].str.split(', ').str[1].str.split('.').str[0]

df_Train['Title'].value_counts()

df_Train["Title"] = df_Train["Title"].replace(['Dr', 'Rev','Col','Major', 'Col','Jonkheer','Dona','Don',
                                               'the Countess', 'Lady', 'Capt', 'Sir','Ms','Mlle','Mme'], 'Other')
df_Test["Title"] = df_Test["Title"].replace(['Dr', 'Rev','Col','Major', 'Col','Jonkheer','Dona','Don',
                                               'the Countess', 'Lady', 'Capt', 'Sir','Ms','Mlle','Mme'], 'Other')

# Title
autoplot(df_Train['Title'],df_Train['Survived'],df_Train,colors1,labels1)
pd.crosstab(df_Train['Title'], df_Train['Survived']).apply(lambda r: r/r.sum(), axis=1)
# Drop no needed columns
df_Train = df_Train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])

# Dataset split to Categorical (Nominal,Ordinal,Binary) and Numeric Vars
df_Cat_Bin = df_Train[['Sex']].iloc[:]
df_Cat_Nom = df_Train[['Embarked','Title','Deck','Family']].iloc[:]
df_Cat_Ord = df_Train[['Pclass']].iloc[:]
df_Num = df_Train[['Age','Fare']].iloc[:]

# Categorical Output
y = df_Train['Survived'].iloc[:]
# LABEL ENCODING - ONE HOT ENCODING CATEGORICAL FEATURES

# Categorical Binary Features Encoding
df_Cat_Bin_Ld = df_Cat_Bin.replace({'male': 0, 'female': 1})

# Categorical Nominal Features Encoding
df_Cat_Nom_OHEd = pd.get_dummies(df_Cat_Nom.astype('str'))

# Categorical Ordinal Features Encoding
df_Cat_Ord_OHEd = pd.get_dummies(df_Cat_Ord.astype('str'))

# All Categorical Features
df_Cat = pd.concat([df_Cat_Bin_Ld,df_Cat_Nom_OHEd,df_Cat_Ord_OHEd],axis=1)
# ALL the Selected IVs
X = pd.concat([df_Num,df_Cat],axis=1)
columns=X.columns
X.head(10)
# Correlation Matrix
corr = X.corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,cmap = "coolwarm",annot=True,annot_kws = {'size': 6})
plt.title("Correlation")
plt.show()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Age Imputing: Taking care of missing values
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
X_train = pd.DataFrame(data = imputer.fit_transform(X_train),
                                 columns = X_train.columns,
                                 index = X_train.index)
X_test = pd.DataFrame(data = imputer.fit_transform(X_test),
                                 columns = X_test.columns,
                                 index = X_test.index)
# Fix Fare skewness
print('Fare Train Skew: %0.2f \nFare Test Skew: %0.2f' %
       (X_train['Fare'].skew(),X_test['Fare'].skew()))

# Using log
X_train['Fare'] = np.log(X_train['Fare'] + 1)
X_test['Fare'] = np.log(X_test['Fare'] + 1)

print('Fare Train Fixed Skew: %0.2f \nFare Test fixed Skew: %0.2f' %
       (X_train['Fare'].skew(),X_test['Fare'].skew()))
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Choosing Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
# Used GridSearchCV for parameter tuning
CF = [None]*9
Names = ['Logistic Regression','SVM linear','SVM rbf','Naive Bayes','kNN','Decision Tree','Random Forest','Gradient Boosting','Ada Boost']
CF[0] = LogisticRegression(penalty='l2',solver='newton-cg')
CF[1] = SVC(kernel = 'linear', random_state = 0,probability=True)
CF[2] = SVC(kernel = 'rbf', random_state = 0,probability=True)
CF[3] = GaussianNB()
CF[4] = KNeighborsClassifier(n_neighbors=15,leaf_size=10,metric='manhattan')
CF[5] = DecisionTreeClassifier(max_depth = 3, min_samples_split = 4,min_samples_leaf=2,random_state = 0)
CF[6] = RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_split=7,min_samples_leaf=2,random_state = 0)
CF[7] = GradientBoostingClassifier(learning_rate=0.05,min_samples_leaf=2,random_state = 0)
CF[8] = AdaBoostClassifier(random_state = 0)
# Classification Metrics
Classifiers = ['Logistic Regression','SVM linear','SVM rbf','Naive Bayes','k-NN','Decision Tree','Random Forest','Gradient Boosting','Ada Boost']
Cols = ['Accuracy','Recall','Precision','f1 score','AUC ROC score']
Scores = pd.DataFrame(index=Classifiers,columns=Cols).astype('float')
for i in range(len(CF)):
    classifier = CF[i]
    classifier.fit(X_train, y_train)
    c_probs = classifier.predict_proba(X_test)
    c_probs = c_probs[:, 1]
    
    y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
    Scores.Accuracy[i] = accuracy_score(y_test,y_pred)
    Scores.Recall[i] = recall_score(y_test,y_pred)
    Scores.Precision[i] = precision_score(y_test,y_pred)
    Scores['f1 score'][i] = f1_score(y_test,y_pred)
    Scores['AUC ROC score'][i] = roc_auc_score(y_test,c_probs)
    
print(Scores)
# Feature Importance plots
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for i in range(4):
    plt.subplot(2, 2, i+1)
    classifier = CF[i+5]
    classifier.fit(X_train, y_train)     

    FImportances = pd.DataFrame(data=classifier.feature_importances_,index=columns,columns=['Importance']).sort_values(by=['Importance'])
    plt.barh(range(FImportances.shape[0]),FImportances['Importance'],color = '#78C850')
    plt.yticks(range(FImportances.shape[0]), FImportances.index)
    plt.title('Feature Importances: %s' % (Names[i+5]))
# ROC - Curves for models
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.3, wspace=0.3)    
for i in range(len(CF)):
    plt.subplot(3, 3, i+1)
    #i=4
    classifier = CF[i]
    classifier.fit(X_train, y_train)  
     
    # Predict probabilities
    r_probs = [0 for _ in range(len(y_test))]
    c_probs = classifier.predict_proba(X_test)

    # Keep probabilities for the positive outcome only
    c_probs = c_probs[:, 1]

    # Calculate AUROC
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    r_auc = roc_auc_score(y_test, r_probs)
    c_auc = roc_auc_score(y_test, c_probs)
    #print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
    #print('%s: AUROC = %.3f' % (Names[i],c_auc))

    # Calculate ROC curve
    r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
    c_fpr, c_tpr, _ = roc_curve(y_test, c_probs)
    plt.plot(r_fpr, r_tpr, linestyle='--',c='r', label='Random Prediction (AUROC = %0.3f)' % r_auc)
    plt.plot(c_fpr, c_tpr, marker='.',c='b', label='%s (AUROC = %0.3f)' % (Names[i],c_auc))

    plt.title('ROC Plot')
    plt.xlabel('False Positive Rate - 1 - Specificity')
    plt.ylabel('True Positive Rate - Sensitivity')
    plt.legend(fontsize='small')
# Cap Curve
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.3, wspace=0.3)    
for i in range(len(CF)):
    plt.subplot(3, 3, i+1)
    
    total = len(y_test)
    class_1_count = np.sum(y_test)
    class_0_count = total - class_1_count

    plt.plot([0, total], [0, class_1_count], c = 'r', linestyle = '--', label = 'Random Model')

    plt.plot([0, class_1_count, total], 
             [0, class_1_count, class_1_count], 
             c = 'grey', linewidth = 2, label = 'Perfect Model')

    classifier = CF[i]
    classifier.fit(X_train, y_train)  
    c_probs = classifier.predict_proba(X_test)

    # Keep probabilities for the positive outcome only
    c_probs = c_probs[:, 1]

    model_y = [y for _, y in sorted(zip(c_probs, y_test), reverse = True)]
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total + 1)

    from sklearn.metrics import auc
    # Area under Random Model
    a = auc([0, total], [0, class_1_count])

    # Area between Perfect and Random Model
    aP = auc([0, class_1_count, total], [0, class_1_count, class_1_count]) - a

    # Area between Trained and Random Model
    aR = auc(x_values, y_values) - a

    AR = aR / aP

    plt.plot(x_values, y_values, c = 'g', label = '%s (AR = %0.3f)' % (Names[i],AR), linewidth = 4)

    # Plot information
    plt.xlabel('Total observations')
    plt.ylabel('Class 1 observations')
    plt.title('Cumulative Accuracy Profile')
    plt.legend(fontsize='small')