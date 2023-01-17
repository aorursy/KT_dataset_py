# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Importing the dataset
df_Train = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Dataset Information
df_Train.info()
df_Train['TotalCharges']
# Convert column TotalCharger from object to float
df_Train['TotalCharges'] = df_Train['TotalCharges'].apply(pd.to_numeric,errors='coerce')
# Missing Values
df_Train.isnull().sum()
# First DataFrame rows
df_Train.head(10)
# Describing The Data
df_Train.describe()
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
               
def autoplot(X,hue,data,colors):
    fig, ax = plt.subplots(1,2,figsize=(15, 10))
    
    plt.subplot(1,2,1)
    ax[0] = sns.barplot(x=X.value_counts().index,
                        y=(X.value_counts()/len(X))*100,
                        data=data,palette='Blues_d')    
    ax[0].set_xlabel(X.name,fontsize=13)
    ax[0].set_ylabel("Percentage",fontsize=13)
    autolabel(ax[0].patches,ax[0],'percentage')
    
    plt.subplot(1,2,2)
    ax[1] = sns.countplot(x=X,hue=hue,data=df_Train,palette=colors,order = X.value_counts().index)   
    ax[1].set_ylabel("Number of Occurrences",fontsize=13)
    ax[1].set_xlabel(X.name,fontsize=13)
    autolabel(ax[1].patches,ax[1],'count')   
    
# Constants that we will use later
colors1 =['#C03028','#78C850']#Churn: No/Yes
# Churn
Churn = pd.crosstab(df_Train['Churn'],df_Train['Churn']).sum()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(Churn, labels=Churn.index, autopct='%1.1f%%',colors=colors1)
plt.legend(title='Churn',fontsize=10,title_fontsize=10)
# Gender
autoplot(df_Train['gender'],df_Train['Churn'],df_Train,colors1)
pd.crosstab(df_Train['gender'], df_Train['Churn']).apply(lambda r: r/r.sum(),axis=1)
# PhoneService-MultipleLines-InternetService
IVs = ['PhoneService','MultipleLines','InternetService']
for i in range(len(IVs)):    
    autoplot(df_Train[IVs[i]],df_Train['Churn'],df_Train,colors1)
for i in range(len(IVs)):    
    print(pd.crosstab(df_Train[IVs[i]], df_Train['Churn']).apply(lambda r: r/r.sum(), axis=1))
    print('\n')
# OnlineSecurity-OnlineBackup-DeviceProtection-TechSupport-StreamingTV-StreamingMovies
IVs = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in range(len(IVs)):
    autoplot(df_Train[IVs[i]],df_Train['Churn'],df_Train,colors1)
for i in range(len(IVs)):
    print(pd.crosstab(df_Train[IVs[i]], df_Train['Churn']).apply(lambda r: r/r.sum(), axis=1))
    print('\n')
#SeniorCitizen-Partner-Dependents
IVs = ['SeniorCitizen','Partner','Dependents']
for i in range(len(IVs)):
    autoplot(df_Train[IVs[i]],df_Train['Churn'],df_Train,colors1)
for i in range(len(IVs)):
    print(pd.crosstab(df_Train[IVs[i]], df_Train['Churn']).apply(lambda r: r/r.sum(), axis=1))
    print('\n')
# Contract-PaperlessBilling-PaymentMethod
IVs = ['Contract','PaperlessBilling','PaymentMethod']
df_Train['PaymentMethod'] = df_Train['PaymentMethod'].replace({'Bank transfer (automatic)':'Bank transfer Auto',
                                                               'Credit card (automatic)':'Credit card Auto'})
for i in range(len(IVs)):    
    autoplot(df_Train[IVs[i]],df_Train['Churn'],df_Train,colors1)
for i in range(len(IVs)):    
    print(pd.crosstab(df_Train[IVs[i]], df_Train['Churn']).apply(lambda r: r/r.sum(), axis=1))
    print('\n')
# Tenure
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.distplot(df_Train['tenure'], ax=ax1)
sns.boxplot(df_Train['tenure'], ax=ax2)
print('Mean Tenure = %0.2f\nMedian Tenure = %0.2f' % (df_Train['tenure'].mean(),df_Train['tenure'].median()))
# Tenure - Churn
ax = sns.FacetGrid(df_Train, hue='Churn',palette=colors1,aspect=2,height=5)
ax = ax.map(sns.kdeplot, "tenure",shade= True)
ax.fig.legend(title='Churn',fontsize=12,title_fontsize=12)    
    
fig, ax = plt.subplots()
ax = sns.boxplot(x='Churn', y='tenure', data=df_Train)

T_0 = df_Train['tenure'][df_Train['Churn'] == 'No'].mean()
T_1 = df_Train['tenure'][df_Train['Churn'] == 'Yes'].mean()
print('Mean Tenure No Churn: %0.1f \nMean Tenure Churn: %0.1f' % (T_0,T_1))
# MonthlyCharges
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.distplot(df_Train['MonthlyCharges'], ax=ax1)
sns.boxplot(df_Train['MonthlyCharges'], ax=ax2)
print('Mean MonthlyCharges = %0.2f\nMedian MonthlyCharges = %0.2f' % (df_Train['MonthlyCharges'].mean(),df_Train['MonthlyCharges'].median()))
# MonthlyCharges - Churn
ax = sns.FacetGrid(df_Train, hue='Churn',palette=colors1,aspect=2,height=5)
ax = ax.map(sns.kdeplot, "MonthlyCharges",shade= True)
ax.fig.legend(title='Churn',fontsize=12,title_fontsize=12)    
    
fig, ax = plt.subplots()
ax = sns.boxplot(x='Churn', y='MonthlyCharges', data=df_Train)

M_0 = df_Train['MonthlyCharges'][df_Train['Churn'] == 'No'].mean()
M_1 = df_Train['MonthlyCharges'][df_Train['Churn'] == 'Yes'].mean()
print('Mean MonthlyCharges No Churn: %0.1f \nMean MonthlyCharges Churn: %0.1f' % (M_0,M_1))
# TotalCharges
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.distplot(df_Train['TotalCharges'], ax=ax1)
sns.boxplot(df_Train['TotalCharges'], ax=ax2)
print('Mean TotalCharges = %0.2f\nMedian TotalCharges = %0.2f' % (df_Train['TotalCharges'].mean(),df_Train['TotalCharges'].median()))
# TotalCharges - Churn
ax = sns.FacetGrid(df_Train, hue='Churn',palette=colors1,aspect=2,height=5)
ax = ax.map(sns.kdeplot, "TotalCharges",shade= True)
ax.fig.legend(title='Churn',fontsize=12,title_fontsize=12)    
    
fig, ax = plt.subplots()
ax = sns.boxplot(x='Churn', y='TotalCharges', data=df_Train)

TC_0 = df_Train['TotalCharges'][df_Train['Churn'] == 'No'].mean()
TC_1 = df_Train['TotalCharges'][df_Train['Churn'] == 'Yes'].mean()
print('Mean TotalCharges No Churn: %0.1f \nMean TotalCharges Churn: %0.1f' % (TC_0,TC_1))
# Online Services 
IVs = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

OnlineServices = df_Train[IVs].replace({'No internet service':2,'No': 0, 'Yes': 1})
df_Train['OnlineServices'] = OnlineServices.sum(axis=1)
df_Train['OnlineServices'] = df_Train['OnlineServices'].replace({12:'No Int. Service'})

autoplot(df_Train['OnlineServices'],df_Train['Churn'],df_Train,colors1)
pd.crosstab(df_Train['OnlineServices'], df_Train['Churn']).apply(lambda r: r/r.sum(), axis=1)
# MonthChTenure = MonthlyCharges * Tenure 
# MonthChTenure - TotalCharges
df_Train['MonthChTenure'] = df_Train['MonthlyCharges']*df_Train['tenure']

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.distplot(df_Train['TotalCharges'], ax=ax1)
sns.distplot(df_Train['MonthChTenure'], ax=ax2)
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 10),sharex=True)
sns.boxplot(df_Train['TotalCharges'], ax=ax1)
sns.boxplot(df_Train['MonthChTenure'], ax=ax2)
# TotalCharges: Delete missing values
df_Train[df_Train['TotalCharges'].isnull()].loc[:,('MonthlyCharges','tenure')]
df_Train = df_Train.drop(df_Train['MonthlyCharges'][df_Train['TotalCharges'].isnull()].index)
# Dataset split to Categorical (Nominal,Binary) and Numeric Vars
df_Cat_Bin = df_Train[['gender','Partner','SeniorCitizen','Dependents','PhoneService','PaperlessBilling']].iloc[:]
df_Cat_Nom = df_Train[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies','StreamingTV','Contract','PaymentMethod']].iloc[:]
df_Num = df_Train[['tenure','MonthlyCharges','TotalCharges']].iloc[:]

# Categorical Output
y = df_Train['Churn'].iloc[:]
# LABEL ENCODING - ONE HOT ENCODING

# Categorical Binary Features Encoding
from sklearn.preprocessing import LabelEncoder
df_Cat_Bin_Ld = df_Cat_Bin.apply(LabelEncoder().fit_transform)

# Categorical Nominal Features Encoding
df_Cat_Nom_OHEd = pd.get_dummies(df_Cat_Nom)

# All Categorical Features
df_Cat = pd.concat([df_Cat_Bin_Ld,df_Cat_Nom_OHEd],axis=1)

# Categorical Outpout Encoding
y_Ld = y.replace({'No': 0, 'Yes': 1})

# ALL the Selected IVs
X = pd.concat([df_Num,df_Cat],axis=1)
columns=X.columns
X.head()
# Correlation Matrix
plt.figure(figsize=(20, 20))
corr = X.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,cmap = "coolwarm",annot=True,annot_kws = {'size': 6})
plt.title("Correlation")
plt.show()
X = X.drop(columns=['OnlineSecurity_No internet service','OnlineBackup_No internet service',
                    'DeviceProtection_No internet service','TechSupport_No internet service',
                    'StreamingMovies_No internet service','StreamingTV_No internet service'])

X = X.drop(columns=['PhoneService'])

X = X.drop(columns=['TotalCharges'])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_Ld, test_size = 0.2, random_state = 0)
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
CF[0] = LogisticRegression(solver='newton-cg')
CF[1] = SVC(kernel = 'linear', random_state = 0,probability=True)
CF[2] = SVC(kernel = 'rbf', random_state = 0,probability=True)
CF[3] = GaussianNB()
CF[4] = KNeighborsClassifier(n_neighbors=20,metric='minkowski')
CF[5] = DecisionTreeClassifier(max_depth=5,min_samples_leaf=2,random_state = 0)
CF[6] = RandomForestClassifier(n_estimators=150,min_samples_split=4,max_depth=9,min_samples_leaf=2,random_state = 0)
CF[7] = GradientBoostingClassifier(loss='exponential',min_samples_leaf=2,learning_rate=0.05,random_state = 0)
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
columns=X.columns
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