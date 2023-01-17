import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_color_codes()

%matplotlib inline

from sklearn import metrics



#setting up for customized printing

from IPython.display import Markdown, display

from IPython.display import HTML

def printmd(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))

    

#function to display dataframes side by side    

from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)
bankLoan = pd.read_csv("../input/Bank_Personal_Loan_Modelling.csv")

bankLoan.head()
print('The total number of rows :', bankLoan.shape[0])

print('The total number of columns :', bankLoan.shape[1])
bankLoan.info()
bankLoan.drop('ID', axis = 1, inplace=True)

bankLoan.drop('ZIP Code', axis = 1, inplace=True)
print(bankLoan.isna().sum())

print('=============================')

print(bankLoan.isnull().sum())

print('=============================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"NO Missing"** values in the data', color="blue")
printmd('**Find the names of columns having negative values**', color="brown")

print([bankLoan[bankLoan[col] < 0].Experience.value_counts() for col in bankLoan.columns if any(bankLoan[col] < 0)])

print('=============================')

printmd('**CONCLUSION**: There are **"NEGATIVE"** values in the **"Experience"** column', color="blue")
printmd('**Replace the negative values with nan**', color="brown")

bankLoan = bankLoan.assign(Experience = lambda x: x.Experience.where(x.Experience.ge(0)))

print(bankLoan.Experience.isna().sum())



print('=============================')



printmd('**Since the imputation needs to be done in only 1 columns (Experience), we will use fillna imputation method**', color="brown")



print(f'Median of Experience Column is {bankLoan.Experience.median()}')



print('=============================')



print(f'Mean of Experience Column before imputation is {bankLoan.Experience.mean()}')

bankLoan.Experience = bankLoan.Experience.fillna(bankLoan.Experience.median())

print(f'Mean of Experience Column after imputation is {bankLoan.Experience.mean()}')

bankLoan.describe().transpose()
bankLoan[bankLoan['Personal Loan'] == 0]['Age']
f, axes = plt.subplots(1, 3, figsize=(20, 8))

age = sns.distplot(bankLoan['Age'], color="red", ax=axes[0], kde=True, hist_kws={"edgecolor":"k"})

age.set_xlabel("Age",fontsize=20)



exp = sns.distplot(bankLoan['Experience'], color='green', ax = axes[1], kde=True, hist_kws={"edgecolor":"k"})

exp.set_xlabel("Experience",fontsize=20)



income = sns.distplot(bankLoan['Income'], color='blue', ax = axes[2], kde=True, hist_kws={"edgecolor":"k"})

income.set_xlabel("Income",fontsize=20)



f, axes = plt.subplots(1, 2, figsize=(11, 6))



ccavg = sns.distplot(bankLoan['CCAvg'], color="brown", ax=axes[0], kde=True, hist_kws={"edgecolor":"k"})

ccavg.set_xlabel("CCAvg",fontsize=20)



mort = sns.distplot(bankLoan['Mortgage'], color="teal", ax=axes[1], kde=True, hist_kws={"edgecolor":"k"})

mort.set_xlabel("Mortgage",fontsize=20)

pd.DataFrame.from_dict(dict(

    {

        'Age':bankLoan.Age.skew(), 

        'Experience': bankLoan.Experience.skew(), 

        'Income': bankLoan.Income.skew(),

        'CCAvg': bankLoan.CCAvg.skew(),

        'Mortgage': bankLoan.Mortgage.skew()        

    }), orient='index', columns=['Skewness'])
f, axes = plt.subplots(1, 3, figsize=(10, 8))

income = sns.boxplot(bankLoan['Income'], color="olive", ax=axes[0], orient='v')

income.set_xlabel("Income",fontsize=20)



ccavg = sns.boxplot(bankLoan['CCAvg'], color='lightgreen', ax=axes[1], orient='v')

ccavg.set_xlabel("CCAvg",fontsize=20)



mort = sns.boxplot(bankLoan['Mortgage'], color='lightblue', ax=axes[2], orient='v')

mort.set_xlabel("Mortgage",fontsize=20)

plData = pd.DataFrame(bankLoan['Personal Loan'].value_counts(), columns=['Personal Loan'])

saData = pd.DataFrame(bankLoan['Securities Account'].value_counts(), columns=['Securities Account'])

cdacctData = pd.DataFrame(bankLoan['CD Account'].value_counts(), columns=['CD Account'])

onlineData = pd.DataFrame(bankLoan['Online'].value_counts(), columns=['Online'])

ccData = pd.DataFrame(bankLoan['CreditCard'].value_counts(), columns=['CreditCard'])



cat = pd.concat([plData,saData,onlineData,cdacctData,ccData], axis=1)

display(cat)

print('=============================')

edu = pd.DataFrame(bankLoan.Education.value_counts(), columns=['Education'])

display(edu.sort_index())

print('=============================')

fam = pd.DataFrame(bankLoan.Family.value_counts(), columns=['Family'])

display(fam.sort_index())
f, axes = plt.subplots(1, 5, figsize=(20, 6))

pl = sns.countplot(bankLoan['Personal Loan'], color="orange", ax=axes[0])

pl.set_xlabel("Personal Loan",fontsize=20)



secacct = sns.countplot(bankLoan['Securities Account'], color='lightgreen', ax = axes[1])

secacct.set_xlabel("Securities Account",fontsize=20)



cdacct = sns.countplot(bankLoan['CD Account'], color='lightblue', ax = axes[2])

cdacct.set_xlabel("CD Account",fontsize=20)



online = sns.countplot(bankLoan['Online'], color='silver', ax = axes[3])

online.set_xlabel("Online",fontsize=20)



cc = sns.countplot(bankLoan['CreditCard'], color='teal', ax = axes[4])

cc.set_xlabel("CreditCard",fontsize=20)



f, axes = plt.subplots(1, 2, figsize=(10, 6))

family = sns.countplot('Family',data=bankLoan, color='darkgreen', ax=axes[0])

family.set_xlabel("Family",fontsize=20)



edu = sns.countplot('Education',data=bankLoan, color='tomato', ax = axes[1])

edu.set_xlabel("Education",fontsize=20)
f, axes = plt.subplots(1, 6, figsize=(20, 6))



secacct = sns.countplot('Securities Account', data=bankLoan, hue='Personal Loan', palette='Accent', ax = axes[0])

secacct.set_xlabel("Securities Account",fontsize=20)



cdacct = sns.countplot(bankLoan['CD Account'], data=bankLoan, hue='Personal Loan', palette='Accent_r', ax = axes[1])

cdacct.set_xlabel("CD Account",fontsize=20)



online = sns.countplot(bankLoan['Online'], data=bankLoan, hue='Personal Loan', palette='BuGn_r', ax = axes[2])

online.set_xlabel("Online",fontsize=20)



cc = sns.countplot(bankLoan['CreditCard'], data=bankLoan, hue='Personal Loan', palette='BuPu_r', ax = axes[3])

cc.set_xlabel("CreditCard",fontsize=20)



#f, axes = plt.subplots(1, 2, figsize=(10, 6))

family = sns.countplot('Family',data=bankLoan, palette='Dark2', hue='Personal Loan', ax=axes[4])

family.set_xlabel("Family",fontsize=20)



edu = sns.countplot('Education',data=bankLoan, palette='OrRd_r', hue='Personal Loan', ax = axes[5])

edu.set_xlabel("Education",fontsize=20)





printmd('**Count of Personal Loans availed**', color='brown')

secacct = pd.DataFrame(bankLoan[bankLoan['Personal Loan'] == 1]['Securities Account'].value_counts(), columns=['Securities Account'])

cdacct = pd.DataFrame(bankLoan[bankLoan['Personal Loan'] == 1]['CD Account'].value_counts(), columns=['CD Account'])

online = pd.DataFrame(bankLoan[bankLoan['Personal Loan'] == 1]['Online'].value_counts(), columns=['Online'])

cc = pd.DataFrame(bankLoan[bankLoan['Personal Loan'] == 1]['CreditCard'].value_counts(), columns=['CreditCard'])



cat = pd.concat([secacct,cdacct,online,cc], axis=1)

display(cat)
f, axes = plt.subplots(1, 3, figsize=(20, 6))



plt1 = sns.boxplot('Family', 'Income', data=bankLoan, hue='Personal Loan', palette='Set1', ax=axes[0])

plt1.set_xlabel("Family",fontsize=20)

plt1.set_ylabel("Income",fontsize=20)



plt2 = sns.boxplot('Family', 'CCAvg', data=bankLoan, hue='Personal Loan', palette='YlOrBr_r', ax=axes[1])

plt2.set_xlabel("Family",fontsize=20)

plt2.set_ylabel("CCAvg",fontsize=20)



plt3 = sns.boxplot('Family', 'Mortgage', data=bankLoan, hue='Personal Loan', palette='viridis', ax=axes[2])

plt3.set_xlabel("Family",fontsize=20)

plt3.set_ylabel("Mortgage",fontsize=20)
f, axes = plt.subplots(1, 2, figsize=(14, 6))



plt1 = sns.boxplot('Education', 'Income', data=bankLoan, hue='Personal Loan', palette='Set1', ax=axes[0])

plt1.set_xlabel("Education",fontsize=20)

plt1.set_ylabel("Income",fontsize=20)



plt2 = sns.boxplot('Education', 'CCAvg', data=bankLoan, hue='Personal Loan', palette='YlOrBr_r', ax=axes[1])

plt2.set_xlabel("Education",fontsize=20)

plt2.set_ylabel("CCAvg",fontsize=20)
sns.catplot('Family', 'Income', data=bankLoan, hue='Personal Loan', col='Education', kind='box', palette='Set1')



sns.catplot('Family', 'CCAvg', data=bankLoan, hue='Personal Loan', col='Education', kind='box', palette='YlOrBr_r')



sns.catplot('Family', 'Mortgage', data=bankLoan, hue='Personal Loan', col='Education', kind='box', palette='viridis')
sns.pairplot(bankLoan[['Age','Experience','Income','CCAvg', 'Mortgage', 'Family', 'Personal Loan']], hue='Personal Loan', diag_kind = 'kde', palette='rocket')
corrData = bankLoan.corr()

f, axes = plt.subplots(1, 1, figsize=(10, 8))

sns.heatmap(corrData,cmap='YlGnBu', ax=axes, annot=True, fmt=".2f", linecolor='white', linewidths=0.3, square=True)

plt.xticks(rotation=60)
bankLoan['Education'] = bankLoan['Education'].astype(dtype='category')

bankLoan['Family'] = bankLoan['Family'].astype(dtype='category')
# Create correlation matrix

corr_matrix = bankLoan.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

f, axes = plt.subplots(1, 1, figsize=(10, 8))

sns.heatmap(upper,cmap='YlGnBu', annot=True, fmt=".2f", ax=axes, linecolor='white', linewidths=0.3, square=True)

plt.xticks(rotation=60)
# Find index of feature columns with correlation greater than 0.98

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

printmd('List of columns with correlation higher than 0.95', color='brown')

display(to_drop)



printmd('Removing **"Experience"** column due to **Multicollinearity**', color='brown')



bankLoanNew = bankLoan.drop('Experience', axis = 1)

bankLoanNew.info()
f, axes = plt.subplots(1, 3, figsize=(20, 6))



mort = sns.distplot(bankLoanNew['Mortgage'], color="darkgreen", ax=axes[0], kde=True, hist_kws={"edgecolor":"k"})

mort.set_xlabel("Mortgage",fontsize=20)

mort.set_title('Without Log Tranformation')



firstLogTransform = np.log1p(bankLoanNew['Mortgage'])

mort = sns.distplot(firstLogTransform, color="darkgreen", kde=True, ax=axes[1], hist_kws={"edgecolor":"k"})

mort.set_xlabel("Mortgage",fontsize=20)

mort.set_title('After First Log Tranformation')



secondLogTransform = np.log1p(firstLogTransform)

mort = sns.distplot(secondLogTransform, color="darkgreen", kde=True,ax=axes[2], hist_kws={"edgecolor":"k"})

mort.set_xlabel("Mortgage",fontsize=20)

mort.set_title('After Second Log Tranformation')



mortSkew = pd.DataFrame.from_dict(dict(

    {

        'Mortgage Without Log Transformation':bankLoan.Mortgage.skew(), 

        'Mortgage After First Log Transformation': firstLogTransform.skew(), 

        'Mortgage After Second Log Transformation': secondLogTransform.skew()

    }), orient='index', columns=['Skewness'])



display(mortSkew)



bankLoanNew['Mortgage'] = secondLogTransform
f, axes = plt.subplots(1, 2, figsize=(15, 6))



ccavg = sns.distplot(bankLoanNew['CCAvg'], color="darkorange", ax=axes[0], kde=True, hist_kws={"edgecolor":"k"})

ccavg.set_xlabel("CCAvg",fontsize=20)

ccavg.set_title('Without Log Tranformation')



firstLogTransform = np.log1p(bankLoanNew['CCAvg'])

ccavg = sns.distplot(firstLogTransform, color="darkorange", kde=True, ax=axes[1], hist_kws={"edgecolor":"k"})

ccavg.set_xlabel("CCAvg",fontsize=20)

ccavg.set_title('After First Log Tranformation')



ccavgSkew = pd.DataFrame.from_dict(dict(

    {

        'CCAvg Without Log Transformation':bankLoan.CCAvg.skew(), 

        'CCAvg After First Log Transformation': firstLogTransform.skew()

    }), orient='index', columns=['Skewness'])



display(ccavgSkew)



bankLoanNew['CCAvg'] = firstLogTransform
from sklearn.preprocessing import OneHotEncoder



onehotencoder = OneHotEncoder(categories='auto')

encodedData = onehotencoder.fit_transform(bankLoanNew[['Family','Education']]).toarray() 

encodedFeatures = pd.DataFrame(encodedData, columns= onehotencoder.get_feature_names(['Family','Education']))

encodedFeatures.head(2)
printmd('''Dropping last encoded feature in each attribute i.e **Family_4**, **Education_3** can be **DROPPED** as information for these features

can be obtained from others''', color='brown')



encodedFeatures.drop(['Family_4', 'Education_3'], axis=1, inplace=True)

bankLoanNew.drop(['Family', 'Education'], axis=1, inplace=True)
encodedFeatures.head(2)
bankLoanNew = pd.concat([bankLoanNew,encodedFeatures],axis=1)

bankLoanNew.head(2)
printmd('**As "Personal Loan" attribute is imbalanced, STRATIFYING the same to maintain the same percentage of distribution**', color='brown')

X = bankLoanNew.loc[:, bankLoanNew.columns != 'Personal Loan']

y = bankLoanNew['Personal Loan']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size =.30, random_state=1)



printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns')

print(f'Testing set has {X_test.shape[0]} rows and {X_test.shape[1]} columns')



printmd('**Original Set Target Value Distribution**', color='brown')



print("Original Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(bankLoanNew.loc[bankLoanNew['Personal Loan'] == 1]), (len(bankLoanNew.loc[bankLoanNew['Personal Loan'] == 1])/len(bankLoanNew.index)) * 100))

print("Original Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(bankLoanNew.loc[bankLoanNew['Personal Loan'] == 0]), (len(bankLoanNew.loc[bankLoanNew['Personal Loan'] == 0])/len(bankLoanNew.index)) * 100))



printmd('**Training Set Target Value Distribution**', color='brown')



print("Training Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))



printmd('**Testing Set Target Value Distribution**', color='brown')

print("Test Personal Loan '1' Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Personal Loan '0' Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)

X_test_scaled = scalar.fit_transform(X_test)
f, axes = plt.subplots(1, 1, figsize=(10, 8))

sns.heatmap(pd.DataFrame(X_train_scaled).corr().abs(),cmap='YlGnBu', ax=axes, annot=True, fmt=".2f",xticklabels=X_train.columns, yticklabels=X_train.columns, linecolor='white', linewidths=0.3, square=True)

plt.xticks(rotation=60)
# function for model fitting, prediction and calculating different scores

def Modelling_Prediction_Scores(model, algoName):

    model.fit(X_train_scaled, y_train)

    #predict on train and test

    y_train_pred = model.predict(X_train_scaled)

    y_test_pred = model.predict(X_test_scaled)



    #predict the probabilities on train and test

    y_train_pred_proba = model.predict_proba(X_train_scaled) 

    y_test_pred_proba = model.predict_proba(X_test_scaled)



    #get Accuracy Score for train and test

    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)

    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)

    accdf = pd.DataFrame([[accuracy_train, accuracy_test, ]], columns=['Training', 'Testing'], index=['Accuracy'])



    #get Mean Squared Error on train and test

    mse_train = metrics.mean_squared_error(y_train, y_train_pred)

    mse_test = metrics.mean_squared_error(y_test, y_test_pred)

    msedf = pd.DataFrame([[mse_train, mse_test, ]], columns=['Training', 'Testing'], index=['Mean Squared Error'])



    #get Precision Score on train and test

    precision_train = metrics.precision_score(y_train, y_train_pred)

    precision_test = metrics.precision_score(y_test, y_test_pred)

    precdf = pd.DataFrame([[precision_train, precision_test, ]], columns=['Training', 'Testing'], index=['Precision'])



    #get Recall Score on train and test

    recall_train = metrics.recall_score(y_train, y_train_pred)

    recall_test = metrics.recall_score(y_test, y_test_pred)

    recdf = pd.DataFrame([[recall_train, recall_test, ]], columns=['Training', 'Testing'], index=['Recall'])



    #get F1-Score on train and test

    f1_score_train = metrics.f1_score(y_train, y_train_pred)

    f1_score_test = metrics.f1_score(y_test, y_test_pred)

    f1sdf = pd.DataFrame([[f1_score_train, f1_score_test, ]], columns=['Training', 'Testing'], index=['F1 Score'])



    #get Area Under the Curve (AUC) for ROC Curve on train and test

    roc_auc_score_train = metrics.roc_auc_score(y_train, y_train_pred)

    roc_auc_score_test = metrics.roc_auc_score(y_test, y_test_pred)

    rocaucsdf = pd.DataFrame([[roc_auc_score_train, roc_auc_score_test, ]], columns=['Training', 'Testing'], index=['ROC AUC Score'])



    #get Area Under the Curve (AUC) for Precision-Recall Curve on train and test

    precision_train, recall_train, thresholds_train = metrics.precision_recall_curve(y_train, y_train_pred_proba[:,1])

    precision_recall_auc_score_train = metrics.auc(recall_train, precision_train)

    precision_test, recall_test, thresholds_test = metrics.precision_recall_curve(y_test,y_test_pred_proba[:,1])

    precision_recall_auc_score_test = metrics.auc(recall_test, precision_test)

    precrecaucsdf = pd.DataFrame([[precision_recall_auc_score_train, precision_recall_auc_score_test]], columns=['Training', 'Testing'], index=['Precision Recall AUC Score'])



    #calculate the confusion matrix 

    #print('tn, fp, fn, tp')

    confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])



    #display confusion matrix in a heatmap

    f, axes = plt.subplots(1, 2, figsize=(20, 8))

    hmap = sns.heatmap(confusion_matrix_test, cmap='YlGnBu', annot=True, fmt=".0f", ax=axes[0], )

    hmap.set_xlabel('Predicted', fontsize=15)

    hmap.set_ylabel('Actual', fontsize=15)



    #plotting the ROC Curve and Precision-Recall Curve

    fpr, tpr, threshold = metrics.roc_curve(y_test,y_test_pred_proba[:,1])

    plt.plot(fpr, tpr, marker='.', label='ROC Curve')

    plt.plot(recall_test, precision_test, marker='.', label='Precision Recall Curve')

    plt.axes(axes[1])

    plt.title(algoName, fontsize=15)

    # axis labels

    plt.xlabel('ROC Curve - False Positive Rate \n Precision Recall Curve - Recall', fontsize=15)    

    plt.ylabel('ROC Curve - True Positive Rate \n Precision Recall Curve - Precision', fontsize=15)

    # show the legend

    plt.legend()

    # show the plot

    plt.show()



    #concatenating all the scores and displaying as single dataframe

    consolidatedDF= pd.concat([accdf, msedf,precdf,recdf,f1sdf, rocaucsdf, precrecaucsdf])



    printmd('**Confusion Matrix**', color='brown')

    display_side_by_side(confusion_matrix_test, consolidatedDF)

    

    return confusion_matrix_test, consolidatedDF
from sklearn.linear_model import LogisticRegression



# Fit the model on train

logRegModel = LogisticRegression()

cmLR, dfLR = Modelling_Prediction_Scores(logRegModel, 'Logistic Regression')
# Fit the model on train

# C is inverse of lambda (Regularization Parameter). Hence lower the C value will strenthen the lambda parameter.

logRegModel = LogisticRegression(solver = 'liblinear', C=0.2)

cmLR, dfLR = Modelling_Prediction_Scores(logRegModel, 'Logistic Regression')
from sklearn.naive_bayes import GaussianNB



# Fit the model on train

gnb = GaussianNB()

cmNB, dfNB = Modelling_Prediction_Scores(gnb, 'Gaussian Naive Bayes Classifier')
#plot the f1-scores for different values of k for a model and see which is optimal

def Optimal_k_Plot(model):

    # creating odd list of K for KNN

    myList = list(range(3,20))



    # subsetting just the odd ones

    klist = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold accuracy scores

    scores = []



    # perform accuracy metrics for values from 3,5....19

    for k in klist:        

        model.n_neighbors = k

        model.fit(X_train_scaled, y_train)

        # predict the response

        y_test_pred = knn.predict(X_test_scaled)        

        test_score= metrics.f1_score(y_test, y_test_pred)

        scores.append(test_score)



    # determining best k

    optimal_k = klist[scores.index(max(scores))]

    print("The optimal number of neighbors is %d" % optimal_k)



    import matplotlib.pyplot as plt

    # plot misclassification error vs k

    plt.plot(klist, scores)

    plt.xlabel('Number of Neighbors K')

    plt.ylabel('AUC Score')

    plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

Optimal_k_Plot(knn)
knn = KNeighborsClassifier(n_neighbors=3)

cmKNN, dfKNN = Modelling_Prediction_Scores(knn, 'KNN Classifier')
knn = KNeighborsClassifier(weights='distance',p=2)

Optimal_k_Plot(knn)
knn = KNeighborsClassifier(n_neighbors=3,weights='distance',p=2)

cmKNN, dfKNN = Modelling_Prediction_Scores(knn, 'KNN Classifier')
knn = KNeighborsClassifier(weights='distance',p=1)

Optimal_k_Plot(knn)
from sklearn.neighbors import KNeighborsClassifier



# Fit the model on train

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)

cmKNN, dfKNN = Modelling_Prediction_Scores(knn, 'KNN Classifier')