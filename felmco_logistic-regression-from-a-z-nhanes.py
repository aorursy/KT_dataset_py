# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Library for plotting

import seaborn as sns

from pandas.plotting import radviz

from pandas.plotting import parallel_coordinates



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/NHANES_CAT_NEW_130.CSV")
#We will check the number of rows of the dataset, the type of data of every column and also to check 

#if the columns contain null values

df.info()
# We will check if we have a balanced or an unbalanced dataset by checking the number of records for each class.

df.HYPCLASS.value_counts()

# After running this sentence, we can notice that we will be working with an unbalanced dataset.
# Number of patient by Class

plt.figure(figsize=(10,6))

labels = ['0 - No Hypertension', '1 - Yes Hypertension']



ax = sns.countplot(x='HYPCLASS',data=df,palette='RdBu_r')



ax.set_title('Number of individuals by Class')

ax.set_xticklabels(labels)

ax.set_xlabel('Hypertension Class')

ax.set_ylim(0,18000)



#Bar values

rects = ax.patches

#To get the value labels from value_counts()

v_labels = df['HYPCLASS'].value_counts()

# Now make some labels with the values 

labels = ["%d" % v_labels[i] for i in range(len(rects))]



for rect, label in zip(rects, labels):

       height = rect.get_height()

       ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
labels = ['0 - No Hypertension', '1 - Yes Hypertension']

ax = sns.countplot(x='HYPCLASS',hue='GENDER',data=df, palette='RdBu_r')

ax.set_xticklabels(labels)

ax.set_title('Hypertensive Population by Gender: 1-Male | 2-Female')
labels = ['0 - No Hypertension', '1 - Yes Hypertension']

ax = sns.countplot(x='HYPCLASS',hue='RACE',data=df)

ax.set_xticklabels(labels)

ax.set_title('Hypertensive Population by Race')

print(" 1: Mexican American\n","2: Other Hispanic\n","3: Non-Hispanic White\n","4: Non-Hispanic Black\n","5: Other Race - Including Multi-Racial\n")
labels = ['1-Mexican American','2-Other Hispanic','3-Non-Hispanic White','4-Non-Hispanic Black','5-Other Race - Including Multi-Racial']

ax = sns.factorplot(x='RACE',y='AGERANGE',col='GENDER', data = df,hue='HYPCLASS',kind='bar')

ax.set_xticklabels(labels,rotation=90)
plt.figure(figsize=(30,10))

labels = ['1-Mexican American','2-Other Hispanic','3-Non-Hispanic White','4-Non-Hispanic Black','5-Other Race - Including Multi-Racial']

ax = sns.factorplot(x='RACE',y='BMIRANGE',col='GENDER',data = df,hue='HYPCLASS',kind='bar')

ax.set_xticklabels(labels,rotation=90)
plt.figure(figsize = (16,5))

sns.heatmap(df.drop('SEQN',axis=1).corr(),annot=True)
# Visualizing the trend of the Hypclass to every variable in a plane (where the forces acting ). 

def rad_viz(df,labels):

    fig = radviz(df, labels, color=sns.color_palette())

    plt.show()



rad_viz(df.drop('SEQN',axis=1),'HYPCLASS') # Specify which column contains the labels
# Parallel coordinates allows to see clusters in data. Points that tend to cluster will appear closer together  

def pcoord_viz(df, labels):

    fig = parallel_coordinates(df, labels, color=sns.color_palette())

    plt.xticks(rotation=60)

    plt.show()



pcoord_viz(df.drop('SEQN',axis=1),'HYPCLASS') # Specify which column contains the labels
df.drop('SEQN',axis=1,inplace=True)



X = df.drop('HYPCLASS',axis=1)

y = df['HYPCLASS']
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression





logmodel = LogisticRegression()



# The "accuracy" scoring is proportional to the number of correct

# classifications

rfecv = RFECV(estimator=logmodel, step=1, cv=StratifiedKFold(),

              scoring='roc_auc')

rfecv.fit(X, y)



print("Optimal number of features : %d" % rfecv.n_features_)



print("Optimal number of features : %d" % rfecv.n_features_)

print("Selected features mask: ")

print(rfecv.support_)

print(X.columns)

print("Total features:",rfecv.support_.size)





# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# Compute chi-squared stats between each non-negative feature and class

from sklearn.feature_selection import chi2



scores, pvalues = chi2(X, y)

pvalues=["{0:.7f}".format(x)for x in pvalues]



p_values = pd.concat([pd.DataFrame(X.columns,columns=['Feature']),pd.DataFrame(pvalues,columns=['p-value']),pd.DataFrame(scores,columns=['Score'])], axis = 1)

print(p_values)
# We will read the dataset again to prevent any undesire change from the preivious DEA and Feature selection.

df = pd.read_csv("../input/NHANES_CAT_NEW_130.CSV")
#We eliminate the sequence number

df.drop('SEQN',axis=1,inplace=True)
#Visualization of the columns before the creation of the dummy variables

df.describe().transpose()
#Create dummy Variables and eliminate the first column to prevent Multicollinearity. This is simply redundancy in the information contained in predictor variables.

trainDfDummies = pd.get_dummies(df, columns=['GENDER', 'AGERANGE', 'RACE', 'BMIRANGE','KIDNEY','SMOKE','DIABETES'], drop_first=True)
trainDfDummies.describe().transpose()
X = trainDfDummies.drop(['HYPCLASS'],axis=1)

y = trainDfDummies['HYPCLASS']
# Independent Variables

X.head()
# Dependent variable

y.head()
# Split our dataset in 70% for training the model and 30% to evaluate the model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



#Default Logistic regression Model

logmodel = LogisticRegression()



# Param grid will receive all the hyper parameters will be evaluated with the LR model and the scoring metric 

# is "roc_auc"

param_grid = {'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'], 'C':[0.1,1, 10, 100, 1000],'class_weight':['','balanced'],'max_iter':[100,500,1000,5000,8000,10000]}

grid = GridSearchCV(logmodel,param_grid,refit=True,verbose=3,scoring='roc_auc')

grid.fit(X_train,y_train)
# get the best parameters

grid.best_params_
# Function to run the Logistic regression model with the best parameters obtained 

#in the previous step with the GridSearch algorithm

def modelevaluation(X,y,model):

    global prob

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



    from collections import Counter

    print('# of real cases in the Test data (P = 1) and (N = 0)',sorted(Counter(y_test).items()))

    

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    

    prob = model.predict_proba(X_test)

    

    #Uses jaccard_similarity_score

    # J(y_true,y_predict) = (y_true intersect y_predict) / (y_true Union y_predict)

    print('accuracy_score: ',accuracy_score(y_test,predictions))

    print('accuracy_score (number of correctly classified samples): ',accuracy_score(y_test,predictions,normalize=False))

    

    print('Zero one loss: ',zero_one_loss(y_test,predictions))

    print('Zero one loss (number of imperfectly predicted subsets): ',zero_one_loss(y_test,predictions,normalize=False))

    

    print(confusion_matrix(y_test,predictions))

    

    tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()

    print("TN:",tn)

    print("FN:",fn)

    print("TP:",tp)

    print("FP:",fp)

    

    print(classification_report(y_test,predictions))

   

    #ROC curve is insensitive to imbalanced classes

    fpr, tpr, thresholds = roc_curve(y_test,predictions,pos_label=1)

    #fpr, tpr, thresholds = roc_curve(y,predictions,pos_label=1)

    print("FPR: ",fpr)

    print("TPR: ",tpr)

    print("AUC: ",auc(fpr, tpr))

    

    #Compute Area Under the Curve (AUC) from prediction scores

    #the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one

    print("ROC_AUC_SCORE: ",roc_auc_score(y_test,predictions))

    

    #Plot of the AUC

    plt.figure()

    lw = 2

    plt.plot(fpr,tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))



    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.show()
from sklearn.metrics import classification_report, confusion_matrix,auc,roc_curve,accuracy_score,roc_auc_score,zero_one_loss



X = trainDfDummies.drop(['HYPCLASS'],axis=1)

y = trainDfDummies['HYPCLASS']



# Best params: {'C': 0.1, 'class_weight': '', 'max_iter': 100, 'solver': 'sag'}

logmodel = LogisticRegression(C=0.1,class_weight='',max_iter=100,solver='sag',warm_start=True)



modelevaluation(X,y,logmodel)
#Print the intercept and the coefficients

print("Intercept:")

print(logmodel.intercept_)



print("\nCoefficients:")

print(logmodel.coef_[0].transpose())



#Calculate the Odd Ratios for clinical interpretation

print("\nOdd Ratios:")

df_oodsr = pd.DataFrame({'Features':X.columns,'coefficient':logmodel.coef_[0],'Odds Ratio':np.exp(logmodel.coef_[0])},columns=['Features','coefficient','Odds Ratio'])



print(df_oodsr)



X[(X.GENDER_2==0) & 

(X.AGERANGE_2==1) &

(X.AGERANGE_3==0) &

(X.AGERANGE_4==0) &

(X.AGERANGE_5==0) &

(X.AGERANGE_6==0) &

(X.RACE_2==0) &

(X.RACE_3==1) &

(X.RACE_4==0) &

(X.RACE_5==0) &

(X.BMIRANGE_2==0) &

(X.BMIRANGE_3==1) &

(X.BMIRANGE_4==0) &

(X.KIDNEY_2==1) &

(X.SMOKE_2==1) &

(X.DIABETES_2==0) &

(X.DIABETES_3==1) ]
X.iloc[6439,:]
# The calculated probability is:

print(logmodel.predict_proba(X)[6439][1])