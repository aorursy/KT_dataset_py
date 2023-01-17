import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from collections import Counter
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from scipy import stats
from scipy.stats import norm, skew,kurtosis
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train_isnull = pd.DataFrame(round((train.isnull().sum().sort_values(ascending=False)/train.shape[0])*100,1)).reset_index()
train_isnull.columns = ['Columns', '% of Missing Data']
train_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
cm = sns.light_palette("Red", as_cmap=True)
train_isnull = train_isnull.style.background_gradient(cmap=cm)
train_isnull
test_isnull = pd.DataFrame(round((test.isnull().sum().sort_values(ascending=False)/test.shape[0])*100,1)).reset_index()
test_isnull.columns = ['Columns', '% of Missing Data']
test_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
cm = sns.light_palette("blue", as_cmap=True)
test_isnull = test_isnull.style.background_gradient(cmap=cm)
test_isnull
#if you run the code below, we observe that about 70% has Embarked 'S'.
#We will Fill Embarked nan values of dataset set with 'S' 
train.Embarked.value_counts()/train.shape[0]
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
#Fill Fare missing values with the median value
group_dataset = [train,test]
for df in group_dataset:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # Fare feature cleaning
    df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

#Plot Age distribution of the Train dataset
f, ax = plt.subplots(figsize = (10,5))
ax = sns.kdeplot(train.Fare,shade=True)
ax.axvline(train.Fare.mean(),color ='g',linestyle = '--')
ax.legend(['Fare','Fare mean'],fontsize=12)
ax.set_xlabel('Fare',fontsize=12,color='black')
ax.set_title('Fare Distribution',color='black',fontsize=14)
y_axis = ax.axes.get_yaxis().set_visible(False) # turn off the y axis label
sns.despine(left=True)
#Name feature cleaning
import re
group_dataset = [train,test]
for df in group_dataset:
    df['Title'] = df.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    # Convert to categorical values Title 
    df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 ,
                                         "Mme":1, "Mlle":1, "Mrs":2, "Mr":3, "Rare":4})
f,ax = plt.subplots(figsize=(10,5))
#Plotting the result of Train dataset
sns.countplot(x='Title', data=train, palette="hls",orient='v',ax=ax,order = [3,1,2,0,4])
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.2, i.get_height()+5, \
            str(round((i.get_height()/train.Title.shape[0])*100))+'%', fontsize=12,
                color='black') 
ax.set_xlabel("Title", fontsize=12)
ax.set_xticklabels(['Mr','Mlle-Miss-Ms-Mme','Mrs','Master','Rare'])
ax.set_title("Title Frequency", fontsize=14)
x_axis = ax.axes.get_yaxis().set_visible(False)
sns.despine(left=True)
plt.show()
# Age feature cleaning ...Again this is the work of Yassine Ghouzam....
group_dataset = [train,test]
for df in group_dataset:
    nan_age_index = list(df["Age"][df["Age"].isnull()].index)
    for i in nan_age_index :
        age_med = df["Age"].median()
        age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & 
                                   (df['Parch'] == df.iloc[i]["Parch"]) & 
                                   (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            df['Age'].iloc[i] = age_pred
        else :
            df['Age'].iloc[i] = age_med
#Plot Age distribution of Train dataset
f, ax = plt.subplots(figsize = (10,5))
ax = sns.kdeplot(train.Age,shade=True)
ax.axvline(train.Age.mean(),color ='r',linestyle = '--')
ax.legend(['Age','Age mean'],fontsize=12)
ax.set_xlabel('Age',fontsize=12,color='black')
ax.set_title('Age Distribution',color='black',fontsize=14)
y_axis = ax.axes.get_yaxis().set_visible(False) # turn off the y axis label
sns.despine(left=True)
# Family size feature cleaning and transform 
group_dataset = [train,test]
for df in group_dataset:
    df["family_size"] = df["SibSp"] + train["Parch"] + 1
    family_size_num = (0,1,3,5,7)
    family_size_label = ['Single','small','medium','large']
    df['family_size'] = pd.cut(df['family_size'],family_size_num,labels=family_size_label)

f,ax = plt.subplots(figsize=(10,5))
#Plotting the result
sns.countplot(x='family_size', data=train, palette="hls",orient='v',ax=ax)
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.3, i.get_height()+5, \
            str(round((i.get_height()/train.Title.shape[0])*100))+'%', fontsize=12,
                color='black') 
ax.set_xlabel("Family Size", fontsize=12)
ax.set_title("Family Size Frequency", fontsize=14)
x_axis = ax.axes.get_yaxis().set_visible(False)
sns.despine(left=True)
plt.show()
#Drop unecessary columns
col_drop = ['Cabin','Name','Ticket']
group_dataset = [train,test]
for df in group_dataset:
    df.drop(columns=col_drop,axis=1,inplace=True)
colormap = plt.cm.RdBu
f, ax = plt.subplots(figsize=(14,10))
sns.heatmap(train.corr(),cmap= colormap,annot=True,ax=ax,annot_kws ={'fontsize':12})
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':13}
ax.tick_params(**kwargs)
ax.tick_params(**kwargs,axis='x')
plt.title ('Pearson Correlation Matrix', color = 'black',fontsize=18)
plt.tight_layout()
plt.show()
col_dummies = ['Embarked','Sex','family_size','Title','Pclass']

train = pd.get_dummies(train,columns=col_dummies,drop_first=True)
test = pd.get_dummies(test,columns=col_dummies,drop_first=True)
train.dropna(inplace=True) 
train.Survived =train.Survived.astype('int')
train.drop('PassengerId',axis=1,inplace=True)
X = train.drop('Survived',axis= 1)
y = train['Survived']

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Standardizing the dataset to normalize them
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
def model_select(classifier,kfold_n):
    cv_result = []
    cv_means = []
    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits= kfold_n)
    cv_result.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
    cv_means.append(np.mean(cv_result))
    return cv_means
# 
model_type = [LogisticRegression(),SVC(),KNeighborsClassifier(),GaussianNB(),RandomForestClassifier(),
              AdaBoostClassifier(),GradientBoostingClassifier(),DecisionTreeClassifier(),ExtraTreesClassifier(),
             MLPClassifier(),LinearDiscriminantAnalysis()]
model_score = [model_select(i,5) for i in model_type]

classifier = ['Logistic Regression','SVC','KNeighbors','Naive Bayes','Random Forest', 
             'AdaBoost','Gradient Boosting','Decision Tree','Extra Trees','Multiple Layer Perceptron',
              'Linear Discriminant']
# XGB classifier fitting
gbm = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05,silent=False,n_jobs=4)
gbm.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_test, y_test)], verbose=False)
y_pred = gbm.predict(X_test)
## Accuracy Score
acc_score_1 = accuracy_score(y_pred,y_test)
ml_model = pd.DataFrame(model_score,classifier).reset_index()
ml_model.columns=['Model','acc_score']
ml_model.loc[11] = ['xgboost',acc_score_1]
ml_model.sort_values('acc_score',ascending = False,inplace=True)
ml_model.reset_index(drop=True,inplace = True)
ml_model
f, ax = plt.subplots(figsize=(10,8))
sns.barplot('acc_score','Model',data=ml_model, ax=ax,palette='RdBu_r',edgecolor=".2")
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.01, i.get_y()+.55, \
        str(round((i.get_width()), 2)), fontsize=12, color='black') 
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title('Model & Accuracy Score',fontsize=16)
sns.despine(bottom=True)
plt.show()
#Initialising the ANN
def keras_model(kernel_init,act_func,optimizer):
    classifier = Sequential()
    #kernel_init = "lecun_normal"
    #act_func = 'selu'
    ##Adding the input layer
    classifier.add(Dense(9,input_shape=(X_train.shape[1],),kernel_initializer=kernel_init,activation=act_func))
    ## Adding the Second layer
    classifier.add(Dense(9,kernel_initializer=kernel_init,activation=act_func))
    ## Adding the Third layer
    classifier.add(Dense(9,kernel_initializer=kernel_init,activation=act_func))
    ## Adding the Forth layer
    classifier.add(Dense(9,kernel_initializer=kernel_init,activation=act_func))
    ## Adding the Fifth layer
    classifier.add(Dense(9,kernel_initializer=kernel_init,activation=act_func))
    ## Adding Output Layer
    classifier.add(Dense(1,kernel_initializer=kernel_init,activation='sigmoid'))

    ## Compiling the ANN
    classifier.compile(optimizer=optimizer,loss = 'binary_crossentropy',metrics=['accuracy'])
    return classifier
#Fitting the model
clf = keras_model('lecun_normal','selu','adam')
clf.fit(X_train,y_train,batch_size=10,epochs=150,validation_data=(X_test,y_test))
y_pred_2 = clf.predict_classes(X_test,batch_size=32)
y_pred = (y_pred_2 > 0.5)
# This function will return 1 (Survive) or 0 (Not survive)..This will help during the evalution on the test data during submission
def y_pred_val (input):
    y_pred_1 = []
    for i in list(input):
        if i == True:
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)
    return y_pred_1
accuracy_score(y_test,y_pred_2)
test_1 = test.drop('PassengerId',axis=1)
test_1 = sc.transform(test_1)
y_test_pred = clf.predict_classes(test_1,batch_size=32)
y_test_pred = (y_test_pred > 0.5)
y_test_pred_2 = y_pred_val(y_test_pred)
submit = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": y_test_pred_2})
submit.to_csv("../working/submit.csv", index=False)