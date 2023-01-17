#Importing Necessary Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk



from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

#Reading data frame

df_app=pd.read_csv('../input/googleplaystore.csv')

df_app.head(5)
print("The Google play store dataset contains %d rows and %d columns." %(df_app.shape[0],df_app.shape[1]))
#Dropping Data frame which has NAN values

df_app=df_app.dropna()

print("The Google play store dataset contains %d rows and %d columns after dropping NAN." %(df_app.shape[0],df_app.shape[1]))
#Checking if there are any duplicates rows present in dataset that has same App

# False= No duplicate

# True=Duplicate

df_app.duplicated(subset='App').value_counts()
#Dropping the duplicates

df_app=df_app.drop_duplicates(subset='App')
print("The Google play store dataset contains %d rows and %d columns after dropping NAN and duplicates." %(df_app.shape[0],df_app.shape[1]))
#Checking the data types of dataset

df_app.dtypes
#Converting the Installs column into integer

df_app['Installs']=df_app['Installs'].apply(lambda a:a.split('+')[0])   #Removes '+' from Installs

se=df_app['Installs'].apply(lambda a:a.split(','))                      #Removes ',' from Installs 



def add_list(x):

    sum=' '

    for i in range(0,len(x)):

        sum+=x[i]

    return int(sum)  



df_app['Installs']=se.apply(lambda a:add_list(a))                      #Convert str to int values 

df_app.head(5)
#Removing Currency symbol from the Price and making it float

def remove_curr(x):

    if x !='0':

        x=x.split('$')[1]

    return float(x)   



df_app['Price']=df_app['Price'].apply(lambda a:remove_curr(a))  #Removes '$' from Price

df_app.head(5)
#Checking the number of apps that available based on type: Free v/s Paid

df_app['Type'].value_counts()
#Number of free and paid Apps available

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)

sns.countplot(x='Type',data=df_app)

plt.title("Number of Apps Available: Free v/s Paid")



#Most installed apps based on Category

plt.subplot(1,2,2)

sns.barplot(x='Type',y='Installs',data=df_app,ci=None)

plt.title("Number of Apps installed: Free v/s Paid")

plt.tight_layout()
#Checking the number of Apps available on playstore based on category

plt.figure(figsize=(12,12))

sns.countplot(y='Category',data=df_app)

plt.title("Number of Apps available based on Category")
#Most installed apps based on Category

plt.figure(figsize=(12,12))

sns.barplot(x='Installs',y='Category',data=df_app,ci=None)

plt.title("Number of Apps installed based on Category")
#Apps available based on Content rating

plt.figure(figsize=(10,10))

sns.countplot(x='Content Rating',data=df_app,)

plt.xticks(rotation=45)

plt.title("Number of Apps available based on Content rating")
#Apps installed based on Content rating

plt.figure(figsize=(10,10))

sns.barplot(x='Content Rating',y='Installs',data=df_app,ci=None)

plt.xticks(rotation=45)

plt.title("Number of Apps installed based on Content rating")
#Android Version of the most available apps

plt.figure(figsize=(15,15))

sns.countplot(y='Android Ver',data=df_app)

plt.title("Android Version's available")
#Android  version of most installed apps

plt.figure(figsize=(15,15))

sns.barplot(x='Installs',y='Android Ver',data=df_app,ci=None)

plt.title("Android Versions of installed Apps")
#Ratings of Apps and the number of installed

plt.figure(figsize=(15,15))

sns.barplot(y='Installs',x='Rating',data=df_app,ci=None)

plt.xticks(rotation=45)

plt.title("Number of Apps and ratings ")
#Most download  Paid apps

df_type=df_app[df_app['Type']=='Paid']

df_type.sort_values(by='Installs',ascending=False)['App'].head(20)
#Top 20 apps that are installed most in Category Communication

df_com=df_app[df_app['Category']=='COMMUNICATION']

df_com.sort_values(by='Installs',ascending=False)['App'].head(20)
#Top 20 apps that are installed most in Category Social

df_soc=df_app[df_app['Category']=='SOCIAL']

df_soc.sort_values(by='Installs',ascending=False)['App'].head(20)
#Top 20 apps that are installed most in Category Video Player

df_vp=df_app[df_app['Category']=='VIDEO_PLAYERS']

df_vp.sort_values(by='Installs',ascending=False)['App'].head(20)
#Reading CSV file that contains reviews for Apps

df=pd.read_csv('../input/googleplaystore_user_reviews.csv')

df.head(10)
#Size of data frame

df.shape
#Class labels available

df['Sentiment'].value_counts()
#Checking if there are any missing values

sns.heatmap(df.isna())
#Dropping the missing values from the data frame

df=df.dropna()

df.shape
#Reviews and Labels

reviews=df['Translated_Review']

labels=df['Sentiment']
# User-defined function

def cleanhtml(sentence): #function to clean the word of any html-tags

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return cleaned

#Stop words and Lemmatizer

Stop=set(stopwords.words('english'))

WrdLem=WordNetLemmatizer()

print(Stop)
#Cleaning the reviews(removing html tags,punctuation,Lemmatizations)

Cleaned_sent=[]

for sent in reviews:

    r1=[]

    sent=cleanhtml(sent)

    sent=cleanpunc(sent)

    sent=sent.lower()

    for  word in sent.split():

        if ((word.isalpha()) & (len(word)>2)):

            if word not in Stop:

                w=WrdLem.lemmatize(word)

                r1.append(w)

            else:

                continue

        else:

            continue

    str1 = (" ".join(r1))        

     

    Cleaned_sent.append(str1)



df['Cleaned_text']=Cleaned_sent

df.head(5)    
#Defining some user defined function



def plot_cm_rates(y_test, Y_pred):



    #Plotting Confusion matrix

    x=confusion_matrix(y_test,Y_pred)

    cm_df=pd.DataFrame(x,index=['Negative','Neutral','Positive'],columns=['Negative','Neutral','Positive'])



    sns.set(font_scale=1,color_codes=True,palette='deep')

    sns.heatmap(cm_df,annot=True,annot_kws={"size":16},fmt='d',cmap="YlGnBu")

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.title("Confusion Matrix ")





def plot_miss_error(cv_scores,hyperparam):

    

    # changing to misclassification error

    MSE = [1 - x for x in cv_scores]



    # determining best k

    optimal_k = hyperparam[MSE.index(min(MSE))]

    print('\nThe optimal value of hyper parameter is %f.' % optimal_k)

    

    # plot misclassification error vs K 

    plt.figure(figsize=(8,8))

    plt.plot(hyperparam, MSE)



    for xy in zip(hyperparam, np.round(MSE,3)):

        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')



    plt.xlabel('Values of Hyperparameter')

    plt.ylabel('Misclassification Error')

    plt.title("Missclassification error v/s Hyperparameter")

    plt.show()

    

    return optimal_k





def train_test_accuracy(Classifier,X_train,y_train,X_test,y_test):

    

    #Train Model Fitting

    Classifier.fit(X_train,y_train)

    pred_train = Classifier.predict(X_train)

    

    #Train Accuracy

    train_acc = accuracy_score(y_train, pred_train, normalize=True) * float(100)

    

    #Test Accuracy

    pred_test = Classifier.predict(X_test)

    test_acc = accuracy_score(y_test, pred_test, normalize=True) * float(100)

    

    #Printing train and test accuracy

    print('\n****Train accuracy = %f%%' % (train_acc))

    print('\n****Test accuracy =  %f%%' % (test_acc))

    

    #plotting Confusion matrix

    plot_cm_rates(y_test,pred_test)
#Splitting the data into train and test

X_train,X_test,y_train,y_test=train_test_split(df['Cleaned_text'].values,labels,test_size=0.3,random_state=0)
#Size of training and test data

print("The number of data points used in  training model is %d "%(X_train.shape[0]))

print("The number of data points used in testing model is %d" %(X_test.shape[0]))
#Train Vector

bow=CountVectorizer()

X_train_bow=bow.fit_transform(X_train)



#Test vector

X_test_bow=bow.transform(X_test)
#Hyper-Parameter 

C=[10**-4,10**-2,10**0,10**2,10**4]
#Hyper Parameter tunning

cv_scores=[]

for c in C:

    LR=LogisticRegression(C=c,solver='newton-cg',multi_class='ovr')

    scores=cross_val_score(LR,X_train_bow,y_train,cv=3,scoring='accuracy')

    cv_scores.append(scores.mean())

    
#Plotting Misclassification error

optimal=plot_miss_error(cv_scores,C)
#Model Fitting based on  optimal value and Plotting Confusion Matrix

classifier1=LogisticRegression(C=optimal,solver='newton-cg',multi_class='ovr')



train_test_accuracy(classifier1,X_train_bow,y_train,X_test_bow,y_test)
#Hyper parameter 

lr=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#Hyper paramete tunning

cv_scores=[]

for l in lr:

    XGB=XGBClassifier(learning_rate=l)

    scores=cross_val_score(XGB,X_train_bow,y_train,cv=3,scoring='accuracy')

    cv_scores.append(scores.mean())

    
#Plotting misclassification error

optimal=plot_miss_error(cv_scores,lr)
#Model Fitting based on  optimal value and plotting the confusion matrix

classifier1=XGBClassifier(learning_rate=optimal)

train_test_accuracy(classifier1,X_train_bow,y_train,X_test_bow,y_test)
#Train vector

tfidf=TfidfVectorizer()

X_train_tfidf=tfidf.fit_transform(X_train)



#Test Vector

X_test_tfidf=tfidf.transform(X_test)
#Hyper Parameter tunning

cv_scores=[]

for c in C:

    LR=LogisticRegression(C=c,multi_class='ovr',solver='newton-cg')

    scores=cross_val_score(LR,X_train_tfidf,y_train,cv=3,scoring='accuracy')

    cv_scores.append(scores.mean())

    
#Plotting misclassification error

optimal=plot_miss_error(cv_scores,C)
#Model Fitting based on  optimal value and Confusion matrix

classifier1=LogisticRegression(C=optimal,multi_class='ovr',solver='newton-cg')



train_test_accuracy(classifier1,X_train_tfidf,y_train,X_test_tfidf,y_test)
#Hyper parameter

cv_scores=[]

for l in lr:

    XGB=XGBClassifier(learning_rate=l)

    scores=cross_val_score(XGB,X_train_tfidf,y_train,cv=3,scoring='accuracy')

    cv_scores.append(scores.mean())

    
#Plotting misclassification error

optimal=plot_miss_error(cv_scores,lr)
#Model Fitting based on  optimal value and Confusion Matrix

classifier1=XGBClassifier(learning_rate=optimal)



train_test_accuracy(classifier1,X_train_tfidf,y_train,X_test_tfidf,y_test)