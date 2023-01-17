# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

from wordcloud import STOPWORDS

import nltk 

import matplotlib.pyplot as plt

import seaborn as sns

import scikitplot as skplt

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/spam.csv',encoding='latin-1')

df=df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])

df.head(5)
#spliting the labels and the data seperately

df_labels=df['v1']

df_labels.head(5)
#stopwords

stopwords= STOPWORDS

stopwords = list(stopwords)

STOPWORDS = nltk.corpus.stopwords.words('english')

stopwords=stopwords+STOPWORDS



ham_dataset=df[df.v1 == 'ham']

spam_dataset=df[df.v1 == 'spam']

ham_words =''

spam_words=''



for words in ham_dataset.v2:

    txt = words.lower()

    tokens = nltk.word_tokenize(txt)

    for word in tokens:

        ham_words = ham_words + word +" "

for words in spam_dataset.v2:

    txt = words.lower()

    tokens = nltk.word_tokenize(txt)

    for word in tokens:

        spam_words = spam_words + word +" "

def gen_wordcloud(wordcloud):

    plt.figure( figsize=(10,8), facecolor='k')

    plt.imshow(wordcloud)

    plt.tight_layout(pad=0)

    plt.axis('off')

    plt.show()

#wordcloud =WordCloud(background_color='white',width=500, height=500,stopwords=stopwords,max_words=500,max_font_size=50,random_state=42).generate(ham_words)

#gen_wordcloud(wordcloud)

#wordcloud =WordCloud(background_color='white',width=500, height=500,stopwords=stopwords,max_words=500,max_font_size=50,random_state=42).generate(spam_words)

#gen_wordcloud(wordcloud)

#plotting ham and spam data % in pie chart 

count_Class=pd.value_counts(df.v1, sort= True)



# Data to plot

labels = 'Ham', 'Spam'

sizes = [count_Class[0], count_Class[1]]

colors = ['gold', 'yellowgreen'] # 'lightcoral', 'lightskyblue'

explode = (0.1, 0.1)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
#splitting the test and train data 

trainset, testset, trainlabel, testlabel = train_test_split(df, df_labels, test_size=0.33, random_state=42)

print(trainset.shape[1])

print(testset.shape)

print("The Trainset consists of {} records and {} features".format(trainset.shape[0],trainset.shape[1]))

print("The Testset consists of {} records and {} features".format(testset.shape[0],trainset.shape[1]))
#extracting n-grams from the text data

countvect= CountVectorizer(ngram_range=(2,2),)

x_counts = countvect.fit(trainset.v2)

#preparing for training set

x_train_df =countvect.transform(trainset.v2)

#preparing for test set

x_test_df = countvect.transform(testset.v2)

#Creating the model using naive bayes

clf=MultinomialNB()

clf.fit(x_train_df,trainset.v1)

predicted_values = clf.predict(x_test_df)

predictions=dict()

acurracy = accuracy_score(testset.v1,predicted_values)

predictions['Naive Bayes']=acurracy*100

confusionmatrix = confusion_matrix(testset.v1,predicted_values)

print("The accuracy of the model is {}%".format(acurracy*100 ))

print(confusionmatrix)

skplt.metrics.plot_confusion_matrix(testset.v1,predicted_values, normalize=True)

plt.show()
#using K-Nearest to predict 

KNN=KNeighborsClassifier()

KNN.fit(x_train_df,trainset.v1)

predictedValues = KNN.predict(x_test_df)

print(predictedValues)

acurracy_KNN = accuracy_score(testset.v1,predictedValues)

predictions['KNN']=acurracy_KNN*100

print("The accuracy of the model is {}%".format(acurracy_KNN*100 ))

confusion_matrix_KNN = confusion_matrix(testset.v1,predictedValues)

print(confusion_matrix_KNN)

skplt.metrics.plot_confusion_matrix(testset.v1,predictedValues, normalize=True)

plt.show()
#using Decision Tree Classifier to model 

DT=DecisionTreeClassifier()

DT.fit(x_train_df,trainset.v1)

predicted_values_DT = DT.predict(x_test_df)

print(predicted_values_DT)

acurracy_DT = accuracy_score(testset.v1,predicted_values_DT)

predictions['DecisionTree']=acurracy_DT*100

print("The accuracy of the model is {}%".format(acurracy_DT*100 ))

#print(testset.v1)

confusion_matrix_DT = confusion_matrix(testset.v1,predicted_values_DT)

print(confusion_matrix_DT)

skplt.metrics.plot_confusion_matrix(testset.v1,predicted_values_DT, normalize=True)

plt.show()
#Training the model using SVM Classifier

SVM = svm.SVC()

SVM.fit(x_train_df,trainset.v1)

predicted_values_svm=SVM.predict(x_test_df)

print(predicted_values_svm)

acurracy_SVM = accuracy_score(testset.v1,predicted_values_svm)

predictions['SVM']=acurracy_SVM*100

print("The accuracy of the model is {}%".format(acurracy_SVM*100 ))

confusion_matrix_SVM = confusion_matrix(testset.v1,predicted_values_svm)

print(confusion_matrix_SVM)

skplt.metrics.plot_confusion_matrix(testset.v1,predicted_values_svm, normalize=True)

plt.show()
#Predicting using RandomForest

RF = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)

RF.fit(x_train_df,trainset.v1)

predicted_values_RF = RF.predict(x_test_df)

print(predicted_values_RF)

acurracy_RF = accuracy_score(testset.v1,predicted_values_RF)

predictions['RandomForest']=acurracy_RF*100

print("The accuracy of the model is {}%".format(acurracy_RF*100 ))

#print(testset.v1)

confusion_matrix_RF = confusion_matrix(testset.v1,predicted_values_RF)

print(confusion_matrix_RF)

skplt.metrics.plot_confusion_matrix(testset.v1,predicted_values_RF, normalize=True)

plt.show()
#modelling using Multi-layer perceptron

MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

MLP.fit(x_train_df,trainset.v1)

predicted_values_MLP = MLP.predict(x_test_df)

print('Predicted Values {}'.format(predicted_values_MLP))

accuracy_MLP = accuracy_score(testset.v1,predicted_values_MLP)

predictions['Neural Networks']=accuracy_MLP*100

print("The accuracy of the model is {}".format(accuracy_MLP*100))

skplt.metrics.plot_confusion_matrix(testset.v1,predicted_values_MLP, normalize=True)

plt.show()
#using XGBoost to model and predict

XGB = XGBClassifier()

XGB.fit(x_train_df,trainset.v1)

predicted_values_XGB=XGB.predict(x_test_df)

print(predicted_values_XGB)

accuracy_XGB = accuracy_score(testset.v1,predicted_values_XGB)

predictions['XGBoost']=accuracy_XGB*100

print("The accuracy of the model is {}".format(accuracy_XGB*100))

skplt.metrics.plot_confusion_matrix(testset.v1,predicted_values_XGB, normalize=True)

plt.show()
fig, (ax1) = plt.subplots(ncols=1, sharey=True,figsize=(15,5))

df=pd.DataFrame(list(predictions.items()),columns=['Algorithms','Percentage'])

display(df)

sns.pointplot(x="Algorithms", y="Percentage", data=df,ax=ax1);
#pr, tpr, thresholds = roc_curve(testset.v1,predicted_values_XGB, pos_label=2)

test_prediction = testset.v1.tolist()

predicted_values = predicted_values_XGB.tolist()

test_prediction = [1 if pred=="spam" else 0 for pred in test_prediction]

predicted_values = [1 if pred=="spam" else 0 for pred in predicted_values]

fpr, tpr, thresholds = roc_curve(test_prediction,predicted_values)

roc_auc = auc(fpr, tpr)

print("The ROC Accuracy is {}".format(roc_auc))

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()