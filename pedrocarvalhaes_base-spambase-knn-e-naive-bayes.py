import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
train_data = pd.read_csv("../input/spambase/train_data(spam).csv", sep=r'\s*,\s*',
        engine='python',
        na_values="")
train_data.shape
train_data
train_data.isnull().sum().sum()
test_data = pd.read_csv("../input/spambase/test_featuress(spam).csv", sep=r'\s*,\s*',
        engine='python',
        na_values="")
test_data.shape
test_data

train_dataNospam = train_data.query('ham == 1')
train_dataSpam = train_data.query('ham == 0')
train_dataNospam.mean()
train_dataSpam.mean()
train_dataNospam = train_dataNospam.drop(columns = 'capital_run_length_average')
train_dataNospam = train_dataNospam.drop(columns = 'capital_run_length_longest')
train_dataNospam = train_dataNospam.drop(columns = 'capital_run_length_total')

train_dataSpam = train_dataSpam.drop(columns = 'capital_run_length_average')
train_dataSpam = train_dataSpam.drop(columns = 'capital_run_length_longest')
train_dataSpam = train_dataSpam.drop(columns = 'capital_run_length_total')
mean_nospam = pd.DataFrame(data = train_dataNospam.mean())
mean_spam = pd.DataFrame(data = train_dataSpam.mean())
list(train_dataNospam)
features = list(train_dataNospam)[0:len(list(train_dataNospam))-1]

N = len(mean_nospam)
nospam_means=[]
spam_means=[]

for i in range(len(mean_nospam)):
    nospam_means.append(mean_nospam[0][i])
    spam_means.append(mean_spam[0][i])

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, nospam_means, width, color='r')

rects2 = ax.bar(ind + width, spam_means, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean')
ax.set_xlabel('Features')
ax.set_xticks(ind + width / 10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(train_dataNospam))

ax.legend((rects1[0], rects2[0]), ('Nospam', 'Span'), prop={'size':20})

plt.rcParams['figure.figsize'] = [30,10]
ax.tick_params(labelsize=20)


N = len(mean_nospam)
sub=[]

for i in range(len(mean_nospam)):
    sub.append(abs(mean_nospam[0][i] - mean_spam[0][i]))
    

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, sub, width, color='r')


# add some text for labels, title and axes ticks
ax.set_ylabel('Diferença')
ax.set_xlabel('Features')
ax.set_xticks(ind + width / 10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(train_dataNospam))

ax.legend(['Diferenças das Médias '], prop={'size':20})
plt.rcParams['figure.figsize'] = [30,10]
ax.tick_params(labelsize=20)
N = len(mean_nospam)
sub=[]

for i in range(len(mean_nospam)):
    sub.append(abs(mean_nospam[0][i] - mean_spam[0][i])/mean_nospam[0][i])
    

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, sub, width, color='r')


# add some text for labels, title and axes ticks
ax.set_ylabel('Diferença')
ax.set_xlabel('Features')
ax.set_xticks(ind + width / 10)
plt.xticks(rotation=90)

ax.set_xticklabels(list(train_dataNospam))

ax.legend(['Diferenças das Médias divididas pela média do Nospam'], prop={'size':20})
plt.rcParams['figure.figsize'] = [30,10]
ax.tick_params(labelsize=20)
train_data['word_freq_our'].mean()*train_data['word_freq_our'].std()
Xtrain_data = train_data[['word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order',
                          'word_freq_receive','word_freq_addresses','word_freq_free','word_freq_business',
                          'word_freq_email','word_freq_credit','word_freq_font','word_freq_000','word_freq_money','char_freq_!',
                          'char_freq_$', 'char_freq_#']]
Xtrain_data.shape
Xtrain_data.head()
Ytrain_data = train_data.ham
Xtest_data = test_data[['word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order',
                          'word_freq_receive','word_freq_addresses','word_freq_free','word_freq_business',
                          'word_freq_email','word_freq_credit','word_freq_font','word_freq_000','word_freq_money','char_freq_!',
                          'char_freq_$', 'char_freq_#']]
Xtest_data.shape
Xtest_data.head()
k=1
v=[]
K=[]
while k<=20:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain_data, Ytrain_data)
    scores = cross_val_score(knn, Xtrain_data, Ytrain_data, cv=10)
    x=np.mean(scores)
    print(x)
    v.append(x)
    K.append(k)
    k+=1
print(np.amax(v),np.argmax(v), K[np.argmax(v)])
vetor = pd.DataFrame(data = v)
plt.plot(K, vetor)
plt.tick_params(labelsize=20)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain_data,Ytrain_data)
scores = cross_val_score(knn, Xtrain_data, Ytrain_data, cv=10)
scores
np.mean(scores)
Ytest_dataPred = knn.predict(Xtest_data)
Ytrain_dataPred = knn.predict(Xtrain_data)
predicted = pd.DataFrame(data = Ytest_dataPred)
predicted[0].value_counts()
Test = []
for i in range(len(test_data['Id'])):
    Test.append(test_data['Id'][i][0:len(test_data['Id'][i])-1])
result = np.vstack((Test,Ytest_dataPred)).T
x = ['Id','ham']
Result = pd.DataFrame(columns=x ,data = result)
Result.to_csv('resultados_SpambaseKnn.csv', index=False)
Result
Xtrain_dataBNB = train_data.iloc[:,0:54]
Xtrain_dataBNB['Knn'] = Ytrain_dataPred
a = list(train_data)[0:len(list(train_data))-1]
for i in range(len(a)):
    l=[]
    #m=[]
    print(a[i])
    for j in train_data['{}'.format(a[i])]:
        if j>train_data['{}'.format(a[i])].mean():
            l.append(1)
        else:
            l.append(0)
        
        #if j>(j-train_data['{}'.format(a[i])].mean())/train_data['{}'.format(a[i])].std():
        #    m.append(1)
        #else:
        #    m.append(0)
        
        
    Xtrain_dataBNB['{}{}'.format(a[i],'*')] = l
    #Xtrain_dataBNB['{}{}'.format(a[i],'**')] = m
    
        
Xtrain_dataBNB = Xtrain_dataBNB.drop(columns= 'ham*')
#Xtrain_dataBNB = Xtrain_dataBNB.drop(columns= 'ham**')

    
Ytrain_dataBNB = train_data.ham
Ytrain_dataBNB.shape
Xtest_dataBNB = test_data.iloc[:,0:54]

Xtest_dataBNB['Knn'] = Ytest_dataPred
a = list(test_data)[0:len(list(test_data))-1]
for i in range(len(a)):
    l=[]
    #m=[]
    print(a[i])
    for j in test_data['{}'.format(a[i])]:
        if j>test_data['{}'.format(a[i])].mean():
            l.append(1)
        else:
            l.append(0)
            
        #if j>(j-test_data['{}'.format(a[i])].mean())/test_data['{}'.format(a[i])].var():
        #    m.append(1)
        #else:
        #    m.append(0)
            
    Xtest_dataBNB['{}{}'.format(a[i],'*')] = l
    #Xtest_dataBNB['{}{}'.format(a[i],'**')] = m

BNB = BernoulliNB(alpha=1)
BNB.fit(Xtrain_dataBNB,Ytrain_dataBNB)
scores = cross_val_score(BNB, Xtrain_dataBNB, Ytrain_dataBNB, cv=10)
x=np.mean(scores)
print(x)
Ytest_dataPredBNB = BNB.predict(Xtest_dataBNB)
predictedBNB = pd.DataFrame(data = Ytest_dataPredBNB)
predictedBNB[0].value_counts()
TestBNB = []
for i in range(len(test_data['Id'])):
    TestBNB.append(test_data['Id'][i][0:len(test_data['Id'][i])-1])
resultBNB = np.vstack((TestBNB,Ytest_dataPredBNB)).T
x = ['Id','ham']
ResultBNB = pd.DataFrame(columns=x ,data = resultBNB)
ResultBNB.to_csv('resultados_SpambaseBNB.csv', index=False)
ResultBNB
