# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from wordcloud  import WordCloud,STOPWORDS

from nltk.corpus import stopwords 

import matplotlib.pyplot as plt

import seaborn as sns

import nltk as nlt
test_data = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')

train_data = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')
train_data.head(10)
test_data.head(10)
train_data['Premise_Length'] = train_data['premise'].apply(lambda x: len(x))

train_data['Hypothesis_Length'] = train_data['hypothesis'].apply(lambda x: len(x))

test_data['Premise_Length'] = test_data['premise'].apply(lambda x: len(x))

test_data['Hypothesis_Length'] = test_data['hypothesis'].apply(lambda x: len(x))

#We will create a dictionary for our language colum so we can reparse our encoding if needed

langs = train_data['language'].value_counts().to_frame().index.to_list()

lang_dic = {langs[i]:i for i in range(0,len(langs))}

train_data['language'] = train_data['language'].replace(lang_dic)

test_data['language'] = test_data['language'].replace(lang_dic)
#We will add a feature wich will represent the amount of similar tokens between  the premise and the hypothesis

tokenizer = nlt.RegexpTokenizer(r"\w+")

def similar_tokens(sir):

    tokens_p = tokenizer.tokenize(sir[0])

    tokens_h = tokenizer.tokenize(sir[1])

    tokens_p = set(tokens_p)

    tokens_h = set(tokens_h)

    return len(tokens_p.intersection(tokens_h))

def dissimilar_tokens(sir):

    tokens_p = tokenizer.tokenize(sir[0])

    tokens_h = tokenizer.tokenize(sir[1])

    tokens_p = set(tokens_p)

    tokens_h = set(tokens_h)

    total = len(tokens_p)+len(tokens_h)

    return total - len(tokens_p.intersection(tokens_h))

def num_of_words(sir,unique = 0):

    tokens_p = tokenizer.tokenize(sir)

    if unique:

        tokens_p = set(tokens_p)

    return len(tokens_p)

def average_word_length(sir):

    tokens_p = tokenizer.tokenize(sir)

    lengths = [len(word) for word in tokens_p]

    avg_len = np.array(lengths).sum()/len(lengths)

    return avg_len

    



train_data['Similar_Tokens#'] = train_data[['premise','hypothesis']].apply(similar_tokens,axis=1) 

train_data['Dissimilar_Tokens#'] = train_data[['premise','hypothesis']].apply(dissimilar_tokens,axis=1) 

train_data['premise_#_of_words'] = train_data['premise'].apply(num_of_words) 

train_data['premise_#_of_unique_words'] = train_data['premise'].apply(num_of_words,unique=1) 

train_data['hypothesis_#_of_words'] = train_data['hypothesis'].apply(num_of_words) 

train_data['hypothesis_#_of_unique_words'] = train_data['hypothesis'].apply(num_of_words,unique=1) 

train_data['hypothesis_avg_word_length'] = train_data['hypothesis'].apply(average_word_length)

train_data['premise_avg_word_length'] = train_data['premise'].apply(average_word_length) 



test_data['Similar_Tokens#'] = test_data[['premise','hypothesis']].apply(similar_tokens,axis=1) 

test_data['Dissimilar_Tokens#'] = test_data[['premise','hypothesis']].apply(dissimilar_tokens,axis=1) 

test_data['premise_#_of_words'] = test_data['premise'].apply(num_of_words) 

test_data['premise_#_of_unique_words'] = test_data['premise'].apply(num_of_words,unique=1) 

test_data['hypothesis_#_of_words'] = test_data['hypothesis'].apply(num_of_words) 

test_data['hypothesis_#_of_unique_words'] = test_data['hypothesis'].apply(num_of_words,unique=1) 

test_data['hypothesis_avg_word_length'] = test_data['hypothesis'].apply(average_word_length)

test_data['premise_avg_word_length'] = test_data['premise'].apply(average_word_length) 
train_data
#Also I Want to have the most common token in each label

def keep_track(my_dict,key):

    if key in my_dict:

        my_dict[key] += 1

    else:

        my_dict[key] = 1

    return my_dict





#the new insight we will try to extract

most_common_premise_hypothesis_token_0_label = []

most_common_premise_hypothesis_token_1_label = []

most_common_premise_hypothesis_token_2_label = []

most_common_premise_hypothesis =  [most_common_premise_hypothesis_token_0_label,most_common_premise_hypothesis_token_1_label,most_common_premise_hypothesis_token_2_label]



for label in range (0,3):

    label_data = train_data[train_data['label']==label]

    for i in range(0,15):

        if i == 0:

            stop_words = set(stopwords.words(str.lower(langs[i])))

            stop_words.add('uh')

        zero_label_dic = {}





        language_labeld = label_data[label_data.language==i]



        for sen in language_labeld['premise']:

            tokens = tokenizer.tokenize(sen)

            for token in tokens:

                zero_label_dic = keep_track(zero_label_dic,token)

        inverse = [(value, key) for key, value in zero_label_dic.items() if str.lower(key) not in stop_words]

        most_common_premise_hypothesis[label].append(max(inverse)[1])

        zero_label_dic={}

        for sen in language_labeld['hypothesis']:

            tokens = tokenizer.tokenize(sen)

            for token in tokens:

                zero_label_dic = keep_track(zero_label_dic,token)

        inverse = [(value, key) for key, value in zero_label_dic.items()]

        inverse.sort()

        most_common_premise_hypothesis[label].append(max(inverse)[1])

    

#================================================================================================================

#Now That We Have Our Most Common Token For Each Label Lets Create A Boolean Feature That Tells Us Does The Sample Contain The Most Common Word Or Not

hypothesis_status_column = []

premise_status_column = []



for index in range(0,train_data.shape[0]):

    lang = train_data.iloc[index,4]

    lbl = train_data.iloc[index,5]

    if train_data.iloc[index,1].find(most_common_premise_hypothesis[lbl][lang]) != -1:

        premise_status_column.append(1)

    else:

        premise_status_column.append(0)

    if train_data.iloc[index,2].find(most_common_premise_hypothesis[lbl][lang+1]) != -1:

        hypothesis_status_column.append(1)

    else:

        hypothesis_status_column.append(0)



#mct = most common token    

train_data['premise_contains_mct'] = premise_status_column

train_data['hypothesis_contains_mct'] = hypothesis_status_column







train_data
train_data.describe()
plt.figure(figsize=(20,11))

sns.set_style('darkgrid')

ax =sns.countplot(train_data.language)

ax.set_xticklabels(labels=langs)

ax.set_title("Counts Of Different Languages In Our Data")

plt.show()
plt.figure(figsize=(20,11))

ax =sns.countplot(train_data.label)

ax.set_title("Counts Of Different Labels In Our Data")

plt.show()
plt.figure(figsize=(20,11))

ax =sns.scatterplot(x=train_data['Premise_Length'],y=train_data['Hypothesis_Length'],size=train_data['label'],palette='Blues',hue=train_data['label'])

ax.set_title("The Spread Of The Premise And Hypothesis Lengths In Our Data Via Label")

plt.show()
plt.figure(figsize=(20,11))

ax =sns.scatterplot(x=train_data['hypothesis_avg_word_length'],y=train_data['premise_avg_word_length'],size=train_data['label'],palette='Blues',hue=train_data['label'])

ax.set_title("The Spread Of The Premise And Hypothesis Average Word Lengths In Our Data Via Label")

plt.show()
plt.figure(figsize=(20,11))

ax =sns.jointplot(x=train_data['Similar_Tokens#'],y=train_data['label'],cmap='mako',height=12,kind='kde',n_levels=10)

plt.show()
plt.figure(figsize=(20,11))

ax =sns.scatterplot(x=train_data['Hypothesis_Length'],y=train_data['Dissimilar_Tokens#'],size=train_data['label'],palette='Blues',hue=train_data['label'])

ax.set_title("The Spread Of The Hypothesis Length And Dissimilar Tokens In Our Data Via Label")

plt.show()
plt.figure(figsize=(20,11))

ax =sns.scatterplot(x=train_data['hypothesis_#_of_words'],y=train_data['Dissimilar_Tokens#'],size=train_data['label'],palette='Blues',hue=train_data['label'])

ax.set_title("The Spread Of The Hypothesis Length And Dissimilar Tokens In Our Data Via Label")

plt.show()
fig = plt.figure(figsize=(20,11))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_data['Dissimilar_Tokens#'],train_data['hypothesis_#_of_words'],train_data['Hypothesis_Length'],s=10,cmap='coolwarm',c=train_data['label'])

ax.set_xlabel('Dissimilar_Tokens',fontsize=13)

ax.set_ylabel('hypothesis_#_of_words',fontsize=13)

ax.set_zlabel('Hypothesis_Length',fontsize=13)

plt.show()
train_data = train_data[train_data['premise_avg_word_length']<25]

train_data = train_data[train_data['Premise_Length']<400]

train_data = train_data[train_data['Hypothesis_Length']<150]

train_data = train_data[train_data['Dissimilar_Tokens#']<60]

train_data = train_data[train_data['hypothesis_#_of_words']<40]

train_data = train_data[train_data['Dissimilar_Tokens#']<80]



plt.figure(figsize=(20,11))

cor = train_data.corr('pearson')

ax =sns.heatmap(cor,cmap="coolwarm",annot=True)

ax.set_title("Correlations Of Different Features In Our Data")

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score as ascore

from sklearn.metrics import f1_score as f1

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from keras.layers import Dense

from keras import Sequential
target = train_data.pop('label')
#Our features will be all feature with reasonable correlation







selector = SelectKBest(chi2,k=6)

selector.fit(train_data[train_data.columns[4:]],target)

X = selector.transform(train_data[train_data.columns[4:]])

selected_features = [train_data.columns[4:][i] for i in range(0,len(train_data.columns[4:])) if selector.get_support()[i] == True]



train_x,test_x,train_y,test_y = train_test_split(X,target)

#train_data

selected_features
def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = KNeighborsClassifier(n_neighbors = n)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(ascore(pred,test_y))

    return results
n_list = [10,20,30,50,80,130,210,350,560]

result = optimal_n(train_x,test_x,train_y,test_y,n_list)

plt.figure(figsize=(20,11))

ax =sns.lineplot(x=np.arange(len(n_list)),y=result)

n_list.insert(0,1)

ax.set_xticklabels(n_list)

ax.set_title('KNN Accuracy Depending On Number Of Neighbors',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
def optimal_e(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = RandomForestClassifier(max_leaf_nodes = n,random_state=42)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(ascore(pred,test_y))

    return results
n_list = [2,3,5,8,13,21,35,56,91,147]

result = optimal_e(train_x,test_x,train_y,test_y,n_list)

plt.figure(figsize=(20,11))

ax = sns.lineplot(x=np.arange(0,10),y=result)

#n_list.insert(0,1)

ax.set_xticklabels(labels = n_list)

ax.set_title('RandomForest Accuracy Depending On Number Of Estimators',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()

def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = AdaBoostClassifier(n_estimators = n,random_state=42,learning_rate=0.05)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(ascore(pred,test_y))

    return results
ee_list = [2,3,5,8,13,21,35,56,91,147,300]

result = optimal_n(train_x,test_x,train_y,test_y,ee_list)

plt.figure(figsize=(20,11))

ax =sns.lineplot(x=np.arange(len(ee_list)),y=result)

n_list.insert(0,1)

ax.set_xticklabels(labels = ee_list)

ax.set_title('AdaBoost Accuracy Depending On Number Of Max Leaf Nodes',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = DecisionTreeClassifier(max_leaf_nodes = n,random_state=42,criterion='entropy')

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(ascore(pred,test_y))

    return results
ee_list = [2,3,5,8,13,21,35,56,91,147,300]

result = optimal_n(train_x,test_x,train_y,test_y,ee_list)

plt.figure(figsize=(20,11))

ax =sns.lineplot(x=np.arange(len(ee_list)),y=result)

n_list.insert(0,1)

ax.set_xticklabels(labels = ee_list)

ax.set_title('Decision Tree Accuracy Depending On Number Of Max Leaf Nodes',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
model = Sequential()

model.add(Dense(10,activation='sigmoid',input_dim=len(selected_features)))

model.add(Dense(16,activation='tanh'))

model.add(Dense(16,activation='sigmoid'))

model.add(Dense(1,activation='tanh'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics='accuracy')
model.fit(train_x,train_y,epochs=10,verbose=False)
rf_model = RandomForestClassifier(n_estimators=96,random_state=42)

ADA_model = AdaBoostClassifier(n_estimators=96,random_state=42,learning_rate=0.3)

dt_model = DecisionTreeClassifier(max_leaf_nodes = 21,random_state=42,criterion='entropy')



X = train_data[selected_features].append(test_data[selected_features])

rf_model.fit(train_data[selected_features],target)

ADA_model.fit(train_data[selected_features],target)

dt_model.fit(train_data[selected_features],target)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



predictions = rf_model.predict(test_data[selected_features])*0.2 + dt_model.predict(test_data[selected_features])*0.5 + ADA_model.predict(test_data[selected_features])*0.3

predictions = (np.round(predictions)).astype('int64')

predictions



sm = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')

sm = sm['prediction'].to_list()



cf_m = confusion_matrix(sm,predictions)

plt.figure(figsize=(20,11))

ax = sns.heatmap(cf_m,annot=True,fmt='d')

result = pd.DataFrame({'id':test_data['id'].to_list(),'prediction':predictions})
result.to_csv('submission.csv',index=False)