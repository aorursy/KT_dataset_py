%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import stopwords 



import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

from bs4 import BeautifulSoup



import string

from string import digits

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer

from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder





from sklearn.model_selection import cross_validate

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split

from sklearn import metrics



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB





class color:

     BOLD = '\033[1m'

     UNDERLINE = '\033[4m'

     END = '\033[0m'

        

import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/train.csv')

df_train.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
df_train['type'].unique()
print (color.BOLD + "Train data:" + color.END)

print ("Number of columns: " + str (df_train.shape[1]))

print ("Number of rows: " + str (df_train.shape[0]))



print(color.BOLD + '\nTest data: ' + color.END)

print ("Number of columns:" + str (test_df.shape[1]))

print ("Number of rows:" +  str (test_df.shape[0]))
print('Datasets Total=',len(df_train) + len(test_df))
print('Training set consist of', round(len(df_train)/8675*100),'% data')

print('Testing set consist of', round(len(test_df)/8675*100),'% data')
print(df_train.info(), '\n')

print(test_df.info())
df_train.posts.iloc[0][0:2000]
len(df_train.iloc[1,1].split('|||'))
train_val_count=df_train['type'].value_counts()

train_val_count
plt.figure(figsize=(12,4))



sns.barplot(train_val_count.index, train_val_count.values,palette= 'Accent_r', ec='black' )





plt.ylabel('Number of occurrences per type', fontsize=10)

plt.xlabel('Personality types', fontsize=10)

plt.title('Total posts for each personality type')

plt.show()
I_E= df_train['type'].map(lambda type: type[0]).value_counts()

N_S= df_train['type'].map(lambda type: type[1]).value_counts()

T_F= df_train['type'].map(lambda type: type[2]).value_counts()

J_P= df_train['type'].map(lambda type: type[3]).value_counts()
print(color.BOLD  + 'Introversion (I) – Extroversion (E):' +color.END ,'\n',I_E, '\n')

print(color.BOLD  + 'Intuition (N) – Sensing (S):' +color.END, '\n', N_S, '\n')

print(color.BOLD  + 'Thinking (T) – Feeling (F):' +color.END, '\n', T_F, '\n')

print(color.BOLD  + 'Judging (J) – Perceiving (P):' +color.END,'\n', J_P)
print(color.BOLD +"Introverts and Extroverts Percentages:" + color.END)

Introversion_perc= 4998/len(df_train)

print('Introversion_percentage is:',round(Introversion_perc *100),'%')

Extroversion_perc= 1508/len(df_train)

print('Extroversion_percentage is:',round(Extroversion_perc *100),'%','\n')



print(color.BOLD +"Intuition and Sensing Percentages:" + color.END)

Intuition_perc= 5612/len(df_train)

print('Intuition_percentage is:',round(Intuition_perc *100),'%')

Sensing_perc= 894/len(df_train)

print('Sensing_percentage is:',round(Sensing_perc *100),'%','\n')



print(color.BOLD +"Thinking and Feeling Percentages:" + color.END)

Thinking_perc= 3518/len(df_train)

print('Thinking_percentage is:',round(Thinking_perc *100),'%')

Feeling_perc= 2988/len(df_train)

print('Feeling_percentage is:',round(Feeling_perc *100),'%','\n')



print(color.BOLD +"Judging and Perceiving Percentages:" + color.END)

Judging_perc= 3932/len(df_train)

print('Thinking_percentage is:',round(Judging_perc *100),'%')

Perceiving_perc= 2574/len(df_train)

print('Perceiving_percentage is:',round(Perceiving_perc *100),'%')
temp = {'Introverts':[77],

        'Extroverts':[23],

        'Intuition':[86],

        'Sensing':[14],

        'Thinking':[54],

        'Feeling':[46],

        'Judging':[60],

        'Perceiving':[40]}



results = pd.DataFrame.from_dict(temp, orient='index', columns=['Percentages'])

results
my_func= lambda x: float(x)

results['Percentages']=results['Percentages'].apply(my_func)
results.plot(kind='bar',colormap='PuRd_r')



plt.title('Total percentage posts for each personality',size=12)

plt.xlabel('Personality types', size = 12)

plt.ylabel('Number of posts available per type', size = 12)

plt.show()
p = df_train.copy()

z = test_df.copy()
p['clean'] = p['posts'].apply(lambda x: ' '.join(x.split('|||')))

p.head()
z['clean'] = z['posts'].apply(lambda x: ' '.join(x.split('|||')))

z.head()
p.shape,z.shape
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

p['clean'] = p['clean'].replace(to_replace = pattern_url, value = subs_url, regex = True)

p.head()
z['clean'] = z['clean'].replace(to_replace = pattern_url, value = subs_url, regex = True)

z.tail()
p['clean_2'] = p['clean'].str.lower()

p.head()
p['clean_2'] = p['clean_2'].apply(lambda x : x.translate(str.maketrans(' ',' ',string.punctuation)))

p.head()
p['clean_2'] = p['clean_2'].apply(lambda x : x.translate(str.maketrans(' ',' ',digits)))

p.head()
p['clean_2'] = p['clean_2'].str.strip()

p.head()
tokeniser = TreebankWordTokenizer()



p['tokens'] = p['clean_2'].apply(tokeniser.tokenize)

p.head()
stemmer = SnowballStemmer('english')



def mbti_stemmer(words, stemmer):

    return [stemmer.stem(word) for word in words] 
p['stem'] = p['tokens'].apply(mbti_stemmer, args=(stemmer, ))

p.head()
Stops = set(stopwords.words('english'))
p['no_stop'] = p['stem'].apply(lambda x: [word for word in list(x) if word not in Stops])

p.head()
p['no_stop'] = p['no_stop'].apply(lambda x: ' '.join(x))

p.head()
unique_type = list(p['type'].unique())

encoder = LabelEncoder().fit(unique_type)



codes = []



for i in range(0, len(p)):

    codes.append(p['type'][i])

coder = encoder.transform(codes)
list(p['type'].unique())
p['codes'] = coder

p.head()
#Features

X = p.clean 



#Labels

y = p.codes
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=14)
Vect = CountVectorizer(ngram_range=(1, 1), stop_words='english', lowercase = True, max_features = 5000)
LRmodel = LogisticRegression(class_weight="balanced", C=0.005, penalty = "l2")

pipe = Pipeline([('vec', Vect), ('model', LRmodel)])
pipe.fit(X_train, y_train)
print(LRmodel.intercept_[0])

print(LRmodel.coef_)
y_pred_train= pipe.predict(X_train)
print('Accuracy: '+ str(metrics.accuracy_score(y_train, y_pred_train)))

print('Precision: '+ str( metrics.precision_score(y_train, y_pred_train, average='macro')))

print('Recall: '+ str(metrics.recall_score(y_train, y_pred_train, average='macro')))

print('F1_Score: '+ str( metrics.f1_score(y_train, y_pred_train, average='macro')))
y_pred_test = pipe.predict(X_test)
print('Accuracy: '+ str(metrics.accuracy_score(y_test, y_pred_test)))

print('Precision: '+ str( metrics.precision_score(y_test, y_pred_test, average='macro')))

print('Recall: '+ str(metrics.recall_score(y_test, y_pred_test, average='macro')))

print('F1_Score: '+ str( metrics.f1_score(y_test, y_pred_test, average='macro')))
cm_logistic_reg = np.array(metrics.confusion_matrix(y_test, y_pred_test))

cm_logistic = pd.DataFrame(cm_logistic_reg, index=['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP',

                                       'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP',

                                       'ISFJ', 'ISFP', 'ISTJ', 'ISTP'], 

                            columns=['predict_ENFJ','predict_ENFP','predict_ENTJ',

                                     'predict_ENTP','predict_ESFJ','predict_ESFP',

                                     'predict_ESTJ','predict_ESTP','predict_INFJ',

                                     'predict_INFP','predict_INTJ','predict_INTP',

                                     'predict_ISFJ','predict_ISFP','predict_ISTJ',

                                     'predict_ISTP'])

cm_logistic
fig, ax = plt.subplots(figsize=(10,8)) 

sns.heatmap(cm_logistic, robust=True, annot=True, linewidth=0.3, 

            fmt='', cmap='BrBG', vmax=303, ax=ax)

plt.title('Confusion Matrix for Logistic Classifier', fontsize=10,

          fontweight='bold', y=1.00)



plt.xticks(fontsize=10)

plt.yticks(rotation=0, fontsize=10);
print(metrics.classification_report(y_test, y_pred_test, target_names=unique_type))
test_preds = pipe.predict(z['clean'])
test_preds
z['type'] = encoder.inverse_transform(test_preds)

z.head()
z['E or I'] = z.apply(lambda x: x['type'][0], axis = 1)

z['N or S'] = z.apply(lambda x: x['type'][1], axis = 1)

z['T or F'] = z.apply(lambda x: x['type'][2], axis = 1)

z['J or P'] = z.apply(lambda x: x['type'][3], axis = 1)



mind = z['E or I'].astype(str).apply(lambda x: x[0] == 'E').astype('int')

energy = z['N or S'].astype(str).apply(lambda x: x[0] == 'N').astype('int')

nature = z['T or F'].astype(str).apply(lambda x: x[0] == 'T').astype('int')

tactics = z['J or P'].astype(str).apply(lambda x: x[0] == 'J').astype('int')



z.head()
df_LogReg = pd.DataFrame({"id":test_df['id'], "mind":mind, "energy":energy, "nature":nature, 'tactics':tactics})

df_LogReg.head()
df_LogReg.shape
df_LogReg.to_csv('EDSA_Team_8_Classification1.csv', index = False)
LRmodel2 = LogisticRegression(class_weight="balanced", C=0.004, penalty = "l2")

pipe2 = Pipeline([('vec', Vect), ('model', LRmodel2)])
pipe2.fit(X_train, y_train)
y_pred_train2= pipe2.predict(X_train)
print('Accuracy: '+ str(metrics.accuracy_score(y_train, y_pred_train2)))

print('Precision: '+ str( metrics.precision_score(y_train, y_pred_train2, average='macro')))

print('Recall: '+ str(metrics.recall_score(y_train, y_pred_train2, average='macro')))

print('F1_Score: '+ str( metrics.f1_score(y_train, y_pred_train2, average='macro')))
y_pred_test2 = pipe2.predict(X_test)
print('Accuracy: '+ str(metrics.accuracy_score(y_test, y_pred_test2)))

print('Precision: '+ str( metrics.precision_score(y_test, y_pred_test2, average='macro')))

print('Recall: '+ str(metrics.recall_score(y_test, y_pred_test2, average='macro')))

print('F1_Score: '+ str( metrics.f1_score(y_test, y_pred_test2, average='macro')))
con_matrix_test2= metrics.confusion_matrix(y_test, y_pred_test2)



cm_logistic_df = pd.DataFrame(con_matrix_test2, index=['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP',

                                       'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP',

                                       'ISFJ', 'ISFP', 'ISTJ', 'ISTP'], 

                            columns=['predict_ENFJ','predict_ENFP','predict_ENTJ',

                                     'predict_ENTP','predict_ESFJ','predict_ESFP',

                                     'predict_ESTJ','predict_ESTP','predict_INFJ',

                                     'predict_INFP','predict_INTJ','predict_INTP',

                                     'predict_ISFJ','predict_ISFP','predict_ISTJ',

                                     'predict_ISTP'])

fig, ax = plt.subplots(figsize=(10,8)) 

sns.heatmap(cm_logistic_df, robust=True, annot=True, linewidth=0.3, 

            fmt='', cmap='BrBG', vmax=303, ax=ax)

plt.title('Confusion Matrix for Logistic Classifier for the test dataset', fontsize=10,

          fontweight='bold', y=1.00)



plt.xticks(fontsize=10)

plt.yticks(rotation=0, fontsize=10)

plt.show()
print(metrics.classification_report(y_test, y_pred_test2, target_names=unique_type))
test_preds2 = pipe2.predict(z['clean'])
test_preds2
z['type'] = encoder.inverse_transform(test_preds2)

z.head()
z['E or I'] = z.apply(lambda x: x['type'][0], axis = 1)

z['N or S'] = z.apply(lambda x: x['type'][1], axis = 1)

z['T or F'] = z.apply(lambda x: x['type'][2], axis = 1)

z['J or P'] = z.apply(lambda x: x['type'][3], axis = 1)



mind = z['E or I'].astype(str).apply(lambda x: x[0] == 'E').astype('int')

energy = z['N or S'].astype(str).apply(lambda x: x[0] == 'N').astype('int')

nature = z['T or F'].astype(str).apply(lambda x: x[0] == 'T').astype('int')

tactics = z['J or P'].astype(str).apply(lambda x: x[0] == 'J').astype('int')



z.head()

df_LogReg2 = pd.DataFrame({"id":test_df['id'], "mind":mind, "energy":energy, "nature":nature, 'tactics':tactics})

df_LogReg2.head()
df_LogReg2.shape
df_LogReg2.to_csv('EDSA_Team_8_Classification2.csv', index = False)
# Fit and score a Random Forest Classifier on SMOTEd data using the parameters identified by the grid search

random= RandomForestClassifier(min_samples_leaf=2, min_samples_split=3, n_estimators=79, 

                             criterion='entropy', bootstrap='False', n_jobs= -1, random_state=123)

random_pipe= Pipeline([('vec', Vect), ('model', random)])
random_pipe.fit(X_train, y_train)
y_pred_random=random_pipe.predict(X_train)
print('Accuracy: '+ str(metrics.accuracy_score(y_train, y_pred_random)))

print('Precision: '+ str( metrics.precision_score(y_train, y_pred_random, average='macro')))

print('Recall: '+ str(metrics.recall_score(y_train, y_pred_random, average='macro')))

print('F1_Score: '+ str( metrics.f1_score(y_train, y_pred_random, average='macro')))
print("Classification Report:")

print(metrics.classification_report(y_train, y_pred_random, target_names=unique_type))
y_pred_random_test = random_pipe.predict(X_test)
print('Accuracy: '+ str(metrics.accuracy_score(y_test, y_pred_random_test)))

print('Precision: '+ str( metrics.precision_score(y_test, y_pred_random_test, average='macro')))

print('Recall: '+ str(metrics.recall_score(y_test, y_pred_random_test, average='macro')))

print('F1_Score: '+ str( metrics.f1_score(y_test, y_pred_random_test, average='macro')))
con_matrix_random= metrics.confusion_matrix(y_test, y_pred_random_test)



cm_logistic_rf = pd.DataFrame(con_matrix_random, index=['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP',

                                       'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP',

                                       'ISFJ', 'ISFP', 'ISTJ', 'ISTP'], 

                            columns=['predict_ENFJ','predict_ENFP','predict_ENTJ',

                                     'predict_ENTP','predict_ESFJ','predict_ESFP',

                                     'predict_ESTJ','predict_ESTP','predict_INFJ',

                                     'predict_INFP','predict_INTJ','predict_INTP',

                                     'predict_ISFJ','predict_ISFP','predict_ISTJ',

                                     'predict_ISTP'])



fig, ax = plt.subplots(figsize=(10,8)) 

sns.heatmap(cm_logistic_rf, robust=True, annot=True, linewidth=0.3, 

            fmt='', cmap='BrBG', vmax=303, ax=ax)

plt.title('Confusion Matrix for Random Forest Classifier ', fontsize=10,

          fontweight='bold', y=1.00)



plt.xticks(fontsize=10)

plt.yticks(rotation=0, fontsize=10)

plt.show()
print("Classification Report:")

print(metrics.classification_report(y_test, y_pred_random_test, target_names=unique_type))
test_preds_random = random_pipe.predict(z['clean'])
z['type'] = encoder.inverse_transform(test_preds_random)

z.head()
z['E or I'] = z.apply(lambda x: x['type'][0], axis = 1)

z['N or S'] = z.apply(lambda x: x['type'][1], axis = 1)

z['T or F'] = z.apply(lambda x: x['type'][2], axis = 1)

z['J or P'] = z.apply(lambda x: x['type'][3], axis = 1)



mind = z['E or I'].astype(str).apply(lambda x: x[0] == 'E').astype('int')

energy = z['N or S'].astype(str).apply(lambda x: x[0] == 'N').astype('int')

nature = z['T or F'].astype(str).apply(lambda x: x[0] == 'T').astype('int')

tactics = z['J or P'].astype(str).apply(lambda x: x[0] == 'J').astype('int')



z.head()
df_random = pd.DataFrame({"id":test_df['id'], "mind":mind, "energy":energy, "nature":nature, 'tactics':tactics})

df_random.head()
df_random.shape
df_random.to_csv('EDSA_Team_8_Classification_random.csv', index = False)
D = {'1':12.52206, '2': 21.72241, '3':19.58775,'4':13.27084,

    '5': 7.04959, '6':6.81062 ,'7':6.76286, '8': 6.66727,

    '9':6.52388, '10': 5.79104, '11':5.63173 ,'12':5.23311, '13': 4.93871,

     '14':4.93074, '15': 4.91481, '16':4.88295 ,'17':4.80329,

    '18':9.85352, '19': 4.82719, '20':4.84312 ,'21':7.54346}



submission = list(D.keys())           

kaggle_score = list(D.values())        

plt.plot(submission, kaggle_score)

plt.show()