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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import nltk as nlp
f_data = pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')

f_data.head(3)
f_data.columns
def get_hour(ts):

    return ts.hour

def get_day(ts):

    return ts.weekday()

def get_month(ts):

    return ts.month

def get_year(ts):

    return ts.year
eng_data = f_data.copy()

eng_data['CreationDate'] = pd.to_datetime(eng_data['CreationDate'])

eng_data['Creation_Hour'] = eng_data['CreationDate'].apply(get_hour)

eng_data['Creation_Day'] = eng_data['CreationDate'].apply(get_day)

eng_data['Creation_Month'] = eng_data['CreationDate'].apply(get_month)

eng_data['Creation_Year'] = eng_data['CreationDate'].apply(get_year)

eng_data.drop(columns=['CreationDate'],inplace=True)
eng_sw =set(nlp.corpus.stopwords.words('english'))

def get_char_amount(val):

    return len(val)



def average_word_length(val):

    splited = val.split(' ')

    char_count = 0

    word_count = 0

    for word in splited:

        if word not in eng_sw:

            char_count =char_count + len(word)

            word_count = word_count+1

    return char_count/word_count





def number_of_words(val):

    splited = val.split(' ')

    char_count = 0

    word_count = 0

    for word in splited:

        if word not in eng_sw:

            word_count = word_count+1

    return word_count    
eng_data['Body_Char_Length'] = eng_data['Body'].apply(get_char_amount)

eng_data['Title_Char_Length'] = eng_data['Title'].apply(get_char_amount)

eng_data['Body_Avg_Word_Length'] = eng_data['Body'].apply(average_word_length)

eng_data['Title_Avg_Word_Length'] = eng_data['Title'].apply(average_word_length)

eng_data['Body_Num_Of_Words'] = eng_data['Body'].apply(number_of_words)

eng_data['Title_Num_Of_Words'] = eng_data['Title'].apply(number_of_words)
def find_most_common_words(ser):

    word_aux = {}

    for sample in ser:

        splited = sample.split(' ')

        for word in splited:

            if word not in eng_sw:

                if word in word_aux:

                    word_aux[word] += 1

                else:

                    word_aux[word] = 1

    

    return word_aux    

    
wa = find_most_common_words(eng_data[eng_data['Y']=='HQ']['Title'])

Top_5_words_hq_titles = sorted(wa, key=wa.get, reverse=True)[:5]

wa = find_most_common_words(eng_data[eng_data['Y']=='HQ']['Body'])

Top_5_words_hq_bodies = sorted(wa, key=wa.get, reverse=True)[5:10]





wa = find_most_common_words(eng_data[eng_data['Y']=='LQ_CLOSE']['Title'])

Top_5_words_lq_titles = sorted(wa, key=wa.get, reverse=True)[:5]

wa = find_most_common_words(eng_data[eng_data['Y']=='LQ_CLOSE']['Body'])

Top_5_words_lq_bodies = sorted(wa, key=wa.get, reverse=True)[6:11]
def contains_hq_title_words(val):

    splited = set(val.split(' '))

    return len(splited.intersection(Top_5_words_hq_titles))

def contains_hq_body_words(val):

    splited = set(val.split(' '))

    return len(splited.intersection(Top_5_words_hq_bodies))

    

def contains_lq_title_words(val):

    splited = set(val.split(' '))

    return len(splited.intersection(Top_5_words_lq_titles))

def contains_lq_body_words(val):

    splited = set(val.split(' '))

    return len(splited.intersection(Top_5_words_lq_bodies))
eng_data['Title_Contains_Top_5_hq_Words'] =eng_data['Title'].apply(contains_hq_title_words)

eng_data['Body_Contains_Top_5_hq_Words'] =eng_data['Title'].apply(contains_hq_body_words)

eng_data['Title_Contains_Top_5_lq_Words'] =eng_data['Title'].apply(contains_lq_title_words)

eng_data['Body_Contains_Top_5_lq_Words'] =eng_data['Title'].apply(contains_lq_body_words)

def tag_cleaner(val):

    splited = val.split('><')

    clean = []

    for tag in splited:

        tag =  tag.replace('<',' ')

        tag =  tag.replace('>',' ')

        clean.append(tag)

    return set(clean)
# all unique tags:

unique_tags = set()

for tag in eng_data['Tags']:

    unique_tags = unique_tags|tag_cleaner(tag)
google_related      = []

prog_lang_related   = []

app_related         = []

ml_dl_related       = []

cs_am_related       = []

db_related          = []

web_dev_related     = []

electronics_related = []



prog_langs = ['swift','basic','c#','f#','c++','java','python','kotlin','camel','coffee','perl',

             'lisp','ruby','visual-studio','azure','assembly','go','haskell',',rust','.net','spyder',

             'jcl','sap','opengl','jenkins','apache','verilog','numpy']



web_keys = ['js','.j','net','docker','server','web','webpage','chrome','firefox','rest','api',

           'angular','react','node','facebook','twitter','amazon-ses','chromium','browser','ntp','svn',

           'xml','explorer','kivy','php']



app_android = ['android','apk','sdk','ipad','iphone','ios']



db_keys = ['db','sql','query','mongo','nosql','json','database','cloud']

ml_dl_keys = ['tensorflow','machine_learning','deep_learning','scatter-plot','opencv','lda',

             'mlmodel','regression','principal-components','pca','pytorch','sklearn','face-recognition',

             ]

electronics_keys = ['esp8266','cpu','ram','core','tsu','gpu','arduino','raspberry']

cs_am_keys  = ['optimization','x509','class','array','sort','algorithm','code','runtime','header-files',

             'calculus','theory','geometry','polynomial']



c_unique_tags = unique_tags.copy()

c_unique_tags = list(c_unique_tags)

c_unique_tags = [tag.strip() for tag in c_unique_tags]



#google realted

for tag in c_unique_tags:

    if 'google'  in tag:

        google_related.append(tag)

c_unique_tags = [tag for tag in c_unique_tags if tag not in google_related]



#prog_lang realted

for tag in c_unique_tags:

    s_flag = False

    for plang in prog_langs:

        if s_flag is True:

            break;

        elif tag.find(plang) != -1:

            prog_lang_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in prog_lang_related]



#web_dev realted

for tag in c_unique_tags:

    s_flag = False

    for wk in web_keys:

        if s_flag is True:

            break;

        elif tag.find(wk) != -1:

            web_dev_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in web_dev_related]



#phone/app_dev realted

for tag in c_unique_tags:

    s_flag = False

    for wk in app_android:

        if s_flag is True:

            break;

        elif tag.find(wk) != -1:

            app_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in app_related]



#db realted

for tag in c_unique_tags:

    s_flag = False

    for wk in db_keys:

        if s_flag is True:

            break;

        elif tag.find(wk) != -1:

            db_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in db_related]



#ml_dl realted

for tag in c_unique_tags:

    s_flag = False

    for wk in ml_dl_keys:

        if s_flag is True:

            break;

        elif tag.find(wk) != -1:

            ml_dl_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in ml_dl_related]





#electronics realted

for tag in c_unique_tags:

    s_flag = False

    for wk in electronics_keys:

        if s_flag is True:

            break;

        elif tag.find(wk) != -1:

            electronics_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in electronics_related]



#cs_am realted

for tag in c_unique_tags:

    s_flag = False

    for wk in cs_am_keys:

        if s_flag is True:

            break;

        elif tag.find(wk) != -1:

            cs_am_related.append(tag)

            s_flag=True

            continue

c_unique_tags = [tag for tag in c_unique_tags if tag not in cs_am_related]

google_related_col = []    

prog_lang_related_col = []  

app_related_col = []        

ml_dl_related_col = []      

cs_am_related_col = []      

db_related_col = []         

web_dev_related_col = []    

electronics_related_col = []

other_related_col = []
for tag in eng_data['Tags']:

    clean_tag = list(tag_cleaner(tag))

    google_related_score =0   

    prog_lang_related_score =0  

    app_related_score =0        

    ml_dl_related_score =0      

    cs_am_related_score =0      

    db_related_score =0         

    web_dev_related_score =0    

    electronics_related_score =0 

    other_related_score =0 



    for tg in clean_tag:

        zero_count = 0

        if tg in google_related:

            google_related_score = google_related_score +1

        else:

            zero_count = zero_count+1

        if tg in prog_lang_related:

            prog_lang_related_score = prog_lang_related_score+1

        else:

            zero_count = zero_count+1

        if tg in app_related:

            app_related_score=app_related_score+1

        else:

            zero_count = zero_count+1

        if tg in ml_dl_related:

            ml_dl_related_score = ml_dl_related_score + 1

        else:

            zero_count = zero_count+1

        if tg in cs_am_related:

            cs_am_related_score=cs_am_related_score+1

        else:

            zero_count = zero_count+1

        if tg in db_related:

            db_related_score=db_related_score+1

        else:

            zero_count = zero_count+1

        if tg in web_dev_related:

            web_dev_related_score=web_dev_related_score+1

        else:

            zero_count = zero_count+1

        if tg in electronics_related:

            electronics_related_score=electronics_related_score+1

        else:

            zero_count = zero_count+1

        if zero_count == 8:

            other_related_score = other_related_score + 1

    google_related_col.append(google_related_score)

    prog_lang_related_col.append(prog_lang_related_score) 

    app_related_col.append(app_related_score)       

    ml_dl_related_col.append(ml_dl_related_score)      

    cs_am_related_col.append(cs_am_related_score)  

    db_related_col.append(db_related_score)        

    web_dev_related_col.append(web_dev_related_score)  

    electronics_related_col.append(electronics_related_score) 

    other_related_col.append(other_related_score)

eng_data['Google_Related'] = google_related_col

eng_data['Programing_Lang_Related'] = prog_lang_related_col

eng_data['App/Phone_Related'] = app_related_col

eng_data['ML/DL_Related'] = ml_dl_related_col

eng_data['CS/AM_Related'] = cs_am_related_col

eng_data['DB/Storage_Related'] = db_related_col

eng_data['WebApp/Dev_Related'] = web_dev_related_col

eng_data['Electronics_Related'] = electronics_related_col

eng_data['Unclassified_Related'] = other_related_col
#last feature we will add is the number of tags in a question

num_of_tags = []



for tag in eng_data['Tags']:

    clean_tag = list(tag_cleaner(tag))

    num_of_tags.append(len(clean_tag))

eng_data['Number_Of_Tags'] = num_of_tags
#lets get rid of the text data the we will no longer use

eng_data.drop(columns=['Body','Tags','Title','Id'],inplace=True)
#lets transform our target label from nominal to numeric where we can use a ordinal scale to show the ranking 

target_labels = eng_data.Y.value_counts().to_frame().reset_index()['index'].to_list()

tl_dic = {target_labels[num-1]:num for num in np.arange(1,4)}

eng_data.Y.replace(tl_dic,inplace=True)
eng_data.head(4)
plt.figure(figsize=(20,11))

correlations = eng_data.corr('pearson')

ax =sns.heatmap(correlations,cmap='Greens',annot=True)
#removal_of_outliers

nm = eng_data['Body_Avg_Word_Length']

eng_data['Body_Avg_Word_Length'] = eng_data['Body_Avg_Word_Length'][nm.between(nm.quantile(0.10),nm.quantile(0.85))]

nm = eng_data['Title_Avg_Word_Length']

eng_data['Title_Avg_Word_Length'] = eng_data['Title_Avg_Word_Length'][nm.between(nm.quantile(0.10),nm.quantile(0.85))]



nm = eng_data['Body_Char_Length']

eng_data['Body_Char_Length'] = eng_data['Body_Char_Length'][nm.between(nm.quantile(0.10),nm.quantile(0.85))]

nm = eng_data['Title_Char_Length']

eng_data['Title_Char_Length'] = eng_data['Title_Char_Length'][nm.between(nm.quantile(0.10),nm.quantile(0.85))]





nm = eng_data['Body_Num_Of_Words']

eng_data['Body_Num_Of_Words'] = eng_data['Body_Num_Of_Words'][nm.between(nm.quantile(0.10),nm.quantile(0.85))]

nm = eng_data['Title_Num_Of_Words']

eng_data['Title_Num_Of_Words'] = eng_data['Title_Num_Of_Words'][nm.between(nm.quantile(0.10),nm.quantile(0.85))]

plt.figure(figsize=(20,11))



ax = sns.barplot(x=eng_data['Creation_Year'],y=eng_data['Number_Of_Tags'],hue=eng_data['Y'])

plt.figure(figsize=(20,11))



ax = sns.boxplot(x=eng_data['Creation_Hour'],y=eng_data['Body_Num_Of_Words'])

plt.figure(figsize=(20,11))



ax = sns.distplot(eng_data[eng_data['Y'] == 3]['Body_Num_Of_Words'],hist=True,kde_kws={'lw':3.5},label='HQ')

ax = sns.distplot(eng_data[eng_data['Y'] == 2]['Body_Num_Of_Words'],hist=True,kde_kws={'lw':3.5},label='E_LQ')

ax = sns.distplot(eng_data[eng_data['Y'] == 1]['Body_Num_Of_Words'],hist=True,kde_kws={'lw':3.5},label='LQ')

ax.legend(prop={'size':20})
plt.figure(figsize=(20,11))



ax = sns.distplot(eng_data[eng_data['Y'] == 3]['Creation_Hour'],hist=True,kde_kws={'lw':3.5},label='HQ')

ax = sns.distplot(eng_data[eng_data['Y'] == 2]['Creation_Hour'],hist=True,kde_kws={'lw':3.5},label='E_LQ')

ax = sns.distplot(eng_data[eng_data['Y'] == 1]['Creation_Hour'],hist=True,kde_kws={'lw':3.5},label='LQ')

ax.plot([12,12],[0,0.12],color='r',linestyle='--',linewidth=3,label='$\sigma=%.2f$'%(12))

ax.legend(prop={'size':20})
fig,axs = plt.subplots(2,2)

fig.set_figwidth(19)

fig.set_figheight(11)

sns.countplot(eng_data[eng_data['Programing_Lang_Related']>0]['Programing_Lang_Related'],hue=eng_data['Y'],

             ax=axs[0,0])

axs[0,0].legend(['HQ','E_LQ','LQ'])



ax = sns.countplot(eng_data[eng_data['WebApp/Dev_Related']>0]['WebApp/Dev_Related'],hue=eng_data['Y'],ax=

                  axs[0,1])

ax.legend(['HQ','E_LQ','LQ'])



ax = sns.countplot(eng_data[eng_data['App/Phone_Related']>0]['App/Phone_Related'],hue=eng_data['Y'],ax=

                  axs[1,0])

ax.legend(['HQ','E_LQ','LQ'])



ax = sns.countplot(eng_data[eng_data['DB/Storage_Related']>0]['DB/Storage_Related'],hue=eng_data['Y'],ax=

                  axs[1,1])

ax.legend(['HQ','E_LQ','LQ'])

plt.show()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score as f1

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
target = eng_data.pop('Y')

features = ['Creation_Year','Title_Avg_Word_Length','App/Phone_Related','Number_Of_Tags','Electronics_Related',

           'DB/Storage_Related','ML/DL_Related','CS/AM_Related']
y = target.copy()

X = eng_data[features].copy()
X.Title_Avg_Word_Length = X.Title_Avg_Word_Length.fillna(X.Title_Avg_Word_Length.mean())
x_train,x_test,y_train,y_test = train_test_split(X,y)
Rf_model = RandomForestRegressor(n_estimators=20)

Rf_model.fit(x_train,y_train)

pred = Rf_model.predict(x_test)

pred = np.round(pred)

rf_score = (f1(pred,y_test,average='macro'))

print(rf_score)
ada_model = AdaBoostClassifier(learning_rate=0.03)

ada_model.fit(x_train,y_train)

pred = ada_model.predict(x_test)

pred = np.round(pred)

ada_score = (f1(pred,y_test,average='macro'))

print(ada_score)
knn_model = KNeighborsClassifier(n_neighbors=300)

knn_model.fit(x_train,y_train)

pred = knn_model.predict(x_test)

pred = np.round(pred)

knn_score = (f1(pred,y_test,average='macro'))

print(knn_score)
tree_model = DecisionTreeClassifier(max_leaf_nodes=35)

tree_model.fit(x_train,y_train)

pred = tree_model.predict(x_test)

pred = np.round(pred)

tree_score = (f1(pred,y_test,average='macro'))

print(tree_score)
from keras import Sequential

from keras.layers import Dense
fcnn_model = Sequential()

fcnn_model.add(Dense(8,activation='tanh',input_dim = len(features)))

fcnn_model.add(Dense(16,activation='tanh'))

fcnn_model.add(Dense(16,activation='tanh'))

fcnn_model.add(Dense(1,activation='tanh'))



fcnn_model.compile(optimizer='adam',loss='categorical_crossentropy')
fcnn_model.fit(x_train,y_train,epochs=10)
fcnn_pred = fcnn_model.predict(x_test)

fcnn_pred = np.round(fcnn_pred)

print((f1(fcnn_pred,y_test,average='macro')))
tree_model = DecisionTreeClassifier(max_leaf_nodes=35)

tree_model.fit(X,y)

pred = tree_model.predict(X)

pred = np.round(pred)

tree_score = (f1(pred,y,average='macro'))



cf_matrix = confusion_matrix(pred,y)



ax = sns.heatmap(cf_matrix,cmap='Blues',annot=True,fmt='g')