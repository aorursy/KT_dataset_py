import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier



%matplotlib inline

sns.set(style="darkgrid")
data = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv') 
data.head()
data.shape
def make_label_encoder(original_feature , new_feature) : 

    enc  = LabelEncoder()

    enc.fit(data[original_feature])

    data[new_feature] = enc.transform(data[original_feature])

    data.drop([original_feature],axis=1, inplace=True)
def make_countplot(feature) :

    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("prism", 3)) 
def make_kdeplot(feature) : 

    sns.kdeplot(data[feature], shade=True)
def make_pie(feature) : 

    plt.pie(data[feature].value_counts(),labels=list(data[feature].value_counts().index),

        autopct ='%1.2f%%' , labeldistance = 1.1,explode = [0.05 for i in range(len(data[feature].value_counts()))] )

    plt.show()
data.columns
for col in data.columns : 

    print('Unique values for Column {0}     is      {1}'.format(col ,len(data[col].unique())))
make_label_encoder('video_id' , 'video_id Code')
data.head()
data['trending_date'].unique()
len(data['trending_date'].unique())
year_list = []

month_list = []

day_list = []

for x in range(data.shape[0]) :

    year_list.append(data['trending_date'][x][:2])

    month_list.append(data['trending_date'][x][6:])

    day_list.append(data['trending_date'][x][3:5])
data.insert(16,'Year',year_list)

data.insert(17,'Month',month_list)

data.insert(18,'Day',day_list)
data.head()
ax = sns.countplot(x="Year", data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark",3))
ax = sns.countplot(x="Month", data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark",3))
ax = sns.kdeplot(data['Day'], shade=True)
data.drop(['trending_date'],axis=1, inplace=True)
year_dict = {'17': 0,'18':1}

data['Year'] = data['Year'].map(year_dict)

data.head()
len(data['title'].unique())
all_words = []

for x in range(data.shape[0]) : 

    all_words = all_words  +  data['title'][x].split()
len(all_words)
all_words_series = pd.Series(all_words)
top_words =  all_words_series.value_counts()[:30] 

top_words
top_words = list(top_words.index)
top_words
top_words.remove('-')

top_words.remove('|')

top_words.remove('The')

top_words.remove('&')

top_words.remove('the')

top_words.remove('of')

top_words.remove('and')

top_words.remove('in')

top_words.remove('on')

top_words.remove('a')

top_words.remove('with')

top_words.remove('In')

top_words.remove('To')

top_words.remove('A')
top_words
l1 = []

l2 = []

l3 = []

l4 = []

l5 = []

l6 = []

l7 = []

l8 = []

l9 = []

l10 = []

l11 = []

l12 = []

l13 = []



this_list = []

for x in range(data.shape[0]) : 

    this_list  =  data['title'][x].split()

    

    if ('Video)' in this_list) or ('Video]' in this_list)  : 

        l1.append(1)

    else : 

        l1.append(0)

        

    if ('(Official' in this_list) or ( 'Official'in this_list)  or ( '[Official' in this_list)  : 

        l2.append(1)

    else : 

        l2.append(0)

        

    if 'Trailer' in this_list : 

        l3.append(1)

    else : 

        l3.append(0)



    if 'You' in this_list : 

        l4.append(1)

    else : 

        l4.append(0)

    if 'Last' in this_list : 

        l5.append(1)

    else : 

        l5.append(0)

    if 'Star' in this_list : 

        l6.append(1)

    else : 

        l6.append(0)

    if 'My' in this_list : 

        l7.append(1)

    else : 

        l7.append(0)

    if 'Me' in this_list : 

        l8.append(1)

    else : 

        l8.append(0)

    if  'I'  in this_list : 

        l9.append(1)

    else : 

        l9.append(0)

    if 'to' in this_list : 

        l10.append(1)

    else : 

        l10.append(0)

    if '2018' in this_list : 

        l11.append(1)

    else : 

        l11.append(0)

    if 'Music' in this_list : 

        l12.append(1)

    else : 

        l12.append(0)

 

    if 'ft.' in this_list : 

        l13.append(1)

    else : 

        l13.append(0)
len(l12)
data.insert(18,'word 1',l1)

data.insert(19,'word 2',l2)

data.insert(20,'word 3',l3)

data.insert(21,'word 4',l4)

data.insert(22,'word 5',l5)

data.insert(23,'word 6',l6)

data.insert(24,'word 7',l7)

data.insert(25,'word 8',l8)

data.insert(26,'word 9',l9)

data.insert(27,'word 10',l10)

data.insert(28,'word 11',l11)

data.insert(29,'word 12',l12)

data.insert(30,'word 13',l13)
data.drop(['title'],axis=1, inplace=True)

data.head()
len(data['channel_title'].unique())
make_label_encoder('channel_title' , 'channel Code')
data.head()
data['category_id'].unique()
len(data['category_id'].unique())
make_countplot('category_id')
len(data['publish_time'].unique())
data['publish_time'][:10]
publish_quarter = []



for x in range(data.shape[0]):

    publish_hour = int(str(data['publish_time'][x])[11:13])

    if publish_hour >=0 and publish_hour < 6  : 

        publish_quarter.append(1)

    elif publish_hour >=6 and publish_hour < 12  : 

        publish_quarter.append(2)

    elif publish_hour >=12 and publish_hour < 18  : 

        publish_quarter.append(3)

    else: 

        publish_quarter.append(4)
len(publish_quarter)
data.insert(30,'Publish Quarter',publish_quarter)
data.head()
make_label_encoder('publish_time' , 'publish time code')
data.head()
len(data['tags'].unique())
data['tags'][:10]
data['tags'][10]
make_label_encoder('tags' , 'tags code')
data.head()
len(data['views'].unique())
data['views'].min()
data['views'].max()
make_kdeplot('views')
def feature_sectors(data , maxx , feature , new_featre):

    step = (maxx- data[feature].min())/10

    new_list = []

    minn = data[feature].min()

    for x in range(data.shape[0]) : 

        if data[feature][x] <= (minn + step):

            new_list.append(1)

        elif data[feature][x] <= (minn + (2*step)):

            new_list.append(2)            

        elif data[feature][x] <= (minn + (3*step)):

            new_list.append(3)            

        elif data[feature][x] <= (minn + (4*step)):

            new_list.append(4)            

        elif data[feature][x] <= (minn + (5*step)):

            new_list.append(5)            

        elif data[feature][x] <= (minn + (6*step)):

            new_list.append(6)            

        elif data[feature][x] <= (minn + (7*step)):

            new_list.append(7)            

        elif data[feature][x] <= (minn + (8*step)):

            new_list.append(8)            

        elif data[feature][x] <= (minn + (9*step)):

            new_list.append(9)            

        else:

            new_list.append(10)            

    data.insert(data.shape[1], new_featre , new_list)   

    

feature_sectors(data ,50000000 , 'views' , 'views sector')
data.head()
make_countplot('views sector')
data.drop(['views sector'],axis=1, inplace=True)
feature_sectors(data ,1000000 , 'views' , 'views sector')
data.head()
make_countplot('views sector')
data.drop(['views sector'],axis=1, inplace=True)
feature_sectors(data ,10000000 , 'views' , 'views sector')
data.head()
make_countplot('views sector')
make_kdeplot('likes')
feature_sectors(data ,500000 , 'likes' , 'likes sectors')
make_countplot('likes sectors')
make_kdeplot('dislikes')
feature_sectors(data ,50000 , 'dislikes' , 'dislikes sectors')
make_countplot('dislikes sectors')
data.drop(['dislikes sectors'],axis=1, inplace=True)
feature_sectors(data ,10000 , 'dislikes' , 'dislikes sectors')
make_countplot('dislikes sectors')
data.head()
make_kdeplot('comment_count')
feature_sectors(data ,10000 , 'comment_count' , 'comment_count sectors')
make_countplot('comment_count sectors')
data.drop(['thumbnail_link'],axis=1, inplace=True)
make_label_encoder('comments_disabled','comments_disabled code')

make_label_encoder('ratings_disabled','ratings_disabled code')

make_label_encoder('video_error_or_removed','video_error_or_removed')
data.head()
len(data['description'].unique())
data.drop(['description'],axis=1, inplace=True)
make_countplot('category_id')
len(data['category_id'].unique())
data.info()
X = data.drop(['category_id'], axis=1, inplace=False)

y = data['category_id']
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 

GBCModel.fit(X_train, y_train)
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth=20,random_state=33) #criterion can be entropy

DecisionTreeClassifierModel.fit(X_train, y_train)



#Calculating Details

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))
#Calculating Prediction

y_pred = DecisionTreeClassifierModel.predict(X_test)

y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)

print('Predicted Value for DecisionTreeClassifierModel is : ' , y_pred[:20])