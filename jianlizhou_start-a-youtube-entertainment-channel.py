#install the lifelines package

!pip install lifelines
#Import modules

import numpy as np

import pandas as pd

import matplotlib as mpl

from datetime import datetime

from datetime import timedelta

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.ticker import PercentFormatter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from IPython.display import display

import seaborn as sns

import re

import collections 

from nltk.corpus import stopwords

from wordcloud import WordCloud

from lifelines import KaplanMeierFitter

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

%matplotlib inline

sns.set()

import warnings

warnings.filterwarnings('ignore')
#Data cleansing and transformation



#load data

USvideo=pd.read_csv('../input/youtube-new/USvideos.csv')



#convert category_ids to more meaningful category names 

category_name=pd.read_csv('../input/category-id-names/category_id_names.csv') # the category information was found online

category_d=dict(zip(category_name.id,category_name.category))

USvideo['category']=USvideo['category_id'].map(category_d)



#convert "trending_date" and "publish_time" to Date/DateTime

#calculate the day of the week and the month of each trending day

USvideo['trending_date']=pd.to_datetime(USvideo['trending_date'],format="%y.%d.%m")

USvideo['publish_time']=pd.to_datetime(USvideo['publish_time'],format='%Y-%m-%dT%H:%M:%S')

USvideo['publish_time'] = USvideo['publish_time'].dt.tz_convert(None)

USvideo['trending_weekday']=USvideo['trending_date'].dt.weekday_name

USvideo['trending_month']=USvideo['trending_date'].dt.month_name()



#calculate video title lengths

USvideo['title_length']=USvideo['title'].apply(lambda x:len(x))



#calculate ratios of number of comments to number of views

USvideo['comment/views']=USvideo['comment_count']/USvideo['views']



#calculate ratios of number of likes to number of views

USvideo['likes/views']=USvideo['likes']/USvideo['views']



#calculate how long a video stayed on the trending list

USvideo['trendingday_rank']=USvideo.groupby('video_id')['trending_date'].rank(method='dense')

USvideo['trendingdays']=USvideo.groupby('video_id')['trendingday_rank'].transform(max)

USvideo['trendingdays']=USvideo['trendingdays'].apply(lambda x: int(x))



#calculate how fast a video got on the trending list after publication

USvideo['age']=(USvideo['trending_date']-USvideo['publish_time']).dt.days

USvideo['age']=np.where(USvideo['age']==-1,0,USvideo['age'])

USvideo['speed']=USvideo.groupby('video_id')['age'].transform(min)



#calculate the rate of growth in views: set the rate of the first trending day to 1 

USvideo['views_growth_rate']=USvideo.sort_values('trending_date').groupby('video_id')['views'].pct_change()

USvideo['views_growth_rate']=USvideo['views_growth_rate']+1

USvideo['views_growth_rate']=np.where(USvideo['views_growth_rate'].isnull(),1,USvideo['views_growth_rate'])

USvideo=USvideo.round({'views_growth_rate':2})



#delete columns that will not be used in the following analysis and drop duplicated rows

columns_del=['category_id','thumbnail_link', 'comments_disabled', 'ratings_disabled',

       'video_error_or_removed', 'description']

USvideo=USvideo.drop(columns=columns_del)

USvideo=USvideo.drop_duplicates()



#data preview

print("The cleaned dataframe has {} rows, {} columns, and {} null value.\n".format(USvideo.shape[0],USvideo.shape[1],USvideo.isnull().sum().sum()))

USvideo=USvideo.sort_values(by='trending_date')

pd.set_option('display.max_columns', 100)

print("The first 3 rows of the dataframe:\n")

display(USvideo.head(3))
#Daily frequency of the 'Entertainment' category



#Figure 1:data preparation 

USvideo1=USvideo.groupby(['trending_date','category']).size()

USvideo1_all=USvideo1/USvideo1.groupby(level=0).sum()

USvideo1_all=USvideo1_all.to_frame().rename(columns={0:'proportion'}).reset_index()



#Figure 1:data visualization

fig, ax = plt.subplots(figsize=(8,4))

categorylist=USvideo['category'].unique().tolist()



#plot "entertainment"

df_e=USvideo1_all.loc[USvideo1_all['category']=='Entertainment',:]

ax.plot(df_e['trending_date'],df_e['proportion'],color='darkorange')



#plot other categories

for i in categorylist:

    if i!='Entertainment':

        df=USvideo1_all.loc[USvideo1_all['category']==i,:]

        ax.plot(df['trending_date'],df['proportion'],color='lightgray')

        

#customize figure legend

custom_lines = [Line2D([0], [0], color='darkorange', lw=2),

                Line2D([0], [0], color='lightgray', lw=2)]

ax.legend(custom_lines, ['Entertainment', 'Other Categories'])



#customize axes and title 

plt.title("Figure 1. 'Entertainment' appears most frequently among the 16 trending categories\n",fontsize=13,y=-0.35)

ax.set(xlabel='date',ylabel='percentage of videos')

ax.set_ylim(0,0.4)



#change y axis values to percentage

ax.yaxis.set_major_formatter(PercentFormatter(1))



#add a horizontal lines to show the mean percentage of the 'Entertainment' category

ent_mean=USvideo1_all.loc[USvideo1_all['category']=='Entertainment','proportion'].mean()

ax.axhline(y=ent_mean,xmin=0,xmax=1,linestyle='--')

plt.show()
#Average views by category



#Figure 2: data preparation

USvideo2=USvideo.groupby('category')['views'].mean().sort_values(ascending=True)



#Figure 2: data visualization

colors=['lightgray']*16

colors[11]='darkorange'

plt.figure(figsize=(8,4))

USvideo2.plot(kind='barh',color=colors,width=0.75)

plt.xlabel("average views per video")

plt.title("Figure 2.\'Entertainment\' is the 5th most-viewed category\n",fontsize=13,y=-0.35)

E_views=round(USvideo2['Entertainment']/1000000,2)

plt.text(2200000,10.6,'{} million views/video'.format(E_views),color='black',fontsize=12)

plt.show()
#Additional data cleansing and transformation



#Now I will focus only on the "entertainment" category

USvideo3=USvideo.loc[USvideo['category']=='Entertainment'].copy()



#clean "tags" 

USvideo3['tags']=USvideo3['tags'].str.lower().str.replace('"','')

USvideo3['tags_new']=USvideo3['tags'].apply(lambda x: x.split('|') )

USvideo3['tags_new']=np.where(USvideo3['tags']=='[none]','',USvideo3['tags_new'])



#calculate the number of tags used in each video

USvideo3['tags_number']=USvideo3['tags_new'].apply(lambda x:len(x))

USvideo3['tags_number']=np.where(USvideo3['tags']=='[none]',0,USvideo3['tags_number'])



#clean "title"

USvideo3['title_new']=USvideo3['title'].str.lower().str.replace("[^a-zA-Z0-9]", " ")



#remove the first 29 days from the dataset 

#Why? We don't know if the trending videos on the earlier days were trending before  

#those days or not, and thus the calculated "trendingdays" may be inaccurate for those early videos.

# 29 was chosen because it was the maximum days a video stayed trending.

first=min(USvideo3['trending_date'])+timedelta(days=29)

USvideo4=USvideo3.loc[USvideo3['trending_date'] > first,:].copy()



#create a smaller subset by picking only the first trending day information of the videos

#Why? Some videos appeared multiple times in the dataset due to the fact that they were trending for more than 1 day.

#USvideo4 created above will still be used, but we also need one record per video for machine learning.

USvideo5=USvideo4.loc[USvideo4.trendingday_rank==1.0,:].copy()
#Wordcloud of title

st=stopwords.words('english')

titles=' '.join( i for i in USvideo5.title_new)

cloud = WordCloud(stopwords=st,max_words=100,background_color="white").generate(titles)

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off")

plt.title("Figure 3. Wordcloud generated from titles of 'Entertainment' videos",y=-0.2,fontsize=13)

plt.show()
#wordcloud of tags



#write a function to count tag frequency

def create_tagcounter(data):

    tagslist=data['tags_new'].tolist()

    tags_flatlist=[item for sublist in tagslist for item in sublist]

    tags_d=collections.Counter(tags_flatlist)

    return tags_d



#generate wordcloud from tag frequency

cloud2=WordCloud(stopwords=st,max_words=100,background_color="white").generate_from_frequencies(create_tagcounter(USvideo5))

plt.imshow(cloud2, interpolation='bilinear')

plt.axis("off")

plt.title("Figure 4. Wordcloud generated from tags of 'Entertainment' videos",y=-0.2,fontsize=13)

plt.show()
# growth rate of views (views/views on the previous day)

plt.figure(figsize=(8,4))

sns.scatterplot(x='trendingday_rank',y='views_growth_rate',data=USvideo4,s=85,alpha=0.5)

plt.ylim(0,20)

plt.xticks(range(0,30,1))

plt.yticks(range(0,20,2))

plt.xlabel('days on trending')

plt.ylabel('growth rate of views')

plt.title("Figure 5. \'Entertainment\' witnesses the fastest growth rate of views on the 2nd trending day \n ",fontsize=13,y=-0.35)

plt.show()
#survival analysis of entertainment videos on the trending list



#data preparation: The "birth" event is the presence on the trending list and the "death" event is the exit of the trending list

#Censoring occur if the video is still on the trending list at the time of data collection (Jun 14, 2018)

USvideo5['event']=np.where(USvideo5['trending_date']==max(USvideo5['trending_date']),0,1)



#fit model

kmf = KaplanMeierFitter()

kmf.fit(USvideo5['trendingdays'], event_observed=USvideo5['event'],label='Kaplan Meier Estimate')

kmf.plot()

plt.xticks(range(0,30,2))

plt.xlabel("timeline(days)")

plt.ylabel("probability of being on the trending list")

plt.title("Figure 6. Duration of 'Entertainment' videos on the trending list",fontsize=13,y=-0.35)

plt.show()
#Other interesting facts about the entertainment category

plt.figure(figsize=(8,4))



#pie chart: percentage of videos that got on the trending list within 1 day

plt.axes([0.08, 0.5, 0.25, 0.5])

groups=['speed<= 1 day','speed >1 day']

size=[(USvideo5['speed']<=1).sum(),(USvideo5['speed']>1).sum()]

center_circle=plt.Circle((0,0),0.7,color='white')

plt.pie(size,labels=groups,colors=['lightsteelblue','lightgray'],labeldistance=1.05,startangle=0,textprops={'fontsize': 11})

plt.gcf().gca().add_artist(center_circle) 

plt.axis('equal')

plt.text(-0.38,-0.05,'76.5%',fontsize=18,weight='bold',color='cornflowerblue')



#histogram: distribution of number of tags

plt.axes([0.6, 0.55, 0.35, 0.4])

sns.distplot(USvideo5['tags_number'])

plt.xlabel('number of tags')

plt.ylabel('proportion')



#waffle chart: number of 'likes' per 100 views 

plt.axes([0,0,0.4,0.4])

temp=np.zeros((10,10))

temp[9,:3]=1

plt.matshow(temp,fignum=0,cmap=mpl.colors.ListedColormap(['lightgray', 'lightsteelblue']))

plt.xticks(np.arange(-0.5,10,1),labels=[])

plt.yticks(np.arange(-0.5,10,1),labels=[])

likes_num=round(USvideo5['likes/views'].mean(),2)*100

plt.text(0,4.5,'3 "likes" / 100 views',fontsize=9,weight='bold',color='cornflowerblue')

plt.title("Figure 7. Other facts about the 'Entertainment' category",x=1,y=-0.4,fontsize=13)



#boxplot: distribution of title length

plt.axes([0.55,0.05,0.4,0.3])

sns.boxplot(USvideo5['title_length'],color='lightsteelblue')

plt.xlabel('length of title')

plt.show()
#Can my video last for more than 6 days on the trending list?

#Target: trendingdays>= 6 days or trendingdays<6 days

#Predictors:videos' 1st trending days' dynamic features and static features



#data preparation

#select relevant columns

newcolumns=['trendingdays','views', 'comment/views',

       'likes/views','speed','tags_number', 'title_length','trending_weekday',"trending_month"] 



#remove 2018 June videos as I wasn't sure if those videos kept trending after the last data collection day (June 14, 2019).

USvideo6=USvideo5.loc[USvideo5.trending_month!='June',newcolumns]



#creat the binary "Y" column 

USvideo6['6_days']=np.where(USvideo6.trendingdays>=6,1,0)



#more data transformation

USvideo6['views']=np.log(USvideo6['views'])

USvideo6.rename(columns={'views':'log_views'},inplace=True)

USvideo6=pd.get_dummies(USvideo6,drop_first=True)

USvideo7=USvideo6.drop(columns='trendingdays')



#train/test split

X=USvideo7.drop(columns='6_days')

y=USvideo7['6_days']

X_train1, X_test1, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)



#data standardization

X_train2=X_train1.iloc[:,:6]

X_train3=X_train1.iloc[:,6:]

X_test2=X_test1.iloc[:,:6]

X_test3=X_test1.iloc[:,6:]

X_train4 = StandardScaler().fit_transform(X_train2)

X_train5 = pd.DataFrame(X_train4, index=X_train2.index, columns=X_test2.columns)

X_test4 = StandardScaler().fit_transform(X_test2)

X_test5 = pd.DataFrame(X_test4, index=X_test2.index, columns=X_test2.columns)

X_train=pd.concat([X_train3,X_train5],axis=1)

X_test=pd.concat([X_test3,X_test5],axis=1)



#let's look at the correlation heatmap first

plt.figure(figsize=(8,5))

mask = np.zeros_like(USvideo7.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax=sns.heatmap(USvideo7.corr(),mask=mask,cmap='Blues')

    ax.set_title("Figure 8. Correlation Heatmap of 'Entertaiment' video variables",fontsize=13)
#dummy classifier

dummy = DummyClassifier(random_state=2)

dummy.fit(X_train, y_train)

y_pred_dummy=dummy.predict(X_test)

y_pred_prob_dummy=dummy.predict_proba(X_test)[:,1]
#logistic regression

lr=LogisticRegression(solver='lbfgs')

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

y_pred_prob_lr=lr.predict_proba(X_test)[:,1]
#decision tree

tree1 = DecisionTreeClassifier(random_state = 12)



#find best parameters

param_tree = {"max_depth": range(1,10),

           "min_samples_split": range(2,10,1),

           "max_leaf_nodes": range(2,5)}

grid_tree = GridSearchCV(tree1,param_tree,cv=5)

grid_tree.fit(X_train,y_train)

bp_tree=grid_tree.best_params_



#fit model

tree2=DecisionTreeClassifier(random_state = 12,max_depth=bp_tree['max_depth'],min_samples_split=bp_tree['min_samples_split'],max_leaf_nodes=bp_tree['max_leaf_nodes'])

tree2.fit(X_train,y_train)

y_pred_tree=tree2.predict(X_test)

y_pred_prob_tree=tree2.predict_proba(X_test)[:,1]
#Support vector classifier

svc_kernel = SVC(kernel = 'rbf')



#find best parameters

param_svc_kernel = {'C': [1,10,100,1000,10000],'gamma':[0.001,0.001,0.1,1,10]}

grid_svc_kernel = GridSearchCV(svc_kernel, param_svc_kernel, cv=5, n_jobs=2)

grid_svc_kernel.fit(X_train, y_train)

bp_svc=grid_svc_kernel.best_params_



#fit model

svc2=SVC(kernel = 'rbf',C=bp_svc['C'],gamma=bp_svc['gamma'],probability=True)

svc2.fit(X_train,y_train)

y_pred_svc=svc2.predict(X_test)

y_pred_prob_svc=svc2.predict_proba(X_test)[:,1]
#K nearest neighbor

knn = KNeighborsClassifier()



# find best parameters

param_knn = {'n_neighbors': range(1,20)}

grid_knn = GridSearchCV(knn, param_knn, cv=5)

grid_knn.fit(X_train, y_train)

bp_knn=grid_knn.best_params_



#fit model

knn2=KNeighborsClassifier(n_neighbors = bp_knn['n_neighbors'])

knn2.fit(X_train,y_train)

y_pred_knn=knn2.predict(X_test)

y_pred_prob_knn=knn2.predict_proba(X_test)[:,1]
#XGBoost classifier

xgb = XGBClassifier()



#find best parameters

parameters = {

     "eta"    : [0.05, 0.15,0.25, 0.30 ] ,

     "max_depth"        : [ 3, 4, 5, 6],

     "min_child_weight" : [ 1, 3, 5, 7 ],

     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

     }

grid_xgb = GridSearchCV(xgb,

                    parameters, 

                    cv=3)

grid_xgb.fit(X_train, y_train)

bp_xgb=grid_xgb.best_params_



#fit model 

xgb1 = XGBClassifier(eta=bp_xgb['eta'],max_depth=bp_xgb['max_depth'],min_child_weight=bp_xgb['min_child_weight'],gamma=bp_xgb['gamma'],colsample_bytree=bp_xgb['colsample_bytree'])

xgb1.fit(X_train, y_train)

y_pred_prob_xgb = xgb1.predict(X_test)

y_pred_xgb= [round(value) for value in y_pred_prob_xgb]
# model performance summary



#plot roc curves and calculate roc_auc score for each model

y_pred_list=[y_pred_dummy,y_pred_lr,y_pred_svc,y_pred_tree,y_pred_knn,y_pred_xgb]

y_pred_prob_list=[y_pred_prob_dummy,y_pred_prob_lr,y_pred_prob_svc,y_pred_prob_tree,y_pred_prob_knn,y_pred_prob_xgb]

model_list=['Dummy Classifier','Logistic Regression','SVC','Decision Tree','KNN','XGBoost']

for i in range(len(y_pred_list)):

    fpr, tpr, threshold = roc_curve(y_test,y_pred_prob_list[i])

    with sns.axes_style("white"):

        plt.plot(fpr,tpr,label=model_list[i]+"(auc=" + "{0:0.2f}".format(roc_auc_score(y_test,y_pred_prob_list[i]))+")")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("Figure 9. ROC curves of models predicting if a video will stay trending for at least 6 days ",fontsize=13,y=-0.3)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.show()