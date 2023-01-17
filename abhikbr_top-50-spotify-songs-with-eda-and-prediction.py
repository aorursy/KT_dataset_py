# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objects as go

import plotly.express as px





from sklearn.model_selection import train_test_split,cross_val_score, KFold

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler,LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB

from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import confusion_matrix, classification_report

from sklearn import metrics

from scipy import stats

from sklearn.preprocessing import StandardScaler
songs=pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

songs.head()
songs.info()
songs.describe()
songs.drop('Unnamed: 0',axis=1,inplace=True)
plt.figure(figsize=(12,7))

sns.countplot(x='Popularity',data=songs,palette="viridis")
plt.figure(figsize=(12,6))

sns.heatmap(songs.corr(),annot=True)
plt.figure(figsize=(12,6))

sns.lineplot(x="Loudness..dB..",y='Energy',data=songs)
plt.figure(figsize=(12,6))

sns.lineplot(x="Valence.",y='Energy',data=songs)
songs['Genre'].value_counts()
plt.figure(figsize=(25,15))

order=['dance pop','pop','latin','edm','canadian hip hop','panamanian pop','dfw rap','canadian pop','brostep','electropop','reggaeton','reggaeton flow','country rap',

      'atl hip hop','escape room','australian pop','trap music','r&b en espanol','big room','pop house','boy band']

sns.countplot(y=songs['Genre'],data=songs,orient="h",order=order,palette="rainbow")
most_popular=songs[songs['Popularity']>89]

medium_popular=songs[(songs['Popularity']>79) & (songs['Popularity']<90) ]

less_popular=songs[(songs['Popularity']>69) & (songs['Popularity']<80)]
most_popular['Type']=most_popular.apply(lambda x:'most popular',axis=1)

medium_popular['Type']=medium_popular.apply(lambda x:'medium popular',axis=1)

less_popular['Type']=less_popular.apply(lambda x:'less popular',axis=1)
popular_divided=pd.concat([most_popular,medium_popular,less_popular])

popular_divided.tail()
import matplotlib.gridspec as gridspec
fig3 = plt.figure(figsize=(15,10))

gs = fig3.add_gridspec(2, 2)

f3_ax1 = fig3.add_subplot(gs[0, :])

f3_ax2 = fig3.add_subplot(gs[-1:, -1])

f3_ax3 = fig3.add_subplot(gs[-1, -2])

sns.lineplot(x=most_popular['Popularity'],y=most_popular['Length.'],data=most_popular,ax=f3_ax1)

f3_ax1.set_title("Most Popular")

sns.lineplot(x=medium_popular['Popularity'],y=medium_popular['Length.'],data=medium_popular,ax=f3_ax2)

f3_ax2.set_title("Medium Popular")

sns.lineplot(x=less_popular['Popularity'],y=less_popular['Length.'],data=less_popular,ax=f3_ax3)

f3_ax3.set_title("Less Popular")
fig3 = plt.figure(figsize=(15,10))

gs = fig3.add_gridspec(2, 2)

f3_ax4 = fig3.add_subplot(gs[0, :])

f3_ax5 = fig3.add_subplot(gs[-1:, -1])

f3_ax6 = fig3.add_subplot(gs[-1, -2])

sns.violinplot(x=most_popular['Popularity'],y=most_popular['Energy'],data=most_popular,ax=f3_ax4).set_title("Most Popular")

sns.violinplot(x=medium_popular['Popularity'],y=medium_popular['Energy'],data=medium_popular,ax=f3_ax5).set_title("Medium Popular")

sns.violinplot(x=less_popular['Popularity'],y=less_popular['Energy'],data=less_popular,ax=f3_ax6).set_title("Less Popular")
fig3 = plt.figure(figsize=(15,10))

gs = fig3.add_gridspec(2, 2)

f3_ax4 = fig3.add_subplot(gs[0, :])

f3_ax5 = fig3.add_subplot(gs[-1:, -1])

f3_ax6 = fig3.add_subplot(gs[-1, -2])

sns.boxplot(x=most_popular['Popularity'],y=most_popular['Beats.Per.Minute'],data=most_popular,ax=f3_ax4).set_title("Most Popular")

sns.boxplot(x=medium_popular['Popularity'],y=medium_popular['Beats.Per.Minute'],data=medium_popular,ax=f3_ax5).set_title("Medium Popular")

sns.boxplot(x=less_popular['Popularity'],y=less_popular['Beats.Per.Minute'],data=less_popular,ax=f3_ax6).set_title("Less Popular")
songs['Genre'].unique()
most_popular['Popularity']=most_popular['Popularity'].astype(int)

medium_popular['Popularity']=medium_popular['Popularity'].astype(int)

less_popular['Popularity']=less_popular['Popularity'].astype(int)
fig3 = plt.figure(figsize=(20,15))

gs = fig3.add_gridspec(2, 2)

f3_ax7 = fig3.add_subplot(gs[0, :])

f3_ax8 = fig3.add_subplot(gs[-1:, -1])

f3_ax9 = fig3.add_subplot(gs[-1, -2])

sns.barplot(x='Popularity',y='Genre',data=most_popular,ax=f3_ax7,orient="h")

f3_ax7.set_title("Most Popular")

sns.barplot(x='Popularity',y='Genre',data=medium_popular,ax=f3_ax8,orient="h")

f3_ax8.set_title("Medium Popular")

sns.barplot(x='Popularity',y='Genre',data=less_popular,ax=f3_ax9,orient="h")

f3_ax9.set_title("Less Popular")
dance_pop=popular_divided[popular_divided['Genre']=="dance pop"]

dance_pop.tail()
fig3 = plt.figure(figsize=(20,15))

gs = fig3.add_gridspec(2, 2)

f3_ax10 = fig3.add_subplot(gs[0, :])

f3_ax11 = fig3.add_subplot(gs[-1:, -1])

f3_ax12 = fig3.add_subplot(gs[-1, -2])

sns.barplot(x='Popularity',y='Artist.Name',data=most_popular,ax=f3_ax10,orient="h",palette="rainbow")

f3_ax10.set_title("Most Popular")

sns.barplot(x='Popularity',y='Artist.Name',data=medium_popular,ax=f3_ax11,orient="h",palette="rainbow")

f3_ax11.set_title("Medium Popular")

sns.barplot(x='Popularity',y='Artist.Name',data=less_popular,ax=f3_ax12,orient="h",palette="rainbow")

f3_ax12.set_title("Less Popular")
plt.figure(figsize=(15,10))

sns.pairplot(popular_divided,hue='Type')
cm = sns.light_palette("green", as_cmap=True)

table=pd.pivot_table(popular_divided,index=['Type','Artist.Name','Genre'])

s = table.style.background_gradient(cmap=cm)

s
billie_ellish=songs[(songs['Artist.Name']=="Billie Eilish")]

billie_ellish
fig3 = plt.figure(figsize=(15,10))

gs = fig3.add_gridspec(2, 2)

f3_ax4 = fig3.add_subplot(gs[0, :])

f3_ax5 = fig3.add_subplot(gs[-1:, -1])

f3_ax6 = fig3.add_subplot(gs[-1, -2])

sns.boxplot(x=most_popular['Popularity'],y=most_popular['Speechiness.'],data=most_popular,ax=f3_ax4).set_title("Most Popular")

sns.boxplot(x=medium_popular['Popularity'],y=medium_popular['Speechiness.'],data=medium_popular,ax=f3_ax5).set_title("Medium Popular")

sns.boxplot(x=less_popular['Popularity'],y=less_popular['Speechiness.'],data=less_popular,ax=f3_ax6).set_title("Less Popular")
fig3 = plt.figure(figsize=(15,10))

gs = fig3.add_gridspec(2, 2)

f3_ax4 = fig3.add_subplot(gs[0, :])

f3_ax5 = fig3.add_subplot(gs[-1:, -1])

f3_ax6 = fig3.add_subplot(gs[-1, -2])

sns.lineplot(x=most_popular['Popularity'],y=most_popular['Acousticness..'],data=most_popular,ax=f3_ax4).set_title("Most Popular")

sns.lineplot(x=medium_popular['Popularity'],y=medium_popular['Acousticness..'],data=medium_popular,ax=f3_ax5).set_title("Medium Popular")

sns.lineplot(x=less_popular['Popularity'],y=less_popular['Acousticness..'],data=less_popular,ax=f3_ax6).set_title("Less Popular")
fig3 = plt.figure(figsize=(15,10))

gs = fig3.add_gridspec(2, 2)

f3_ax4 = fig3.add_subplot(gs[0, :])

f3_ax5 = fig3.add_subplot(gs[-1:, -1])

f3_ax6 = fig3.add_subplot(gs[-1, -2])

sns.boxplot(x=most_popular['Popularity'],y=most_popular['Valence.'],data=most_popular,ax=f3_ax4).set_title("Most Popular")

sns.boxplot(x=medium_popular['Popularity'],y=medium_popular['Valence.'],data=medium_popular,ax=f3_ax5).set_title("Medium Popular")

sns.boxplot(x=less_popular['Popularity'],y=less_popular['Valence.'],data=less_popular,ax=f3_ax6).set_title("Less Popular")
songs.isnull().sum()
less=['Shawn Mendes','Lauv']

medium=['Ariana Grande','Ed Sheeran','Lil Nas X','DJ Snake','Lewis Capaldi','Chris Brown','Y2K','Jhay Cortez','Tones and I','Ali Gatie','J Balvin',

 'The Chainsmokers', 'Ariana Grande','Maluma','Young Thug','Katy Perry','Martin Garrix','Ed Sheeran','Jonas Brothers','Kygo','Lady Gaga','Khalid','ROSALÍA','Marshmello',

'Nicky Jam','Marshmello','The Chainsmokers']

most=['Anuel AA','Post Malone','Lil Tecca','SamSmith','Bad Bunny','Drake','J Balvin','Post Malone','Lizzo','MEDUZA','Lil Nas X','Lunay','Daddy Yankee',

      'Taylor Swift']

common=['Billie Eilish','Sech']
def encoding(x):

    if x in less:

        return 0

    elif x in medium:

        return 1

    elif x in common:

        return 2

    elif x in most:

        return 3
songs['Artist_Dummy']=songs['Artist.Name'].apply(encoding)

songs.head()
songs['Genre'].unique()
final=pd.get_dummies(songs,columns=['Genre'],drop_first=True)

final.head()

final.drop(['Artist.Name','Track.Name','Loudness..dB..'],axis=1,inplace=True)
final=final.fillna(0)
final.isna().sum()
X=final.drop('Popularity',axis=1)

y=final['Popularity']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=101)
#Linear Regression

regression=LinearRegression()

regression.fit(X_train,y_train)

y_pred=regression.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df_output)
#Checking the accuracy of Linear Regression

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#KNN



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=101)



sc=StandardScaler()

sc.fit(X_train)

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)

# sorted(sklearn.neighbors.VALID_METRICS['brute'])

error=[]

for i in range(1,30):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(10,10))

plt.plot(range(1,30),error,color='black',marker='o',markerfacecolor='cyan',markersize=10)

plt.title('Error Rate K value')

plt.xlabel('K Value')

plt.ylabel('Mean error')
knn=KNeighborsClassifier(n_neighbors=18)

knn.fit(X_train,y_train)

pred_i=knn.predict(X_test)

#Checking the accuracy of Linear Regression

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_i))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_i))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_i)))
df_output_Knn = pd.DataFrame({'Actual': y_test, 'Predicted': pred_i})

print(df_output_Knn)
#GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred=gnb.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df_output)
score=cross_val_score(gnb,X_train,y_train,scoring='accuracy',cv=2).mean()

print(score)
#MultinominalNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

y_pred_mnb=mnb.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_mnb})

print(df_output)
score_mnb=cross_val_score(mnb,X_train,y_train,scoring='accuracy',cv=3).mean()

print(score_mnb)
# Linear SVM model 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)

LinSVC = LinearSVC(penalty='l2', loss='squared_hinge', dual=True)

LinSVC.fit(X_train, y_train)

y_pred_svm=LinSVC.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_svm})

print(df_output)
# Testing the accuracy

scores_svm=cross_val_score(LinSVC,X_train,y_train,scoring='accuracy',cv=3).mean()

print(scores_svm)