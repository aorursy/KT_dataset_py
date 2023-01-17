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

import plotly.express as ex

import plotly.figure_factory as ff
p_data = pd.read_csv('/kaggle/input/top-1000-patreons/patreon2.csv')

p_data.head(3)
p_data.Theme = p_data.Theme.apply(lambda x: ' '.join(x.split(' ')[:len(x.split(' '))-1]))

p_data.head(3)
p_data.Patrons = p_data.Patrons.apply(lambda x: ''.join(x.split(',')))

p_data.Patrons = p_data.Patrons.astype('int64')

p_data.head(3)
#different earning type

Earnings = p_data.Earnings

Earnings = Earnings[Earnings.notna()]

earning_types = Earnings.apply(lambda x: ' '.join(x.split(' ')[1:])).to_frame()

earning_types.Earnings.value_counts()
p_data['Per Month'] = 0



p_data.loc[p_data[p_data['Earnings'].str.contains('per month',na=False)].index,'Per Month'] = 1



def other_earning_methods(sir):

    if sir == np.nan or type(sir) ==float:

        return 'NaN'

    elif sir.find('month') != -1:

        return 0

    else:

        return 1



def get_earning_amount(sir):

    if type(sir) == float:

        return 'NaN'

    amount = sir.split(' ')[0]

    amount = ''.join(amount[1:].split(','))

    return amount



p_data['Per Product'] = p_data.Earnings.apply(other_earning_methods)

p_data['Earning Amount'] = p_data.Earnings.apply(get_earning_amount)

p_data.drop(columns=['Earnings'],inplace=True)

p_data.head(3)

Per_patron = p_data['Per patron']

Per_patron = Per_patron[Per_patron.notna()]

earning_types = Per_patron.apply(lambda x: ' '.join(x.split(' ')[1:])).to_frame()

earning_types['Per patron'].value_counts()
p_data['Per Patron Earning'] = p_data['Per patron'].apply(get_earning_amount)

p_data.drop(columns=['Per patron'],inplace=True)

p_data.head(3)

def num_of_words(sir):

    return len(sir.split(' '))

def average_word_length(sir):

    aux =0

    splited = sir.split(' ')

    for i in splited:

        aux+=len(i)

    return aux/len(splited)



adult = ['adult','futanari','18+','nsfw','dating','gta','erotic','nude','sex','love','harem','monster']

def is_adult(sir):

    lowerd = sir.lower()

    for word in adult:

        if lowerd.find(word)!=-1:

            return 1

    return 0



gaming = ['game','rpg','warcraft','sims','nintendo','league','wow','vr','mod','dnd','minecraft']

def is_gaming(sir):

    lowerd = sir.lower()

    for word in gaming:

        if lowerd.find(word)!=-1:

            return 1

    return 0



video = ['video','youtube','acting']

def is_video(sir):

    lowerd = sir.lower()

    for word in video:

        if lowerd.find(word)!=-1:

            return 1

    return 0





music = ['ukulele','music','audio','guitar']

def is_music(sir):

    lowerd = sir.lower()

    for word in music:

        if lowerd.find(word)!=-1:

            return 1

    return 0



art = ['art','paint','animation','comics','sketch']

def is_art(sir):

    lowerd = sir.lower()

    for word in art:

        if lowerd.find(word)!=-1:

            return 1

    return 0



entertainment = ['entertainment','book','reactions','media','cartoons','tarot','post','meme','comedy']

def is_entertainment(sir):

    lowerd = sir.lower()

    for word in entertainment:

        if lowerd.find(word)!=-1:

            return 1

    return 0





education = ['book','lesson ','technology','essay','philosophy','science']

def is_education(sir):

    lowerd = sir.lower()

    for word in education:

        if lowerd.find(word)!=-1:

            return 1

    return 0



formal = ['journalism','news','interviews','radio']

def is_formal(sir):

    lowerd = sir.lower()

    for word in formal:

        if lowerd.find(word)!=-1:

            return 1

    return 0
p_data['Creator # Of Words'] = p_data.Creator.apply(num_of_words)

p_data['Creator Avg Word Length'] = p_data.Creator.apply(average_word_length)

p_data['Theme # Of Words'] = p_data.Theme.apply(num_of_words)

p_data['Theme Avg Word Length'] = p_data.Theme.apply(average_word_length)



p_data['Podcast'] = p_data.Theme.apply(lambda x: 1 if x.lower().find('podcast')!= -1 else 0)

p_data['Adult'] = p_data.Theme.apply(is_adult)

p_data['Gaming'] = p_data.Theme.apply(is_gaming)

p_data['Video'] = p_data.Theme.apply(is_video)

p_data['Music'] = p_data.Theme.apply(is_music)

p_data['Art'] = p_data.Theme.apply(is_art)

p_data['Entertainment'] = p_data.Theme.apply(is_entertainment)

p_data['Education'] = p_data.Theme.apply(is_education)

p_data['Formal'] = p_data.Theme.apply(is_formal)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

import plotly.graph_objs as go



vectize = TfidfVectorizer()

vectize.fit(p_data['Theme'])

tfidf_fets = vectize.transform(p_data['Theme'])



svd_model = TruncatedSVD(n_components=350)

svd_model.fit(tfidf_fets)

tfidf_fets=svd_model.transform(tfidf_fets)



ex_var = svd_model.explained_variance_ratio_

variance_cum = np.cumsum(ex_var)

data = [go.Scatter(x=np.arange(0,len(variance_cum)),y=variance_cum)]

layout = dict(title='Decomposed Tfidf Explained Variance',

             xaxis_title='# Componenets',yaxis_title='Explained Variance')

go.Figure(data=data,layout=layout)
p_data = p_data.replace('NaN',np.nan)

p_data.isna().sum()
info = p_data.describe()

info.loc['kurt'] = p_data.kurt()

info.loc['skew'] = p_data.skew()

info
plt.figure(figsize=(20,11))

ax = sns.distplot(p_data['Patrons'])

ax.set_title('Distribution Of Patron Counts',fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (p_data['Patrons'].mean(),), r'$\mathrm{median}=%.2f$' % (p_data['Patrons'].median(),),

         r'$\sigma=%.2f$' % (p_data['Patrons'].std(),)))

props = dict(boxstyle='round', facecolor='blue', alpha=0.2)

ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)

plt.show()
p_data = p_data[p_data['Patrons']<14000]
plt.figure(figsize=(20,11))

ax = sns.distplot(p_data['Patrons'])

ax.set_title('Distribution Of Patron Counts Without Outliers',fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (p_data['Patrons'].mean(),), r'$\mathrm{median}=%.2f$' % (p_data['Patrons'].median(),),

         r'$\sigma=%.2f$' % (p_data['Patrons'].std(),)))

props = dict(boxstyle='round', facecolor='blue', alpha=0.2)

ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)

plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(p_data['Days Running'])

ax.set_title('Distribution Of Total Days Running',fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (p_data['Days Running'].mean(),), r'$\mathrm{median}=%.2f$' % (p_data['Days Running'].median(),),

         r'$\sigma=%.2f$' % (p_data['Days Running'].std(),)))

props = dict(boxstyle='round', facecolor='blue', alpha=0.2)

ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)

plt.show()
earning_data = p_data[p_data['Earning Amount'].notna()]

earning_data['Per Patron Earning'] = earning_data['Per Patron Earning'].astype('float64')

earning_data['Earning Amount']     = earning_data['Earning Amount'].astype('int64')



plt.figure(figsize=(20,11))

ax = sns.distplot(earning_data['Earning Amount'])

ax.set_title('Distribution Of Total Earnings',fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (earning_data['Earning Amount'].mean(),), r'$\mathrm{median}=%.2f$' % (earning_data['Earning Amount'].median(),),

         r'$\sigma=%.2f$' % (earning_data['Earning Amount'].std(),)))

props = dict(boxstyle='round', facecolor='blue', alpha=0.2)

ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)

plt.show()

earning_data = earning_data[earning_data['Earning Amount']<50000]
plt.figure(figsize=(20,11))

ax = sns.distplot(earning_data['Earning Amount'])

ax.set_title('Distribution Of Total Earnings Without Ourliers',fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (earning_data['Earning Amount'].mean(),), r'$\mathrm{median}=%.2f$' % (earning_data['Earning Amount'].median(),),

         r'$\sigma=%.2f$' % (earning_data['Earning Amount'].std(),)))

props = dict(boxstyle='round', facecolor='blue', alpha=0.2)

ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)

plt.show()

earning_data = earning_data[earning_data['Per Patron Earning']<12]


plt.figure(figsize=(20,11))

ax = sns.distplot(earning_data['Per Patron Earning'])

ax.set_title('Distribution Of Per Patron Earning',fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (earning_data['Per Patron Earning'].mean(),), r'$\mathrm{median}=%.2f$' % (earning_data['Per Patron Earning'].median(),),

         r'$\sigma=%.2f$' % (earning_data['Per Patron Earning'].std(),)))

props = dict(boxstyle='round', facecolor='blue', alpha=0.2)

ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)

plt.show()

ex.scatter(earning_data,y='Earning Amount',x='Days Running',title='Speard of earnings based on days running',height=900)
ex.scatter(earning_data,y='Earning Amount',x='Per Patron Earning',color='Patrons',title='Speard of earnings based on earnings per patron and number of patrons',height=900)
ex.pie(earning_data,'Per Month',title='Proportion Of Per Month And Per Product Patreons ')
earning_cor = earning_data.corr('pearson')

ex.imshow(earning_cor,height=900)

earning_cor = p_data.corr('pearson')

ex.imshow(earning_cor,height=900)

ex.scatter_3d(p_data,x='Patrons',y='Days Running',z='Creator # Of Words',title='Prior To Clustring',height=900)
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy  import dendrogram,linkage

hclustr = AgglomerativeClustering(affinity='manhattan',linkage='average',n_clusters=3)



hclustr.fit(p_data[['Podcast','Adult','Gaming','Video','Music','Art','Entertainment','Education','Formal']])



#hclustr.labels_

shclustr = linkage(p_data[['Podcast','Adult','Gaming','Video','Music','Art','Entertainment','Education','Formal']],method='average')

plt.figure(figsize=(20,13))

ax = plt.subplot(111)

d = dendrogram(shclustr,orientation='right',ax=ax)

ax.set_title('Clustering By The Key Feature Of The Patreon',fontsize=20)

ax.axes.yaxis.set_visible(False)

plt.show()
p_data['Cluster'] = hclustr.labels_
ex.scatter_3d(p_data,x='Patrons',y='Days Running',z='Creator # Of Words',color='Cluster',title='Prior To Clustring',height=900)
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=3)

kmeans.fit(p_data[['Podcast','Adult','Gaming','Video','Music','Art','Entertainment','Education','Formal','Patrons']])

p_data['Cluster'] = kmeans.labels_

ex.scatter_3d(p_data,x='Patrons',y='Days Running',z='Creator # Of Words',color='Cluster',title='Clustered',height=900)
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler



def RMSE(Y,Y_HAT):

    return np.sqrt(mean_squared_error(Y_HAT,Y))
#Tfidf features

tfidf = pd.DataFrame(tfidf_fets)
features = ['Podcast','Adult','Gaming','Video','Music','Art','Entertainment','Education','Formal','Patrons']

earnings = p_data[p_data['Earning Amount'].notna()]

earnings['Earning Amount'] = earnings['Earning Amount'].astype('int')



#the outliers we fixed earlier

earnings = earnings[earnings['Earning Amount']<50000]



Y = earnings['Earning Amount']

X = earnings[features]

indexs = [index for index in Y.index if index not in [985, 986, 987, 996]]

XTF = tfidf.loc[indexs,:]
knnr_model = KNeighborsRegressor(n_neighbors=20)

knnr_model.fit(XTF,Y.loc[indexs])

tfidf_knn_prediction = knnr_model.predict(XTF)



#add to Features

X = X.loc[indexs,:]

Y = Y.loc[indexs]

X['Knnr_Tfidf'] =tfidf_knn_prediction

X
train_x,test_x,train_y,test_y = train_test_split(X,Y)





LR_Pipe = Pipeline(steps = [('model',LinearRegression())])

Knn_Pipe = Pipeline(steps = [('scale',StandardScaler()),('model',KNeighborsRegressor(n_neighbors=20))])

RF_Pipe = Pipeline(steps = [('model',RandomForestRegressor(random_state=42,n_estimators=150))])



LR_Pipe.fit(train_x,train_y)

Knn_Pipe.fit(train_x,train_y)

RF_Pipe.fit(train_x,train_y)



LR_Prediction  = LR_Pipe.predict(test_x)

Knn_Prediction = Knn_Pipe.predict(test_x)

RF_Prediction  = RF_Pipe.predict(test_x)



LR_Score = RMSE(test_y,LR_Prediction)

Knn_Score = RMSE(test_y,Knn_Prediction)

RF_Score = RMSE(test_y,RF_Prediction)



print('LinearRegression RMSE: {} , Knn RMSE: {} , RandomForest RMSE: {}'.format(LR_Score,Knn_Score,RF_Score))
plt.figure(figsize=(25,15))

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=LR_Prediction,label='LinearRegression Prediction',color='g')

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=RF_Prediction,label='KNN Prediction',color='r')
Blended_Prediction = 0.6*LR_Prediction + 0.4*Knn_Prediction
print("Blended RMSE :",RMSE(Blended_Prediction,test_y))
plt.figure(figsize=(25,15))

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=Blended_Prediction,label='Blended Prediction',color='r')

from xgboost import XGBRegressor



holdout_x = train_x.sample(10)

holdout_y = train_y.sample(10)

train_x = train_x.drop(index=holdout_x.index)

train_y = train_y.drop(index=holdout_y.index)
XGB_model = XGBRegressor(n_estimators = 500,learning_rate=0.03,random_state=42,gamma=0.3)

XGB_model.fit(train_x,train_y,early_stopping_rounds=4,eval_set=[(holdout_x,holdout_y)],verbose=False)

XGB_predictions = XGB_model.predict(test_x)
plt.figure(figsize=(25,15))

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=XGB_predictions,label='XGB Prediction',color='r')

from sklearn.preprocessing import PolynomialFeatures



PR_pipe = Pipeline(steps = [('PF',PolynomialFeatures(2)),('model',LinearRegression())])

PR_pipe.fit(train_x,train_y)

PR_Prediction = PR_pipe.predict(test_x)

 

plt.figure(figsize=(25,15))

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=PR_Prediction,label='PolynomialRegression Prediction',color='r')

print("Plynomial Regression RMSE: {}".format(RMSE(test_y,PR_Prediction)))
from sklearn.linear_model import RidgeCV



Ridge_pipe = Pipeline(steps = [('model',RidgeCV())])

Ridge_pipe.fit(train_x,train_y)

Ridge_Prediction = Ridge_pipe.predict(test_x)

 

plt.figure(figsize=(25,15))

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=Ridge_Prediction,label='Ridge Regression Prediction',color='r')

from sklearn.ensemble import StackingRegressor



Stack_Reg = StackingRegressor(estimators=[('LR',LR_Pipe),('RF',RF_Pipe),('KNN',Knn_Pipe),("XGB",XGB_model)],final_estimator=RandomForestRegressor(n_estimators = 500,random_state=42),

                             passthrough=True)



Stack_Reg.fit(train_x,train_y)



stack_pred =Stack_Reg.predict(test_x)







plt.figure(figsize=(25,15))

ax = sns.pointplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.pointplot(x=np.arange(0,len(test_y)),y=stack_pred,label='PolynomialRegression Prediction',color='r')
print("Blended Regression RMSE: {}".format(RMSE(test_y,stack_pred)))
plt.figure(figsize=(25,15))

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=test_y,label='Actual Values',lw=2)

ax = sns.lineplot(x=np.arange(0,len(test_y)),y=Blended_Prediction,label='Blended Prediction',color='r')
