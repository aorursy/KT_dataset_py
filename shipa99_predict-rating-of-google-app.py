import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
df = pd.read_csv('../input/googleplaystore.csv')
df.head()
df.isna().sum()
df.dropna(inplace=True)
CategoryVal = df["Category"].unique()
CategoryDict = {}
for i in range(len(CategoryVal)):
    CategoryDict[CategoryVal[i]] = i
df["Category_id"] = df["Category"].map(CategoryDict).astype(int)

sorted(CategoryDict.items(), key=itemgetter(1))
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1048576
        return(x)
    elif 'k' in size:
        x = size[:-1]
        x = float(x)*1024
        return(x)
    else:
        return None

df["Size"] = df["Size"].map(change_size)
df.Size.fillna(method = 'ffill', inplace = True)
df['Installs'] = [(i[:-1].replace(',','')) for i in df['Installs']]
df.Installs = pd.to_numeric(df.Installs)
RatingL = df['Content Rating'].unique()
RatingDict = {}
for i in range(len(RatingL)):
    RatingDict[RatingL[i]] = i
df['Content Rating Id'] = df['Content Rating'].map(RatingDict).astype(int)

sorted(RatingDict.items(), key=itemgetter(1))
GenresL = df['Genres'].unique()
GenresDict = {}
for i in range(len(GenresL)):
    GenresDict[GenresL[i]] = i
df['Genres_id'] = df['Genres'].map(GenresDict).astype(int)

sorted(GenresDict.items(), key=itemgetter(1))
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price

df['Price'] = df['Price'].map(price_clean).astype(float)
df['Reviews'] = df['Reviews'].astype(int)
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1

df['Type'] = df['Type'].map(type_cat)
df.head()
plt.figure(figsize=(10,10))
g = sns.countplot(y="Category",data=df, palette = "Set2")
plt.title('Number of apps in each category',size = 20);
plt.figure(figsize=(10,10))
g = sns.barplot(x="Installs", y="Category", data=df, palette = "Set2", capsize=.6)
plt.title('Installations in each сategory',size = 20);
data = [go.Histogram(
        x = df.Rating,
        xbins = {'start': 1, 'size': 0.1, 'end' :5}
)]

print('Average app rating = ', np.mean(df['Rating']))
plotly.offline.iplot(data, filename='overall_rating_distribution')
f = stats.f_oneway(df.loc[df.Category == 'BUSINESS']['Rating'].dropna(), 
               df.loc[df.Category == 'FAMILY']['Rating'].dropna(),
               df.loc[df.Category == 'GAME']['Rating'].dropna(),
               df.loc[df.Category == 'PERSONALIZATION']['Rating'].dropna(),
               df.loc[df.Category == 'LIFESTYLE']['Rating'].dropna(),
               df.loc[df.Category == 'FINANCE']['Rating'].dropna(),
               df.loc[df.Category == 'EDUCATION']['Rating'].dropna(),
               df.loc[df.Category == 'MEDICAL']['Rating'].dropna(),
               df.loc[df.Category == 'TOOLS']['Rating'].dropna(),
               df.loc[df.Category == 'PRODUCTIVITY']['Rating'].dropna(),
               df.loc[df.Category == 'COMMUNICATION']['Rating'].dropna()
              )

print(f)

groups = df.groupby('Category').filter(lambda x: len(x) > 286).reset_index()
array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))
plt.figure(figsize=(4,4))
df.Type.value_counts().plot(kind="pie")
print(df.Type.value_counts())
paid_mean_price =  df[df['Type'] == 1].groupby('Category')['Price'].mean().sort_values(ascending=False)
paid_mean_price
plt.figure(figsize=(18,10))
sns.barplot(x=paid_mean_price[:10].index, y=paid_mean_price[:10].get_values())
paid_apps = df[df.Price>0]
p = sns.jointplot( "Price", "Rating", paid_apps)
trace0 = go.Box(
    y=np.log10(df['Installs'][df.Type==1]),
    name = 'Paid',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )

)
trace1 = go.Box(
    y=np.log10(df['Installs'][df.Type==0]),
    name = 'Free',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
layout = go.Layout(
    title = "Number of downloads of paid apps Vs free apps",
    yaxis= {'title': 'Number of downloads (log-scaled)'}
)
data = [trace0, trace1]
plotly.offline.iplot({'data': data, 'layout': layout})
groups = df.groupby('Category').filter(lambda x: len(x) >= 50).reset_index()


sns.set_style("darkgrid")
ax = sns.jointplot(df['Size'], df['Rating'])
data = [{
    #'x': type_groups.get_group(t)['Rating'], 
    'x' : df['Installs'],
    'type':'scatter',
    'y' : df['Size'],
    #'name' : t,
    'mode' : 'markers',
    'showlegend': False,
    'text' : df['Size'],
    } for t in set(df.Type)]


layout = {'title':"Installs vs Size", 
          'xaxis': {'title' : 'Installs'},
          'yaxis' : {'title' : 'Size (in MB)'},
         'plot_bgcolor': 'rgb(0,0,0)'}

plotly.offline.iplot({'data': data, 'layout': layout})
plt.figure(figsize=(12,10))
corrmat = df.corr()
p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
df_copy = df.copy()

df_copy = df_copy[df_copy.Rating > 2]
df_copy = df_copy[df_copy.Installs > 0]

df_copy['Installs'] = np.log10(df['Installs'])

sns.lmplot("Rating", "Installs", data=df_copy)
ax = plt.gca()
_ = ax.set_title('Rating Vs Number of Downloads')
df_copy = df.copy()

df_copy = df_copy[df_copy.Reviews > 10]
df_copy = df_copy[df_copy.Installs > 0]

df_copy['Installs'] = np.log10(df['Installs'])
df_copy['Reviews'] = np.log10(df['Reviews'])

sns.lmplot("Reviews", "Installs", data=df_copy)
ax = plt.gca()
_ = ax.set_title('Number of Reviews Vs Number of Downloads (Log scaled)')
reviews_df = pd.read_csv('../input/googleplaystore_user_reviews.csv')
reviews_df.head()
merged_df = pd.merge(df, reviews_df, on = "App", how = "inner")
merged_df.head()
from wordcloud import WordCloud
wc = WordCloud(background_color="white", max_words=200, colormap="tab20")
# generate word cloud

'''from nltk.corpus import stopwords
stop = stopwords.words('english')
stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
            'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']'''

#merged_df = merged_df.dropna(subset=['Translated_Review'])
merged_df['Translated_Review'] = merged_df['Translated_Review'].apply(lambda x: " ".join(x for x in str(x).split(' ')))
#print(any(merged_df.Translated_Review.isna()))
merged_df.Translated_Review = merged_df.Translated_Review.apply(lambda x: x if 'app' not in x.split(' ') else np.nan)
merged_df.dropna(subset=['Translated_Review'], inplace=True)


free = merged_df.loc[merged_df.Type==0]['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(free)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Top words in free app reviews',size = 20);
plt.show()
paid = merged_df.loc[merged_df.Type==1]['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc = WordCloud(background_color="white", max_words=200, colormap="Set2")
wc.generate(''.join(str(paid)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Top words in paid app reviews',size = 20);
plt.show()
def Evaluationmatrix(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))
def Evaluationmatrix_dict(y_true, y_predict, name = 'All features'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
    return dict_matrix

dff = df.drop(labels = ['Last Updated', 'Current Ver', 'Android Ver', 'App', 'Content Rating'], axis = 1)
dff_1 = pd.get_dummies(dff, columns=['Category', 'Genres'])
dff_1.head()
X_1 = dff_1.drop(labels = ['Rating', 'Category_id', 'Genres_id'],axis = 1)
y_1 = dff_1['Rating']
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.30)
model = LinearRegression()
model.fit(X_train_1,y_train_1)
Results_1 = model.predict(X_test_1)
print ('Mean: ' + str(Results_1.mean()))
print ('Standart deviation: ' + str(Results_1.std()))
resultsdf = pd.DataFrame()
resultsdf = resultsdf.from_dict(Evaluationmatrix_dict(y_test_1,Results_1),orient = 'index')
resultsdf = resultsdf.transpose()
r2_score(y_test_1, Results_1)
dff_2 = pd.get_dummies(dff, columns=['Category'])
dff_2.head()
X_2 = dff_2.drop(labels = ['Rating', 'Category_id', 'Genres_id', 'Genres'],axis = 1)
y_2 = dff_2['Rating']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.30)
model = LinearRegression()
model.fit(X_train_2,y_train_2)
Results_2 = model.predict(X_test_2)
print ('Mean: ' + str(Results_2.mean()))
print ('Standart deviation: ' + str(Results_2.std()))
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_2,Results_2, name = 'w/o genres'),ignore_index = True)
r2_score(y_test_2, Results_2)
dff_3 = pd.get_dummies(dff, columns=['Genres'])
dff_3.head()
X_3 = dff_3.drop(labels = ['Rating', 'Category_id', 'Genres_id', 'Category'],axis = 1)
y_3 = dff_3['Rating']
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.30)
model = LinearRegression()
model.fit(X_train_3,y_train_3)
Results_3 = model.predict(X_test_3)
print ('Mean :' + str(Results_3.mean()))
print ('Standart deviation :' + str(Results_3.std()))
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_3,Results_3, name = 'w/o categories'),ignore_index = True)
r2_score(y_test_3, Results_3)
resultsdf
resultsdf.set_index('Series Name', inplace = True)

plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')
plt.subplot(3,1,2)
resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')
plt.subplot(3,1,3)
resultsdf['Mean Squared Log Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'Mean Squared Log Error')
plt.show()
plt.figure(figsize=(12,7))
plt.grid()
sns.regplot(Results_1,y_test_1,color='teal', label = 'all features', marker = 'x')
sns.regplot(Results_3,y_test_3,color='orange',label = 'w/o genres')
plt.legend()
plt.title('Linear model')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()
from sklearn.ensemble import RandomForestRegressor

X_5 = dff.drop(labels = ['Rating', 'Category', 'Genres'],axis = 1)
y_5 = dff.Rating
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, y_5, test_size=0.30)
model_2 = RandomForestRegressor()
model_2.fit(X_train_5, y_train_5)
Results_5 = model_2.predict(X_test_5)
print ('Mean :' + str(Results_5.mean()))
print ('Standart deviation :' + str(Results_5.std()))
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_5,Results_5, name = 'RFR model'),ignore_index = True)
resultsdf
resultsdf.set_index('Series Name', inplace = True)

plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')
plt.subplot(3,1,2)
resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')
plt.subplot(3,1,3)
resultsdf['Mean Squared Log Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'Mean Squared Log Error')
plt.show()
plt.figure(figsize=(12,7))
plt.grid()
sns.regplot(Results_5, y_test_5, color='orange', label = 'RFR')
sns.regplot(Results_1, y_test_1, color='teal', label = 'linear regression', marker = 'x')
plt.legend()
plt.title('RFR model vs Linear Regression')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()
