import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.style.use('ggplot')

import random
df = pd.read_csv("../input/boston-ds/crime.csv", index_col = None, encoding='windows-1252', parse_dates = ['OCCURRED_ON_DATE', 'YEAR', 'DAY_OF_WEEK'], engine='python')

df.head()
print(f'The dataset contains %s rows and %s columns' % (df.shape[0],df.shape[1]), '\n')

print('The columns and the its values types:\n')

df.info()
codes = pd.read_csv("../input/boston-ds/offense_codes.csv", index_col = None, encoding='windows-1252', engine='python')

codes.CODE.value_counts() #This line is for checking of whether codes are unique or not

codes.drop_duplicates(subset=['CODE'], keep='first', inplace=True) #Since there are duplicates, let's drop them
codes.head()
print(f'The dataset contains %s rows and %s columns' % (codes.shape[0],codes.shape[1]), '\n')

print('The columns and the its values types:\n')

codes.info()
top = df.OFFENSE_CODE.value_counts().to_frame().reset_index(level=0)
top.columns.values[0] = 'CODE'

top.columns.values[1] = 'TOTAL_AMOUNT'

top.head(5)
code_top = top.merge(codes, on='CODE', how = 'left')
code_top.head(10)
gr = df.loc[:,['OFFENSE_CODE','OFFENSE_CODE_GROUP']]

gr.info()

code_tg = pd.merge(code_top, gr, left_on='CODE', right_on='OFFENSE_CODE', how='inner')

code_tg.drop_duplicates(subset=['CODE'], keep='first', inplace=True)

code_tg.reset_index(drop=True,inplace=True)

code_tg.head(5)
code_tg.head(20).plot(kind = 'barh', x = 'NAME', y = 'TOTAL_AMOUNT', figsize=(12, 12))



plt.gca().invert_yaxis() 



plt.xlabel('Number of Reports')

plt.ylabel('Offense Type')

df.sort_values(['OCCURRED_ON_DATE'], ascending=True, inplace=True)

plt.title('Boston Offense Rating: ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[-1]))



# This loop automatically add the value of each position to the each bar:

for index, value in enumerate(code_tg.head(20)['TOTAL_AMOUNT']):

    label = format(int(value), ',')

    plt.annotate(label, xy=(value - 100, index + 0.10), ha='right', color='white')
top_gr = code_tg.groupby(['OFFENSE_CODE_GROUP'], as_index=False).sum(axis=1)

top_gr = top_gr[['OFFENSE_CODE_GROUP','TOTAL_AMOUNT']].sort_values(['TOTAL_AMOUNT'], ascending=False).reset_index(drop=True)
top_gr.head(10)
top_gr.head(10).plot(kind = 'barh', x = 'OFFENSE_CODE_GROUP', y = 'TOTAL_AMOUNT', figsize=(12, 5))



plt.gca().invert_yaxis() 



plt.xlabel('Number of Reports')

plt.ylabel('Offense Group')

plt.title('Boston Offense Groups Rating: ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[-1]))





# This loop automatically add the value of each position to the each bar:

for index, value in enumerate(top_gr.head(10)['TOTAL_AMOUNT']):

    label = format(int(value), ',')

    plt.annotate(label, xy=(value - 300, index + 0.13), ha='right', color='white')
shtng = df[(df.SHOOTING == 'Y') & (df.DISTRICT.notnull())]
import folium

import folium.plugins as plugins



latitude = list(shtng.Lat)[1] # This is to initiate the latitude start point for the map

longitude = list(shtng.Long)[1] # This is to initiate the longitude start point for the map



latitudes = list(shtng.Lat) #create the list of all reported latitudes

longitudes = list(shtng.Long) #create the list of all reported longitudes



shooting_map = folium.Map(location = [latitude, longitude], zoom_start = 12) # instantiate a folium.map object



shooting = plugins.MarkerCluster().add_to(shooting_map) # instantiate a mark cluster object for the incidents in the dataframe



# loop through the dataframe and add each data point to the mark cluster

for lat, lng, label, in zip(shtng.Lat, shtng.Long, shtng.DISTRICT):

    if (not np.isnan(lat)) & (not np.isnan(lng)): # also, we check a non-nullness of the coordinates

        folium.Marker(

            location=[lat, lng],

#             icon=None,

            popup=label,

            icon=folium.Icon(icon='exclamation-sign')

        ).add_to(shooting)



# display the map

shooting_map
# re-assemble the dataset for the more convenient plotting process

top_sh = shtng.DISTRICT.value_counts().to_frame().reset_index(level=0)

top_sh.columns.values[0] = 'DISTRICT'

top_sh.columns.values[1] = 'NUMBER'

top_sh.plot(kind = 'barh', x = 'DISTRICT', y = 'NUMBER', figsize=(12, 7))



# invert y-axis

plt.gca().invert_yaxis()



# Name axis and title

plt.xlabel('Number of Shooting Reports')

plt.ylabel('Districts')



plt.title('Boston Top Shooting Districts: ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[-1]))



# Lop for values plotting

for index, value in enumerate(top_sh['NUMBER']):

    label = format(int(value), ',')

    plt.annotate(label, xy=(value - 1, index + 0.11),

                 ha='right', 

                 color='white'

                )

    

# Loop for arrows plotting. Notice that the arrowhead will always point on the bottom-right bar's corner.

# Also, here I separately defined a starting arrows' point to maximize the procedural plotting 

xy_label = (250,5)

for index, value in enumerate(top_sh['NUMBER']):

    plt.annotate('',

             xy=(value, index + 0.3),

             xytext=xy_label,

             xycoords='data',

             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2',

                             connectionstyle='arc3', 

                             color='xkcd:blue', 

                             lw=2

                            )

            )

    if index == 2: # We want to plot only top 3 the most shooting districts, so we need to interrupt the loop here.

        break



# This dictionary I built using googling method.

dict0 = {'C11' : 'DORCHESTER', 'B3' : 'MATTAPAN', 'B2' : 'ROXBURY'} 



# Plot the district name decoding it using our dictionary.

for index, value in enumerate(top_sh['NUMBER']):

    v = top_sh.loc[top_sh['NUMBER']==value]['DISTRICT'].astype('str')

    plt.annotate('[ ' + dict0[v[index]] + ' ]',

             xy=(value - 15, index + 0.13),

             rotation=0,

             va='bottom',

             ha='right',

             color = 'white'

            )

    if index == 2:

        break

        

# Plot the annotation text. Here I used xy_label defined earlier for automation.

plt.annotate('The Most Shooting Districts', # text to display

             xy=(xy_label[0],xy_label[1] + 0.5),

             rotation=0,

             va='bottom',

             ha='center',

            )

        

plt.show()
shooting_hmap = folium.Map(location=[df.Lat[100],df.Long[100]], 

                       tiles = "Stamen Toner",

                      zoom_start = 12)



from folium.plugins import HeatMap   



hm = df.loc[:,['Lat','Long','SHOOTING']]

hm.dropna(axis=0, inplace=True)

hlimit = hm.shape[0]

hm = hm.sample(hlimit)

hdata = []

for ln, lt in zip(hm.Lat, hm.Long):

    hdata.append((ln,lt))

HeatMap(hdata, 

        gradient = {0.01: 'blue', 0.15: 'lime', 0.25: 'red'},

        blur = 15,

        radius=5).add_to(shooting_hmap)



shooting_hmap
shtngh = df[(df.SHOOTING == 'Y') & (df.HOUR.notnull())]

shtngh.sort_values(['OCCURRED_ON_DATE'], ascending=True, inplace=True)

shtngh.head()
# plot the histogram

plt.figure(figsize=(9, 5))

plt.hist(shtngh.HOUR, bins=range(24))

plt.title('Shooting Time Distribution in Boston: ' + str(shtngh.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(shtngh.OCCURRED_ON_DATE.dt.date.iloc[-1]))

plt.xticks(range(24))



# Decrease Arrow

plt.annotate('',

             xy=(10, 25), # Arrow head

             xytext=(1, 120), # Starting point

             xycoords='data', # Use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='angle3, angleA=110,angleB=0', color='xkcd:blue', lw=2)

            ) # Arrow props



# After Midday Arrow

plt.annotate('',

             xy=(16, 85),

             xytext=(15, 100),

             xycoords='data',

             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:blue', lw=2)

            )



# Latenight Madness Arrow

plt.annotate('',

             xy=(21.5, 190),

             xytext=(20, 65),

             xycoords='data',

             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:blue', lw=2)

            )



# Annotate Text

plt.annotate('Gradual decrease ',

             xy=(2.5, 38),

             rotation=-40,

             va='bottom',

             ha='left',

            )



# Annotate Text

plt.annotate('After midday peak', # text to display

             xy=(15, 100),

             rotation=0,

             va='bottom',

             ha='right',

            )



# Annotate Text

plt.annotate('Latenight madness',

             xy=(19.5, 90),

             rotation=79,

             va='bottom',

             ha='left',    

            )



plt.show()
vd = df[(df.OFFENSE_CODE_GROUP == 'Verbal Disputes') & (df.HOUR.notnull())]

vd.sort_values(['OCCURRED_ON_DATE'], ascending=True, inplace=True)

vd.shape
plt.figure(figsize=(9, 5))

plt.hist(vd.HOUR, bins=range(24))

plt.title('Verbal Disputes Rate in Boston:'  + str(vd.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(vd.OCCURRED_ON_DATE.dt.date.iloc[-1]))

plt.xticks(range(24))

plt.annotate('',

             xy=(21.5, 1400),

             xytext=(5, 200),

             xycoords='data',

             arrowprops=dict(arrowstyle='->', connectionstyle='arc, angleA=90, angleB=-95, armA=40, armB=60, rad=45.0', color='xkcd:blue', lw=2)

            )

plt.annotate('Verbal disputes gradually increases', # text to display

             xy=(3, 1200),

             rotation=0,

             va='bottom',

             ha='left',

            )

plt.annotate('all day long and reaches its peak', # text to display

             xy=(3, 1100),

             rotation=0,

             va='bottom',

             ha='left'

            )

plt.annotate('at midnight', # text to display

             xy=(3, 1000),

             rotation=0,

             va='bottom',

             ha='left',

            )

plt.show()
pred = df.loc[:,['SHOOTING', 'OFFENSE_CODE_GROUP']]

pred.replace(np.nan, 0, inplace=True)

pred.replace('Y', 1, inplace=True)

groups_dummy = pd.get_dummies(pred['OFFENSE_CODE_GROUP'])

pred = pd.concat([pred,groups_dummy], axis=1)

pred.drop(['OFFENSE_CODE_GROUP'], inplace=True, axis=1)
y = pred[['SHOOTING']]

X = pred.iloc[:,1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)

LR
yhat = LR.predict(x_test)

yhat
# Evaluation using Jaccard Index

from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test, yhat)
yhat_prob = LR.predict_proba(x_test)

yhat_prob
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, yhat, labels=[1,0]))
from sklearn.metrics import log_loss

log_loss(y_test, yhat_prob)
feature_importance=pd.concat([pd.DataFrame(X.columns), pd.DataFrame(LR.coef_.T)], axis = 1)

feature_importance.columns = ['features', 'importance']

feature_importance.sort_values(['importance'], ascending=False, inplace=True)

feature_importance = feature_importance.reset_index(drop=True)

feature_importance.head(8)
shtng_gr = shtng.loc[:,['OCCURRED_ON_DATE']]

shtng_gr['Amount'] = 1

shtng_gr['Date'] = pd.DatetimeIndex(shtng_gr.OCCURRED_ON_DATE).normalize()

shtng_gr.drop(['OCCURRED_ON_DATE'], axis=1, inplace=True)

shtng_gr['YM'] = pd.to_datetime(shtng_gr["Date"], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))

shtng_gr['YM'] = pd.to_datetime(shtng_gr["YM"])

shtng_gr.drop(['Date'], axis=1, inplace=True)

shtng_gr = shtng_gr.groupby(['YM'], as_index=False).sum()

shtng_gr.reset_index(drop=False, inplace=True)

shtng_gr.head()
from numpy.polynomial.polynomial import polyfit

px = np.asarray(shtng_gr.index)

b, m = polyfit(shtng_gr.index, shtng_gr.Amount, 1)

shtng_gr.plot(kind='scatter',x='index', y='Amount', rot='90', figsize=(10, 6), alpha = 1, c='xkcd:salmon')

plt.plot(px, b + m * px, '-', c='xkcd:blue')

plt.xticks(shtng_gr.index, shtng_gr.YM.dt.date, rotation=90)

plt.ylabel('Amount of Shooting Reports / Month')

plt.xlabel('Months')

plt.title('Total Shooting Reports in Boston: ' + str(shtng_gr.YM.dt.date.iloc[0]) + ' : ' + str(shtng_gr.YM.dt.date.iloc[-1]))



plt.annotate('Regression Line : Insignificant growth',                      # s: str. Will leave it blank for no text

             xy=(20, 27),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(12, 46),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:blue', lw=2)

            )



plt.show()
ma = df[(df.OFFENSE_CODE_GROUP == 'Medical Assistance') & (df.OCCURRED_ON_DATE.notnull())]
ma_gr = ma.loc[:,['OCCURRED_ON_DATE']]

ma_gr['Amount'] = 1

ma_gr['Date'] = pd.DatetimeIndex(ma_gr.OCCURRED_ON_DATE).normalize()

ma_gr.drop(['OCCURRED_ON_DATE'], axis=1, inplace=True)

ma_gr['YM'] = pd.to_datetime(ma_gr["Date"], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))

ma_gr['YM'] = pd.to_datetime(ma_gr["YM"])

ma_gr.drop(['Date'], axis=1, inplace=True)

ma_gr = ma_gr.groupby(['YM'], as_index=False).sum()

ma_gr.reset_index(drop=False, inplace=True)

ma_gr.head()
import seaborn as sns

plt.figure(figsize=(15, 10))

ax = sns.regplot(x='index', y='Amount', data=ma_gr, color='green', marker='+', scatter_kws={'s': 50,'color':'xkcd:salmon', 'alpha' : 1})



ax.set(xlabel='Months', ylabel='Amount of Medical Assistance Reports / Month')

ax.set_title('Total Medical Assistance Reports in Boston: ' + str(ma_gr.YM.dt.date.iloc[0]) + ' : ' + str(ma_gr.YM.dt.date.iloc[-1]))

ax.set_ylim(200)

plt.xticks(ma_gr.index, ma_gr.YM.dt.date, rotation=90)



plt.annotate('Steady Growth',

             xy=(25, 600),

             xytext=(12, 350),

             xycoords='data',

             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:salmon', lw=2)

            )





plt.show()