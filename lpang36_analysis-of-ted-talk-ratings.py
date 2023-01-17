# Hide warnings if there are any
import warnings
warnings.filterwarnings('ignore')
import ast

# Load in Python libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD
import scipy.stats
from scipy.stats import gaussian_kde
import re
import ast
import random
from sklearn import preprocessing

matplotlib.style.use('ggplot')

df = pd.read_csv('../input/ted_main.csv')

%matplotlib inline
df.sample()

df['ratings'] = df['ratings'].apply(lambda x: eval(str(x))) #turns stringified dictionary into python dictionary

counter = {'Funny':0, 'Beautiful':0, 'Ingenious':0, 'Courageous':0, 'Longwinded':0, 'Confusing':0, 'Informative':0, 'Fascinating':0, 'Unconvincing':0, 'Persuasive':0, 'Jaw-dropping':0, 'OK':0, 'Obnoxious':0, 'Inspiring':0}

for i in range(len(df['ratings'])):
    for j in range(len(df['ratings'][i])):
        counter[df['ratings'][i][j]['name']] += df['ratings'][i][j]['count']
    
frequencies = list(counter.values())
descr = counter.keys()
descriptors = [x for _,x in sorted(zip(frequencies,counter.keys()), reverse=True)]
neg_descriptors = {"Confusing", "Unconvincing", "Longwinded", "Obnoxious", "OK"}
neg_indices  = [x for x in range (len(descriptors)) if descriptors[x] in neg_descriptors]
frequencies.sort(reverse=True)

indices = np.arange(len(descriptors))
bar = plot.bar(indices, frequencies, 0.8)
[bar[i].set_color('b') for i in neg_indices]
plot.xticks(indices, descriptors, rotation=45, ha="right")
plot.show()
df['aggregateRatings'] = df['ratings'].apply(lambda x: \
                                            x[0]['count']+ \
                                            x[1]['count']- \
                                            x[2]['count']+ \
                                            x[3]['count']- \
                                            x[4]['count']- \
                                            x[5]['count']+ \
                                            x[6]['count']+ \
                                            x[7]['count']+ \
                                            x[8]['count']+ \
                                            x[9]['count']+ \
                                            x[10]['count']+ \
                                            x[11]['count']- \
                                            x[12]['count']- \
                                            x[13]['count'])
                                            

ar = df['aggregateRatings']
df.plot.scatter(x='languages',y='aggregateRatings',color='black')
#define convenience function for trend lines
def plotTrendLine(x,y,data=df,color='red',logx=False,logy=False):
    oldx = np.reshape(data[x].values,(-1,))
    oldy = np.reshape(data[y].values,(-1,))
    tempx = oldx if not logx else np.log10(oldx)
    tempy = oldy if not logy else np.log10(oldy)
    
    idx = np.isfinite(tempx) & np.isfinite(tempy)
    z = np.polyfit(tempx[idx],tempy[idx],1)
    tempy = z[0]*tempx+z[1]
    plot.plot(oldx,tempy,color=color)
    return z
plotTrendLine('languages','aggregateRatings');
df.plot.scatter(x='views',y='aggregateRatings',color='black')
plotTrendLine('views','aggregateRatings',logx=True);
plot.xscale('log')
df.plot.scatter(x='comments',y='aggregateRatings',color='black')
plotTrendLine('comments','aggregateRatings',logx=True);
plot.xscale('log')
df['totalRatings'] = df['ratings'].apply(lambda x: sum([x[i]['count'] for i in range(len(x))]))

df['avgPerRating'] = df['aggregateRatings']/df['totalRatings']
plot.hist(x=df['avgPerRating'],bins=50,range=(-1,1),color='black')
#convenience function for density curve plotting
def plotDensityCurve(x,linspace,covariance_factor=0.25,multiplier=1,data=df,color='red'):
    tempx = np.reshape(data[x].values,(-1,)) if data is not None else x
    density = gaussian_kde(tempx)
    xs = np.linspace(linspace[0],linspace[1],linspace[2])
    density.covariance_factor = lambda: covariance_factor
    density._compute_covariance()
    plot.plot(xs,density(xs)*multiplier)
plotDensityCurve('avgPerRating',(-1,1,50),multiplier=100)
df.plot.scatter(x='comments',y='avgPerRating',color='black')
coeffs = plotTrendLine('comments','avgPerRating',logx=True);
plot.xscale('log')
print("Slope: "+str(coeffs[0]))
df.plot.scatter(x='views',y='avgPerRating',color='black')
coeffs = plotTrendLine('views','avgPerRating',logx=True);
plot.xscale('log')
print("Slope: "+str(coeffs[0]))
df['related_talks'] = df['related_talks'].apply(lambda x: eval(str(x)))

groups = {}
for i in range(len(df)):
    cur = df['title'][i]
    related = [df['related_talks'][i][j]['title'] for j in range(len(df['related_talks'][i]))]
    groups[cur] = set(related)
    groups[cur].add(cur)
    for rel in related:
        if rel in groups and rel != cur:
            groups[cur].union(groups.pop(rel))
                   
groups = [g for g in groups.values() if len(g) > 0]
lens = [len(t) for t in groups]

plot.hist(lens,color='black')
print("Number of groups: " + str(len(groups)))
print("Largest Group Size: " + str(max(lens)))
print("Smallest Group Size: " + str(min(lens)))
print("Average Group Size: " + str(np.mean(lens)))
title_to_id = {}
for i in range(len(df)):
    title_to_id[df['title'][i]] = i

group_avg = []
for g in groups:
    group_avg.append([df['avgPerRating'][title_to_id[title]] for title in g])

group_std = [np.std(nums) for nums in group_avg]
plot.hist(group_std,color='black')
print("Standard deviation for the set: " + str(np.std(df['avgPerRating'])))
print("Average standard deviation among groups: " + str(np.mean(group_std)))
df.sample()
mldf = df[['num_speaker','duration','comments','languages','views', 'film_date', 'published_date']]
mldf['descriptionSentiment'] = df['description'].apply(lambda x:TextBlob(re.sub(r'[^\x00-\x7f]',r'',x)).sentiment.polarity)
print("Sentiment: ")
print(mldf['descriptionSentiment'].head())
print("Description: ")
print(df['description'].head())
df['tags'] = df['tags'].apply(lambda x:eval(str(x)))
all_tags = {}
count = 0
for talk in df['tags']:
    for tag in talk:
        if not tag in all_tags:
            all_tags[tag] = count
            count = count+1
onehot = np.zeros((0,count))
for talk in df['tags']:
    temp = np.zeros((1,count))
    for tag in talk:
        temp[0,all_tags[tag]] = 1
    onehot = np.concatenate((onehot,temp),0)
mldf_np = mldf.as_matrix()
all_y = np.reshape(df['avgPerRating'].as_matrix(),(-1,1))
all_x = np.concatenate((mldf_np,onehot),1)
combined = np.concatenate((all_x,all_y),1)
np.random.shuffle(combined)
data_size = np.shape(all_y)[0]
train_size = (int)(data_size*0.75)
feature_size = np.shape(all_x)[1]
x_train = combined[0:train_size,0:feature_size]
y_train = np.reshape(combined[0:train_size,feature_size],(-1,1))
x_val = combined[train_size:data_size,0:feature_size]
y_val = np.reshape(combined[train_size:data_size,feature_size],(-1,1))
import keras.optimizers as op
from keras.layers import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(feature_size,)))
model.add(Dense(1,kernel_regularizer=l2(0.01)))
model.compile(loss='mean_squared_error',optimizer="adam")
history = model.fit(x=x_train,y=y_train,batch_size=64,epochs=50,validation_data=(x_val,y_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = np.arange(1,len(loss)+1)
plot.plot(epochs,loss,color='red')
plot.plot(epochs,val_loss,color='blue')
all_weights_temp = model.layers[1].get_weights()[0]
all_weights = []
for weight in all_weights_temp:
    all_weights.append(weight[0])
existing_weights = all_weights[:6]
tag_weights = all_weights[6:]
fig,ax = plot.subplots()
index = np.arange(0,6)
ax.bar(index,existing_weights)
ax.set_xticklabels(('','num_speaker','duration','comments','languages','views','sentiment'))
plot.show()
plot.hist(x=tag_weights,bins=30,range=(-0.07,0.07),color='black')
plotDensityCurve(tag_weights,(-0.07,0.07,30),0.25,5,data=None)
inverted_tags = dict([v,k] for k,v in all_tags.items())
tag_columns = ('features',)
tag_df = pd.DataFrame({'features': tag_weights})
best_tags = tag_df.sort_values('features',ascending=False).head(10).index.tolist()
for tag in best_tags:
    print(inverted_tags[tag]) if tag in inverted_tags else ""
worst_tags = tag_df.sort_values('features',ascending=True).head(10).index.tolist()
for tag in worst_tags:
    print(inverted_tags[tag]) if tag in inverted_tags else ""
s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'theme'
theme_df = df.drop('tags', axis=1).join(s)
theme_df.head()
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()
pop_themes.columns = ['theme', 'talks']
top_themes = pop_themes.head(20)
top_themes
top_theme_list = []
for theme in top_themes['theme'].tolist():
    top_theme_list.append((theme,tag_df['features'].tolist()[all_tags[theme]]))
sorted_list = sorted(top_theme_list,key=lambda x: -x[1])
theme_list = []
weight_list = []
for pair in sorted_list:
    theme_list.append(pair[0])
    weight_list.append(pair[1])
fig,ax = plot.subplots()
index = np.arange(0,20)
ax.bar(index,weight_list)
ax.set_xticklabels(theme_list)
plot.xticks(index)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plot.show()