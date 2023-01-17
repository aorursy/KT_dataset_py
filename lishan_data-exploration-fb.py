
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_train = pd.read_csv('../input/train.csv')
#most of the checkins has the accuracy being under 100 (probably 100 meters)
import matplotlib.pyplot as plt
plt.hist(df_train.accuracy)
plt.show()
#take part of the data to do experiment
df_train_sub=df_train[:20000].copy()
df_test=df_train[20000:23000].copy()
def process(df,keep_good): #to return the training data and testing data
    if keep_good:
        df_processed=df[df['accuracy']<200].copy()  #rule out the checkins with low quality
    else:
        df_processed=df
    weight=[20,40,0,0]
    df_processed['x']=weight[0]*df_processed['x']
    df_processed['y']=weight[1]*df_processed['y']
    minute= (df_processed['time'] % (24*60))*1.0/(24*60) * 2 * np.pi
    df_processed['minute_sin'] = (np.sin(minute) + 1).round(4)
    df_processed['minute_cos'] = (np.cos(minute) + 1).round(4)
    df_processed['minute_sin'] =weight[2]*df_processed['minute_sin']
    df_processed['minute_cos'] =weight[2]*df_processed['minute_cos']
    del minute
    weekday = 2 * np.pi * ((df_processed['time'] // 1440) % 7) / 7
    df_processed['weekday_sin'] = (np.sin(weekday) + 1).round(4)
    df_processed['weekday_cos'] = (np.cos(weekday) + 1).round(4)
    df_processed['weekday_sin']=weight[3]*df_processed['weekday_sin']
    df_processed['weekday_cos']=weight[3]*df_processed['weekday_cos']
    del weekday
    X=df_processed.drop(['row_id','time','place_id','accuracy'],axis=1)
    Y=df_processed['place_id']
    return X,Y
    
    

train_x,train_y=process(df_train_sub,False)
test_x,test_y=process(df_test,False)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=37,weights="distance")
clf.fit(train_x,train_y)
pred=clf.predict(test_x)
np.mean(pred==test_y.values)