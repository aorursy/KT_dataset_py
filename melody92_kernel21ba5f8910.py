import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')
train_x = pd.read_csv('../input/mmm-course-hackathon-2020/train.csv')
test_x = pd.read_csv('../input/mmm-course-hackathon-2020/test.csv')
# plot few paths
first_traj = train_x[train_x['traj_ind']==1]
second_traj = train_x[train_x['traj_ind']==14]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(first_traj['x'], first_traj['y'], first_traj['z'], c='red', s=60)
ax.scatter(second_traj['x'], second_traj['y'], second_traj['z'], c='blue', s=60)
ax.view_init(30, 30)
plt.show()
train_x['label'].value_counts()
#check if there any missing values.
train_x.isnull().sum()
#check correlation
sns.set_style("whitegrid")
sns.heatmap(train_x.corr(), annot=True)
def add_max_cordinate(df,cordinate='z'):
    max_feat_by_traj_id = pd.DataFrame(df.groupby('traj_ind')[cordinate].max()).reset_index()
    max_feat_by_traj_id = max_feat_by_traj_id.rename(columns = {cordinate:'max_'+ cordinate})
    df = df.merge(max_feat_by_traj_id, left_on='traj_ind',right_on='traj_ind' )
    return df

def add_min_cordinate(df,cordinate='z'):
    max_feat_by_traj_id = pd.DataFrame(df.groupby('traj_ind')[cordinate].min()).reset_index()
    max_feat_by_traj_id = max_feat_by_traj_id.rename(columns = {cordinate:'min_'+ cordinate})
    df = df.merge(max_feat_by_traj_id, left_on='traj_ind',right_on='traj_ind' )
    return df
def add_distances_x_y_start_to_end(df):
    df = df.sort_values(by=['traj_ind','time_stamp'])
    distances_per_traj_ind = {}
    for i in df['traj_ind'].unique():
        x1 = df[df['traj_ind']==i].tail(1)['x']
        x2 = df[df['traj_ind']==i].head(1)['x']
        y1 = df[df['traj_ind']==i].tail(1)['y']
        y2 = df[df['traj_ind']==i].head(1)['y']

        d2 = np.square( x2.values - x1.values )  + np.square( y2.values - y1.values ) 
        distances = np.sqrt( d2 )
        distances_per_traj_ind[i] = distances

    distances_per_traj_ind_df = pd.DataFrame(distances_per_traj_ind).T
    distances_per_traj_ind_df = distances_per_traj_ind_df.reset_index().rename(columns={'index':'traj_ind'})
    
    df = df.merge(distances_per_traj_ind_df, on='traj_ind')
    df = df.rename(columns={0:'distance_x_y'})
    return df
def calculate_starting_angle(df):
    angle_per_traj_ind = []
    for i in df['traj_ind'].unique():
        print
        if(len(df[df['traj_ind']==i])>=3):
            a = np.array(df[df['traj_ind']==i].head(3).reset_index().iloc[0][['x','y','z']])
            b = np.array(df[df['traj_ind']==i].head(3).reset_index().iloc[1][['x','y','z']])
            c = np.array(df[df['traj_ind']==i].head(3).reset_index().iloc[2][['x','y','z']])


            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            angle_per_traj_ind.append([i,angle])
            
        else:
            angle_per_traj_ind.append([i,np.nan])     
    angle_per_traj_ind = pd.DataFrame(angle_per_traj_ind)
    angle_per_traj_ind = angle_per_traj_ind.rename(columns={0:'traj_ind'})
    angle_per_traj_ind = angle_per_traj_ind.iloc[:,0:2]
    df = df.merge(angle_per_traj_ind, on='traj_ind')
    df.columns = [*df.columns[:-1], 'strating_angle']
    return df
def time_to_pick(df):
    import datetime
    from datetime import date
    
    df['time_to_peak'] = np.nan
    for i in df['traj_ind'].unique():
        #print('starting ' + str(i) )
        time_peak = df[(df['traj_ind']==i) & (df['z']==df['max_z'])]['time_stamp'] 
        time_start = df[df['traj_ind']==i].head(1)['time_stamp'] 
        if(str(time_peak)==str(time_start)):
            miliseconds=0
        else:
            delta = time_peak - time_start
            #df.loc[df['traj_ind']==i, 'time_to_peak'] = delta.microseconds
            time_peak = datetime.datetime.strptime(str(time_peak.values[0]), '%Y-%m-%d %H:%M:%S.%f')
            time_start = datetime.datetime.strptime(str(time_start.values[0]), '%Y-%m-%d %H:%M:%S.%f')

            #print(time_peak , time_start)
            delta = time_peak - time_start
            #print(i,delta)
            miliseconds = delta.microseconds
        df.loc[df['traj_ind']==i, 'time_to_peak'] = miliseconds
    return df
def add_distances_x_z_start_to_end(df):
    df = df.sort_values(by=['traj_ind','time_stamp'])
    distances_per_traj_ind = {}
    for i in df['traj_ind'].unique():
        x1 = df[df['traj_ind']==i].tail(1)['x']
        x2 = df[df['traj_ind']==i].head(1)['x']
        y1 = df[df['traj_ind']==i].tail(1)['z']
        y2 = df[df['traj_ind']==i].head(1)['z']

        d2 = np.square( x2.values - x1.values )  + np.square( y2.values - y1.values ) 
        distances = np.sqrt( d2 )
        distances_per_traj_ind[i] = distances

    distances_per_traj_ind_df = pd.DataFrame(distances_per_traj_ind).T
    distances_per_traj_ind_df = distances_per_traj_ind_df.reset_index().rename(columns={'index':'traj_ind'})
    
    df = df.merge(distances_per_traj_ind_df, on='traj_ind')
    df = df.rename(columns={0:'distance_x_z'})
    return df
def add_instances_count(df):
    instances_count_by_traj = df.groupby('traj_ind').count()['x'].reset_index()
    instances_count_by_traj.rename(columns={'x':'count_instances'} , inplace=True)
    df = df.merge(instances_count_by_traj, on='traj_ind')
    return df
for feat in ['x','y','z']:
    train_x = add_max_cordinate(train_x,feat)
    train_x = add_min_cordinate(train_x,feat)
    test_x = add_max_cordinate(test_x,feat)
    test_x = add_min_cordinate(test_x,feat)

#train_x = time_to_pick(train_x)
#test_x = time_to_pick(test_x)
train_x = add_distances_x_y_start_to_end(train_x)
test_x = add_distances_x_y_start_to_end(test_x)
#train_x = calculate_starting_angle(train_x)
#test_x = calculate_starting_angle(test_x)
train_x = add_instances_count(train_x)
test_x = add_instances_count(test_x)
#train_x = add_distances_x_z_start_to_end(train_x)
#test_x = add_distances_x_z_start_to_end(test_x)
#train_x
dict_anomaly ={}

dict_anomaly['std_each_feature_label_0'] = train_x[train_x['label']==0].iloc[:,3:].std()
dict_anomaly['mean_each_feature_label_0'] = train_x[train_x['label']==0].iloc[:,3:].mean()

dict_anomaly['std_each_feature_label_1'] = train_x[train_x['label']==1].iloc[:,3:].std()
dict_anomaly['mean_each_feature_label_1'] = train_x[train_x['label']==1].iloc[:,3:].mean()

dict_anomaly['std_each_feature_label_2'] = train_x[train_x['label']==2].iloc[:,3:].std()
dict_anomaly['mean_each_feature_label_2'] = train_x[train_x['label']==2].iloc[:,3:].mean()
dict_anomaly['std_each_feature_label_2']
train_x['grade_anomaly'] = 0
for i in range(3):
    grade_anomaly = train_x[train_x['label']==i].apply(lambda x: abs((x[3:-1] - dict_anomaly['mean_each_feature_label_'+str(i)]) / dict_anomaly['std_each_feature_label_'+str(i)]).sum(),axis=1)
    train_x.loc[train_x['label']==i, 'grade_anomaly'] = grade_anomaly
    
#save data
train_x.to_pickle("data_train.pkl")
test_x.to_pickle("data_test.pkl")
#sns.boxplot(y = train_x['grade_anomaly'] , x=train_x['label'])
#train_x[train_x['label']==1]['grade_anomaly'].describe()
#load data
train_x = pd.read_pickle("data_train.pkl")
real_test = pd.read_pickle("data_test.pkl")
real_test.head()
len(train_x)
len_label_0 = len(train_x[train_x['label']==0])
len_label_1 = len(train_x[train_x['label']==1])
len_label_2 = len(train_x[train_x['label']==2])

train_x_copy = train_x.copy()

index_anomaly = train_x[train_x['label']==0].nlargest(int(len_label_0*0.1) , 'grade_anomaly')['traj_ind'].unique()
index_anomaly2 = train_x[train_x['label']==1].nlargest(int(len_label_1*0.015) , 'grade_anomaly')['traj_ind'].unique()
index_anomaly3 = train_x[train_x['label']==2].nlargest(int(len_label_2*0.015) , 'grade_anomaly')['traj_ind'].unique()

train_x_copy = train_x_copy[~train_x_copy['traj_ind'].isin(index_anomaly)]
train_x_copy = train_x_copy[~train_x_copy['traj_ind'].isin(index_anomaly2)]
train_x_copy = train_x_copy[~train_x_copy['traj_ind'].isin(index_anomaly3)]
train_x = train_x_copy
#train_x[train_x['count_instances']==1]['label'].value_counts()
### major = 1
#sns.boxplot(train_x[train_x['count_instances']==1]['label'] , train_x[train_x['count_instances']==1]['y'])
#deal with 1 instances :
#train_x.loc[train_x['count_instances']==1,'label']  = train_x[train_x['count_instances']==1].apply(lambda x: 2 if x['y']>0.35 else 1, axis=1)
#train_x[train_x['count_instances']==1]['label'].value_counts()
# import os
# os.remove("data.pkl")
#train_x_norm['strating_angle'] = train_x_norm['strating_angle']*100
#train_x['xyz_mul'] = train_x['x']*train_x['y']*train_x['z']
#real_test['xyz_mul'] = real_test['x']*real_test['y']*real_test['z']
train_x['combine_distance_max_z'] = train_x['distance_x_y']*train_x['max_z']
real_test['combine_distance_max_z'] = real_test['distance_x_y']*real_test['max_z']
real_test.head()
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
train_x = train_x.fillna(0)
x_features = ['max_x','max_z','min_z' ,'distance_x_y','count_instances','label']#,'strating_angle','time_to_peak']
y_features = ['max_x','max_z','min_z' ,'distance_x_y','count_instances','traj_ind']#,'strating_angle','time_to_peak']
save_traj_ind = real_test[y_features].drop_duplicates()['traj_ind']
real_test = real_test[y_features[:-1]].drop_duplicates()
train_x = train_x[x_features].drop_duplicates()
real_test.head()
# this part was for validation (not including in this notebook anymore..)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train_x.drop('label', axis=1), train_x['label'], test_size=0.2)

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# len(X_test)
Y = train_x['label']
train_x_norm = scaler.fit_transform(train_x.drop('label', axis=1))
real_test = scaler.transform(real_test)
X = pd.DataFrame(train_x_norm, columns=x_features[:-1])
Y = pd.get_dummies(Y)
real_test.shape
from keras.models import Sequential
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Conv1D
#LEARNING_RATE =0.005

# define the keras model (fc)
model = Sequential()
#model.add(Conv1D(filters=64, kernel_size=7, activation='relu',input_dim=X.shape[1] ))
model.add(Dense(60, activation='relu', input_dim=X.shape[1]))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_x['label']),
                                                 train_x['label'])
class_weights
model.fit(X, Y, epochs=50, batch_size=120, class_weight=class_weights)
results_predictions = model.predict_proba(real_test)
c = pd.DataFrame(results_predictions, index=save_traj_ind)
c = c.reset_index().rename(columns={'traj_ind':'trajectory_ind' ,0:'type_0',1:'type_1',2:'type_2'})
c.to_csv('subs/final_result_v6.csv' , index=False)
best = pd.read_csv('final_result_v6.csv')
best[1:].idxmax(axis=0)
best['true_label'] = best.apply(lambda x: x[1:].idxmax() , axis=1)
traj_for_change = best[(best['true_label']=='0') & (best['0']<0.6)]['traj_ind']
fix_indexs = best[(best['true_label']=='0') & (best['0']<0.6)].index
zeros = best.iloc[fix_indexs]['0']
ones = best.iloc[fix_indexs]['1']
best.loc[best['traj_ind'].isin(traj_for_change) , '0' ] = ones
best.loc[best['traj_ind'].isin(traj_for_change) , '1' ] = zeros
best = best.reset_index().rename(columns={'traj_ind':'trajectory_ind' ,'0':'type_0','1':'type_1','2':'type_2'})
best = best[['trajectory_ind',	'type_0',	'type_1',	'type_2']]
best.to_csv('subs/final_result_v32.csv' , index=False)
