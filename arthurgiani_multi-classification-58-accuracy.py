import pandas as pd

import numpy as np

import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier

import gc

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold



import os

print(os.listdir("../input/competicao-dsa-machine-learning-sep-2019"))
#Loading datasets



train = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_treino.csv')

test = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_teste.csv')

target = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/y_treino.csv')
#Checking shape



train.shape, test.shape, target.shape
#Checking missing values in train/test dataset



train.isnull().sum()
test.isnull().sum()
target.isnull().sum()
train.describe()
test.describe()
target.describe()
#Number of labels per surface class



labels_per_surface = px.histogram(target, x='surface',color='surface',title='Labels per surface class').update_xaxes(categoryorder='total descending')

labels_per_surface.show()
#Visualizing variable behaviour in one random series_id, for example, tiled surface



tiled_surface = train[train['series_id']==5]
train.iloc[:, 3:]
cols_subplot = tiled_surface.columns[3:] #just the important features at the moment



plt.figure(figsize=(26, 16))

for index, col in enumerate(cols_subplot):

    plt.subplot(3, 4, index + 1)

    plt.plot(tiled_surface[col])

    plt.title(col)
fine_concrete_surface = train[train['series_id']==0]

plt.figure(figsize=(26, 16))

for index, col in enumerate(cols_subplot):

    plt.subplot(3, 4, index + 1)

    plt.plot(fine_concrete_surface[col])

    plt.title(col)
fine_concrete_surface = train[train['series_id']==48]

plt.figure(figsize=(26, 16))

for index, col in enumerate(cols_subplot):

    plt.subplot(3, 4, index + 1)

    plt.plot(fine_concrete_surface[col])

    plt.title(col)

    

#Mesma superficie, medidos em horários diferentes, produziram comportamento diferente em relação a orientação do robô
#Correlação entre variáveis



f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(train.iloc[:, 3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#Plotando distribuição das variáveis



def plot_distr_features(train_data, test_data, label_train, label_test):

    

    index = 0

    

    plt.figure(figsize=(26,16))

    

    features = train_data.columns[3:]

    

    for f in features:

        plt.subplot(4,4, index+1)

        sns.kdeplot(train_data[f],bw=1, label = label_train)

        sns.kdeplot(test_data[f], bw=1,label=label_test)

        plt.title(f)

        index += 1

    plt.show()

        

    

plot_distr_features(train,test,'train', 'test')
#Plotando distribuição das variáveis



def plot_class_dist(dataset):

    

    index = 0

    

    plt.figure(figsize=(26,16))

    

    features = dataset.columns[3:]

    classes = target['surface'].unique().tolist()

    df_aux = dataset.merge(target, on='series_id', how='inner')

    

    for f in features:

        plt.subplot(5,2, index+1)

        

        for c in classes:

            df_class = df_aux[df_aux['surface']==c]

            sns.kdeplot(df_class[f], bw=0.5, label = c)

        index += 1

    plt.show()
plot_class_dist(train)

#requires normalization
def quaternion_to_euler(df):

    

    x = df['orientation_X']

    y = df['orientation_Y']

    z = df['orientation_Z']

    w = df['orientation_W']

    

    roll = np.arctan2(2*(w*x + y*z),1 - 2*(x*x + y*y))

    pitch = np.arcsin(2*(w*y - z*x))

    yawl = np.arctan2(2*(w*z + x*y),1 - 2*(y*y + z*z))

    

    return roll, pitch,yawl
train['euler_x'], train['euler_y'], train['euler_z'] = zip(*train.apply(quaternion_to_euler,axis=1))
test['euler_x'], test['euler_y'], test['euler_z'] = zip(*test.apply(quaternion_to_euler,axis=1))
#Distribution in roll, pitch and yawl



euler_cols = ['euler_x', 'euler_y', 'euler_z']



plt.figure(figsize=(26,10))



for index,col in enumerate(euler_cols):

    

    plt.subplot(1,3,index+1)

    

    sns.kdeplot(train[col], label=col)

    sns.kdeplot(test[col], label=col)

    plt.title(col)

plt.show()
def relationship_between_variables(df):

    

    data = pd.DataFrame()

    

    df['ang_velocity'] = (df['angular_velocity_X']**2 + df['angular_velocity_Y']**2 + df['angular_velocity_Z']**2)** 0.5

    

    df['linear_acc'] = (df['linear_acceleration_X']**2 + df['linear_acceleration_Y']**2 + df['linear_acceleration_Z']**2)**0.5

 

    df['abs_position_xyz'] = (df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2)**0.5

    

    df['acc_vs_vel'] = df['linear_acc'] / df['ang_velocity']

    

    #This parameters will be used to calculate another parameters grouped by series_id, such as mean, std, median etc..

    

    

    for col in df.columns:

        if col not in ['row_id', 'series_id', 'measurement_number']:

            data[col + '_avg'] = df.groupby(['series_id'])[col].mean()

            data[col + '_median'] = df.groupby(['series_id'])[col].median()

            data[col + '_std'] = df.groupby(['series_id'])[col].std()

            data[col + '_max'] = df.groupby(['series_id'])[col].max()

            data[col + '_min'] = df.groupby(['series_id'])[col].min()

            data[col + '_range'] = data[col + '_max'] - data[col + '_min']

            data[col + '_q25'] = df.groupby(['series_id'])[col].quantile(0.25)

            data[col + '_q75'] = df.groupby(['series_id'])[col].quantile(0.75)

            data[col + '_q95'] = df.groupby(['series_id'])[col].quantile(0.95)

            data[col + '_iqr'] = data[col + '_q75'] - data[col + '_q25']

            

    return data

    

    
train_feat = relationship_between_variables(train)

test_feat = relationship_between_variables(test)
train_feat.reset_index(level=0, inplace=True)

test_feat.reset_index(level=0, inplace=True)
#Normalizing Data



scaler = StandardScaler()



train_col_names = train_feat.columns

test_col_names = test_feat.columns



x_train_norm = scaler.fit_transform(train_feat)

x_test_norm = scaler.fit_transform(test_feat)



x_train_final = pd.DataFrame(x_train_norm, columns=train_col_names)

x_test_final = pd.DataFrame(x_test_norm, columns=test_col_names)
len(train_feat) == len(target)



#Same lenght. No Need to concatenate
#Label encoding

le = LabelEncoder()

target['surface'] = le.fit_transform(target['surface'])
#Cross validation technique



folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for times, (trn_idx, val_idx) in enumerate(folds.split(x_train_final.values,target['surface'].values)):

    model = RandomForestClassifier(n_estimators=500, n_jobs = -1)

    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

    model.fit(x_train_final.iloc[trn_idx],target['surface'][trn_idx])

    print("Fold: {} score: {}".format(times,model.score(x_train_final.iloc[val_idx],target['surface'][val_idx])))
surfaces_predict = model.predict(x_test_final)
surfaces_predict = le.inverse_transform(surfaces_predict)
submission = pd.DataFrame({'series_id': test_feat['series_id'].unique(), 'surface':surfaces_predict})
submission.to_csv('submission.csv',index=False)