import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plot

%matplotlib inline  

import os

import seaborn as sns # Plot

import warnings

warnings.filterwarnings("ignore")

sns.set(style="darkgrid") # Style seaborn

sns.set(color_codes=True) # Color seaborn 



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.pipeline import make_pipeline

#Importing Confusion Matrix

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import linkage,cophenet,dendrogram,fcluster

from scipy.spatial.distance import pdist

from scipy.spatial import distance_matrix





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Cargando el la ruta de los archivos

Path_train = "../input/learn-together/train.csv"

Path_test = "../input/learn-together/test.csv"

#send = pd.read_csv("../input/comp-01/best_submission.csv",index_col='Id')# ->>"test.csv"

#Cargando los data sets

df_train = pd.read_csv(Path_train,index_col='Id')# ->> "train.csv"

df_test = pd.read_csv(Path_test,index_col='Id')# ->>"test.csv"

del df_train['Soil_Type7']

del df_test['Soil_Type7']

del df_train['Soil_Type15']

del df_test['Soil_Type15']

data_train = df_train.copy()

data_test = df_test.copy()
def split(df, headSize):

    hd = df.head(headSize)

    tl = df.tail(len(df)-headSize)

    return hd, tl



# Function for comparing different approaches

def score_dataset(data):

    y = data.Cover_Type.values # <- Target

    X = data # Data

    X.drop(['Cover_Type'], axis=1, inplace=True) # Drop target

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

      

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, random_state=2019)

    

    RF = RandomForestClassifier(n_jobs =  -1, n_estimators = 1000,random_state = 1)

    RF.fit(X=X_train, y=y_train)

    pred = RF.predict(X_val)

    mae = mean_absolute_error(pred,y_val ) 

    acc = round(accuracy_score(y_val, pred),4)

    return acc

    

# Function for data join    

def data_join(x,y,z,c=0):

    if (c == 0):

        frames = [x,y]

        data_join = pd.concat(frames,join='inner',axis=1)

        return data_join

    else:

        frames = [x,y,z]

        data_join = pd.concat(frames,join='inner',axis=1)

        return data_join



# Visual data comparison function 

def compare_plot(data):

    sns.pairplot(data, palette="RdBu",diag_kind="kde",hue='Cover_Type',height=2.5)

    plt.show()

    

# Multiple histogram    

def histogram(data):

    f, ax = plt.subplots(figsize=(10, 10))

    data.hist(ax=ax,align='mid',orientation='vertical',bins=52)

    plt.show()    
data_train.iloc[:,:10].head().transpose()
data_train.iloc[:,10:52].head()
group_1 = data_join(data_train.iloc[:,:10], data_train[['Cover_Type']],None)

compare_plot(group_1)

data = abs(data_train.copy())

training_score = score_dataset(df_train.copy())

transformed_training_score = score_dataset(data)

print("Previous Accuracy :",training_score,"Current Accuracy :",transformed_training_score)

print("Train :",df_train.copy().shape)

print("Test :",df_test.copy().shape)
train = abs(df_train.copy())

cols_1 = ['Elevation','Aspect','Slope']

cols_2 = ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology']

cols_3 = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']

cols_4 = ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']



# Elevation

train['Elevation_sin_'] = abs(np.ceil(np.sin(train['Elevation'])))

train['Elevation_cos_'] = abs(np.ceil(np.cos(train['Elevation'])))

train['Elevation_tanh_'] = abs(np.ceil(np.tanh(train['Elevation'])))

train['Elevation_bin_round_100'] = np.array(np.floor(np.array(train['Elevation']) / 100))

train['Elevation_bin_round_1000'] = np.array(np.floor(np.array(train['Elevation']) / 1000))

train['Elevation_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Elevation']))))

train['Elevation_log'] = np.ceil(np.log(np.floor(np.array(1+train['Elevation']))))

# Aspect

train['Aspect_sin_'] = abs(np.ceil(np.sin(train['Aspect'])))

train['Aspect_cos_'] = abs(np.ceil(np.cos(train['Aspect'])))

train['Aspect_tanh_'] = abs(np.ceil(np.tanh(train['Aspect'])))

train['Aspect_bin_round_100'] = np.array(np.floor(np.array(train['Aspect']) / 100))

train['Aspect_bin_round_1000'] = np.array(np.floor(np.array(train['Aspect']) / 1000))

train['Aspect_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Aspect']))))

train['Aspect_log'] = np.ceil(np.log(np.floor(np.array(1+train['Aspect']))))

# Slope

train['Slope_sin_'] = abs(np.ceil(np.sin(train['Slope'])))

train['Slope_cos_'] = abs(np.ceil(np.cos(train['Slope'])))

train['Slope_tanh_'] = abs(np.ceil(np.tanh(train['Slope'])))

train['Slope_bin_round_100'] = np.array(np.floor(np.array(train['Slope']) / 100))

train['Slope_bin_round_1000'] = np.array(np.floor(np.array(train['Slope']) / 1000))

train['Slope_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Slope']))))

train['Slope_log'] = np.ceil(np.log(np.floor(np.array(1+train['Slope']))))

# Basic statistics - Elev_Asp_Slope

train['Elev_Asp_Slope_sum'] = train[cols_2].sum(axis=1)

train['Elev_Asp_Slope_mean'] = train[cols_2].mean(axis=1)

train['Elev_Asp_Slope_std'] = np.round(train[cols_2].std(axis=1),3)

train['Elev_Asp_Slope_min'] = train[cols_2].min(axis=1)

train['Elev_Asp_Slope_max'] = train[cols_2].max(axis=1)

train['Elev_Asp_Slope_median'] = train[cols_2].median(axis=1)

train['Elev_Asp_Slope_quantile'] = train[cols_2].quantile(axis=1)

# Horizontal_Distance_To_Hydrology

train['Horizontal_Distance_To_Hydrology_sin_'] = abs(np.ceil(np.sin(train['Horizontal_Distance_To_Hydrology'])))

train['Horizontal_Distance_To_Hydrology_cos_'] = abs(np.ceil(np.cos(train['Horizontal_Distance_To_Hydrology'])))

train['Horizontal_Distance_To_Hydrology_tanh_'] = abs(np.ceil(np.tanh(train['Horizontal_Distance_To_Hydrology'])))

train['Horizontal_Distance_To_Hydrology_bin_round_100'] = np.array(np.floor(np.array(train['Horizontal_Distance_To_Hydrology']) / 100))

train['Horizontal_Distance_To_Hydrology_bin_round_1000'] = np.array(np.floor(np.array(train['Horizontal_Distance_To_Hydrology']) / 1000))

train['Horizontal_Distance_To_Hydrology_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Horizontal_Distance_To_Hydrology']))))

train['Horizontal_Distance_To_Hydrology_log'] = np.ceil(np.log(np.floor(np.array(1+train['Horizontal_Distance_To_Hydrology']))))

# Vertical_Distance_To_Hydrology

train['Vertical_Distance_To_Hydrology_sin_'] = abs(np.ceil(np.sin(train['Vertical_Distance_To_Hydrology'])))

train['Vertical_Distance_To_Hydrology_cos_'] = abs(np.ceil(np.cos(train['Vertical_Distance_To_Hydrology'])))

train['Vertical_Distance_To_Hydrology_tanh_'] = abs(np.ceil(np.tanh(train['Vertical_Distance_To_Hydrology'])))

train['Vertical_Distance_To_Hydrology_bin_round_100'] = np.array(np.floor(np.array(train['Vertical_Distance_To_Hydrology']) / 100))

train['Vertical_Distance_To_Hydrology_bin_round_1000'] = np.array(np.floor(np.array(train['Vertical_Distance_To_Hydrology']) / 1000))

train['Vertical_Distance_To_Hydrology_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Vertical_Distance_To_Hydrology']))))

train['Vertical_Distance_To_Hydrology_log'] = np.ceil(np.log(np.floor(np.array(1+train['Vertical_Distance_To_Hydrology']))))

# Basic statistics - Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_sum'] = train[cols_1].sum(axis=1)

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_mean'] = train[cols_1].mean(axis=1)

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_std'] = np.round(train[cols_1].std(axis=1))

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_min'] = train[cols_1].min(axis=1)

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_max'] = train[cols_1].max(axis=1)

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_median'] = train[cols_1].median(axis=1)

train['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_quantile'] = train[cols_1].quantile(axis=1)

# Horizontal_Distance_To_Roadways

train['Horizontal_Distance_To_Roadways_sin_'] = abs(np.ceil(np.sin(train['Horizontal_Distance_To_Roadways'])))

train['Horizontal_Distance_To_Roadways_cos_'] = abs(np.ceil(np.cos(train['Horizontal_Distance_To_Roadways'])))

train['Horizontal_Distance_To_Roadways_tanh_'] = abs(np.ceil(np.tanh(train['Horizontal_Distance_To_Roadways'])))

train['Horizontal_Distance_To_Roadways_bin_round_100'] = np.array(np.floor(np.array(train['Horizontal_Distance_To_Roadways']) / 100))

train['Horizontal_Distance_To_Roadways_bin_round_1000'] = np.array(np.floor(np.array(train['Horizontal_Distance_To_Roadways']) / 1000))

train['Horizontal_Distance_To_Roadways_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Horizontal_Distance_To_Roadways']))))

train['Horizontal_Distance_To_Roadways_log'] = np.ceil(np.log(np.floor(np.array(1+train['Horizontal_Distance_To_Roadways']))))

# Hillshade_9am

train['Hillshade_9am_sin_'] = abs(np.ceil(np.sin(train['Hillshade_9am'])))

train['Hillshade_9am_cos_'] = abs(np.ceil(np.cos(train['Hillshade_9am'])))

train['Hillshade_9am_tanh_'] = abs(np.ceil(np.tanh(train['Hillshade_9am'])))

train['Hillshade_9am_bin_round_100'] = np.array(np.floor(np.array(train['Hillshade_9am']) / 100))

train['Hillshade_9am_bin_round_1000'] = np.array(np.floor(np.array(train['Hillshade_9am']) / 1000))

train['Hillshade_9am_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Hillshade_9am']))))

train['Hillshade_9am_log'] = np.ceil(np.log(np.floor(np.array(1+train['Hillshade_9am']))))

# Hillshade_Noon

train['Hillshade_Noon_sin_'] = abs(np.ceil(np.sin(train['Hillshade_Noon'])))

train['Hillshade_Noon_cos_'] = abs(np.ceil(np.cos(train['Hillshade_Noon'])))

train['Hillshade_Noon_tanh_'] = abs(np.ceil(np.tanh(train['Hillshade_Noon'])))

train['Hillshade_Noon_bin_round_100'] = np.array(np.floor(np.array(train['Hillshade_Noon']) / 100))

train['Hillshade_Noon_bin_round_1000'] = np.array(np.floor(np.array(train['Hillshade_Noon']) / 1000))

train['Hillshade_Noon_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Hillshade_Noon']))))

train['Hillshade_Noon_log'] = np.ceil(np.log(np.floor(np.array(1+train['Hillshade_Noon']))))

# Hillshade_3pm

train['Hillshade_3pm_sin_'] = abs(np.ceil(np.sin(train['Hillshade_3pm'])))

train['Hillshade_3pm_cos_'] = abs(np.ceil(np.cos(train['Hillshade_3pm'])))

train['Hillshade_3pm_tanh_'] = abs(np.ceil(np.tanh(train['Hillshade_3pm'])))

train['Hillshade_3pm_bin_round_100'] = np.array(np.floor(np.array(train['Hillshade_3pm']) / 100))

train['Hillshade_3pm_bin_round_1000'] = np.array(np.floor(np.array(train['Hillshade_3pm']) / 1000))

train['Hillshade_3pm_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Hillshade_3pm']))))

train['Hillshade_3pm_log'] = np.ceil(np.log(np.floor(np.array(1+train['Hillshade_3pm']))))

# Horizontal_Distance_To_Fire_Points

train['Horizontal_Distance_To_Fire_Points_sin_'] = abs(np.ceil(np.sin(train['Horizontal_Distance_To_Fire_Points'])))

train['Horizontal_Distance_To_Fire_Points_cos_'] = abs(np.ceil(np.cos(train['Horizontal_Distance_To_Fire_Points'])))

train['Horizontal_Distance_To_Fire_Points_tanh_'] = abs(np.ceil(np.tanh(train['Horizontal_Distance_To_Fire_Points'])))

train['Horizontal_Distance_To_Fire_Points_bin_round_100'] = np.array(np.floor(np.array(train['Horizontal_Distance_To_Fire_Points']) / 100))

train['Horizontal_Distance_To_Fire_Points_bin_round_1000'] = np.array(np.floor(np.array(train['Horizontal_Distance_To_Fire_Points']) / 1000))

train['Horizontal_Distance_To_Fire_Points_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Horizontal_Distance_To_Fire_Points']))))

train['Horizontal_Distance_To_Fire_Points_log'] = np.ceil(np.log(np.floor(np.array(1+train['Horizontal_Distance_To_Fire_Points']))))

# Basic statistics - Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_sum'] = train[cols_3].sum(axis=1)

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_mean'] = train[cols_3].mean(axis=1)

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_std'] = np.round(train[cols_3].std(axis=1),3)

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_min'] = train[cols_3].min(axis=1)

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_max'] = train[cols_3].max(axis=1)

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_median'] = train[cols_3].median(axis=1)

train['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_quantile'] = train[cols_3].quantile(axis=1)

# Basic statistics - Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_sum'] = train[cols_4].sum(axis=1)

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_mean'] = train[cols_4].mean(axis=1)

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_std'] = np.round(train[cols_4].std(axis=1),3)

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_min'] = train[cols_4].min(axis=1)

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_max'] = train[cols_4].max(axis=1)

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_median'] = train[cols_4].median(axis=1)

train['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_quantile'] = train[cols_4].quantile(axis=1)

print("Train Size : ",train.shape)

print("previous accuracy :",training_score,"current accuracy :",score_dataset(train.copy()))

print("New features have been added...")

#d_train.iloc[:,52:].head()
test = abs(df_test.copy())

cols_1 = ['Elevation','Aspect','Slope']

cols_2 = ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology']

cols_3 = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']

cols_4 = ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']



# Elevation

test['Elevation_sin_'] = abs(np.ceil(np.sin(test['Elevation'])))

test['Elevation_cos_'] = abs(np.ceil(np.cos(test['Elevation'])))

test['Elevation_tanh_'] = abs(np.ceil(np.tanh(test['Elevation'])))

test['Elevation_bin_round_100'] = np.array(np.floor(np.array(test['Elevation']) / 100))

test['Elevation_bin_round_1000'] = np.array(np.floor(np.array(test['Elevation']) / 1000))

test['Elevation_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Elevation']))))

test['Elevation_log'] = np.ceil(np.log(np.floor(np.array(1+test['Elevation']))))

# Aspect

test['Aspect_sin_'] = abs(np.ceil(np.sin(test['Aspect'])))

test['Aspect_cos_'] = abs(np.ceil(np.cos(test['Aspect'])))

test['Aspect_tanh_'] = abs(np.ceil(np.tanh(test['Aspect'])))

test['Aspect_bin_round_100'] = np.array(np.floor(np.array(test['Aspect']) / 100))

test['Aspect_bin_round_1000'] = np.array(np.floor(np.array(test['Aspect']) / 1000))

test['Aspect_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Aspect']))))

test['Aspect_log'] = np.ceil(np.log(np.floor(np.array(1+test['Aspect']))))

# Slope

test['Slope_sin_'] = abs(np.ceil(np.sin(test['Slope'])))

test['Slope_cos_'] = abs(np.ceil(np.cos(test['Slope'])))

test['Slope_tanh_'] = abs(np.ceil(np.tanh(test['Slope'])))

test['Slope_bin_round_100'] = np.array(np.floor(np.array(test['Slope']) / 100))

test['Slope_bin_round_1000'] = np.array(np.floor(np.array(test['Slope']) / 1000))

test['Slope_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Slope']))))

test['Slope_log'] = np.ceil(np.log(np.floor(np.array(1+test['Slope']))))

# Basic statistics - Elev_Asp_Slope

test['Elev_Asp_Slope_sum'] = test[cols_2].sum(axis=1)

test['Elev_Asp_Slope_mean'] = test[cols_2].mean(axis=1)

test['Elev_Asp_Slope_std'] = np.round(test[cols_2].std(axis=1),3)

test['Elev_Asp_Slope_min'] = test[cols_2].min(axis=1)

test['Elev_Asp_Slope_max'] = test[cols_2].max(axis=1)

test['Elev_Asp_Slope_median'] = test[cols_2].median(axis=1)

test['Elev_Asp_Slope_quantile'] = test[cols_2].quantile(axis=1)

# Horizontal_Distance_To_Hydrology

test['Horizontal_Distance_To_Hydrology_sin_'] = abs(np.ceil(np.sin(test['Horizontal_Distance_To_Hydrology'])))

test['Horizontal_Distance_To_Hydrology_cos_'] = abs(np.ceil(np.cos(test['Horizontal_Distance_To_Hydrology'])))

test['Horizontal_Distance_To_Hydrology_tanh_'] = abs(np.ceil(np.tanh(test['Horizontal_Distance_To_Hydrology'])))

test['Horizontal_Distance_To_Hydrology_bin_round_100'] = np.array(np.floor(np.array(test['Horizontal_Distance_To_Hydrology']) / 100))

test['Horizontal_Distance_To_Hydrology_bin_round_1000'] = np.array(np.floor(np.array(test['Horizontal_Distance_To_Hydrology']) / 1000))

test['Horizontal_Distance_To_Hydrology_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Horizontal_Distance_To_Hydrology']))))

test['Horizontal_Distance_To_Hydrology_log'] = np.ceil(np.log(np.floor(np.array(1+test['Horizontal_Distance_To_Hydrology']))))

# Vertical_Distance_To_Hydrology

test['Vertical_Distance_To_Hydrology_sin_'] = abs(np.ceil(np.sin(test['Vertical_Distance_To_Hydrology'])))

test['Vertical_Distance_To_Hydrology_cos_'] = abs(np.ceil(np.cos(test['Vertical_Distance_To_Hydrology'])))

test['Vertical_Distance_To_Hydrology_tanh_'] = abs(np.ceil(np.tanh(test['Vertical_Distance_To_Hydrology'])))

test['Vertical_Distance_To_Hydrology_bin_round_100'] = np.array(np.floor(np.array(test['Vertical_Distance_To_Hydrology']) / 100))

test['Vertical_Distance_To_Hydrology_bin_round_1000'] = np.array(np.floor(np.array(test['Vertical_Distance_To_Hydrology']) / 1000))

test['Vertical_Distance_To_Hydrology_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Vertical_Distance_To_Hydrology']))))

test['Vertical_Distance_To_Hydrology_log'] = np.ceil(np.log(np.floor(np.array(1+test['Vertical_Distance_To_Hydrology']))))

# Basic statistics - Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_sum'] = test[cols_1].sum(axis=1)

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_mean'] = test[cols_1].mean(axis=1)

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_std'] = np.round(test[cols_1].std(axis=1))

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_min'] = test[cols_1].min(axis=1)

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_max'] = test[cols_1].max(axis=1)

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_median'] = test[cols_1].median(axis=1)

test['Horizontal_Distance_To_Hydrology_+_Vertical_Distance_To_Hydrology_quantile'] = test[cols_1].quantile(axis=1)

# Horizontal_Distance_To_Roadways

test['Horizontal_Distance_To_Roadways_sin_'] = abs(np.ceil(np.sin(test['Horizontal_Distance_To_Roadways'])))

test['Horizontal_Distance_To_Roadways_cos_'] = abs(np.ceil(np.cos(test['Horizontal_Distance_To_Roadways'])))

test['Horizontal_Distance_To_Roadways_tanh_'] = abs(np.ceil(np.tanh(test['Horizontal_Distance_To_Roadways'])))

test['Horizontal_Distance_To_Roadways_bin_round_100'] = np.array(np.floor(np.array(test['Horizontal_Distance_To_Roadways']) / 100))

test['Horizontal_Distance_To_Roadways_bin_round_1000'] = np.array(np.floor(np.array(test['Horizontal_Distance_To_Roadways']) / 1000))

test['Horizontal_Distance_To_Roadways_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Horizontal_Distance_To_Roadways']))))

test['Horizontal_Distance_To_Roadways_log'] = np.ceil(np.log(np.floor(np.array(1+test['Horizontal_Distance_To_Roadways']))))

# Hillshade_9am

test['Hillshade_9am_sin_'] = abs(np.ceil(np.sin(test['Hillshade_9am'])))

test['Hillshade_9am_cos_'] = abs(np.ceil(np.cos(test['Hillshade_9am'])))

test['Hillshade_9am_tanh_'] = abs(np.ceil(np.tanh(test['Hillshade_9am'])))

test['Hillshade_9am_bin_round_100'] = np.array(np.floor(np.array(test['Hillshade_9am']) / 100))

test['Hillshade_9am_bin_round_1000'] = np.array(np.floor(np.array(test['Hillshade_9am']) / 1000))

test['Hillshade_9am_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Hillshade_9am']))))

test['Hillshade_9am_log'] = np.ceil(np.log(np.floor(np.array(1+test['Hillshade_9am']))))

# Hillshade_Noon

test['Hillshade_Noon_sin_'] = abs(np.ceil(np.sin(test['Hillshade_Noon'])))

test['Hillshade_Noon_cos_'] = abs(np.ceil(np.cos(test['Hillshade_Noon'])))

test['Hillshade_Noon_tanh_'] = abs(np.ceil(np.tanh(test['Hillshade_Noon'])))

test['Hillshade_Noon_bin_round_100'] = np.array(np.floor(np.array(test['Hillshade_Noon']) / 100))

test['Hillshade_Noon_bin_round_1000'] = np.array(np.floor(np.array(test['Hillshade_Noon']) / 1000))

test['Hillshade_Noon_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Hillshade_Noon']))))

test['Hillshade_Noon_log'] = np.ceil(np.log(np.floor(np.array(1+test['Hillshade_Noon']))))

# Hillshade_3pm

test['Hillshade_3pm_sin_'] = abs(np.ceil(np.sin(test['Hillshade_3pm'])))

test['Hillshade_3pm_cos_'] = abs(np.ceil(np.cos(test['Hillshade_3pm'])))

test['Hillshade_3pm_tanh_'] = abs(np.ceil(np.tanh(test['Hillshade_3pm'])))

test['Hillshade_3pm_bin_round_100'] = np.array(np.floor(np.array(test['Hillshade_3pm']) / 100))

test['Hillshade_3pm_bin_round_1000'] = np.array(np.floor(np.array(test['Hillshade_3pm']) / 1000))

test['Hillshade_3pm_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Hillshade_3pm']))))

test['Hillshade_3pm_log'] = np.ceil(np.log(np.floor(np.array(1+test['Hillshade_3pm']))))

# Horizontal_Distance_To_Fire_Points

test['Horizontal_Distance_To_Fire_Points_sin_'] = abs(np.ceil(np.sin(test['Horizontal_Distance_To_Fire_Points'])))

test['Horizontal_Distance_To_Fire_Points_cos_'] = abs(np.ceil(np.cos(test['Horizontal_Distance_To_Fire_Points'])))

test['Horizontal_Distance_To_Fire_Points_tanh_'] = abs(np.ceil(np.tanh(test['Horizontal_Distance_To_Fire_Points'])))

test['Horizontal_Distance_To_Fire_Points_bin_round_100'] = np.array(np.floor(np.array(test['Horizontal_Distance_To_Fire_Points']) / 100))

test['Horizontal_Distance_To_Fire_Points_bin_round_1000'] = np.array(np.floor(np.array(test['Horizontal_Distance_To_Fire_Points']) / 1000))

test['Horizontal_Distance_To_Fire_Points_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Horizontal_Distance_To_Fire_Points']))))

test['Horizontal_Distance_To_Fire_Points_log'] = np.ceil(np.log(np.floor(np.array(1+test['Horizontal_Distance_To_Fire_Points']))))

# Basic statistics - Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_sum'] = test[cols_3].sum(axis=1)

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_mean'] = test[cols_3].mean(axis=1)

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_std'] = np.round(test[cols_3].std(axis=1),3)

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_min'] = test[cols_3].min(axis=1)

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_max'] = test[cols_3].max(axis=1)

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_median'] = test[cols_3].median(axis=1)

test['Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points_quantile'] = test[cols_3].quantile(axis=1)

# Basic statistics - Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_sum'] = test[cols_4].sum(axis=1)

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_mean'] = test[cols_4].mean(axis=1)

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_std'] = np.round(test[cols_4].std(axis=1),3)

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_min'] = test[cols_4].min(axis=1)

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_max'] = test[cols_4].max(axis=1)

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_median'] = test[cols_4].median(axis=1)

test['Hillshade_9am_+_Hillshade_Noon_+_Hillshade_3pm_quantile'] = test[cols_4].quantile(axis=1)

print("Test Size : ",test.shape)

print("New features have been added...")
data_train = abs(df_train.copy())

data_group_2 = data_train.iloc[:,3:5]

x = data_join(data_group_2, data_train[['Cover_Type']],None)

compare_plot(x)

data_group_2.hist(bins=60)

fig, ax = plt.subplots()

ax.set_xlabel('Horizontal Distance To Hydrology', fontsize=12)

ax.set_ylabel('Vertical Distance To Hydrology', fontsize=12)

ax.set_title('Scatter',fontsize=18)

x = data_train.iloc[:,3:4]

y = data_train.iloc[:,4:5]

plt.scatter(x,y,c="b",marker='o',alpha=0.4,linewidth=2.0)

data_group_2.head()
z_1 = linkage(data_group_2, method='complete')

c,coph_dist = cophenet(z_1,pdist(data_group_2,'minkowski'))



fig, ax = plt.subplots(figsize = (30,30))

ax.set_xlabel('Horizontal Distance To Hydrology', fontsize=25)

ax.set_ylabel('Vertical Distance To Hydrology', fontsize=25)

ax.set_title('Cluster',fontsize=25)

clusters = fcluster(z_1,2,criterion="distance")

cl_data = pd.DataFrame(np.array(clusters),columns=['Cluster_G2'])

x = data_train.iloc[:,3:4]

y = data_train.iloc[:,4:5]

plt.scatter(x,y,clusters,c="r",marker='o',alpha=0.4,linewidth=2.0)

ax.grid(True)

fig.tight_layout()

plt.show()

fig, ax = plt.subplots()

ax.set_title('Cluster distribution',fontsize=18)

plt.hist(clusters)

plt.show()



# Create new data



m = pd.DataFrame(z_1)

m.columns = ['d1','d2','d3','d4']

col = m.columns

new_reg = m.loc[[15118]] - 100

join_reg = pd.concat([m,new_reg])

join_reg.reset_index(drop=False, inplace = True)

join_reg.reindex(df_train.index, fill_value=0)

join_reg.set_index(df_train.index)

new_cluster_data = pd.DataFrame({ 'Cluster_G2' : cl_data['Cluster_G2'].values,

                                  'Cluster_G2_Data_x' : join_reg.d1.values,

                                  'Cluster_G2_Data_y' : join_reg.d2.values,

                                  'Cluster_G2_Data_xy' : join_reg.d3.values,

                                  'Cluster_G2_Data_ncl' : join_reg.d4.values,

    

},index=df_train.index)



print("Joining Accuracy :",c)

print("Matrix distances Train :",m.shape)

print("Cluster distances Train :",cl_data.shape)

print("Join data_clusters Train :",new_cluster_data.shape)

new_cluster_data.head()
original_data = abs(df_train.copy())

a = pd.merge(original_data,train.iloc[:,53:], right_index=True, left_index=True)

train = pd.merge(a,new_cluster_data, right_index=True, left_index=True)



# Horizontal_Distance_+_Vertical_Distance_To_Hydrology

train['Cluster_G2_HDR_+_HDFP_sin_'] = abs(np.ceil(np.sin(train['Cluster_G2'])))

train['Cluster_G2_HDR_+_HDFP_cos_'] = abs(np.ceil(np.cos(train['Cluster_G2'])))

train['Cluster_G2_HDR_+_HDFP_tanh_'] = abs(np.ceil(np.tanh(train['Cluster_G2'])))

train['Cluster_G2_HDR_+_HDFP_bin_round_100'] = np.array(np.floor(np.array(train['Cluster_G2']) / 100))

train['Cluster_G2_HDR_+_HDFP_bin_round_1000'] = np.array(np.floor(np.array(train['Cluster_G2']) / 1000))

train['Cluster_G2_HDR_+_HDFP_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Cluster_G2']))))

train['Cluster_G2_HDR_+_HDFP_log'] = np.ceil(np.log(np.floor(np.array(1+train['Cluster_G2']))))

print("Train Size : ",train.shape)

print("previous accuracy :",training_score,"current accuracy :",score_dataset(train.copy()))

print("New features have been added...")

train.head()
Cluster_G2 = new_cluster_data

new_concat = pd.concat([Cluster_G2,Cluster_G2])

new_concat = pd.concat([new_concat,new_concat])

new_concat = pd.concat([new_concat,new_concat])

new_concat = pd.concat([new_concat,new_concat])

new_concat = pd.concat([new_concat,new_concat])

new_concat = pd.concat([new_concat,Cluster_G2])

new_concat = pd.concat([new_concat,Cluster_G2])

new_concat = pd.concat([new_concat,Cluster_G2])

new_concat = pd.concat([new_concat,Cluster_G2])

new_concat = pd.concat([new_concat,Cluster_G2])

new_concat = pd.concat([new_concat,Cluster_G2])

first, second = split(new_concat, 565892)

new_test_g2 = first.sort_index(ascending=True)

new_test_cluster = pd.DataFrame({ 'Cluster_G2' : new_test_g2['Cluster_G2'].values,

                                  'Cluster_G2_Data_x' : new_test_g2.Cluster_G2_Data_x.values,

                                  'Cluster_G2_Data_y' : new_test_g2.Cluster_G2_Data_y.values,

                                  'Cluster_G2_Data_xy' : new_test_g2.Cluster_G2_Data_xy.values,

                                  'Cluster_G2_Data_ncl' : new_test_g2.Cluster_G2_Data_ncl.values,

    

},index=df_test.index)

original_data = abs(df_test.copy())

a = pd.merge(original_data,test.iloc[:,52:], right_index=True, left_index=True)

test = pd.merge(a,new_test_cluster, right_index=True, left_index=True)

test['Cluster_G2_HDR_+_HDFP_sin_'] = abs(np.ceil(np.sin(test['Cluster_G2'])))

test['Cluster_G2_HDR_+_HDFP_cos_'] = abs(np.ceil(np.cos(test['Cluster_G2'])))

test['Cluster_G2_HDR_+_HDFP_tanh_'] = abs(np.ceil(np.tanh(test['Cluster_G2'])))

test['Cluster_G2_HDR_+_HDFP_bin_round_100'] = np.array(np.floor(np.array(test['Cluster_G2']) / 100))

test['Cluster_G2_HDR_+_HDFP_bin_round_1000'] = np.array(np.floor(np.array(test['Cluster_G2']) / 1000))

test['Cluster_G2_HDR_+_HDFP_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Cluster_G2']))))

test['Cluster_G2_HDR_+_HDFP_log'] = np.ceil(np.log(np.floor(np.array(1+test['Cluster_G2']))))

print("Join data_clusters Test :",new_test_cluster.shape)

print("Test Size : ",test.shape)

print("New features have been added...")

test.head()
data_train = abs(df_train.copy())

data_group_3 = data_join(data_train[['Horizontal_Distance_To_Roadways']], data_train[['Horizontal_Distance_To_Fire_Points']],None)

data_group_3.head()

x = data_join(data_group_3, data_train[['Cover_Type']],None)

compare_plot(x)

data_group_3.hist(bins=60)

fig, ax = plt.subplots(figsize = (8,8))

ax.set_xlabel('Horizontal_Distance_To_Roadways', fontsize=12)

ax.set_ylabel('Horizontal_Distance_To_Fire_Points', fontsize=12)

ax.set_title('Scatter',fontsize=18)

x = data_group_3.iloc[:,0:1]

y = data_group_3.iloc[:,1:2]

plt.scatter(x,y,c="r",marker='^',alpha=0.25,linewidth=2.0)
z_2 = linkage(data_group_3, method='average')

c,coph_dist = cophenet(z_2,pdist(data_group_3))



fig, ax = plt.subplots(figsize = (30,30))

ax.set_xlabel('Horizontal_Distance_To_Roadways', fontsize=25)

ax.set_ylabel('Horizontal_Distance_To_Fire_Points', fontsize=25)

ax.set_title('Cluster',fontsize=25)

clusters_2 = fcluster(z_2,2,criterion="distance")

cl_data = pd.DataFrame(np.array(clusters_2),columns=['Cluster_G3'])

x = data_group_3.iloc[:,0:1]

y = data_group_3.iloc[:,1:2]

plt.scatter(x,y,clusters_2,c="r",marker='o',alpha=0.4,linewidth=2.0)

ax.grid(True)

fig.tight_layout()

plt.show()

fig, ax = plt.subplots()

ax.set_title('Cluster distribution',fontsize=18)

plt.hist(clusters_2,bins=90)

plt.show()



# Create new data



m = pd.DataFrame(z_2)

m.columns = ['d1','d2','d3','d4']

col = m.columns

new_reg = m.loc[[15118]] - 100

join_reg = pd.concat([m,new_reg])

join_reg.reset_index(drop=False, inplace = True)

join_reg.reindex(df_train.index, fill_value=0)

join_reg.set_index(df_train.index)

new_cluster_data3 = pd.DataFrame({'Cluster_G3' : cl_data['Cluster_G3'].values,

                                  'Cluster_G3_Data_x' : join_reg.d1.values,

                                  'Cluster_G3_Data_y' : join_reg.d2.values,

                                  'Cluster_G3_Data_xy' : join_reg.d3.values,

                                  'Cluster_G3_Data_nc' : join_reg.d4.values,

    

},index=df_train.index)



print("Joining Accuracy :",c)

print("Matrix distances Train :",m.shape)

print("Cluster distances Train :",cl_data.shape)

print("Join data_clusters Train :",new_cluster_data3.shape)

new_cluster_data3.head()
original_data = abs(df_train.copy())

b = pd.merge(original_data,train.iloc[:,53:], right_index=True, left_index=True)

train = pd.merge(b,new_cluster_data3, right_index=True, left_index=True)



# Cluster_Horizontal_Distance_To_Roadways_+_Horizontal_Distance_To_Fire_Points

train['Cluster_G3_HDR_+_HDFP_sin_'] = abs(np.ceil(np.sin(train['Cluster_G3'])))

train['Cluster_G3_HDR_+_HDFP_cos_'] = abs(np.ceil(np.cos(train['Cluster_G3'])))

train['Cluster_G3_HDR_+_HDFP_tanh_'] = abs(np.ceil(np.tanh(train['Cluster_G3'])))

train['Cluster_G3_HDR_+_HDFP_bin_round_100'] = np.array(np.floor(np.array(train['Cluster_G3']) / 100))

train['Cluster_G3_HDR_+_HDFP_bin_round_1000'] = np.array(np.floor(np.array(train['Cluster_G3']) / 1000))

train['Cluster_G3_HDR_+_HDFP_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Cluster_G3']))))

train['Cluster_G3_HDR_+_HDFP_log'] = np.ceil(np.log(np.floor(np.array(1+train['Cluster_G3']))))

print("Train Size : ",train.shape)

print("previous accuracy :",training_score,"current accuracy :",score_dataset(train.copy()))

print("New features have been added...")

train.head()
Cluster_G3 = new_cluster_data3

new_Cluster_G3 = pd.concat([Cluster_G3,Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,new_Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,new_Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,new_Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,new_Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,Cluster_G3])

new_Cluster_G3 = pd.concat([new_Cluster_G3,Cluster_G3])

first, second = split(new_Cluster_G3, 565892)

new_test_g3 = first.sort_index(ascending=True)

new_cluster_test3 = pd.DataFrame({'Cluster_G3' : new_test_g3['Cluster_G3'].values,

                                  'Cluster_G3_Data_x' : new_test_g3.Cluster_G3_Data_x.values,

                                  'Cluster_G3_Data_y' : new_test_g3.Cluster_G3_Data_y.values,

                                  'Cluster_G3_Data_xy' : new_test_g3.Cluster_G3_Data_xy.values,

                                  'Cluster_G3_Data_nc' : new_test_g3.Cluster_G3_Data_nc.values,

    

},index=df_test.index)

original_data = abs(df_test.copy())

a = pd.merge(original_data,test.iloc[:,52:], right_index=True, left_index=True)

test = pd.merge(a,new_cluster_test3, right_index=True, left_index=True)

test['Cluster_G3_HDR_+_HDFP_sin_'] = abs(np.ceil(np.sin(test['Cluster_G3'])))

test['Cluster_G3_HDR_+_HDFP_cos_'] = abs(np.ceil(np.cos(test['Cluster_G3'])))

test['Cluster_G3_HDR_+_HDFP_tanh_'] = abs(np.ceil(np.tanh(test['Cluster_G3'])))

test['Cluster_G3_HDR_+_HDFP_bin_round_100'] = np.array(np.floor(np.array(test['Cluster_G3']) / 100))

test['Cluster_G3_HDR_+_HDFP_bin_round_1000'] = np.array(np.floor(np.array(test['Cluster_G3']) / 1000))

test['Cluster_G3_HDR_+_HDFP_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Cluster_G3']))))

test['Cluster_G3_HDR_+_HDFP_log'] = np.ceil(np.log(np.floor(np.array(1+test['Cluster_G3']))))

print("Join data_clusters Test :",new_cluster_test3.shape)

print("Test Size : ",test.shape)

print("New features have been added...")

test.head()
data_train = abs(df_train.copy())

data_group_4 = data_train.iloc[:,6:9]

sns.set(style="darkgrid")

sns.pairplot(data_group_4, palette ="b",kind ="scatter",

             height=2.5)

plt.show()

data_group_4.head()
z_3 = linkage(data_group_4, method='average')

c,coph_dist = cophenet(z_3,pdist(data_group_4,'chebyshev'))



fig, ax = plt.subplots(figsize = (30,30))

ax.set_xlabel('Horizontal_Distance_To_Roadways', fontsize=25)

ax.set_ylabel('Horizontal_Distance_To_Fire_Points', fontsize=25)

ax.set_title('Cluster',fontsize=25)

clusters_3 = fcluster(z_3,3,criterion="distance")

cl_data = pd.DataFrame(np.array(clusters_3),columns=['Cluster_G4'])

x = data_group_3.iloc[:,0:1]

y = data_group_3.iloc[:,1:2]

plt.scatter(x,y,clusters_3,c="orange",marker='o',alpha=0.4,linewidth=2.0)

ax.grid(True)

fig.tight_layout()

plt.show()

fig, ax = plt.subplots()

ax.set_title('Cluster distribution',fontsize=18)

plt.hist(clusters_3,bins=90)

plt.show()



# Create new data



m = pd.DataFrame(z_3)

m.columns = ['d1','d2','d3','d4']

col = m.columns

new_reg = m.loc[[15118]] - 100

join_reg = pd.concat([m,new_reg])

join_reg.reset_index(drop=False, inplace = True)

join_reg.reindex(df_train.index, fill_value=0)

join_reg.set_index(df_train.index)

new_cluster_data3 = pd.DataFrame({ 'Cluster_G4' : cl_data['Cluster_G4'].values,

                                  'Cluster_G4_Data_x' : join_reg.d1.values,

                                  'Cluster_G4_Data_y' : join_reg.d2.values,

                                  'Cluster_G4_Data_xy' : join_reg.d3.values,

                                  'Cluster_G4_Data_nc' : join_reg.d4.values,

    

},index=df_train.index)



print("Joining Accuracy :",c)

print("Matrix distances Train :",m.shape)

print("Cluster distances Train :",cl_data.shape)

print("Join data_clusters Train :",new_cluster_data3.shape)

new_cluster_data3.head()
original_data = abs(df_train.copy())

c = pd.merge(original_data,train.iloc[:,53:], right_index=True, left_index=True)

train = pd.merge(c,new_cluster_data3, right_index=True, left_index=True)

# Cluster_G4_HDR_+_HDFP

train['Cluster_G4_HDR_+_HDFP_sin_'] = abs(np.ceil(np.sin(train['Cluster_G4'])))

train['Cluster_G4_HDR_+_HDFP_cos_'] = abs(np.ceil(np.cos(train['Cluster_G4'])))

train['Cluster_G4_HDR_+_HDFP_tanh_'] = abs(np.ceil(np.tanh(train['Cluster_G4'])))

train['Cluster_G4_HDR_+_HDFP_bin_round_100'] = np.array(np.floor(np.array(train['Cluster_G4']) / 100))

train['Cluster_G4_HDR_+_HDFP_bin_round_1000'] = np.array(np.floor(np.array(train['Cluster_G4']) / 1000))

train['Cluster_G4_HDR_+_HDFP_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(train['Cluster_G4']))))

train['Cluster_G4_HDR_+_HDFP_log'] = np.ceil(np.log(np.floor(np.array(1+train['Cluster_G4']))))

print("Train Size : ",train.shape)

print("previous accuracy :",training_score,"current accuracy :",score_dataset(train.copy()))

print("New features have been added...")

train.head()
Cluster_G4 = new_cluster_data3

new_Cluster_G4 = pd.concat([Cluster_G4,Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,new_Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,new_Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,new_Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,new_Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,Cluster_G4])

new_Cluster_G4 = pd.concat([new_Cluster_G4,Cluster_G4])

first, second = split(new_Cluster_G4, 565892)

new_test_g4 = first.sort_index(ascending=True)



new_cluster_data4 = pd.DataFrame({'Cluster_G4' : new_test_g4['Cluster_G4'].values,

                                  'Cluster_G4_Data_x' : new_test_g4.Cluster_G4_Data_x.values,

                                  'Cluster_G4_Data_y' : new_test_g4.Cluster_G4_Data_y.values,

                                  'Cluster_G4_Data_xy' : new_test_g4.Cluster_G4_Data_xy.values,

                                  'Cluster_G4_Data_nc' : new_test_g4.Cluster_G4_Data_nc.values,

    

},index=df_test.index)



original_data = abs(df_test.copy())

b = pd.merge(original_data,test.iloc[:,52:], right_index=True, left_index=True)

test = pd.merge(b,new_cluster_data4, right_index=True, left_index=True)



# Cluster_G4_HDR_+_HDFP

test['Cluster_G4_HDR_+_HDFP_sin_'] = abs(np.ceil(np.sin(test['Cluster_G4'])))

test['Cluster_G4_HDR_+_HDFP_cos_'] = abs(np.ceil(np.cos(test['Cluster_G4'])))

test['Cluster_G4_HDR_+_HDFP_tanh_'] = abs(np.ceil(np.tanh(test['Cluster_G4'])))

test['Cluster_G4_HDR_+_HDFP_bin_round_100'] = np.array(np.floor(np.array(test['Cluster_G4']) / 100))

test['Cluster_G4_HDR_+_HDFP_bin_round_1000'] = np.array(np.floor(np.array(test['Cluster_G4']) / 1000))

test['Cluster_G4_HDR_+_HDFP_sqrt'] = np.ceil(np.sqrt(np.floor(np.array(test['Cluster_G4']))))

test['Cluster_G4_HDR_+_HDFP_log'] = np.ceil(np.log(np.floor(np.array(1+test['Cluster_G4']))))

print("Join data_clusters Test :",new_cluster_data4.shape)

print("Test Size : ",test.shape)

print("New features have been added...")

test.head()
original_data = abs(df_train.copy())

col_g5 = original_data.iloc[:,10:52].columns.values

data_group_5 = data_train.iloc[:,10:52]

# Wilderness_Area1	Wilderness_Area2	Wilderness_Area3	Wilderness_Area4

cols=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

train['WA1_+_WA2_+_WA3_+_WA4_sum'] = data_group_5[cols].sum(axis=1)

train['WA1_+_WA2_+_WA3_+_WA4_mean'] = data_group_5[cols].mean(axis=1)

train['WA1_+_WA2_+_WA3_+_WA4_std'] = np.round(data_group_5[cols].std(axis=1))

train['WA1_+_WA2_+_WA3_+_WA4_min'] = data_group_5[cols].min(axis=1)

train['WA1_+_WA2_+_WA3_+_WA4_max'] = data_group_5[cols].max(axis=1)

train['WA1_+_WA2_+_WA3_+_WA4_median'] = data_group_5[cols].median(axis=1)

train['WA1_+_WA2_+_WA3_+_WA4_quantile'] = data_group_5[cols].quantile(axis=1)

print("Train Size : ",train.shape)

print("previous accuracy :",training_score,"current accuracy :",score_dataset(train.copy()))

print("New features have been added...")

#output = pd.DataFrame(new_data)

#output.to_csv('new_train.csv', index=False)

#output.head()
original_data = abs(df_test.copy())

col_g5 = original_data.iloc[:,10:52].columns.values

data_group_5 = df_test.iloc[:,10:52]

# Wilderness_Area1	Wilderness_Area2	Wilderness_Area3	Wilderness_Area4

cols=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

test['WA1_+_WA2_+_WA3_+_WA4_sum'] = data_group_5[cols].sum(axis=1)

test['WA1_+_WA2_+_WA3_+_WA4_mean'] = data_group_5[cols].mean(axis=1)

test['WA1_+_WA2_+_WA3_+_WA4_std'] = np.round(data_group_5[cols].std(axis=1))

test['WA1_+_WA2_+_WA3_+_WA4_min'] = data_group_5[cols].min(axis=1)

test['WA1_+_WA2_+_WA3_+_WA4_max'] = data_group_5[cols].max(axis=1)

test['WA1_+_WA2_+_WA3_+_WA4_median'] = data_group_5[cols].median(axis=1)

test['WA1_+_WA2_+_WA3_+_WA4_quantile'] = data_group_5[cols].quantile(axis=1)

print("Test Size : ",test.shape)

print("New features have been added...")

#output = pd.DataFrame(new_data)

#output.to_csv('new_test.csv', index=False)

#output.head()
data_train = train.copy()

data_test = test.copy()

y = data_train.Cover_Type.values # <- Target

X = data_train # Data

X.drop(['Cover_Type'], axis=1, inplace=True) # Drop target

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_test = scaler.transform(data_test) 

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, random_state=2019)

RF = RandomForestClassifier(n_jobs =  -1, n_estimators = 1000,random_state = 1)

RF.fit(X=X_train, y=y_train)

pred = RF.predict(X_test)

output = pd.DataFrame({'Id': df_test.index,

                       'Cover_Type': pred})

output.to_csv('submission.csv', index=False)

output.head()