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
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("darkgrid")
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm_notebook
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
data = pd.read_csv("../input/mf-accelerator/contest_train.csv")
data.head()
features = data.drop(columns=["ID","TARGET"])
features.head()
target = data.TARGET
count_dict = {}
for col in features.columns:
    count_dict[col] = [features[col].unique().shape[0]]
feats_counts = pd.DataFrame(count_dict).T
feats_counts.columns=["counts"]
feats_counts.head()
sorted_feats = feats_counts.copy().sort_values(by=["counts"])
sorted_feats[sorted_feats.counts < 2].index
sorted_feats[sorted_feats.counts == 3].index
data.TARGET.hist()
features[["FEATURE_207","FEATURE_205","FEATURE_208"]].hist();
features[["FEATURE_257","FEATURE_259"]].hist();
features[["FEATURE_123","FEATURE_131","FEATURE_156"]].hist();
def null_map(data):
    data_to_heat = data.isnull()
    data_to_heat.head()
    with plt.xkcd():
        plt.figure(figsize=(20,14))
        colors = ['#000099', '#ffff00'] 
        sns.heatmap(data_to_heat,cmap = sns.color_palette(colors));
        
null_map(features)
na_prop_dict = {}
for col in features.columns:
    na_prop_dict[col] = [features[col].isnull().astype(int).mean()]
feats_na_prop = pd.DataFrame(na_prop_dict).T
feats_na_prop.columns=["NAN_prop"]
sorted_feats_nan_prop = feats_na_prop.copy().sort_values(by=["NAN_prop"],ascending=False)
sorted_feats_nan_prop.head()
sorted_feats_nan_prop[sorted_feats_nan_prop.NAN_prop > 0.5]
features_new = features.copy().drop(columns = ['FEATURE_249', 'FEATURE_3', 'FEATURE_256', 'FEATURE_144',"FEATURE_189"])
features_new.head()
def LR_feature_selection(features,max_R2=0.9):
    '''
    Функция, которая строит LR каждого признака на все остальные
    и убирает признаки, где R2 > max_R2
    Возвращает итоговые признаки и таблицу R2
    '''
    features = features.copy().fillna(0)

    features_list = list(features.columns)
    R2s = {}
    for col in tqdm_notebook(features.columns):

        model = LinearRegression()
        X = features[features_list].drop(columns = [col])
        y = features[col]

        model.fit(X,y)

        R2 = model.score(X,y)
        R2s[col] = [R2]

        if R2 > max_R2:
            features_list = list(features[features_list].drop(columns = [col]).columns)

    R2_df = pd.DataFrame(R2s).T
    R2_df.columns=["R2"]

    return features[features_list],R2_df
feats_selected, R2_df = LR_feature_selection(features_new)
R2_df.plot()
feats_selected.shape[1]
feats_selected.head()
def RF_feature_selection_validation(features_train,target_train,features_val,target_val, min_metric = 0.01):
    '''
    Строит RF классификатор таргета по каждому из признаков 
    отдельно, отбирая признаки, проверяет на валидации 

    Выводит список отобранных признаков и значения метрики
    '''
    assert len(features_train.columns) == len(features_val.columns) 


    features_train = features_train.copy().fillna(0)
    target_train = target_train.copy()

    features_val = features_val.copy().fillna(0)
    target_val = target_val.copy()

    features_list = list(features_train.columns)
    metric_values = {}
    for col in tqdm_notebook(features_train.columns):

        model = RandomForestClassifier(n_estimators=10,n_jobs=-1)
        X = features_train[[col]]
        y = target_train

        model.fit(X,y)

        predictions = model.predict(features_val[[col]])

        metric = f1_score(target_val, predictions,average='macro')
    
        metric_values[col] = [metric]

        if metric < min_metric:
            features_list = list(features_train[features_list].drop(columns = [col]).columns)

    metric_values = pd.DataFrame(metric_values).T
    metric_values.columns=["f1_average"]

    return features_list, metric_values
feats_train,feats_val,labels_train,labels_val = train_test_split(feats_selected,target, test_size = 0.3,\
                                                                   shuffle=True,random_state=42,\
                                                                   stratify = target)
features_selectedRF,metric_values = RF_feature_selection_validation(feats_train,labels_train,
                                                                      feats_val,labels_val,
                                                                      min_metric = 0.3)
metric_values.plot()
len(features_selectedRF)
X_scaled = StandardScaler().fit_transform(data[features_selectedRF].copy().fillna(0))

X_pca =  PCA(n_components=2).fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca,columns=["component1","component2"])
pca_df['target'] = target
plt.figure(figsize=(14,10))
sns.scatterplot(data=pca_df, x="component1", y="component2", hue="target",alpha=0.5)
X_embedded = TSNE(n_components=2).fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_embedded,columns=["component1","component2"])
tsne_df['target'] = target
plt.figure(figsize=(14,10))
sns.scatterplot(data=tsne_df, x="component1", y="component2", hue="target",alpha=0.5)
