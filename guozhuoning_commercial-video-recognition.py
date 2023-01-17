import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# 导入文件依据运行的环境和平台进行必要的更改

data = pd.read_csv("../input/commercial-vedio-data/commercial_vedio_data.csv", index_col=0)



# Rename

data.rename(columns={'1':'Length', '2':'Move_E', '3':'Move_D', '4':'Frame_E', '5':'Frame_D', '6':'Energe_E', '7':'Energe_D', '8':'ZCR_E', '9':'ZCR_D', '10':'Centroid_E', '11':'Centroid_D', '12':'Rolloff_E', '13':'Rolloff_D', '14':'Flux_E', '15':'Flux_D', '16':'BasFreq_E', '17':'BasFreq_D', '4124':'Edge_E', '4125':'Edge_D', 'labels':'Label'}, inplace=True)



# 特征名和标签名

col_name = data.columns[:-2]

label_name = data.columns[-1]



print ('训练集的标签:{}\n'.format(label_name))

print ('训练集的特征:{}\n'.format(col_name))

print ('训练集的形状:{}\n'.format(data.shape))
# 打印data的前五行数据

data.head()
# Label分布的直方图

sns.distplot(data['Label'], kde=False)
# 描述数据中特征的分布

data.describe()
# 时长Length分布和统计

data.drop(data[data['Length'] > 10000].index.tolist(), inplace=True)



fig, axes = plt.subplots(1, 2)

sns.barplot(x='Label', y='Length', data=data, ax=axes[0])

sns.stripplot(x='Label', y='Length', data=data, ax=axes[1], jitter=True)

plt.show()



facet = sns.FacetGrid(data[['Length', 'Label']], hue='Label', aspect=2)

facet.map(sns.kdeplot, "Length", shade=True)

facet.set(xlim=(0, 500))

facet.add_legend()

facet.set_axis_labels("Length", "Density")
# 缺失值

data.isnull().any()
# 填充缺失值

data = data.fillna(data.mean())
# 重复值

data.drop_duplicates(inplace=True)

data.shape
# Label -1 -> 0

data['Label'] = data['Label'].apply(lambda x:0 if x == -1 else x)

data['Label'].hist()
# 分离特征和标签

X = data.drop(['Label'], axis=1)

Y = data['Label']



# 划分训练集和测试集

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.75)



xtrain.shape
from sklearn.ensemble import RandomForestClassifier

rfcModel = RandomForestClassifier()

rfcModel.fit(xtrain, ytrain)
# 将特征的重要性程度进行排序

N_most_important = 25



imp = np.argsort(rfcModel.feature_importances_)[::-1]

imp_slct = imp[:N_most_important]



FeaturesImportances = zip(col_name, map(lambda x:round(x,5), rfcModel.feature_importances_))

FeatureRank = pd.DataFrame(columns=['Feature', 'Imp'], data=sorted(FeaturesImportances, key=lambda x:x[1], reverse=True)[:N_most_important])
# 重新选择X

xtrain_slct = xtrain.iloc[:,imp_slct]

xtest_slct  = xtest.iloc[:,imp_slct]
# 特征排序图

ax1 = fig.add_subplot(111)

ax1 = sns.barplot(x='Feature', y='Imp', data=FeatureRank)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)



SumImp = FeatureRank

for i in SumImp.index:

    if (i==0):

        SumImp['Imp'][i] = FeatureRank['Imp'][i]

    else:

        SumImp['Imp'][i] = SumImp['Imp'][i-1] + FeatureRank['Imp'][i]

ax2 = ax1.twinx()

plt.step(x=SumImp['Feature'], y=SumImp['Imp'])

from sklearn.decomposition import PCA

pca = PCA(n_components=N_most_important)

pca.fit(xtrain)

pca.explained_variance_ratio_
# 对训练集使用PCA生成新特征，根据累计贡献率，保留前5个主成分

pca1 = PCA(6)

pc = pd.DataFrame(pca1.fit_transform(xtrain))

pc.index = xtrain.index

xtrain_pca = xtrain.join(pc)
# 对测试集进行相同的操作，注意测试集上直接使用pca中的transform函数

pc = pd.DataFrame(pca1.fit_transform(xtest))

pd.index = xtrain.index

xtest_pca = xtest.join(pc)
#  cut_bin对训练集进行分箱

def cut_bin(df,label,max_depth,p):

    df_bin = df[[label]]

    df_feature = df.drop([label],axis=1)

    dict_bin = {}

    for col in df_feature.columns:

        get_model = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=int(p*len(df)))

        get_cut_point = get_model.fit(df[col].values.reshape(-1,1),df[label].values.reshape(-1,1))

        cut_point = get_cut_point.tree_.threshold[get_cut_point.tree_.threshold!=-2]

        

        N_split = np.zeros_like(df[col])

        inter_range = []

        if len(cut_point)==1:

            N_split[np.array(df[col]<cut_point[0])]=1

            N_split[np.array(df[col]>=cut_point[0])]=2

            inter_range=[[1,-100000000,cut_point[0]],[2,cut_point[0],100000000]]

        elif len(cut_point)>1:

            cut_point.sort()

            N_split[np.array(df[col]<cut_point[0])]=1

            inter_range=[[1,-100000000,cut_point[0]]]

            for i in range(len(cut_point)-1):

                N_split[np.array((df[col]>=cut_point[i]) & (df[col]<cut_point[i+1]))]=i+2

                inter_range=inter_range+[[i+2,cut_point[i],cut_point[i+1]]]

            N_split[np.array(df[col]>=cut_point[len(cut_point)-1])]=len(cut_point)+1

            inter_range=inter_range+[[len(cut_point)+1,cut_point[len(cut_point)-1],100000000]]

        else:

            N_split=1

            inter_range=np.array([1,-100000000,100000000]).reshape(1,-1)

        df_bin[col] = N_split

        inter_df = pd.DataFrame(inter_range)

        inter_df.columns=['bin','lower','upper']

        crosstable = pd.crosstab(df_bin[col],df_bin[label])

        crosstable.columns = ['notCommercial','Commercial']

        crosstable['all'] = crosstable['notCommercial']+crosstable['Commercial']

        crosstable['percent'] = crosstable['all']/sum(crosstable['all'])

        crosstable['c_rate'] = crosstable['Commercial']/crosstable['all']

        inter_df = pd.merge(inter_df, crosstable, left_on='bin', right_index=True)

        dict_bin[col] = inter_df

    return df_bin, dict_bin



#  cut_test_bin对测试集进行分箱

def cut_test_bin(df, label, train_dict_bin):

    df_bin = df[[label]]

    df_feature = df.drop([label],axis=1)

    dict_bin = {}

    for col in df_feature.columns:

        train_bin = train_dict_bin[col]

        splited = pd.Series([np.nan]*len(df[col]))

        for i in range(len(train_bin['bin'])):

            splited[((df[col]>=train_bin['lower'][i]) & (df[col]<train_bin['upper'][i])).tolist()]=train_bin['bin'][i]

            df_bin[col]=splited.tolist()

        crosstable = pd.crosstab(df_bin[col],df_bin[label])

        crosstable.columns = ['notCommercial','Commercial']

        crosstable['all'] = crosstable['notCommercial']+crosstable['Commercial']

        crosstable['percent'] = crosstable['all']/sum(crosstable['all'])

        crosstable['c_rate'] = crosstable['Commercial']/crosstable['all']

        inter_df = pd.merge(train_bin[['bin','lower','upper']], crosstable, left_on='bin', right_index=True, how='left')

        dict_bin[col] = inter_df

    return df_bin, dict_bin   
# 使用决策树进行特征分箱

from sklearn.tree import DecisionTreeClassifier

train = xtrain.join(ytrain)

test  = xtest.join(ytest)

new_train, train_dict_bin = cut_bin(train, 'Label', 50, 0.2)

new_test , test_dict_bin  = cut_test_bin(test, 'Label', train_dict_bin)

# 分离特征和标签

xtrain = new_train.drop(['Label'], axis=1)

xtest  = new_test.drop(['Label'] , axis=1)

ytrain = new_train['Label']

ytest  = new_test['Label']
# 随机森林分类器训练模型

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features=16,max_depth=12,n_estimators=2048,n_jobs=-1,random_state=0)

rf.fit(xtrain, ytrain)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score, accuracy_score



# AUC和混淆矩阵评估

ytrain_pred_clf = rf.predict_proba(xtrain)

ytrain_pred = rf.predict(xtrain)

ytest_pred_clf = rf.predict_proba(xtest)

ytest_pred = rf.predict(xtest)



# 评估训练集效果，直观判断是否过拟合

print ('分类模型训练集表现：')

print ('ml train model auc score {:.6f}'.format(roc_auc_score(ytrain, ytrain_pred_clf[:,1])))

print ('------------------------------')

print ('ml train model accuracy score {:.6f}'.format(accuracy_score(ytrain, ytrain_pred)))

print ('------------------------------')

threshold = 0.5

print (confusion_matrix(ytrain, (ytrain_pred_clf>threshold)[:,1]))



# 评估测试集效果

print ('分类模型测试集表现：')

print ('ml model auc score {:.6f}'.format(roc_auc_score(ytest, ytest_pred_clf[:,1])))

print ('------------------------------')

print ('ml model accuracy score {:.6f}'.format(accuracy_score(ytest, ytest_pred)))

print ('------------------------------')

threshold = 0.5

print (confusion_matrix(ytest, (ytest_pred_clf>threshold)[:,1]))



# 随机猜测函数对比

ytest_random_clf = np.random.uniform(low=0.0, high=1.0, size=len(ytest))

print ('random model auc score {:.6f}'.format(roc_auc_score(ytest, ytest_random_clf)))

print ('------------------------------')

print (confusion_matrix(ytest, (ytest_random_clf<=threshold).astype('int')))
## 计算各阈值下假阳性率、真阳性率和AUC

from sklearn.metrics import roc_curve, auc

fpr,tpr,threshold = roc_curve(ytest,ytest_pred_clf[:,1])

roc_auc = auc(fpr,tpr)
## 假阳性率为横坐标，真阳性率为纵坐标做曲线

plt.figure()

lw = 2

plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, 

         label='ROC curve (area = %0.2f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve')

plt.legend(loc="lower right")

plt.show()