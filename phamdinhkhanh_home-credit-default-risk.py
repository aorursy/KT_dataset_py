import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import os

import seaborn as sns



%matplotlib inline



# Checking for all file

os.listdir('../input')
# Training data

app_train = pd.read_csv('../input/application_train.csv')

print('Training data shape: ', app_train.shape)

app_train.head()
# Testing data features

app_test = pd.read_csv('../input/application_test.csv')

print('Testing data shape: ', app_test.shape)

app_test.head()
app_train['TARGET'].value_counts().plot.bar()

n_group = app_train['TARGET'].value_counts()

n_group_sum = n_group.sum()



print('Repaid: {}'.format(n_group[0]))

print('Not repaid: {}'.format(n_group[1]))



print('Repaid: {:.2f} {}'.format(n_group[0]/n_group_sum*100, '%'))

print('Not repaid: {:.2f} {}'.format(n_group[1]/n_group_sum*100, '%'))
def summary_missing(dataset):

    n_miss = dataset.isnull().sum()

    n_obs = dataset.shape[0]

    n_miss_per = n_miss/n_obs*100

    n_miss_tbl = pd.concat([n_miss, n_miss_per], axis = 1).sort_values(1, ascending = False).round(1)

    n_miss_tbl = n_miss_tbl[n_miss_tbl[1] != 0]

    print('No fields: ', dataset.shape[0])

    print('No missing fields: ', n_miss_tbl.shape[0])

    n_miss_tbl = n_miss_tbl.rename(columns = {0:'Number mising Value', 1:'Percentage missing Value'})

    return n_miss_tbl



summary_missing(app_train)
def _tbl_dtype(dataset):

    sum_dtype = pd.DataFrame(dataset.dtypes).sort_values(0).rename(columns = {0:'Data Type'})

    return sum_dtype



table_dtype = _tbl_dtype(app_train)

table_dtype
table_dtype['Data Type'].value_counts()
# Các dòng dữ liệu dạng object

app_train.select_dtypes('object').head()
app_train.select_dtypes('object').apply(pd.Series.nunique)
dtypes_object = table_dtype[table_dtype['Data Type'] == 'object'].index.tolist()

dtypes_object = [col for col in dtypes_object if col not in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']]





def _plot_bar_classes(cols):

    app_train[cols].value_counts().plot.bar()



plt.figure(figsize = (20, 15))    

for i in range(1, 15, 1):

    plt.subplot(5, 3, i)

    _plot_bar_classes(dtypes_object[i-1])

    plt.title(dtypes_object[i-1])
def _per_categorical(col):

    tbl_per = pd.pivot_table(app_train[['TARGET', col]], index = ['TARGET'], columns = [col], aggfunc = len)

    per_categorical = (tbl_per.iloc[0, :]/tbl_per.iloc[1, :]).sort_values(ascending = True)

    print(per_categorical)

    print('-------------------------------------\n')

    return per_categorical



for col in dtypes_object:

    _per_categorical(col)
def _plot_per_categorical(col):

    tbl_per = pd.pivot_table(app_train[['TARGET', col]], index = ['TARGET'], columns = [col], aggfunc = len)

    per_categorical = (tbl_per.iloc[0, :]/tbl_per.iloc[1, :]).sort_values(ascending = True)

    per_categorical.plot.bar()

    plt.title(col)

    return per_categorical



plt.figure(figsize = (20, 15))

i = 0

for col in dtypes_object:

    i += 1

    plt.subplot(5, 3, i)

    _plot_per_categorical(col)
plt.figure(figsize = (15, 7))

i = 0

for col in ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']:

    i += 1

    plt.subplot(2, 1, i)

    _plot_per_categorical(col)
for col in ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']:

    _per_categorical(col)
# Nhóm các giá trị rate gần bằng nhau vào 1 nhóm theo schedule_div.

def _devide_group(col, schedule_div = None, n_groups = 3, *kwargs):

    cols = []

    tbl_per_cat = _per_categorical(col)

    

    if schedule_div is None:

        n_cats = len(tbl_per_cat)

        n_val_incat = int(n_cats/n_groups)

        n_odd = n_cats - n_groups*n_val_incat



        for i in range(n_groups):

            if i == n_groups - 1:

                el = tbl_per_cat[(n_val_incat*i):(n_val_incat*(i+1)+n_odd)].index.tolist()

            else:

                el = tbl_per_cat[(n_val_incat*i):n_val_incat*(i+1)].index.tolist()    

            cols.append(el)

    else:

        idx = 0

        for n_cols in schedule_div:

            el_cols = tbl_per_cat[idx:(idx+n_cols)].index.tolist()

            cols.append(el_cols)

            idx += n_cols

                

    return cols



cols_OCCUPATION_TYPE = _devide_group(col = 'OCCUPATION_TYPE', schedule_div = [1, 7, 9, 1])

cols_OCCUPATION_TYPE
cols_ORGANIZATION_TYPE = _devide_group(col = 'ORGANIZATION_TYPE')

cols_ORGANIZATION_TYPE
def _map_lambda_cats(cols_list, colname, x): 

    cats = list(map(lambda x:colname + '_' + str(x), np.arange(len(cols_list)).tolist()))

    for i in range(len(cols_ORGANIZATION_TYPE)):

        if x in cols_list[i]:

            return cats[i]

        

def _map_cats(cols_list, colname, dataset):                    

    return list(map(lambda x: _map_lambda_cats(cols_list, colname, x), 

                    dataset[colname]))



app_train['ORGANIZATION_TYPE'] = _map_cats(cols_ORGANIZATION_TYPE, 'ORGANIZATION_TYPE', app_train)

pd.Series.unique(app_train['ORGANIZATION_TYPE'])
app_test['ORGANIZATION_TYPE'] = _map_cats(cols_ORGANIZATION_TYPE, 'ORGANIZATION_TYPE', app_test)

pd.Series.unique(app_test['ORGANIZATION_TYPE'])
app_train['OCCUPATION_TYPE'] = _map_cats(cols_OCCUPATION_TYPE, 'OCCUPATION_TYPE', app_train)

app_test['OCCUPATION_TYPE'] = _map_cats(cols_OCCUPATION_TYPE, 'OCCUPATION_TYPE', app_test)
i = 0

plt.figure(figsize = (16, 8))

for col in ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']:

    i += 1

    plt.subplot(2, 1, i)

    _plot_per_categorical(col)
app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)
print('app_train shape: ', app_train.shape)

print('app_test shape: ', app_test.shape)
for fea_name in app_train.columns:

    if fea_name not in app_test.columns:

        print(fea_name)
TARGET = app_train['TARGET']



# Lệnh align theo axis = 1 sẽ lấy những trường xuất hiện đồng thời trong app_train và app_test

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Sau lệnh align biến TARGET bị mất, do đó ta cần gán lại biến này

app_train['TARGET'] = TARGET



print('app_train shape: ', app_train.shape)

print('app_test shape: ', app_test.shape)
app_train.head()
app_train['AMT_INCOME_TOTAL'].describe().plot.box()
app_train['AMT_INCOME_TOTAL'].describe()
def _plot_density(colname):

    plt.figure(figsize = (10, 8))

    sns.kdeplot(app_train[colname][app_train['TARGET'] == 0], label = 'Target = 0')

    sns.kdeplot(app_train[colname][app_train['TARGET'] == 1], label = 'Target = 1')

    plt.xlabel(colname)

    plt.ylabel('Density')

    plt.title('Distribution of %s'%colname)



_plot_density('AMT_INCOME_TOTAL')
def _zoom_3sigma(col, dataset, dataset_apl):

    '''

    col: Tên cột dữ liệu

    dataset: Bảng dữ liệu gốc sử dụng để tính khoảng 3 sigma

    dataset_apl: Bảng dữ liệu mới áp dụng khoảng 3 sigma được lấy từ dataset.

    '''

    xs = dataset[col]

    mu = xs.mean()

    sigma = xs.std()

    low =  mu - 3*sigma

#     low =  0 if low < 0 else low

    high = mu + 3*sigma

    

    def _value(x):

        if x < low: return low

        elif x > high: return high

        else: return x

    xapl = dataset_apl[col]    

    xnew = list(map(lambda x: _value(x), xapl))

    n_low = len([i for i in xnew if i == low])

    n_high = len([i for i in xnew if i == high])

    n = len(xapl)

    print('Percentage of low: {:.2f}{}'.format(100*n_low/n, '%'))

    print('Percentage of high: {:.2f}{}'.format(100*n_high/n, '%'))

    print('Low value: {:.2f}'.format(low))

    print('High value: {:.2f}'.format(high))

    return xnew



# Kiểm tra với biến FLAG_MOBIL

x = _zoom_3sigma('FLAG_MOBIL', app_train, app_train)    
app_train.dtypes.unique()
# Thống kê các giá trị khác biệt trong toàn bộ các biến.

def _count_unique(x):

    return pd.Series.nunique(x)



tbl_dis_val = app_train.apply(_count_unique).sort_values(ascending = False)

tbl_dis_val[tbl_dis_val > 500]
cols_3sigma = tbl_dis_val[tbl_dis_val > 500].index.tolist()

# Loại bỏ biến key là SK_ID_CURR ra khỏi danh sách:

cols_3sigma = cols_3sigma[1:]
# Loại bỏ các outlier bằng 3 sigma

for col in cols_3sigma:

    print(col)

    app_train[col] = _zoom_3sigma(col, app_train, app_train) 

    print('------------------------\n')
for col in cols_3sigma:

    print(col)

    app_test[col] = _zoom_3sigma(col, app_train, app_test) 

    print('------------------------\n')
# Kiểm tra lại biến AMT_INCOME_TOTAL sau khi loại bỏ outlier

app_train['AMT_INCOME_TOTAL'].describe().plot.box()
_plot_density('AMT_INCOME_TOTAL')
from sklearn.preprocessing import MinMaxScaler, Imputer



if 'TARGET' in app_train.columns:

    TARGET = app_train.pop('TARGET')



# Gán train và test vào app_train và app_test; train, test được sử dụng để scale dữ liệu

train = app_train

test = app_test



# Khởi tạo inputer theo phương pháp trung bình

inputer = Imputer(strategy = 'mean')

inputer.fit(train)



# Điền các giá trị NA bằng trung bình

train = inputer.transform(train)

test = inputer.transform(test)



# Khởi tạo scaler theo phương pháp MinMaxScaler trong khoảng [-1, 1]

scaler = MinMaxScaler(feature_range = (-1, 1))

scaler.fit(train)



# Scale dữ liệu trên train và test

train = scaler.transform(train)

test = scaler.transform(test)



# Loại bỏ cột SK_ID_CURR đầu tiên do cột này là key. Khi cần lấy từ app_train và app_test sang

train = train[:, 1:]

test = test[:, 1:]



print('train shape: ', train.shape)

print('test shape: ', test.shape)
app_train['TARGET'] = TARGET

corr_tbl = app_train.corr()

corr_tbl
corr_tbl['TARGET'].sort_values()
pd_train = pd.DataFrame(train, columns = app_train.columns[1:-1])

pd_train['TARGET'] = TARGET

pd_train.head()
corr_tbl_train = pd_train.corr()

corr_tbl_train
corr_tbl_train['TARGET'].sort_values()
# Lấy ra danh sách 15 biến có tương quan lớn nhất tới biến mục tiêu theo trị tuyệt đối.

cols_corr_15 = np.abs(corr_tbl_train['TARGET']).sort_values()[-16:].index.tolist()



# Tính ma trận hệ số tương quan

cols_tbl_15 = pd_train[cols_corr_15].corr()



# Biểu diễn trên biểu đồ heatmap

plt.figure(figsize = (13, 10))

sns.heatmap(cols_tbl_15, cmap = plt.cm.RdYlBu_r, annot = True)
plt.figure(figsize = (20, 5))

for i in range(5):

    _plot_density(cols_corr_15[i])
age_bin = app_train[['TARGET', 'DAYS_BIRTH']]

age_bin['YEAR_OLD'] = -app_train['DAYS_BIRTH']/365



# Phân chia khoảng tuổi thanh 10 khoảng bằng nhau

age_bin['DAYS_BIN'] = pd.cut(age_bin['YEAR_OLD'], bins = np.linspace(20, 70, num = 11))

age_bin.head()
age_bin.groupby(['DAYS_BIN']).mean()
plt.figure(figsize = (8, 6))

age_bin.groupby(['DAYS_BIN']).mean()['TARGET'].plot.barh(color = 'b')

plt.xticks(rotation = '75')

plt.xlabel('Not Repaid rate')
from sklearn.linear_model import LogisticRegression



# Xây dựng mô hình logistic với tham số kiểm soát C = 0.0001

log_reg = LogisticRegression(C = 0.0001)



# Huấn luyện mô hình

log_reg.fit(train, TARGET)
train_pred_prob = log_reg.predict_proba(train)[:, 1]
TARGET.value_counts()/TARGET.value_counts().sum()
from sklearn.metrics import roc_curve, precision_recall_curve

fpr, tpr, thres = precision_recall_curve(TARGET, train_pred_prob)



def _plot_roc_curve(fpr, tpr, thres):

    plt.figure(figsize = (10, 8))

    plt.plot(fpr, tpr, 'b-', label = 'ROC')

    plt.plot([0, 1], [0, 1], '--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')



_plot_roc_curve(fpr, tpr, thres)
from sklearn.metrics import precision_recall_curve

prec, rec, thres = precision_recall_curve(TARGET, train_pred_prob)



def _plot_prec_rec_curve(prec, rec, thres):

    plt.figure(figsize = (10, 8))

    plt.plot(thres, prec[:-1], 'b--', label = 'Precision')

    plt.plot(thres, rec[:-1], 'g-', label = 'Recall')

    plt.xlabel('Threshold')

    plt.ylabel('Probability')

    plt.title('Precsion vs Recall Curve')

    plt.legend()



_plot_prec_rec_curve(prec, rec, thres)
print(cols_corr_15)
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler



# Khởi tạo các preprocessing. Trong đó inputer theo mean, minmax scaler theo khoảng 0, 1 và polynomial features bậc 3.

inputer = Imputer(strategy = 'mean')

minmax_scaler = MinMaxScaler(feature_range = (0, 1))

poly_engineer = PolynomialFeatures(degree = 3)



# Lấy các feature có tương quan lớn nhất đến biến mục tiêu từ app_train và app_test

TARGET = app_train[cols_corr_15[-1]]

train_poly_fea = app_train[cols_corr_15[:-1]]

test_poly_fea = app_test[cols_corr_15[:-1]]



# input dữ liệu missing

inputer = inputer.fit(train_poly_fea)

train_poly_fea = inputer.transform(train_poly_fea)

test_poly_fea = inputer.transform(test_poly_fea)



# Minmax scaler dữ liệu

minmax_scaler = minmax_scaler.fit(train_poly_fea)

train_poly_fea = minmax_scaler.transform(train_poly_fea)

test_poly_fea = minmax_scaler.transform(test_poly_fea)



print('train_poly_fea shape: ', train_poly_fea.shape)

print('test_poly_fea shape: ', test_poly_fea.shape)
# Polynormial features dữ liệu

poly_engineer = poly_engineer.fit(train_poly_fea)

train_poly_fea = poly_engineer.transform(train_poly_fea)

test_poly_fea = poly_engineer.transform(test_poly_fea)



print('train_poly_fea shape: ', train_poly_fea.shape)

print('test_poly_fea shape: ', test_poly_fea.shape)
features = poly_engineer.get_feature_names(input_features = cols_corr_15[:-1])

features[:10]
from sklearn.linear_model import LogisticRegression



# Xây dựng mô hình hồi qui logistic với tham số kiểm soát là C = 0.0001

lg_reg = LogisticRegression(C = 0.0001)

lg_reg.fit(train_poly_fea, TARGET)

lg_reg
# Dự báo xác xuất logistic

train_pred_prob = lg_reg.predict_proba(train_poly_fea)[:, 1]
# Biểu diễn đường roc_curve

from sklearn.metrics import roc_curve

fpr, tpr, thres = roc_curve(TARGET, train_pred_prob)





def _plot_roc_curve(fpr, tpr, thres):

    roc = plt.figure(figsize = (10, 8))

    plt.plot(fpr, tpr, 'b-', label = 'ROC')

    plt.plot([0, 1], [0, 1], '--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    return roc



# Lưu biểu đồ vào p1

p1 = _plot_roc_curve(fpr, tpr, thres)
from sklearn.metrics import auc

#0.7127599620726505

auc(fpr, tpr)
from sklearn.metrics import precision_recall_curve, accuracy_score



prec, rec, thres = precision_recall_curve(TARGET, train_pred_prob)



def _plot_prec_rec_curve(prec, rec, thres):

    plot_pr = plt.figure(figsize = (10, 8))

    plt.plot(thres, prec[:-1], 'b--', label = 'Precision')

    plt.plot(thres, rec[:-1], 'g-', label = 'Recall')

    plt.xlabel('Threshold')

    plt.ylabel('Probability')

    plt.title('Precsion vs Recall Curve')

    return plot_pr



_plot_prec_rec_curve(prec, rec, thres)
# Accuracy

train_pred_label = lg_reg.predict(train_poly_fea)

accuracy_score(TARGET, train_pred_label)
from sklearn.ensemble import RandomForestClassifier



# Khởi tạo rừng cây

rd_classifier = RandomForestClassifier(n_estimators = 100, # Số cây trong rừng cây

                                       max_depth = 5, # Độ sâu của cây

                                       random_state = 123, # Khai báo seed để mô hình không đổi cho các lần chạy sau

                                       verbose = 1, # In log của quá trình huấn luyện

                                       n_jobs = -1 # Sử dụng đa luồng để vận hành mô hình

                                      )

rd_classifier
# Huấn luyện mô hình

rd_classifier.fit(train_poly_fea, TARGET)



# Dự báo trên tập train

train_prob_rd = rd_classifier.predict_proba(train_poly_fea)[:,1]
fpr2, tpr2, thres2 = roc_curve(TARGET, train_prob_rd)

p2 = _plot_roc_curve(fpr, tpr, thres)
plt.figure(figsize = (10, 8))

plt.plot(fpr2, tpr2, 'b-', label = 'Random Forest')

plt.plot(fpr, tpr, 'r-', label = 'Logistic')

plt.plot([0, 1], [0, 1], '--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()
from sklearn.metrics import auc

#0.7300858815170105

auc(fpr2, tpr2)
prec, rec, thres = precision_recall_curve(TARGET, train_pred_prob)

_plot_prec_rec_curve(prec, rec, thres)
# Lấy thông tin về mức độ quan trọng các biến tác động lên biến mục tiêu

feature_importance = rd_classifier.feature_importances_

feature_importance = pd.DataFrame({'importance values': feature_importance})

feature_importance.index = features

feature_importance = feature_importance.sort_values('importance values', ascending = False)

feature_importance[:10]
feature_importance[:10].sort_values('importance values', ascending = True).plot.barh(figsize = (8, 6))

plt.yticks(rotation = 15)

plt.xlabel('Importance values')
feature_importance.iloc[:5, 0].tolist()
from sklearn.metrics import accuracy_score

train_label_rd = rd_classifier.predict(train_poly_fea)

accuracy_score(train_label_rd, TARGET)
np.unique(train_label_rd, return_counts = True)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import gc
lgb_classifier = lgb.LGBMClassifier(n_estimator = 10000, 

                                    objective = 'binary', 

                                    class_weight = 'balanced',

                                    learning_rate = 0.05,

                                    reg_alpha = 0.1,

                                    reg_lambda = 0.1,

                                    subsample = 0.8,

                                    n_job = -1,

                                    random_state = 12

                                   )

lgb_classifier
kfold = KFold(n_splits = 10, shuffle = True, random_state = 12)

valid_scores = []

train_scores = []

count = 0

for train_idx, valid_idx in kfold.split(train_poly_fea):

    count += 1

    # Split train, valid

    train_features, train_labels = train_poly_fea[train_idx], TARGET[train_idx]

    valid_features, valid_labels = train_poly_fea[valid_idx], TARGET[valid_idx]

    lgb_classifier.fit(train_features, train_labels, eval_metric = 'auc',

              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],

              eval_names = ['valid', 'train'], 

              early_stopping_rounds = 100, verbose = 200)

    

    valid_score = lgb_classifier.best_score_['valid']['auc'] 

    train_score = lgb_classifier.best_score_['train']['auc'] 

    

    valid_scores.append(valid_score)

    train_scores.append(train_score)

    

    print('fold time: {}; train score: {}; valid score: {}'.format(count, valid_score, train_score))
# Deep learning với Keras

from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda

from keras.models import Model

from keras.layers.embeddings import Embedding
train.shape
# design neural network

input_els = []

encode_els = []



# Generate a list include many Input layers



for i in range(train.shape[1]):

    # input alway have the shape (*, 1)

    input_els.append(Input(shape = (1,)))

    encode_els.append(input_els[-1])

# encode_els
# concate nate all layers

encode_els = concatenate(encode_els) 



# After completed the input layers, we design the hidden layers

hidden1 = Dense(units = 128, kernel_initializer = 'normal', activation = 'relu')(encode_els)

droplayer1 = Dropout(0.2)(hidden1)

hidden2 = Dense(64, kernel_initializer = 'normal', activation = 'relu')(droplayer1)

droplayer2 = Dropout(0.2)(hidden2)

outputlayer = Dense(1, kernel_initializer = 'normal', activation = 'sigmoid')(droplayer2)



classifier = Model(input = input_els, outputs = [outputlayer])
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()
# split train/valid

from sklearn.model_selection import KFold

count = 0

kfold = KFold(n_splits = 10, shuffle = True, random_state = 12)

valid_scores = []

train_scores = []

for train_idx, valid_idx in kfold.split(train_poly_fea):

    while count < 1:

        count += 1

        # Split train, valid

        train_features, train_labels = train[train_idx], TARGET[train_idx]

        valid_features, valid_labels = train[valid_idx], TARGET[valid_idx]

        classifier.fit(

            [train_features[:, i] for i in range(train.shape[1])], #lấy list toàn bộ các cột

            train_labels,

            epochs=1,

            batch_size=128,

            shuffle=True,

            validation_data=([valid_features[:, i] for i in range(train.shape[1])], valid_labels) 

        )
# DỰ báo trên tập train.

train_prob_nn = classifier.predict([train[:, i] for i in range(train.shape[1])])

train_prob_nn
# np.save('train_prob_nn.npy',train_prob_nn)



fpr4, tpr4, thres4 = roc_curve(TARGET, train_prob_nn)

_plot_roc_curve(fpr4, tpr4, thres4)
from sklearn.metrics import auc

auc(fpr4, tpr4)
prec, rec, thres = precision_recall_curve(TARGET, train_prob_nn)

_plot_prec_rec_curve(prec, rec, thres)
# Save data

# np.save('train1.npy', train)

# np.save('test1.npy', test)

# np.save('TARGET.npy', TARGET)

# import numpy as np

# train = np.load('train.npy')

# print(train.shape)

# test = np.load('test.npy')

# print(test.shape)