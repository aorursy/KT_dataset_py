%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import kurtosis

from scipy.stats import skew



sns.set()
#Preprocessing and train_test_split

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale, StandardScaler, LabelEncoder, Imputer



#Classifiers

#from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import catboost as ctb



#Cross validation

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

train = pd.read_csv("../input/X_train.csv")

test = pd.read_csv("../input/X_test.csv")

target = pd.read_csv("../input/y_train.csv")

sub = pd.read_csv("../input/sample_submission.csv")
train.info()
test.info()
sns.countplot(y = 'surface', data = target)
corr = train.corr()



_ , ax = plt.subplots(figsize =(14, 10))

hm = sns.heatmap(corr, ax= ax, annot= True,linewidths=0.3)
plt.figure(figsize=(26, 16))

for i, col in enumerate(train.columns[3:]):

    plt.subplot(3, 4, i + 1)

    plt.hist(train[col], bins=80)

    plt.hist(test[col], bins=80)

    plt.title(col)
def plotseries(df,series_id,color='Blue'):

    plt.figure(figsize=(26, 16))

    for i, col in enumerate(df.columns[3:]):

        plt.subplot(3, 4, i + 1)

        plt.plot(train.loc[train['series_id'] == series_id, col])

        plt.title(col)
plotseries(train,3)
# from @theoviel at https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

from numpy.fft import rfft, irfft, rfftfreq



def filter_signal(signal, threshold=1e3):

    fourier = rfft(signal)

    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)
# denoise train and test angular_velocity and linear_acceleration data



train_denoised = train.copy()

test_denoised = test.copy()



# train

for col in train.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        # Apply filter_signal function to the data in each series

        denoised_data = train.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))

        

        # Assign the denoised data back to X_train

        list_denoised_data = []

        for arr in denoised_data:

            for val in arr:

                list_denoised_data.append(val)

                

        train_denoised[col] = list_denoised_data

        

# test

for col in test.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        # Apply filter_signal function to the data in each series

        denoised_data = test.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))

        

        # Assign the denoised data back to X_test

        list_denoised_data = []

        for arr in denoised_data:

            for val in arr:

                list_denoised_data.append(val)

                

        test_denoised[col] = list_denoised_data
plotseries(train_denoised,3)
plt.figure(figsize=(20, 20))

plt.subplot(2,1,1)

plt.plot(train.angular_velocity_X[120:250], label="original");

plt.plot(train_denoised.angular_velocity_X[120:250], label="denoised");

plt.title('linear_acceleration_X')

plt.subplot(2,1,2)

plt.plot(train.angular_velocity_Y[120:250], label="original");

plt.plot(train_denoised.angular_velocity_Y[120:250], label="denoised");

plt.title('linear_acceleration_Y')

plt.legend()

plt.show()

for col in train.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        train_denoised[col + '_noise'] = np.abs(train[col] - train_denoised[col])

        

for col in test.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        test_denoised[col + '_noise'] = np.abs(test[col] - test_denoised[col])

        

"""I've used "https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

I do a coordinate transformation from quaternion to Euler" to convert quaternions to euler angles""" 



def quaternion_to_euler(x, y, z, w):



        import math

        t0 = +2.0 * (w * x + y * z)

        t1 = +1.0 - 2.0 * (x * x + y * y)

        X = math.atan2(t0, t1)



        t2 = +2.0 * (w * y - z * x)

        t2 = +1.0 if t2 > +1.0 else t2

        t2 = -1.0 if t2 < -1.0 else t2

        Y = math.asin(t2)



        t3 = +2.0 * (w * z + x * y)

        t4 = +1.0 - 2.0 * (y * y + z * z)

        Z = math.atan2(t3, t4)



        return X, Y, Z



def _kurtosis(x):

    return kurtosis(x)



def CPT5(x):

    den = len(x)*np.exp(np.std(x))

    return sum(np.exp(x))/den



def skewness(x):

    return skew(x)



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    xn_i1 = x[0:len(x)-2]  # xn-1

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def mean_abs(x):

    return sum(abs(x))/len(x)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))

def feat(df):

      

    df['total_angular_velocity'] = (df['angular_velocity_X']**2+df['angular_velocity_Y']**2+df['angular_velocity_Z']**2)**0.5

    df['total_linear_acceleration'] = (df['linear_acceleration_X']**2+df['linear_acceleration_Y']**2+df['linear_acceleration_Z']**2)**0.5

    df['acc_vs_vel'] = df['total_linear_acceleration']/df['total_angular_velocity']

    

    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()

    

    xlist, ylist, zlist = [], [], []

    

    for i in range(len(x)):

        x2, y2, z2 = quaternion_to_euler(x[i],y[i],z[i],w[i])

        xlist.append(x2)

        ylist.append(y2)

        zlist.append(z2)

    

    df['euler_X'] = xlist

    df['euler_Y'] = ylist

    df['euler_Z'] = zlist

    

    df['euler_orientation'] = (df['euler_X']**2 + df['euler_Y']**2 + df['euler_Z']**2)**0.5

    

    def mean_diff(x):

        return np.mean(np.abs(np.diff(x)))

  

    def mean_diff_diff(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    df2 = pd.DataFrame()    

    

    for col in df.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        if 'noise' in col:

            df2[col + '_mean'] = df.groupby(['series_id'])[col].mean()

        

        else:

            df2[col + '_mean'] = df.groupby(['series_id'])[col].mean()

            df2[col + '_min'] = df.groupby(['series_id'])[col].min()

            df2[col + '_max'] = df.groupby(['series_id'])[col].max()

            df2[col + '_std'] = df.groupby(['series_id'])[col].std()

            df2[col + '_range'] = df2[col + '_max'] - df2[col + '_min']

            df2[col + '_max_min_ratio'] = df.groupby(['series_id'])[col].max()/df.groupby(['series_id'])[col].min()

        

            df2[col + '_mean_abs_difference'] =df.groupby(['series_id'])[col].apply(mean_diff)

            df2[col + '_mean_diff_of_abs_diff'] = df.groupby('series_id')[col].apply(mean_diff_diff)

        

        

            df2[col + '_CPT5'] = df.groupby(['series_id'])[col].apply(CPT5) 

            df2[col + '_SSC'] = df.groupby(['series_id'])[col].apply(SSC) 

            df2[col + '_skewness'] = df.groupby(['series_id'])[col].apply(skewness)

            df2[col + '_wave_lenght'] = df.groupby(['series_id'])[col].apply(wave_length)

            df2[col + '_norm_entropy'] = df.groupby(['series_id'])[col].apply(norm_entropy)

            df2[col + '_SRAV'] = df.groupby(['series_id'])[col].apply(SRAV)

            df2[col + '_kurtosis'] = df.groupby(['series_id'])[col].apply(_kurtosis) 

            df2[col + '_mean_abs'] = df.groupby(['series_id'])[col].apply(mean_abs) 

            df2[col + '_zero_crossing'] = df.groupby(['series_id'])[col].apply(zero_crossing) 

        

        

        

    return df2

            
train_fe = feat(train_denoised)

test_fe = feat(test_denoised)
train_fe.fillna(0,inplace=True)

test_fe.fillna(0,inplace=True)

train_fe.replace(-np.inf,0,inplace=True)

train_fe.replace(np.inf,0,inplace=True)

test_fe.replace(-np.inf,0,inplace=True)

test_fe.replace(np.inf,0,inplace=True)
sc = StandardScaler()



X_train = pd.DataFrame(sc.fit_transform(train_fe))

X_test = pd.DataFrame(sc.transform(test_fe))
le = LabelEncoder()

target['surface'] = le.fit_transform(target['surface'])
# https://www.kaggle.com/artgor/where-do-the-robots-drive

import itertools



def plot_confusion_matrix(truth, pred, classes, normalize=False, title='Confusion Matrix',cmap=plt.cm.Blues):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
cv = 3

eval_list= []

pred_list = []

meas_list = []



for i in range (0,cv):



    folds = StratifiedKFold(n_splits=8, shuffle=True, random_state=20)

    predicted_rf = np.zeros((X_test.shape[0],9))

    measured_rf= np.zeros((X_train.shape[0],9))

    score = 0

           

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,target['surface'].values)):

        

        X_tr = X_train.iloc[trn_idx]

        y_tr = target['surface'][trn_idx]

    

        X_valid = X_train.iloc[val_idx]

        y_valid = target['surface'][val_idx]

        

        rfc = RandomForestClassifier(n_estimators=100, min_samples_leaf = 1,max_depth= None,n_jobs=-1,random_state=20)

        rfc.fit(X_tr,y_tr)

        measured_rf[val_idx] = rfc.predict_proba(X_valid)

        y_pred = rfc.predict_proba(X_test)

        predicted_rf += y_pred

        score += rfc.score(X_valid,y_valid)

        

        

        print("Fold: {}, RF Score: {}".format(fold,rfc.score(X_valid,y_valid)))

        

    predicted_rf /= folds.n_splits    

    

    meas_list.append(measured_rf)

    pred_list.append(predicted_rf)

    eval_list.append(score/folds.n_splits) 
plot_confusion_matrix(target['surface'], measured_rf.argmax(1), le.classes_,title ='Confusion Matrix for Random Forest')
indx = eval_list.index(max(eval_list))

                      

print(indx, max(eval_list))

pred_rf = pred_list[indx]

meas_rf = meas_list[indx]
sub['surface'] = le.inverse_transform(pred_rf.argmax(1))

#sub.to_csv('submission_rf_fft.csv', index=False)

sub.head()
meas_rf.shape
folds = StratifiedKFold(n_splits=8, shuffle=True, random_state=20)

predicted_et = np.zeros((X_test.shape[0],9))

measured_et = np.zeros((X_train.shape[0],9))

score_et = 0

           

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,target['surface'].values)):

    

    X_tr = X_train.iloc[trn_idx]

    y_tr = target['surface'][trn_idx]

    

    X_valid = X_train.iloc[val_idx]

    y_valid = target['surface'][val_idx]

    

    

    etc = ExtraTreesClassifier(n_estimators=200,max_depth=12,min_samples_leaf=2,n_jobs=-1,random_state=20)

    etc.fit(X_tr,y_tr)

    measured_et[val_idx] = etc.predict_proba(X_valid)

    et_pred = etc.predict_proba(X_test)

    predicted_et += et_pred

    score_et += etc.score(X_valid,y_valid)/folds.n_splits

        

    print("Fold: {}, ET Score: {}".format(fold,etc.score(X_valid,y_valid)))

        

predicted_et /= folds.n_splits   
plot_confusion_matrix(target['surface'], measured_et.argmax(1), le.classes_,title ='Confusion Matrix for Extra Trees',cmap=plt.cm.YlOrRd)
measured_et.shape
params_lgb = {'num_leaves': 123,

          'min_data_in_leaf': 12,

          'objective': 'multiclass',

          'max_depth': 24,

          'learning_rate': 0.0468035094972387,

          "bagging_freq": 5,

          "bagging_fraction": 0.89330183551903,

          "bagging_seed": 11,

          "verbosity": 0,

          'reg_alpha': 0.9498109326932401,

          'reg_lambda': 0.805849096054620,

          "num_class": 9,

          'nthread': -1,

          'min_split_gain': 0.0099132272405649,

          'subsample': 0.90273588307031,

         }
def fit_predict_lgb(X, X_tst, y, param=None):

    

    lgb_predicted = np.zeros((X_tst.shape[0],9))

    lgb_measured= np.zeros((X.shape[0],9))

    lgb_acc = 0

    

    feature_importance = pd.DataFrame()

    

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X,y)):

                

        lgbm = lgb.LGBMClassifier(**param, n_estimators = 20000, verbose = 0, n_jobs = -1,random_state=20,

                                  early_stopping_rounds=100)

        lgbm.fit(X.iloc[trn_idx],y[trn_idx], eval_set=[(X.iloc[trn_idx],y[trn_idx]), (X.iloc[val_idx],y[val_idx])], 

                 eval_metric='multi_logloss')

        lgb_measured[val_idx] = lgbm.predict_proba(X.iloc[val_idx])

        y_pred = lgbm.predict_proba(X_tst)/folds.n_splits

        lgb_predicted +=y_pred

        lgb_acc += accuracy_score(y[val_idx], lgb_measured[val_idx].argmax(1))/folds.n_splits

    

        print("Fold: {} LGB score: {}".format(fold,accuracy_score(y[val_idx], lgb_measured[val_idx].argmax(1))))

        

       

    return lgb_measured, lgb_predicted, lgb_acc

measured_lgb, predicted_lgb, score_lgb = fit_predict_lgb(X_train,X_test,target['surface'], param=params_lgb)
plot_confusion_matrix(target['surface'], measured_lgb.argmax(1),le.classes_, title ='Confusion Matrix for Training Set with LightGBM',)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)

svc_predicted = np.zeros((X_test.shape[0],9))

svc_measured= np.zeros((X_train.shape[0]))

svc_score = 0

svc_acc = []

svc = SVC(random_state=314,C=1, decision_function_shape= 'ovo', gamma= 'auto',max_iter= -1, probability=True)



for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,target['surface'].values)):



    X_tr = X_train.iloc[trn_idx]

    y_tr = target['surface'][trn_idx]

        

    X_valid= X_train.iloc[val_idx]

    y_valid = target['surface'][val_idx]

    

    

    svc.fit(X_tr,y_tr)

    svc_measured[val_idx] = svc.predict(X_valid)

    svc_score += svc.score(X_valid,y_valid)/folds.n_splits

    

    svc_pred = svc.predict_proba(X_test)

    svc_predicted += svc_pred

    

    svc_acc.append(svc_score)

    

    print("Fold: {}, SVC Score: {}".format(fold,svc.score(X_valid,y_valid)))

    

svc_predicted /= folds.n_splits
plot_confusion_matrix(target['surface'],svc_measured,le.classes_, title ='Confusion Matrix for Training Set with SVC',cmap=plt.cm.YlOrBr)
x_train = np.concatenate((measured_et, meas_rf, measured_lgb), axis=1)

x_test = np.concatenate((predicted_et, pred_rf, predicted_lgb), axis=1)



print("{},{}".format(x_train.shape, x_test.shape))
logistic_regression = LogisticRegression()

logistic_regression.fit(x_train,target['surface'])



logreg_pred = logistic_regression.predict_proba(x_test)
sub['surface'] = le.inverse_transform(logreg_pred.argmax(1))

#sub.to_csv('submission_lr_stack.csv', index=False)

sub.head()