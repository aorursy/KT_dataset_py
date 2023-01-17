# импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import math
import xgboost # conda install py-xgboost
import time
from tqdm import tqdm
fd_001_train = pd.read_csv("/kaggle/input/nasa-cmaps/CMaps/train_FD001.txt",sep=" ",header=None)
fd_001_test = pd.read_csv("/kaggle/input/nasa-cmaps/CMaps/test_FD001.txt",sep=" ",header=None)
fd_001_train.describe()
fd_001_train.drop(columns=[26,27],inplace=True)
fd_001_test.drop(columns=[26,27],inplace=True)
columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
fd_001_train.columns = columns
fd_001_test.columns = columns
# первичное знакомство с данными
#initial acquaintance with data
fd_001_train.describe()
# удалим колонки с константными значениями, как не несущими информацию о состоянии агрегата
#delete columns with constant values ​​that do not carry information about the state of the unit
fd_001_train.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)

# функция для подготовки тренировочных данных и формирования колонки RUL  с информациеей об оставшихся
#до поломки циклах
#function for preparing training data and forming a RUL column with information about the remaining
# before breaking cycles
def prepare_train_data(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]

df = prepare_train_data(fd_001_train)
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()
# функция ошибки для соревновательных данных
#Error Function for Competitive Data
def score(y_true,y_pred,a1=10,a2=13):
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0 :
            score += math.exp(i/a2) - 1   
        else:
            score += math.exp(- i/a1) - 1
    return score
    
   
def score_func(y_true,y_pred):
    print(f' соревновательный счет {round(score(y_true,y_pred),2)}')
    print(f' mean absolute error {round(mean_absolute_error(y_true,y_pred),2)}')
    print(f' root mean squared error {round(mean_squared_error(y_true,y_pred),2) ** 0.5}')
    print(f' R2 score {round(r2_score(y_true,y_pred),2)}')
    return
    
train_df = df.drop(columns = ['unit_number','setting_1','setting_2','P15','NRc'])
# функция для создания и тренировки моделей по алгоритмам "Random forest" и "XGBoost"
#function for creating and training models using the "Random forest" and "XGBoost" algorithms
def train_models(data,model = 'FOREST'):
    X = data.iloc[:,:14].to_numpy() 
    Y = data.iloc[:,14:].to_numpy()
    Y = np.ravel(Y)
    if model == 'FOREST':
         # параметры для моделей подобраны в подобном цикле, с введением в функцию дополнительного параметра param:
         #  parameters for models are selected in a similar cycle, with the introduction 
         # of an additional param parameter into the function:
         #for i in range(1,11):
         #     xgb = train_models(train_df,param=i,model="XGB",)
         #     y_xgb_i_pred = xgb.predict(X_001_test)
         #     print(f'param = {i}')
         #     score_func(y_true,y_xgb_i_pred)
        model = RandomForestRegressor(n_estimators=70, max_features=7, max_depth=5, n_jobs=-1, random_state=1)
        model.fit(X,Y)
        return model
    elif model == 'XGB':
        model = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.018, gamma=0, subsample=0.8,
                           colsample_bytree=0.5, max_depth=3,silent=True)
        model.fit(X,Y)
        return model
    return
    
# функция для совместного отображения реальных и предсказанных значений
#function for joint display of real and predicted values

def plot_result(y_true,y_pred):
    rcParams['figure.figsize'] = 12,10
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    plt.ylabel('RUL')
    plt.xlabel('training samples')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.title('Сравнение реальных и предсказанных значений ')
    plt.show()
    return
fd_001_test.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)
test_max = fd_001_test.groupby('unit_number')['time_in_cycles'].max().reset_index()
test_max.columns = ['unit_number','max']
fd_001_test = fd_001_test.merge(test_max, on=['unit_number'], how='left')
test = fd_001_test[fd_001_test['time_in_cycles'] == fd_001_test['max']].reset_index()
test.drop(columns=['index','max','unit_number','setting_1','setting_2','P15','NRc'],inplace = True)
X_001_test = test.to_numpy()
X_001_test.shape
model_1 = train_models(train_df)
y_pred = model_1.predict(X_001_test)
RUL = pd.read_csv("/kaggle/input/nasa-cmaps/CMaps/RUL_FD001.txt",sep=" ",header=None)
y_true = RUL[0].to_numpy()
score_func(y_true, y_pred)
plot_result(y_true,y_pred)
# для отбрасывания значений в тренировочном массиве используется параметр factor в 
# функции prepare_train_data, в test_data находятся подготовленные к распознаванию сэмплы, в первом столбце которых 
#  - значение времени в циклах, для которого производится предсказание RUL

# to discard values in the training array, use the factor parameter in
# prepare_train_data functions, in test_data are samples prepared for recognition, in the first column of which
# - value of time in cycles for which RUL is predicted
def single_train(test_data,train_data,algorithm):
    y_single_pred = []
    for sample in tqdm(test_data):
        time.sleep(0.01)
        single_train_df = prepare_train_data(train_data, factor = sample[0])
        single_train_df.drop(columns = ['unit_number','setting_1','setting_2','P15','NRc'],inplace = True)
        model = train_models(single_train_df,algorithm)
        y_p = model.predict(sample.reshape(1,-1))[0]
        y_single_pred.append(y_p)
    y_single_pred = np.array(y_single_pred)
    return y_single_pred
y_single_pred = single_train(X_001_test,fd_001_train,'FOREST')
plot_result(y_true,y_single_pred)
score_func(y_true, y_single_pred)
def prepare_test_data(fd_001_test,n=0):
    test = fd_001_test[fd_001_test['time_in_cycles'] == fd_001_test['max'] - n].reset_index()
    test.drop(columns=['index','max','unit_number','setting_1','setting_2','P15','NRc'],inplace = True)
    X_return = test.to_numpy()
    return X_return
N=5
y_n_pred = y_single_pred
for i in range(1,N):
    X_001_test = prepare_test_data(fd_001_test,i)
    y_single_i_pred = single_train(X_001_test,fd_001_train,'FOREST')    
    y_n_pred = np.vstack((y_n_pred,y_single_i_pred))  
y_multi_pred = np.mean(y_n_pred,axis = 0)
score_func(y_true,y_multi_pred)
plot_result(y_true,y_multi_pred)
N=10
#Чтобы повторно не вычислять средний результат для 5 предсказаний, сохраненное значение y_multi_pred 
# заносится в y_n_pred, далее считаются предсказания для 5,6.... строки от последней для данного двигателя

# In order not to recalculate the average result for 5 predictions, the stored value y_multi_pred
# is entered in y_n_pred, then the predictions for 5,6,7 .... lines from the last for the given engine
y_n_pred = y_multi_pred
for i in range(5,N):
    X_001_test = prepare_test_data(fd_001_test,i)
    y_single_i_pred = single_train(X_001_test,fd_001_train,'FOREST')    
    y_n_pred = np.vstack((y_n_pred,y_single_i_pred))  
y_multi_pred_10 = np.mean(y_n_pred,axis = 0)
score_func(y_true,y_multi_pred_10)
plot_result(y_true,y_multi_pred_10)
xgb = train_models(train_df,model="XGB")
y_xgb_pred = xgb.predict(X_001_test)
score_func(y_true,y_xgb_pred)
plot_result(y_true,y_xgb_pred)
y_single_xgb_pred = single_train(X_001_test,fd_001_train,'XGB')
score_func(y_true,y_single_xgb_pred)
plot_result(y_true,y_single_xgb_pred)
N=5
y_n_pred = y_single_xgb_pred
for i in range(1,N):
    X_001_test = prepare_test_data(fd_001_test,i)
    y_single_i_pred = single_train(X_001_test,fd_001_train,'XGB')    
    y_n_pred = np.vstack((y_n_pred,y_single_i_pred)) 
y_5_pred_xgb = np.mean(y_n_pred,axis = 0)
score_func(y_true,y_5_pred_xgb)
plot_result(y_true,y_5_pred_xgb)
compare = pd.DataFrame(list(zip(y_true, y_pred, y_single_pred,y_multi_pred,y_multi_pred_10,y_xgb_pred,y_single_xgb_pred)), 
               columns =['True','Forest_Predicted','Forest_Single_predicted','multi_5','multi_10'
                         ,'XGBoost','XGBoost_single']) 
compare['unit_number'] = compare.index + 1
compare['Predicted_error'] = compare['True'] - compare['Forest_Predicted']
compare['Single_pred_error'] = compare['True'] - compare['Forest_Single_predicted']
compare['multi_5_error'] = compare['True'] - compare['multi_5']
compare['multi_10_error'] = compare['True'] - compare['multi_10']
compare['xgb_error'] = compare['True'] - compare['XGBoost']
compare['xgb_single_error'] = compare['True'] - compare['XGBoost_single']
ax1 = compare.plot(subplots=True, sharex=True, figsize=(20,20))
# формирование целевой переменной label, TTF - время до поломки
TTF = 10
train_df['label'] = np.where(train_df['RUL'] <= TTF, 1, 0 )
train_df.head()
sns.scatterplot(x="Nc", y="T50", hue="label", data=train_df)
plt.title('Диаграмма рассеивания T50 от Nc')
#исключаем свойство RUL и формируем массив признаков и целевой переменной
# exclude the RUL property and form an array of attributes and the target variable
X_class = train_df.iloc[:,:14].to_numpy() 
Y_class = train_df.iloc[:,15:].to_numpy()
Y_class = np.ravel(Y_class)
# Балансировка классов для улучшения работы классификатора

# Class balancing to improve classifier performance
from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler
ros = RandomOverSampler(random_state=0)
ros.fit(X_class, Y_class)
X_resampled, y_resampled = ros.fit_sample(X_class, Y_class)
print('Количество элементов до операции:', len(X_class))
print('Количество элементов после операции:', len(X_resampled))
# Здесь делим данные на обучающую выборку и тестовую , test_size = 0.2 задает долю тестовой выборки = 20%

#
# Here we divide the data into the training sample and the test one, 
#test_size = 0.2 sets the proportion of the test sample = 20%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size = 0.2,random_state = 3)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
forest = RandomForestClassifier(n_estimators=70 ,max_depth = 8, random_state=193)
forest.fit(X_train,y_train)
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def classificator_score(y_,y_p):
    print(f' accuracy score {round(accuracy_score(y_, y_p),2)}')
    print(f' precision score {round(precision_score(y_, y_p),2)}')
    print(f' recall score {round(recall_score(y_, y_p),2)}')
    print(f' F1 score {round(f1_score(y_, y_p),2)}')
    return
classificator_score(y_test,forest.predict(X_test))
y_xgb_pred = model_xgb.predict(X_001_test)
classificator_score(y_test,model_xgb.predict(X_test))
test.head()
X_001_test = test.to_numpy()
# предсказание для X_001_test, время до поломки = TTF =10
predicted = pd.DataFrame()
predicted ['forest'] =  forest.predict(X_001_test)
predicted['XGB'] = y_xgb_pred
predicted['RUL']=RUL[0]
predicted['true_label'] = np.where(y_true <= TTF, 1, 0 )
predicted['unit_number'] = predicted.index + 1
predicted.head()
# истинные значения TTF <= 10
predicted[predicted['true_label'] == 1]
# двигатели, для которых алгоритм классификации RandomForest дал неверные предсказания
# engines for which the RandomForest classification algorithm gave incorrect predictions
predicted[predicted['true_label'] != predicted['forest']]
# двигатели, для которых алгоритм классификации XGBoost дал неверные предсказания
# engines for which the XGBoost classification algorithm gave incorrect predictions
predicted[predicted['true_label'] != predicted['XGB']]
y_true_class = np.where(y_true <= TTF, 1, 0 )
y_pred_class = predicted['forest'].tolist()
def expected_profit(y_true,y_pred):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(y_true)):
        if (y_true[i] != y_pred[i]) & (y_pred[i] == 1):
            FP += 1
        elif (y_true[i] != y_pred[i]) & (y_pred[i] == 0):
            FN += 1
        elif (y_true[i] == y_pred[i]) & (y_pred[i] == 0):
            TN += 1
        else:
            TP += 1
    print(f'TP ={TP}, TN = {TN}, FP = {FP}, FN = {FN}')
    print (f'ожидаемая прибыль {(300 * TP - 200 * FN - 100 * FP) * 1000}')
    return 
        
expected_profit(y_true_class,y_pred_class)
expected_profit(y_true_class,y_xgb_pred)


