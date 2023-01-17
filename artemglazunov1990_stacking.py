!pip install pytorch-tabnet
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
import numpy as np
from sklearn.metrics import f1_score,classification_report
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.model_selection import cross_val_predict

import itertools
from tqdm import tqdm_notebook
import gc

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('/kaggle/input/mf-accelerator/contest_train.csv')
target = data.TARGET
data = data.fillna(0)
features = data.drop(columns=["TARGET","ID"])
features.head()
data_test = pd.read_csv('/kaggle/input/mf-accelerator/contest_test.csv')
features_test = data_test.copy().drop(columns=["ID"])
features_test.head()
features_train,features_val,labels_train,labels_val = train_test_split(features,target, test_size = 0.3,\
                                                                   shuffle=True,random_state=1,\
                                                                   stratify = target)
features_train.head()
class Stacking(BaseEstimator, ClassifierMixin):  
    """Стекинг моделей 
    на основе материалов А. Дьяконова
    """
    

    def __init__(self, models, metamodel,merge=False):
        """
        Инициализация
        models - базовые модели для стекинга
        metamodel - метамодель
        """
        self.models = models
        self.metamodel = metamodel
        self.n = len(models)
        self.meta = None
        self.merge = merge


    def fit(self, X, y=None, p=0.25, random_state=0):
        """
        Обучение стекинга

        p - в каком отношении делить на выборку 
        на подвыборки для базовых и метаалгоритма
        random_state - для воспроизводимости
        merge - слить полученные признаки и исходные при работе метаалгоритма    
        """
        # разбиение на обучение моделей и метамодели
        base, meta, y_base, y_meta = train_test_split(X, y, test_size=p, random_state=random_state,stratify = y)
            
        # заполнение матрицы для обучения метамодели
        self.meta = np.zeros((meta.shape[0], self.n))
        for t, base_model in enumerate(self.models):
            base_model.fit(np.array(base), np.array(y_base))
                
            self.meta[:, t] = base_model.predict(meta).reshape((1,-1))#reshape для работы нейросетей и катбуста
            print(f"Ok {t}")

        # обучение метамодели
        if self.merge:#если обучаем метамодель на объединенной выборке с исходными признаками и новыми
            data_meta_ext = np.concatenate((meta,self.meta),axis=1)
            self.metamodel.fit(data_meta_ext, y_meta)
        else:
            self.metamodel.fit(self.meta, y_meta)
        print("------")
        print("Ok")


        return self
    


    def predict(self, X, y=None):
        """
        Предсказание стекингом
        """
        # заполение матрицы для мета-классификатора
        X_meta = np.zeros((X.shape[0], self.n))
        
        print("------")
        print("Prediction")  



        for t, base_model in enumerate(self.models):
            
            X_meta[:, t] = base_model.predict(X).reshape((1,-1))
          
            print(f"Ok{t}")  
          

        if self.merge:#если объединенная выборка для обучения метамодели
            data_meta_test = np.concatenate((X,X_meta),axis=1)
            res = self.metamodel.predict(data_meta_test)

        else:
            res = self.metamodel.predict(X_meta)
        
        return (res)
class NNWrapper(BaseEstimator, ClassifierMixin):  
    """Обертка для нейросетей для совместимости со стекингом"""

    def __init__(self, model,scaler,X_valid=None,
                 y_valid=None,max_epochs=10, patience=150):
        
        self.model = model
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.max_epochs = max_epochs
        self.patience = patience
        self.scaler = scaler

    def fit(self, X, y=None):
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_train=np.array(X_sc), y_train=np.array(y), X_valid=self.X_valid,y_valid=self.y_valid,
                  max_epochs=self.max_epochs, patience=self.patience) 

        return self

    def predict(self, X_test, y=None):
        X_test_sc = self.scaler.transform(np.array(X_test))
        prediction = self.model.predict(np.array(X_test_sc)).reshape((1,-1))#Ключевая строка
        
        
        return prediction

class LMWrapper(BaseEstimator, ClassifierMixin):  
    """Обертка для линейных и иных простых моделей 
    для использования масштабирования"""

    def __init__(self, model,scaler):
        
        self.model = model
        self.scaler = scaler
        
    def fit(self, X, y=None):
        
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, y) 

        return self

    def predict(self, X_test, y=None):
        X_test_sc = self.scaler.transform(X_test)
        prediction = self.model.predict(X_test_sc)
        
        
        return prediction
ls0 = Lasso(alpha=0.01,random_state=0)
knn1 = LMWrapper(KNeighborsRegressor(n_neighbors=3,),StandardScaler())
knn2 = LMWrapper(KNeighborsRegressor(n_neighbors=10),StandardScaler())
rf2 = RandomForestRegressor(n_estimators=100, max_depth=10,random_state=100)
gbm1 = lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.05, max_depth=7, n_estimators=200, nthread=-1,
                        objective='regression',random_state=0) 
cb_reg1 = CatBoostRegressor(task_type='GPU',random_state=0,
                             iterations=1000,verbose=False)
reg_tabnet = NNWrapper(TabNetRegressor(verbose=0,seed=0),StandardScaler(),max_epochs=100, patience=150,
                       X_valid=np.array(features_val), y_valid = np.array(labels_val).reshape(-1, 1))
clf_cb_1 = CatBoostClassifier(task_type='GPU',random_state=0, loss_function='MultiClass',
                                auto_class_weights="Balanced",iterations=1000,verbose=False)
clf_lr = LMWrapper(LogisticRegression(multi_class="multinomial",class_weight="balanced",
                                        C=1e-1,max_iter=300,random_state=0),StandardScaler())
clf_nb1 =  BernoulliNB(alpha=1,binarize=0.3)
%%time
warnings.filterwarnings("ignore")
models = [ls0,knn1, knn2,rf2,gbm1,cb_reg1,
           reg_tabnet,clf_cb_1,clf_nb1,clf_lr]

meta_model = CatBoostClassifier(task_type='GPU',random_state=0, loss_function='MultiClassOneVsAll',
                                auto_class_weights="Balanced",iterations=1000,verbose=False)

stack = Stacking(models, meta_model,merge=True)
stack.fit(features_train,np.array(labels_train).reshape(-1, 1),p=0.2,random_state=0)# сложно из-за нейросети
preds = stack.predict(features_val)
print(classification_report(labels_val,preds))
print("------")
print(f"Macro f1 score: {f1_score(labels_val,preds,average='macro')}")
