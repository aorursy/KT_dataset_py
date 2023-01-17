import os

import typing as tp



import numpy as np

import pandas as pd



import cv2

import pydicom

from PIL import Image



import sklearn

from sklearn import model_selection

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error



import lightgbm as lgb



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
SEED = 42



if os.path.exists('/kaggle/input'):

    DATA_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

else:

    DATA_DIR = '../data/raw/'
def preprocessing(data: pd.DataFrame, is_test: bool = True) -> pd.DataFrame:

    # Create Common Features.

    features = pd.DataFrame()

    for patient, u_data in data.groupby('Patient'):

        feature = pd.DataFrame({

            'current_FVC': u_data['FVC'],

            'current_Percent': u_data['Percent'],

            'current_Age': u_data['Age'],

            'current_Week': u_data['Weeks'],

            'Patient': u_data['Patient'],

            'Sex': u_data['Sex'].map({'Female': 0, 'Male': 1}),

            'SmokingStatus': u_data['SmokingStatus'].map({'Currently smokes': 0, 'Never smoked': 1, 'Ex-smoker': 2}),

        })

        features = pd.concat([features, feature])

    # Create Label Data.

    if is_test:

        label = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), usecols=['Patient_Week'])

        label['Patient'] = label['Patient_Week'].apply(lambda x: x.split('_')[0])

        label['pred_Weeks'] = label['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

        label['FVC'] = np.nan



        dst_data = pd.merge(label, features, how='left', on='Patient')

    else:

        label = pd.DataFrame({

            'Patient_Week': data['Patient'].astype(str) + '_' + data['Weeks'].astype(str),

            'Patient': data['Patient'],

            'pred_Weeks': data['Weeks'],

            'FVC': data['FVC']

        })



        dst_data = pd.merge(label, features, how='outer', on='Patient')

        dst_data = dst_data.query('pred_Weeks!=current_Week')

        

    dst_data['passed_Weeks'] = dst_data['current_Week'] - dst_data['pred_Weeks']

    return dst_data



train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

train = preprocessing(train, is_test=False)
print(train.shape)

train.head()
class OSICLossForLGBM:

    """

    Custom Loss for LightGBM.

    

    * Objective: return grad & hess of NLL of gaussian

    * Evaluation: return competition metric

    """

    

    def __init__(self, epsilon: float=1) -> None:

        """Initialize."""

        self.name = "osic_loss"

        self.n_class = 2  # FVC & Confidence

        self.epsilon = epsilon

    

    def __call__(self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None) -> float:

        """Calc loss."""

        sigma_clip = np.maximum(preds[:, 1], 70)

        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)

        loss_by_sample = - np.sqrt(2) * Delta / sigma_clip - np.log(np.sqrt(2) * sigma_clip)

        loss = np.average(loss_by_sample, weight)

        

        return loss

    

    def _calc_grad_and_hess(

        self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None

    ) -> tp.Tuple[np.ndarray]:

        """Calc Grad and Hess"""

        mu = preds[:, 0]

        sigma = preds[:, 1]

        

        sigma_t = np.log(1 + np.exp(sigma))

        grad_sigma_t = 1 / (1 + np.exp(- sigma))

        hess_sigma_t = grad_sigma_t * (1 - grad_sigma_t)

        

        grad = np.zeros_like(preds)

        hess = np.zeros_like(preds)

        grad[:, 0] = - (labels - mu) / sigma_t ** 2

        hess[:, 0] = 1 / sigma_t ** 2

        

        tmp = ((labels - mu) / sigma_t) ** 2

        grad[:, 1] = 1 / sigma_t * (1 - tmp) * grad_sigma_t

        hess[:, 1] = (

            - 1 / sigma_t ** 2 * (1 - 3 * tmp) * grad_sigma_t ** 2

            + 1 / sigma_t * (1 - tmp) * hess_sigma_t

        )

        if weight is not None:

            grad = grad * weight[:, None]

            hess = hess * weight[:, None]

        return grad, hess

    

    def loss(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[str, float, bool]:

        """Return Loss for lightgbm"""

        labels = data.get_label()

        weight = data.get_weight()

        n_example = len(labels)

        

        # # reshape preds: (n_class * n_example,) => (n_class, n_example) =>  (n_example, n_class)

        preds = preds.reshape(self.n_class, n_example).T

        # # calc loss

        loss = self(preds, labels, weight)

        

        return self.name, loss, True

    

    def grad_and_hess(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[np.ndarray]:

        """Return Grad and Hess for lightgbm"""

        labels = data.get_label()

        weight = data.get_weight()

        n_example = len(labels)

        

        # # reshape preds: (n_class * n_example,) => (n_class, n_example) =>  (n_example, n_class)

        preds = preds.reshape(self.n_class, n_example).T

        # # calc grad and hess.

        grad, hess =  self._calc_grad_and_hess(preds, labels, weight)



        # # reshape grad, hess: (n_example, n_class) => (n_class, n_example) => (n_class * n_example,) 

        grad = grad.T.reshape(n_example * self.n_class)

        hess = hess.T.reshape(n_example * self.n_class)

        

        return grad, hess
class LGBM_Wrapper():



    def __init__(self):

        self.model = None

        self.importance = None



        self.train_bin_path = 'tmp_train_set.bin'

        self.valid_bin_path = 'tmp_valid_set.bin'



    def _remove_bin_file(self, filename):

        if os.path.exists(filename):

            os.remove(filename)



    def dataset_to_binary(self, train_dataset, valid_dataset):

        # Remove Binary Cache.

        self._remove_bin_file(self.train_bin_path)

        self._remove_bin_file(self.valid_bin_path)

        # Save Binary Cache.

        train_dataset.save_binary(self.train_bin_path)

        valid_dataset.save_binary(self.valid_bin_path)

        # Reload Binary Cache.

        train_dataset = lgb.Dataset(self.train_bin_path)

        valid_dataset = lgb.Dataset(self.valid_bin_path)

        return train_dataset, valid_dataset



    def fit(self, params, train_param,

            X_train, y_train, X_valid, y_valid, categorical=None,

            train_weight=None, valid_weight=None):

        train_dataset = lgb.Dataset(

            X_train, y_train, feature_name=X_train.columns.tolist(),

            categorical_feature=categorical, weight=train_weight

        )

        valid_dataset = lgb.Dataset(

            X_valid, y_valid, weight=valid_weight, 

            categorical_feature=categorical, reference=train_dataset

        )



        train_dataset, valid_dataset = self.dataset_to_binary(train_dataset, valid_dataset)



        self.model = lgb.train(

            params,

            train_dataset,

            valid_sets=[train_dataset, valid_dataset],

            **train_param

        )

        # Remove Binary Cache.

        self._remove_bin_file(self.train_bin_path)

        self._remove_bin_file(self.valid_bin_path)



    def predict(self, data):

        return self.model.predict(data, num_iteration=self.model.best_iteration)



    def model_importance(self):

        imp_df = pd.DataFrame(

            [self.model.feature_importance()],

            columns=self.model.feature_name(),

            index=['Importance']

        ).T

        imp_df.sort_values(by='Importance', inplace=True)

        return imp_df



    def plot_importance(self, filepath, max_num_features=50, figsize=(18, 25)):

        imp_df = self.model_importance()

        # Plot Importance DataFrame.

        plt.figure(figsize=figsize)

        imp_df[-max_num_features:].plot(

            kind='barh', title='Feature importance', figsize=figsize,

            y='Importance', align="center"

        )

        plt.show()

        # plt.savefig(filepath)

        # plt.close('all')
custom_loss = OSICLossForLGBM()



params = {

    'model_params':{

        'num_class': 2,

        # 'objective': 'regression',

        'metric': 'None',

        'boosting_type': 'gbdt',

        'learning_rate': 5e-02,

        'seed': SEED,

        "subsample": 0.4,

        "subsample_freq": 1,

        'max_depth': 1,

        'verbosity': -1

    },

    'train_params': {

        "num_boost_round": 10000,

        "verbose_eval":100,

        "early_stopping_rounds": 100,

        'fobj': custom_loss.grad_and_hess,

        'feval': custom_loss.loss

    }

}





u_idx = train['Patient_Week']



categorical_cols = ['Sex', 'SmokingStatus']

drop_cols = ['Patient', 'Patient_Week', 'FVC']

features = [c for c in train.columns.tolist() if c not in drop_cols]



X = train[features]

y = train['FVC']

groups = train['Patient']



n_fold = 5

g_kfold = model_selection.GroupKFold(n_splits=n_fold)



models = []

oof = np.zeros((train.shape[0], 2))

for i, (train_idx, valid_idx) in enumerate(g_kfold.split(X, y, groups)):

    print('\n' + '#'*20)

    print('#'*5, f' {i+1}-Fold')

    print('#'*20 + '\n')

    

    print(f'Train Size: {len(train_idx)}')

    print(f'Valid Size: {len(valid_idx)}', '\n')

    

    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]

    X_valid, y_valid = X.iloc[valid_idx, :], y.iloc[valid_idx]

    

    lgb_model = LGBM_Wrapper()

    lgb_model.fit(

        params['model_params'],

        params['train_params'],

        X_train,

        y_train,

        X_valid,

        y_valid,

        categorical_cols

    )

    # add to oof

    oof[valid_idx] = lgb_model.predict(X_valid)

    # Add Model

    models.append(lgb_model)
f_imp = np.array([m.model_importance().sort_index().values for m in models])

f_name = models[0].model_importance().sort_index().index



imp_df = pd.DataFrame(f_imp.reshape(-1, len(f_name)).T, index=f_name)

imp_df['AVG_importance'] = imp_df.iloc[:, :len(models)].mean(axis=1)

imp_df['STD_importance'] = imp_df.iloc[:, :len(models)].std(axis=1)

imp_df.sort_values(by='AVG_importance', inplace=True)



imp_df.plot(

    kind='barh', 

    y='AVG_importance', 

    xerr='STD_importance', 

    capsize=4, 

    figsize=(5, 6)

)

plt.tight_layout()

plt.show()
def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70)

    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)

    metric = - (np.sqrt(2) * delta / sigma_clip) - np.log(np.sqrt(2) * sigma_clip)

    return np.mean(metric)



score(train['FVC'], oof[:, 0], oof[:, 1])
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

test = preprocessing(test, is_test=True)
print(test.shape)

test.head()
test_idx = test['Patient_Week'].to_numpy().reshape(-1, 1)

pred = np.mean([m.predict(test[features]) for m in models], axis=0)



pred_df = pd.DataFrame(

    np.concatenate((test_idx, pred), axis=1), 

    columns=['Patient_Week', 'FVC', 'Confidence']

)
submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))



sub_df = submission.drop(columns=['FVC', 'Confidence'])

sub_df = sub_df.merge(pred_df[['Patient_Week', 'FVC', 'Confidence']], on='Patient_Week')

sub_df.columns = submission.columns





if os.path.exists('/kaggle/input'):

    sub_df.to_csv('submission.csv', index=False)



print(sub_df.shape)

sub_df.head()