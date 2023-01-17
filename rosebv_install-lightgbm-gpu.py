!apt-get install -y libboost-all-dev
!pip uninstall -y lightgbm
!git clone --recursive https://github.com/microsoft/LightGBM

!cd LightGBM

!git checkout tags/v2.3.1

!mkdir build
!cd LightGBM && cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .
!cd LightGBM && make -j$(nproc)
!cd LightGBM/python-package && python setup.py install --precompile
!mkdir -p /etc/OpenCL/vendors

!echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
import lightgbm as lgb

from sklearn.datasets import load_iris
iris = load_iris()



lgb_train = lgb.Dataset(iris.data[:100], iris.target[:100])

lgb_eval = lgb.Dataset(iris.data[100:], iris.target[100:], reference=lgb_train)
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'auc',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 1,

    'device': 'gpu'

}



# Run only one round for faster test

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=1,

                valid_sets=lgb_eval,

                early_stopping_rounds=1)

gbm.best_iteration