#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAACoCAMAAABt9SM9AAABs1BMVEX///8zMzMRirIAzGaLW9P3wFIAAADvR28G1qAsLCwlJSUgICDq6uooKCgVFRVvb29ZWVn09PQPDw+fn5+MjIw0KyYXfaAcdJM0LisjZX41Hyg1Iy8OxJMSsFs0LTA0KDArZVMAzmEA3KLxPm0Ah7WGVtcgj1CdaZDGk5f6w0sA0l34QmpmmJ2ndrp9b8X2PGxEgqp6rJDLy8swmP+7u7vq8/8Ukf9FoP/FxcWwsLB7e3vb29uUlJRqampAQEBfX19OTk49PT0aMS4cJTEpMjEsMCKBOUkmKzJiR40qMB7QQ2QsSlbVp0x6U7ZJPF+CgoIkimuZekFzYTsuUzz4vMjydZD60NhdpcPK4Or4yXH50IaSZtV148CO58rl+fNc37bziJ72qbn96+/4xM7wWXz98PO71uOgyNl2sssAcaM0lbn87NH2ujn51pjf1PLXyu+jgNul574u0HWr7dfG8+Rd37fq+vUAjWv0l6vxZoVWY4D74LT88Nr75sDSnomRa5Gz1P+/qOU7bZ+vkt+IuvcnHxB3NszHtOieeNl4OUZo2pVS1Yis6MOB36ZwblVCWWAkfUna1RwbAAAReUlEQVR4nO2diWPbRnbGB7Q15AAgwCHgXaeHd2VSbdV0yxYGbVwkwG2S7fZid1cHLUu0JNuiQtnyISZ1artNum2Sxt7d/sl9DyApSqJ56TSFz9IQGAwGgx/evHkzpExCYsWKFStWrFixYsWKFSvWCOKG0U66kjs7rhvtm9bR82S3p0Q/RYd4t2JXPm5Tz18upS6xKe3Jsjs7fhDtgmRiqAfOcyRMg8p7KxYyxHKIRXl7nzon1eTzk0uzGlEV5BPZgiubFC0CbrKUwQyqh3caBJBjdO2pKhDuEs6JzCPrMThH25EhwR8Oh8o+Nyhvm1S2Gr5wqCIqyCMTlonLiWu0T73YcqkTAAvgw2hW4USlFYFCboWabVgB1eFWTEp9mQpUJDyrUEaqFUJ9kgmIUGZUIpxSSSxBaY3Cj8upG5TgFNXF6sJeHMEyaQDgqY3ma8NFZAJ1moEkgH370jlyGEkuNSn+Eh14CapMDVKlRKjCbbVh8RLFrlryiWxht9IVwj1eFSp+aHvQEw0qe3A2xY4qUyhvw8lBiWRUOOQSMeyAESwwSriQD6epBCA6PqEZLgM8S267wYssl8rUD4zw5kiVueBj4E7RTXVgQecwBIY7gEGk3Pcxt0ppJoRVdrAOp0yQDYjZVPfVagirBLA4liBdWAFUgbnUkPEaCloZ8ShlF54UCWH51ANYjgg9DnqaC00nioeOJYTF0WeVFAKMwKkDTCcNqORqVgbL6MLqWhZ0QgX6pIGwgqOwFAc7KoHOCBUb6LgAFreIW8mcI4RRBTeqUcOi0H+CDLiQDFUlCsblgOFEo6GGFmASh2o2dRiVZZopUwSH/Ql8lop1cKowKYQiA7Ms8Cj74P80hCWEEENjNUtQMUVTgifgwLZG4IdToQrMLr7P4haHByujDzZtHOZNC/aJaxvdKMq1TRynIMcKfQu3TRKOX5YMJXB0gzpgSAyi4ABK4ImYbxtYVzSEGiA4ZJrE4qErgyzbDYvjpd2oykshizo67RO79penvD8+uwwyPG90F61pfHihWLGmSCU7etVUDRO79xj3w77jeYdPskoDanTswzm8pKpqaaDPsqs9Fwylq46jlno7ZKnHxfPMOURe3UmyihsUx+x9YcBO0IccPkujh3N6pJuHczjNwI0PhFWVooL7hTynTKtqLyza8xS4eg6whBKN7k0tUxmmajqRHR/gcN23AZbtOJzYNtEMDZ+r5VfD1rdhuSo22S6pBjE025F1twSwNYvorl7CI76p4d23GfCq6zgwb6n6aKq6DxfWIYd7UFavmLCPBd2S054yYyhLPN8OUziFalUVogvbhCyuu1BSNzSiQTuhVUapetrDgktdPxqnVT8wNT/QCa1aGAz6JvU4LVu0hLFlhdoqzkNMJ7sPi1NPoxwydZz2plUDT9KJUiU0awaYZzvUCwtWgqACoalgSQHJOkZWJ+XAAkuhoqVQ28e5J9RlwYRQpprWXrhBWH7FymokEKwszBMls5zGkh41OP6oMNkm5TDsNahdHWTuJ6EqhuNuBMtTfbsMluFq1IPLEx52Q6eCE9tyCVtUqhhR6QgWxPYGRtiWQ2WbYiwpY9PTANwj0WQZI3CEpWqexzkYsZ4lkqLDTAbC95KAPatUxrIw/yYVmD0bYaV2BxY8DwOeD7SDh91Qi6b01ARYuLAGrQsQFlcFaNoph63hLKMawTIodQEWrdiKZ4QPtwcWTvF4RtB1bx8WzEp0D27N8SjHluNJQQRLQwAArgMr6obwArCIJ+B6jqN7WggrCGExgjNrakSV7sNSdcAcYWjDknpgVTuwSqzdtNMT2InMI/NVwcwpqeguLjd5eJNM5QdhwTSRuA4PYVmmCZhcrsowDdT6w/LQ4CJYOpR327AyNho0XMOze2DhZQGeZcFMUpU7sMApoFOiDonMUItsMITlQk/MlkmmDCfvN+30VA4I2o8dwWIBAcvK0IpfggkKLfPQssr7iwekRMObj2bPZVw7we5J9egxt2GxDiyYC3ttWJEBIywlXIiGEYFSUe6BlQmoH9kepe0laoTlZqniYiq5hyzLguOiX8aKvKhpRyKcD0kwih4Y7k9BztTMIt3Q/E5Vavp06z9L8XhGHCtWrFixLpDkM9Low5+pCtIZqeKMvP4P7fKVM5M0eJGvI1eQmHBmYkow4icl5EA5w3ZBw/zh5mXSM2xR2KrsSEsS8vEfYTBe8bQwjJZx1qxA2VFsix2bleQZ0lgnsGGTHPG4TZpALBjOqnrshklV2x6zEunIqv0B2eOxPyEpQzsizx7zEkzSNS2Q0uPZ5+DPJWTO0od2la4Og2VGD/HGH02gG3Ai8z2NmoFtuGNZw+CneNwHOJmGOgfihR3oxk//ZAL9K9KquELgehlzvJ4jDVrGOra1T6ihH8PR01jsxk9vpsbWzRAWjIViNmNRMT1Ou8RBTmuaYaV11aKOrY1DaxJY4bA9wJux447r48FaXV09gGPvC8y63wfU/R5YgiBLVU0JxokfJoDFVPD7ZafS3j0S2zHVqY4Z8B3SeLD2Wmut1P21+6n7X6zB7treXoqstfZSyHA1tZYKN1JrkNQB4T4sUTQVuH9/9Cc7Aay0JyuSbTAmSeBnFVMS4XJpUexclGVMp4zlJAnuiYlYShChNIPtMCs8kYVZJwBrtYVk9khqlazWU1+u1vdS9S9XU609oNYia/dho5Wq79VTqS/31npgCaIpMsUORu8Hk1iWo3nMtqWMZVswzLu27bC0Zpo6EwAG0JC0CsMI2TQ1iTkmBH/ATzUtv2RClshKlm0y5luqZfV/rGPC2gMG9TWSut+6WU/Vb67u3axDh/syhdtrN2/WkdraGuS1DnRD6BSGo9k0U5JwMnNKsKolUyubWYtm06ao2AralFIRTFG0TdPS0qIGhiU6OqW6wyShomVYuuTCluKyLGCCE6FwWsWsvhoPVquean3RarWAB1jWXr1V/6IOZgTJ/VS9Dj0UkptwHLoiZPXCYszJUNOrGoozkqOfCJbqGNRkYFK2Bh0/vIyhaeAqGQrqLLOgqsFh21Fs2zMy0CoHiom2JJoQ4IQnMkd9n/2fxWjYvpe0qkuir1vBKH1xMliSlDWpJWVphommJJUrXomKbqe0olWkQMvYNCsFgkkptERUHfBP4N8km4lGNpuFEx3nfSsHZwdLYF6JsTIfidUksKCzZUTbCiqmDX6KZSzTrDATXJMeXTHtmWhTkgMeKiOBM9Ns3LZV0bFUwdLFAE6silXMOgkH34Ywqv5N/HlHn2AtBvwEkjjKpHqSOAsHvrSEr+itWHghUdwfDtNSuMNEPJDGI+E25IApoTV1D5+YZd3820JuVCU7yn0GtFhF42VwFuYpwTp1TQQrOb4+C00rbWZtNK/h0en0wYpMJ3dwt7OdC3+T4WsHFkan1BBsjA0Hd8apg7W4sPAwmSw8aMPJLWzgkYeLmD5YWC9sbCw8zG2sF16+zHVgYXSqGFZaYKXBC3hTByu5tAGU1jeAWHJhA142FxaTuQebGw8Lm4uLS4WNB4ubhQ2SJD2whLIRiDjYUCMzqF1TByu3sJ58mPtq86tkYXN9YSmXfPTVZiG3sbSeTG4mX74srD/aBGKbjzbXe2AxpvlQpWKrA2PTaYS1vpBLLi0CrOTLB4WHSx8DLMhaLDx6+PJR4eVG4VFy4yX0yB5Y4OUZzhOd0/JZ168dS9dPCdbi0tLSeuHBg6X13FdLSw8KL5fWAVwSMnOLeGRhaWkDkodLGwdgCaJVhkhw8EOcGNb1j350LH00gNaxfFahUICxDpPk0ia4q0KuEObmoiQ8AkNhYX80bF/U1YcFDxPDuvbjxLF0arCSuY9H1WfsekcCvh0xbBVu+mDl/nxk/ftHXWFr2OWD9fHfTNKcawdb0H8eFsPqA4sFXqUfrumHlYd/iXwTt/L5Pg3JH4VVtgKj3IfW1MNqkGWynK/VeCNBasXlNp78flLMH4IlCoEmiW6fdk09rPwWX84vb32eaOa3GluN/DIvNvKJYq2Wz2/VeDPfILXGAViiXVXMcsbsE3NNP6wGwNpubBe3AFajlq8lPuf5RuPzreV8o1ZrgmV9nj8AK21mGDMlMX30UyNTDyvshgneqG1DNwR7Amq1fJM3eCJPGtAt83DoACyBGQzfT/Gq3mFaUw8r0Uygc1+GpNlsghE1l9GSQueFme3NXlgV28YPY8lHPsg3/bBG04HRUBQFSdfKxuWyrPzIkv6uo19g1czRFBb4lwnWX82Pqpmu5n+FtEQ7CuIPDolTDmtmbEWwhIqRYUJaMA/0xEsKK7Kl+Z7triJY4ODLFS/DjBjWk1dvVmZmHu/g9uM3r+Znop443wNLSEtWYOuZ3vjhcsK6y+8SRIbbb97szM+8evXq7vyTNyvz+7AE0WJUMyqX3rLmyc7reUAGW69XXu2AcT0hMyt8hdztgSVUTM0qR++rX2ZYMxxBzby+C9jmV14/3lm5+/rxE/7qzQFYQtZ20sy0zc6i4OWE9WRnZeXV/M6bnRVA9Wbn7szOzuvH869e7xyEJaR9ySxLkqFEy4GXExa68q5Xh3R+5/WTmfZ+LyxoiikBIy3NdPHywBoSlD5+3N38VfrTjtCYTEFMV5mQxU90XhJY8//xF6PqP/++q7CJlpdJC9H7r31gFb/+YGCluhoG62dzI+tqV78MG5MJmGKruCjfz7Lu3Pr6g4DV+4dOnyWHwbo6vkJY4NoD27BNt383LN6+fefrMWGF69kJfEeg/9sCpwJrX59MDGtuAMUIlpC2HFVVgjLAKh4Rv3PrVohrdFjLtWZtO9+sQbJVqy0n2m+aRAkSbFxQWHO75Jvw0Bz+dJMw7cASJEl3MHS49l+3j+oWCnCNYVm1RjGR54lmI7/daBbzeVzGTWzXGvlEE+EtR28LXDxYQAvIfL+7+83cXHGXzM19t8ufzf22WESEv+w2yGjDuvVe3f7v/xkZVoIs5xO15hbA4rjSvZUvNhPFfKMB2LbA0orNi2lZAOvbp3PPdnefzs093QVw3+zufj9XnPuW98ISpDasO30UoSqObllNju/OFZe3gc9Wk4NNNQEPb25t5xvbW1uJfG15u3kRYc09JU/Jt9/t7pJ7V8l3/N5c8bvib8G8AF4vrPZSYL84606Iaiyf1Wg2thPNLXyDqdFYToBpgTktg6ElgFUz3LyYlnUPdHXuGSRX7z3DjGf3wGF9/32vz+qoH6yvb4eoJgkd8ge39pPE4SHyosAapFFgtVFd/DhrMlijh6fXftLRjffBKhY7Wx8SrNyosP76L2dH1ZWOZv/sJ++zrK4+IFj/+w9dDYV1ZXxNFSzhk+5fL336oxOAhVYFv2E6O3Wwehp2ErDevfth9vnzt89/ePsctmJYg+wKOLVmn+/NkhdXvnxRfxvDGqQXrdbb2dn623eQtOpxNxxoWe9evKjPzj4Hw5qtx5Y1BNbv9lpA6MXz2Ss/7P3uxbvZGNYgWlGMNXulMybGsMZQDCuGJVzrXiIfwxoGS/jHrn4fwxoGS+j8pRf7dQxrKKyOfhHDimHFsGJYMawYVgwrhhXDOhVY7Nf/1NXPrsawhtDq/vXSz//5XGFd/+jHx9JZwNrXp+cL6wL8XzQfEKxTVAxrDMWwxlAMawzFsMZQDGsMxbDG0GnCmothDdW/dPWH2eFsLjmsrthvYlgj60YMa3TFsMZQDGsMxbDGUAxrDMWwxtDUwRIvGqxR/sLivL7fMBgGSzvGt4z+5k8n0P+FsCRzUKPG+NKxE1RaHwbLVY5R/R9PoJDVkK/WHfObJU9IWXcYLHIezRpq8f3+09zTVro0lBWxzsNB0CEPUT77RrH0KN8075y9zSvesEa5Z/ql6QJ+mcRo3+fuHMdtTaLsUFZgW35WZGcmMeuMYlcokyln1zBRqYz0LfPE1dTMGcnRRjOrSIannpEcbTRUsWLFihUrVqxYsWLFihUrVqxYsWLFGk//D9hWxIgl3WcPAAAAAElFTkSuQmCC',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ai4all-project/results/classifier/lasso_min/lassoRandomForest_probs.csv')

df.head()
print(f"data shape: {df.shape}")
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

df["CZB_ID"] = encoder.fit_transform(df["CZB_ID"].fillna('Nan'))

#df["Unnamed: 0"] = encoder.fit_transform(df["Unnamed: 0"].fillna('Nan'))

df.head()
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
y_train = df.COVID19.values
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df.values)

    rmse= np.sqrt(-cross_val_score(model, df.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1) 
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(df.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(df.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(df.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(df, y_train)

xgb_train_pred = model_xgb.predict(df)

xgb_pred = np.expm1(model_xgb.predict(df))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(df, y_train)

lgb_train_pred = model_lgb.predict(df)

lgb_pred = np.expm1(model_lgb.predict(df.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.70 +

               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
test_ID = df['CZB_ID']
sub = pd.DataFrame()

sub['CZB_ID'] = test_ID

sub['COVID19'] = ensemble

sub.to_csv('submission.csv',index=False)