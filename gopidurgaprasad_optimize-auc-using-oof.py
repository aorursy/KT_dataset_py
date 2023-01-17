import pandas as pd, numpy as np, os

import matplotlib.pyplot as plt

from scipy.optimize import minimize, fmin

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization

from functools import partial

from sklearn import metrics
class OptimizeAUC:

    """

    Class for optimizing AUC

    This class is all you need to find best weights for

    any model and for any metric and for any type of predictions

    With very small changes, this class can be used for optimization of

    weights in ensemble models for _any_ type of predictions

    """

    def __init__(self):

        self.coef_ = 0

    

    def _auc(self, coef, X, y):

        """

        This functions calculates and returns AUC

        :param coef: coef list, of the same length as number of models

        :param X: predictions, in this case a 2d array

        :param y: targets, in our case binary 1d array

        """



        # multiply coefficients with every column of the array

        # with predictions.

        # this means: element 1 of coef is multiplied by column 1

        # of the prediction array, element 2 of coef is multiplied

        # by column 2 of the prediction array and so on!

        x_coef = X * coef

        # create predictions by taking row wise sum

        predictions = np.sum(x_coef, axis=1)



        # calculate auc score

        auc_score = metrics.roc_auc_score(y, predictions)



        #return negative auc

        return -1.0 * auc_score

    

    def fit(self, X, y):

        # remember partial from hypeparamer optimization chapter?

        loss_partial = partial(self._auc, X=X, y=y)



        # dirichlet distribution. you can use any distribution you want

        # to initialize the coefficients

        # we want the coefficients to sum to 1

        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)



        # use scipy fmin to minimize the loss function, in our case auc

        self.coef_ = fmin(loss_partial, initial_coef, disp=True)



    def predict(self, X):

        # this is similar to _auc function

        x_coef = X * self.coef_

        predictions = np.sum(x_coef, axis=1)

        return predictions
sample_submission = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
SUB_DIR = "../input/siimsubmitionfiles/Final_Submissions/Final_Submissions"

TF_0OFS = "../input/siimsubmitionfiles/TF-OOFS/TF-OOFS"

TF_SUBS = "../input/siimsubmitionfiles/TF-SUBS/TF-SUBS"
offs_b5_df = pd.read_csv(f"{SUB_DIR}/offs_b5_df.csv")

offs_b5_ns_df = pd.read_csv(f"{SUB_DIR}/offs_b5_ns_df.csv")

offs_b5_ds_df = pd.read_csv(f"{SUB_DIR}/offs_b5_512_ds_df.csv")

offs_b5_10_ds_df = pd.read_csv(f"{SUB_DIR}/offs_b5_10_ds_df.csv")

offs_b5_f_ds_df = pd.read_csv(f"{SUB_DIR}/offs_b5_f_ds_df.csv")

offs_tf_b5_df = pd.read_csv(f"{SUB_DIR}/offs_tf_b5_df.csv")

offs_tf_b5_v1_df = pd.read_csv(f"{SUB_DIR}/offs_tf_b5_v1_df.csv")



offs_b6_df = pd.read_csv(f"{SUB_DIR}/offs_b6_df.csv")

offs_b6_2_df = pd.read_csv(f"{SUB_DIR}/offs_b6_2_ds_df.csv")

offs_b6_10_df = pd.read_csv(f"{SUB_DIR}/offs_b6_10_ds_df.csv")

offs_tf_b6_df = pd.read_csv(f"{SUB_DIR}/offs_tf_b6_df.csv")



offs_d201_df = pd.read_csv(f"{SUB_DIR}/offs_d201_df.csv")

offs_d201_ns_df = pd.read_csv(f"{SUB_DIR}/offs_d201_ns_df.csv")

offs_d201_512_df = pd.read_csv(f"{SUB_DIR}/offs_d201_512_df.csv")

offs_d201_512_ds_df = pd.read_csv(f"{SUB_DIR}/offs_d201_512_ds_df.csv")

offs_tf_d201_df = pd.read_csv(f"{SUB_DIR}/offs_tf_d201_df.csv")



offs_b4_df = pd.read_csv(f"{SUB_DIR}/offs_b4_df.csv")

offs_b4_f_df = pd.read_csv(f"{SUB_DIR}/offs_b4_f_5_ds_df.csv")



offs_b7_224_df = pd.read_csv(f"{SUB_DIR}/offs_b7_224_ds_df.csv")

offs_tf_b7_df = pd.read_csv(f"{SUB_DIR}/offs_tf_b7_df.csv")

offs_b7_512_ds_df = pd.read_csv(f"{SUB_DIR}/offs_b7_512_ds_df.csv")



offs_dpn92_ds_df = pd.read_csv(f"{SUB_DIR}/offs_dpn92_ds_df.csv")



offs_sk50_ds_df = pd.read_csv(f"{SUB_DIR}/offs_sk50_5_ds_df.csv")



offs_meta_df = pd.read_csv(f"{SUB_DIR}/tabular_oof_df.csv").sort_values("image_name")

offs_meta_df = offs_meta_df[offs_meta_df.image_name.isin(offs_dpn92_ds_df.image_name)].reset_index(drop=True)



offs_meta_2_df = pd.read_csv(f"{SUB_DIR}/tabular2_oof_df .csv").sort_values("image_name")

offs_meta_2_df = offs_meta_2_df[offs_meta_2_df.image_name.isin(offs_dpn92_ds_df.image_name)].reset_index(drop=True)
offs_final_df = pd.DataFrame({

    "image_name" : offs_b5_df.image_name.values,

    "target" : offs_b5_df.target.values,



    "b5" : offs_b5_df.prediction.values,

    "b5_ns": offs_b5_ns_df.prediction.values,

    "b5_ds": offs_b5_ds_df.prediction.values,

    "b5_ds_10": offs_b5_10_ds_df.prediction.values,

    "b5_f_ds" : offs_b5_f_ds_df.prediction.values,

    "tf_b5" : offs_tf_b5_df.pred.values,

    "tf_b5_v1": offs_tf_b5_v1_df.pred.values,



    "b6" : offs_b6_df.prediction.values,

    "b6_2" : offs_b6_2_df.prediction.values,

    "b6_10" : offs_b6_10_df.prediction,

    "tf_b6": offs_tf_b6_df.pred.values,



    "b4" : offs_b4_df.prediction.values,

    "b4_f" : offs_b4_f_df.prediction.values,



    "d201" :offs_d201_df.prediction.values,

    "d201_ns" : offs_d201_ns_df.prediction.values,

    "d201_512" : offs_d201_512_df.prediction.values,

    "d201_512_ds" : offs_d201_512_ds_df.prediction.values,

    "tf_d201" : offs_tf_d201_df.pred.values,



    "b7_224_ds" : offs_b7_224_df.prediction.values,

    "tf_b7" : offs_tf_b7_df.pred.values,

    "b7_512_ds" : offs_b7_512_ds_df.prediction.values,



    "dpn92_ds" : offs_dpn92_ds_df.prediction.values,



    "sk50_ds" : offs_sk50_ds_df.prediction.values,



    "meta_sub" : offs_meta_df.prediction.values,

    "meta_sub_2" : offs_meta_2_df.prediction.values

})
train_df = offs_final_df

train_cols = [

    'b5', 'b5_ns', 'b5_ds', 'b5_ds_10', 'b5_f_ds','tf_b5', 'tf_b5_v1', 

    'b6', 'b6_2', 'b6_10', 'tf_b6', 

    'b4', 'b4_f',

    'd201', 'd201_ns', 'd201_512', 'd201_512_ds', 'tf_d201', 

    'b7_224_ds','tf_b7', 'b7_512_ds', 

    'dpn92_ds', 

    'sk50_ds', 

    'meta_sub', 'meta_sub_2' 

]



X = train_df[train_cols]

y = train_df["target"]
opt = OptimizeAUC()

opt.fit(X, y)
opt.coef_
# b5

b5_512_5    = pd.read_csv(f"{SUB_DIR}/Final_B5_512_456_5.csv")["target"]

b5_512_5_ns = pd.read_csv(f"{SUB_DIR}/Final_NS_E5_512_456_5.csv")["target"]

b5_512_5_ds = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_5.csv")["target"]

b5_512_10_ds = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_10.csv")["target"]

b5_512_f_ds = pd.read_csv(f"{SUB_DIR}/F_Final_DS_B5_512_456_5.csv")["target"]

b5_512_5_tf = pd.read_csv(f"{SUB_DIR}/TF-B5-512-V.csv")["target"]

b5_512_5_v1 = pd.read_csv(f"{SUB_DIR}/TF-B5-512.csv")["target"]



# b6

b6_528_5    = pd.read_csv(f"{SUB_DIR}/Final_B6_768_528_5.csv")["target"]

b6_512_2    = pd.read_csv(f"{SUB_DIR}/Final_DSO_B6_768_528_2.csv")["target"]

b6_512_10   = pd.read_csv(f"{SUB_DIR}/Final_DS_B6_768_528_10.csv")["target"]

b6_512_5_tf = pd.read_csv(f"{SUB_DIR}/TF-B6-512-5.csv")["target"]



#b4

b4_300_10      = pd.read_csv(f"{SUB_DIR}/Final_B4_512_380_10.csv")["target"]

b4_f           = pd.read_csv(f"{SUB_DIR}/F_Final_DS_B4_512_380_5.csv")["target"]



# d201

d201_224_5     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_224_5.csv")["target"]

d201_224_5_ns  = pd.read_csv(f"{SUB_DIR}/Final_NS_D201_512_224_5.csv")["target"]

d201_512_5     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_512_5.csv")["target"]

d201_512_5_ds  = pd.read_csv(f"{SUB_DIR}/Final_DS_D201_512_512_5.csv")["target"]

d201_512_5_tf  = pd.read_csv(f"{SUB_DIR}/TF-D201-512-5.csv")["target"]



# b7

b7_224_ds      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_224_5.csv")["target"]

b7_512_ds      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_512_5.csv")["target"]

b7_512_tf      = pd.read_csv(f"{SUB_DIR}/TF-B7-512-5.csv")["target"]



# dpn92

dpn92_ds       = pd.read_csv(f"{SUB_DIR}/Final_DS_dpn92_512_512_5.csv")["target"]



# sk50

sk50_ds        = pd.read_csv(f"{SUB_DIR}/Final_DS_sk50_512_512_5.csv")["target"]



# meta

meta_1 = pd.read_csv(f"{SUB_DIR}/tabular_test_prediction.csv")["target"]

meta_2 = pd.read_csv(f"{SUB_DIR}/tabular2_test_prediction .csv")["target"]
W = opt.coef_



pytorch_sub = np.mean([

         

    # b5

   W[0] * b5_512_5, #    = pd.read_csv(f"{SUB_DIR}/Final_B5_512_456_5.csv")["target"]

   W[1] * b5_512_5_ns,# = pd.read_csv(f"{SUB_DIR}/Final_NS_E5_512_456_5.csv")["target"]

   W[2] * b5_512_5_ds,# = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_5.csv")["target"]

   W[3] * b5_512_10_ds,# = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_10.csv")["target"]

   W[4] * b5_512_f_ds, #= pd.read_csv(f"{SUB_DIR}/F_Final_DS_B5_512_456_5.csv")["target"]

   W[5] * b5_512_5_tf, #= pd.read_csv(f"{SUB_DIR}/TF-B5-512-V.csv")["target"]

   W[6] * b5_512_5_v1, #= pd.read_csv(f"{SUB_DIR}/TF-B5-512.csv")["target"]



    # b6

   W[7] * b6_528_5,#    = pd.read_csv(f"{SUB_DIR}/Final_B6_768_528_5.csv")["target"]

   W[8] * b6_512_2,#    = pd.read_csv(f"{SUB_DIR}/Final_DSO_B6_768_528_2.csv")["target"]

   W[9] * b6_512_10,#   = pd.read_csv(f"{SUB_DIR}/Final_DS_B6_768_528_10.csv")["target"]

   W[10] * b6_512_5_tf,# = pd.read_csv(f"{SUB_DIR}/TF-B6-512-5.csv")["target"]



    #b4

   W[11] * b4_300_10,#      = pd.read_csv(f"{SUB_DIR}/Final_B4_512_380_10.csv")["target"]

   W[12] * b4_f,#           = pd.read_csv(f"{SUB_DIR}/F_Final_DS_B4_512_380_5.csv")["target"]



    # d201

   W[13] * d201_224_5,#     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_224_5.csv")["target"]

   W[14] * d201_224_5_ns,#  = pd.read_csv(f"{SUB_DIR}/Final_NS_D201_512_224_5.csv")["target"]

   W[15] * d201_512_5,#     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_512_5.csv")["target"]

   W[16] * d201_512_5_ds,#  = pd.read_csv(f"{SUB_DIR}/Final_DS_D201_512_512_5.csv")["target"]

   W[17] * d201_512_5_tf,#  = pd.read_csv(f"{SUB_DIR}/TF-D201-512-5.csv")["target"]



    # b7

   W[18] * b7_224_ds,#      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_224_5.csv")["target"]

   W[19] * b7_512_ds,#      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_512_5.csv")["target"]

   W[20] * b7_512_tf,#      = pd.read_csv(f"{SUB_DIR}/TF-B7-512-5.csv")["target"]





    # dpn92

   W[21] * dpn92_ds,#       = pd.read_csv(f"{SUB_DIR}/Final_DS_dpn92_512_512_5.csv")["target"]



    # sk50

   W[22] * sk50_ds,#        = pd.read_csv(f"{SUB_DIR}/Final_DS_sk50_512_512_5.csv")["target"]



    # meta

  W[23] *  meta_1,# = pd.read_csv(f"{SUB_DIR}/tabular_test_prediction.csv")["target"]

  W[24] *  meta_2,# = pd.read_csv(f"{SUB_DIR}/tabular2_test_prediction .csv")["target"]



], axis=0)
sample_submission["target"] = pytorch_sub

sample_submission.to_csv("Final_Pytorch.csv", index=False)
offs_b7_512_15 = pd.read_csv(f"{TF_0OFS}/offs_b7_512_15.csv")

offs_b6_512_15 = pd.read_csv(f"{TF_0OFS}/offs_b6_512_15.csv")

offs_b5_512_15 = pd.read_csv(f"{TF_0OFS}/offs_b5_512_15.csv")



offs_b7_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b7_512_10.csv")

offs_b6_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b6_512_10.csv")

offs_b5_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b5_512_10.csv")

offs_b4_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b4_512_10.csv")

offs_b3_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b3_512_10.csv")

offs_b2_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b2_512_10.csv")

offs_b1_512_10 = pd.read_csv(f"{TF_0OFS}/offs_b1_512_10.csv")



offs_b7_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b7_768_5.csv")

offs_b6_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b6_768_5.csv")

offs_b5_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b5_768_5.csv")

offs_b4_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b4_768_5.csv")

offs_b3_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b3_768_5.csv")

offs_b2_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b2_768_5.csv")

offs_b1_768_5 = pd.read_csv(f"{TF_0OFS}/offs_b1_768_5.csv")



offs_b7_384_15 = pd.read_csv(f"{TF_0OFS}/offs_b7_384_15.csv")

offs_b6_384_15 = pd.read_csv(f"{TF_0OFS}/offs_b6_384_15.csv")

offs_b5_384_15 = pd.read_csv(f"{TF_0OFS}/offs_b5_384_15.csv")
offs_final_tf_df = pd.DataFrame({



    "target" : offs_b7_512_15.target.values,

    "image_name" : offs_b7_512_15.image_name.values,



    "b7_512_15" : offs_b7_512_15.prediction.values,

    "b6_512_15" : offs_b6_512_15.prediction.values,

    "b5_512_15" : offs_b5_512_15.prediction.values,



    "b7_512_10" : offs_b7_512_10.prediction.values,

    "b6_512_10" : offs_b6_512_10.prediction.values,

    "b5_512_10" : offs_b5_512_10.prediction.values,

    "b4_512_10" : offs_b4_512_10.prediction.values,

    "b3_512_10" : offs_b3_512_10.prediction.values,

    "b2_512_10" : offs_b2_512_10.prediction.values,

    "b1_512_10" : offs_b1_512_10.prediction.values,



    "b7_768_5" : offs_b7_768_5.prediction.values,

    "b6_768_5" : offs_b6_768_5.prediction.values,

    "b5_768_5" : offs_b5_768_5.prediction.values,

    "b4_768_5" : offs_b4_768_5.prediction.values,

    "b3_768_5" : offs_b3_768_5.prediction.values,

    "b2_768_5" : offs_b2_768_5.prediction.values,

    "b1_768_5" : offs_b1_768_5.prediction.values,



    "b7_384_15" : offs_b7_384_15.prediction.values,

    "b6_384_15" : offs_b6_384_15.prediction.values,

    "b5_384_15" : offs_b5_384_15.prediction.values,



})
train_df = offs_final_tf_df



train_cols = [

    'b7_512_15', 'b6_512_15', 'b5_512_15',

    'b7_512_10', 'b6_512_10', 'b5_512_10', 'b4_512_10', 'b3_512_10',

    'b2_512_10', 'b1_512_10', 'b7_768_5', 'b6_768_5', 'b5_768_5',

    'b4_768_5', 'b3_768_5', 'b2_768_5', 'b1_768_5', 'b7_384_15',

    'b6_384_15', 'b5_384_15'

]



X = train_df[train_cols]

y = train_df["target"]
opt.coef_
B7_512_15 = pd.read_csv(f"{TF_SUBS}/B7_512_15.csv")["target"]

B6_512_15 = pd.read_csv(f"{TF_SUBS}/B6_512_15.csv")["target"]

B5_512_15 = pd.read_csv(f"{TF_SUBS}/B5_512_15.csv")["target"]



B7_512_10 = pd.read_csv(f"{TF_SUBS}/B7_512_10.csv")["target"]

B6_512_10 = pd.read_csv(f"{TF_SUBS}/B6_512_10.csv")["target"]

B5_512_10 = pd.read_csv(f"{TF_SUBS}/B5_512_10.csv")["target"]

B4_512_10 = pd.read_csv(f"{TF_SUBS}/B4_512_10.csv")["target"]

B3_512_10 = pd.read_csv(f"{TF_SUBS}/B3_512_10.csv")["target"]

B2_512_10 = pd.read_csv(f"{TF_SUBS}/B2_512_10.csv")["target"]

B1_512_10 = pd.read_csv(f"{TF_SUBS}/B1_512_10.csv")["target"]



B7_768_5 = pd.read_csv(f"{TF_SUBS}/B7_768_5.csv")["target"]

B6_768_5 = pd.read_csv(f"{TF_SUBS}/B6_768_5.csv")["target"]

B5_768_5 = pd.read_csv(f"{TF_SUBS}/B5_768_5.csv")["target"]

B4_768_5 = pd.read_csv(f"{TF_SUBS}/B4_768_5.csv")["target"]

B3_768_5 = pd.read_csv(f"{TF_SUBS}/B3_768_5.csv")["target"]

B2_768_5 = pd.read_csv(f"{TF_SUBS}/B2_768_5.csv")["target"]

B1_768_5 = pd.read_csv(f"{TF_SUBS}/B1_768_5.csv")["target"]



B7_384_15 = pd.read_csv(f"{TF_SUBS}/B7_384_15.csv")["target"]

B6_384_15 = pd.read_csv(f"{TF_SUBS}/B6_384_15.csv")["target"]

B5_384_15 = pd.read_csv(f"{TF_SUBS}/B5_384_15.csv")["target"]
W = opt.coef_



tf_sub = np.mean([

    

    W[0] * B7_512_15,#= pd.read_csv(f"{TF_SUBS}/B7_512_15.csv")["target"]

    W[1] * B6_512_15,# = pd.read_csv(f"{TF_SUBS}/B6_512_15.csv")["target"]

    W[2] * B5_512_15,# = pd.read_csv(f"{TF_SUBS}/B5_512_15.csv")["target"]



    W[3] * B7_512_10,# = pd.read_csv(f"{TF_SUBS}/B7_512_10.csv")["target"]

    W[4] * B6_512_10,# = pd.read_csv(f"{TF_SUBS}/B6_512_10.csv")["target"]

    W[5] * B5_512_10,# = pd.read_csv(f"{TF_SUBS}/B5_512_10.csv")["target"]

    W[6] * B4_512_10,# = pd.read_csv(f"{TF_SUBS}/B4_512_10.csv")["target"]

    W[7] * B3_512_10,# = pd.read_csv(f"{TF_SUBS}/B3_512_10.csv")["target"]

    W[8] * B2_512_10,# = pd.read_csv(f"{TF_SUBS}/B2_512_10.csv")["target"]

    W[9] * B1_512_10,# = pd.read_csv(f"{TF_SUBS}/B1_512_10.csv")["target"]



    W[10] * B7_768_5,# = pd.read_csv(f"{TF_SUBS}/B7_768_5.csv")["target"]

    W[11] * B6_768_5,# = pd.read_csv(f"{TF_SUBS}/B6_768_5.csv")["target"]

    W[12] * B5_768_5,# = pd.read_csv(f"{TF_SUBS}/B5_768_5.csv")["target"]

    W[13] * B4_768_5,# = pd.read_csv(f"{TF_SUBS}/B4_768_5.csv")["target"]

    W[14] * B3_768_5,# = pd.read_csv(f"{TF_SUBS}/B3_768_5.csv")["target"]

    W[15] * B2_768_5,# = pd.read_csv(f"{TF_SUBS}/B2_768_5.csv")["target"]

    W[16] * B1_768_5,# = pd.read_csv(f"{TF_SUBS}/B1_768_5.csv")["target"]



    W[17] * B7_384_15,# = pd.read_csv(f"{TF_SUBS}/B7_384_15.csv")["target"]

    W[18] * B6_384_15,# = pd.read_csv(f"{TF_SUBS}/B6_384_15.csv")["target"]

    W[19] * B5_384_15,# = pd.read_csv(f"{TF_SUBS}/B5_384_15.csv")["target"]



], axis=0)
sample_submission["target"] = tf_sub

sample_submission.to_csv("Final_TF.csv", index=False)
train_df = pd.merge(offs_final_df, offs_final_tf_df, how="left", left_on=["image_name", "target"], right_on=["image_name", "target"])



train_cols = [

    'b5', 'b5_ns', 'b5_ds', 'b5_ds_10', 'b5_f_ds',

       'tf_b5', 'tf_b5_v1', 'b6', 'b6_2', 'b6_10', 'tf_b6', 'b4', 'b4_f',

       'd201', 'd201_ns', 'd201_512', 'd201_512_ds', 'tf_d201', 'b7_224_ds',

       'tf_b7', 'b7_512_ds', 'dpn92_ds', 'sk50_ds', 

       #'meta_sub', 'meta_sub_2',

       'b7_512_15', 'b6_512_15', 'b5_512_15', 'b7_512_10', 'b6_512_10',

       'b5_512_10', 'b4_512_10', 'b3_512_10', 'b2_512_10', 'b1_512_10',

       'b7_768_5', 'b6_768_5', 'b5_768_5', 'b4_768_5', 'b3_768_5', 'b2_768_5',

       'b1_768_5', 'b7_384_15', 'b6_384_15', 'b5_384_15'

]



X = train_df[train_cols]

y = train_df["target"]
opt = OptimizeAUC()

opt.fit(X, y)
opt.coef_
W = opt.coef_



pytorch_tf_sub = np.mean([

         

        # b5

    W[0] * b5_512_5, #    = pd.read_csv(f"{SUB_DIR}/Final_B5_512_456_5.csv")["target"]

    W[1] * b5_512_5_ns,# = pd.read_csv(f"{SUB_DIR}/Final_NS_E5_512_456_5.csv")["target"]

    W[2] * b5_512_5_ds,# = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_5.csv")["target"]

    W[3] * b5_512_10_ds,# = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_10.csv")["target"]

    W[4] * b5_512_f_ds, #= pd.read_csv(f"{SUB_DIR}/F_Final_DS_B5_512_456_5.csv")["target"]

    W[5] * b5_512_5_tf, #= pd.read_csv(f"{SUB_DIR}/TF-B5-512-V.csv")["target"]

    W[6] * b5_512_5_v1, #= pd.read_csv(f"{SUB_DIR}/TF-B5-512.csv")["target"]



        # b6

    W[7] * b6_528_5,#    = pd.read_csv(f"{SUB_DIR}/Final_B6_768_528_5.csv")["target"]

    W[8] * b6_512_2,#    = pd.read_csv(f"{SUB_DIR}/Final_DSO_B6_768_528_2.csv")["target"]

    W[9] * b6_512_10,#   = pd.read_csv(f"{SUB_DIR}/Final_DS_B6_768_528_10.csv")["target"]

    W[10] * b6_512_5_tf,# = pd.read_csv(f"{SUB_DIR}/TF-B6-512-5.csv")["target"]



        #b4

    W[11] * b4_300_10,#      = pd.read_csv(f"{SUB_DIR}/Final_B4_512_380_10.csv")["target"]

    W[12] * b4_f,#           = pd.read_csv(f"{SUB_DIR}/F_Final_DS_B4_512_380_5.csv")["target"]



        # d201

    W[13] * d201_224_5,#     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_224_5.csv")["target"]

    W[14] * d201_224_5_ns,#  = pd.read_csv(f"{SUB_DIR}/Final_NS_D201_512_224_5.csv")["target"]

    W[15] * d201_512_5,#     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_512_5.csv")["target"]

    W[16] * d201_512_5_ds,#  = pd.read_csv(f"{SUB_DIR}/Final_DS_D201_512_512_5.csv")["target"]

    W[17] * d201_512_5_tf,#  = pd.read_csv(f"{SUB_DIR}/TF-D201-512-5.csv")["target"]



        # b7

    W[18] * b7_224_ds,#      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_224_5.csv")["target"]

    W[19] * b7_512_ds,#      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_512_5.csv")["target"]

    W[20] * b7_512_tf,#      = pd.read_csv(f"{SUB_DIR}/TF-B7-512-5.csv")["target"]





        # dpn92

    W[21] * dpn92_ds,#       = pd.read_csv(f"{SUB_DIR}/Final_DS_dpn92_512_512_5.csv")["target"]



        # sk50

    W[22] * sk50_ds,#        = pd.read_csv(f"{SUB_DIR}/Final_DS_sk50_512_512_5.csv")["target"]



        # meta

    #W[23] *  meta_1,# = pd.read_csv(f"{SUB_DIR}/tabular_test_prediction.csv")["target"]

    #W[24] *  meta_2,# = pd.read_csv(f"{SUB_DIR}/tabular2_test_prediction .csv")["target"]





    W[23] * B7_512_15,#= pd.read_csv(f"{TF_SUBS}/B7_512_15.csv")["target"]

    W[24] * B6_512_15,# = pd.read_csv(f"{TF_SUBS}/B6_512_15.csv")["target"]

    W[25] * B5_512_15,# = pd.read_csv(f"{TF_SUBS}/B5_512_15.csv")["target"]



    W[26] * B7_512_10,# = pd.read_csv(f"{TF_SUBS}/B7_512_10.csv")["target"]

    W[27] * B6_512_10,# = pd.read_csv(f"{TF_SUBS}/B6_512_10.csv")["target"]

    W[38] * B5_512_10,# = pd.read_csv(f"{TF_SUBS}/B5_512_10.csv")["target"]

    W[39] * B4_512_10,# = pd.read_csv(f"{TF_SUBS}/B4_512_10.csv")["target"]

    W[30] * B3_512_10,# = pd.read_csv(f"{TF_SUBS}/B3_512_10.csv")["target"]

    W[31] * B2_512_10,# = pd.read_csv(f"{TF_SUBS}/B2_512_10.csv")["target"]

    W[32] * B1_512_10,# = pd.read_csv(f"{TF_SUBS}/B1_512_10.csv")["target"]



    W[33] * B7_768_5,# = pd.read_csv(f"{TF_SUBS}/B7_768_5.csv")["target"]

    W[34] * B6_768_5,# = pd.read_csv(f"{TF_SUBS}/B6_768_5.csv")["target"]

    W[35] * B5_768_5,# = pd.read_csv(f"{TF_SUBS}/B5_768_5.csv")["target"]

    W[36] * B4_768_5,# = pd.read_csv(f"{TF_SUBS}/B4_768_5.csv")["target"]

    W[37] * B3_768_5,# = pd.read_csv(f"{TF_SUBS}/B3_768_5.csv")["target"]

    W[38] * B2_768_5,# = pd.read_csv(f"{TF_SUBS}/B2_768_5.csv")["target"]

    W[39] * B1_768_5,# = pd.read_csv(f"{TF_SUBS}/B1_768_5.csv")["target"]



    W[40] * B7_384_15,# = pd.read_csv(f"{TF_SUBS}/B7_384_15.csv")["target"]

    W[41] * B6_384_15,# = pd.read_csv(f"{TF_SUBS}/B6_384_15.csv")["target"]

    W[42] * B5_384_15,# = pd.read_csv(f"{TF_SUBS}/B5_384_15.csv")["target"]





], axis=0)
sample_submission["target"] = pytorch_tf_sub

sample_submission.to_csv("Final_Pytorch_TF.csv", index=False)
W = [1] * 50



simple_avg = np.mean([

         

        # b5

    W[0] * b5_512_5, #    = pd.read_csv(f"{SUB_DIR}/Final_B5_512_456_5.csv")["target"]

    W[1] * b5_512_5_ns,# = pd.read_csv(f"{SUB_DIR}/Final_NS_E5_512_456_5.csv")["target"]

    W[2] * b5_512_5_ds,# = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_5.csv")["target"]

    W[3] * b5_512_10_ds,# = pd.read_csv(f"{SUB_DIR}/Final_DS_B5_512_456_10.csv")["target"]

    W[4] * b5_512_f_ds, #= pd.read_csv(f"{SUB_DIR}/F_Final_DS_B5_512_456_5.csv")["target"]

    W[5] * b5_512_5_tf, #= pd.read_csv(f"{SUB_DIR}/TF-B5-512-V.csv")["target"]

    W[6] * b5_512_5_v1, #= pd.read_csv(f"{SUB_DIR}/TF-B5-512.csv")["target"]



        # b6

    W[7] * b6_528_5,#    = pd.read_csv(f"{SUB_DIR}/Final_B6_768_528_5.csv")["target"]

    W[8] * b6_512_2,#    = pd.read_csv(f"{SUB_DIR}/Final_DSO_B6_768_528_2.csv")["target"]

    W[9] * b6_512_10,#   = pd.read_csv(f"{SUB_DIR}/Final_DS_B6_768_528_10.csv")["target"]

    W[10] * b6_512_5_tf,# = pd.read_csv(f"{SUB_DIR}/TF-B6-512-5.csv")["target"]



        #b4

    W[11] * b4_300_10,#      = pd.read_csv(f"{SUB_DIR}/Final_B4_512_380_10.csv")["target"]

    W[12] * b4_f,#           = pd.read_csv(f"{SUB_DIR}/F_Final_DS_B4_512_380_5.csv")["target"]



        # d201

    W[13] * d201_224_5,#     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_224_5.csv")["target"]

    W[14] * d201_224_5_ns,#  = pd.read_csv(f"{SUB_DIR}/Final_NS_D201_512_224_5.csv")["target"]

    W[15] * d201_512_5,#     = pd.read_csv(f"{SUB_DIR}/Final_D201_512_512_5.csv")["target"]

    W[16] * d201_512_5_ds,#  = pd.read_csv(f"{SUB_DIR}/Final_DS_D201_512_512_5.csv")["target"]

    W[17] * d201_512_5_tf,#  = pd.read_csv(f"{SUB_DIR}/TF-D201-512-5.csv")["target"]



        # b7

    W[18] * b7_224_ds,#      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_224_5.csv")["target"]

    W[19] * b7_512_ds,#      = pd.read_csv(f"{SUB_DIR}/Final_DS_B7_512_512_5.csv")["target"]

    W[20] * b7_512_tf,#      = pd.read_csv(f"{SUB_DIR}/TF-B7-512-5.csv")["target"]





        # dpn92

    W[21] * dpn92_ds,#       = pd.read_csv(f"{SUB_DIR}/Final_DS_dpn92_512_512_5.csv")["target"]



        # sk50

    W[22] * sk50_ds,#        = pd.read_csv(f"{SUB_DIR}/Final_DS_sk50_512_512_5.csv")["target"]



        # meta

    #W[23] *  meta_1,# = pd.read_csv(f"{SUB_DIR}/tabular_test_prediction.csv")["target"]

    #W[24] *  meta_2,# = pd.read_csv(f"{SUB_DIR}/tabular2_test_prediction .csv")["target"]





    W[23] * B7_512_15,#= pd.read_csv(f"{TF_SUBS}/B7_512_15.csv")["target"]

    W[24] * B6_512_15,# = pd.read_csv(f"{TF_SUBS}/B6_512_15.csv")["target"]

    W[25] * B5_512_15,# = pd.read_csv(f"{TF_SUBS}/B5_512_15.csv")["target"]



    W[26] * B7_512_10,# = pd.read_csv(f"{TF_SUBS}/B7_512_10.csv")["target"]

    W[27] * B6_512_10,# = pd.read_csv(f"{TF_SUBS}/B6_512_10.csv")["target"]

    W[38] * B5_512_10,# = pd.read_csv(f"{TF_SUBS}/B5_512_10.csv")["target"]

    W[39] * B4_512_10,# = pd.read_csv(f"{TF_SUBS}/B4_512_10.csv")["target"]

    W[30] * B3_512_10,# = pd.read_csv(f"{TF_SUBS}/B3_512_10.csv")["target"]

    W[31] * B2_512_10,# = pd.read_csv(f"{TF_SUBS}/B2_512_10.csv")["target"]

    W[32] * B1_512_10,# = pd.read_csv(f"{TF_SUBS}/B1_512_10.csv")["target"]



    W[33] * B7_768_5,# = pd.read_csv(f"{TF_SUBS}/B7_768_5.csv")["target"]

    W[34] * B6_768_5,# = pd.read_csv(f"{TF_SUBS}/B6_768_5.csv")["target"]

    W[35] * B5_768_5,# = pd.read_csv(f"{TF_SUBS}/B5_768_5.csv")["target"]

    W[36] * B4_768_5,# = pd.read_csv(f"{TF_SUBS}/B4_768_5.csv")["target"]

    W[37] * B3_768_5,# = pd.read_csv(f"{TF_SUBS}/B3_768_5.csv")["target"]

    W[38] * B2_768_5,# = pd.read_csv(f"{TF_SUBS}/B2_768_5.csv")["target"]

    W[39] * B1_768_5,# = pd.read_csv(f"{TF_SUBS}/B1_768_5.csv")["target"]



    W[40] * B7_384_15,# = pd.read_csv(f"{TF_SUBS}/B7_384_15.csv")["target"]

    W[41] * B6_384_15,# = pd.read_csv(f"{TF_SUBS}/B6_384_15.csv")["target"]

    W[42] * B5_384_15,# = pd.read_csv(f"{TF_SUBS}/B5_384_15.csv")["target"]





], axis=0)
sample_submission["target"] = simple_avg

sample_submission.to_csv("Final_simple_avg.csv", index=False)