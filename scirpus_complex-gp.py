from statsmodels.robust import mad

import matplotlib.pyplot as plt

from scipy.signal import butter

from scipy import signal

from scipy.signal import savgol_filter

import seaborn as sns

from sklearn import *

import pandas as pd 

import numpy as np

import warnings

import scipy

import pywt

import os

import gc

from sklearn.metrics import f1_score

plt.style.use('ggplot')

sns.set_style('darkgrid')

class GP:

    def __init__(self):

        self.classes = 11

        self.class_names = [ 'class_0',

                             'class_1',

                             'class_2',

                             'class_3',

                             'class_4',

                             'class_5',

                             'class_6',

                             'class_7',

                             'class_8',

                             'class_9',

                             'class_10']





    def GrabPredictions(self, data):

        oof_preds = np.zeros((len(data), len(self.class_names)))

        oof_preds[:,0] = self.GP_class_0(data)

        oof_preds[:,1] = self.GP_class_1(data)

        oof_preds[:,2] = self.GP_class_2(data)

        oof_preds[:,3] = self.GP_class_3(data)

        oof_preds[:,4] = self.GP_class_4(data)

        oof_preds[:,5] = self.GP_class_5(data)

        oof_preds[:,6] = self.GP_class_6(data)

        oof_preds[:,7] = self.GP_class_7(data)

        oof_preds[:,8] = self.GP_class_8(data)

        oof_preds[:,9] = self.GP_class_9(data)

        oof_preds[:,10] = self.GP_class_10(data)

        oof_df = pd.DataFrame(oof_preds, columns=self.class_names)

        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)

        return oof_df



    def Output(self, p):

        return 1.0/(1.0+np.exp(-p))

   

    def GP_class_0(self,data):

        return self.Output( -1.394145 +

                            0.100000*np.tanh(np.real(((((data["minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"]))) + (((((data["minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"]))) + (data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_minbatch_msignal"]) + (((((((((((data["abs_minbatch_msignal"]) + (np.cos(((-((((data["minbatch_msignal"]) - (((((data["mean_abs_chgbatch_msignal"]) * 2.0)) * 2.0))))))))))) * 2.0)) - (data["abs_maxbatch_slices2"]))) * 2.0)) * 2.0)))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate((((((((((((data["minbatch_slices2_msignal"]) + (((data["abs_minbatch_slices2_msignal"]) * 2.0)))/2.0)) + ((((data["minbatch_slices2_msignal"]) + ((((data["minbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))/2.0)))) * 2.0)) + (data["minbatch_slices2_msignal"]))) * 2.0)))) +

                            0.100000*np.tanh(np.real((((((complex(3.0)) + ((((data["abs_minbatch_slices2_msignal"]) + ((((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) + (data["abs_maxbatch"]))/2.0)) + (((((((((((np.conjugate(data["minbatch_slices2_msignal"])) * 2.0)) + (data["maxtominbatch_slices2_msignal"]))/2.0)) * 2.0)) + (data["abs_minbatch_slices2_msignal"]))/2.0)))))/2.0)))/2.0)) + (((((data["abs_maxbatch"]) / 2.0)) * (data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((np.where(np.abs(((data["mean_abs_chgbatch_msignal"]) + (np.conjugate(data["meanbatch_slices2"])))) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_msignal"]))) + (data["minbatch_msignal"]))) - (np.sin((data["mean_abs_chgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(2.0)) <= np.abs(((data["abs_minbatch_msignal"]) + ((((data["mean_abs_chgbatch_slices2_msignal"]) + ((((np.sin((np.conjugate(data["mean_abs_chgbatch_slices2_msignal"])))) + (data["minbatch_msignal"]))/2.0)))/2.0)))),complex(1.), complex(0.) )) + (data["abs_minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) + (((((((data["minbatch_msignal"]) + (np.conjugate(((data["minbatch_msignal"]) + (data["maxtominbatch_msignal"])))))) + (data["mean_abs_chgbatch_slices2_msignal"]))) + (complex(0,1)*np.conjugate(((data["stdbatch_msignal"]) / 2.0)))))))) +

                            0.100000*np.tanh(np.real((((((((((data["abs_minbatch_slices2_msignal"]) / 2.0)) + ((((data["abs_minbatch_msignal"]) + (data["abs_minbatch_msignal"]))/2.0)))/2.0)) + (((data["abs_minbatch_slices2_msignal"]) * 2.0)))) + (((((((((((np.conjugate(data["abs_minbatch_msignal"])) + (data["minbatch_slices2_msignal"]))/2.0)) + (((data["abs_minbatch_slices2_msignal"]) * 2.0)))/2.0)) + (data["abs_minbatch_slices2_msignal"]))) + (data["abs_minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((((data["maxtominbatch_slices2_msignal"]) - ((-((data["minbatch_msignal"])))))) - (np.cos((np.conjugate((((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) + (((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )) - (complex(1.0)))))/2.0))))))) + (data["abs_minbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real((((((complex(3.0)) + ((((((data["minbatch_msignal"]) + (((((((((data["minbatch_msignal"]) + ((((data["minbatch_msignal"]) + ((((data["stdbatch_slices2_msignal"]) + (np.sin((data["medianbatch_slices2"]))))/2.0)))/2.0)))) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0)))) + (data["minbatch_msignal"]))/2.0)))/2.0)) + (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) + (complex(0.0))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) - (((np.where(np.abs(((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) + (np.where(np.abs((((data["rangebatch_slices2_msignal"]) + (np.where(np.abs((((data["stdbatch_slices2_msignal"]) + (data["minbatch"]))/2.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) )))/2.0)) > np.abs(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) > np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )) - (np.cos((data["abs_avgbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(((complex(2.0)) - (data["rangebatch_slices2"]))) > np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )) + (((((data["abs_minbatch_msignal"]) + (data["abs_minbatch_slices2_msignal"]))) - (np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + (((data["minbatch_slices2_msignal"]) + (((complex(3.0)) - (np.where(np.abs(complex(8.81193161010742188)) <= np.abs(((np.where(np.abs(data["maxtominbatch_msignal"]) <= np.abs(((data["stdbatch_msignal"]) / 2.0)),complex(1.), complex(0.) )) + (((data["minbatch_slices2_msignal"]) + (((data["meanbatch_slices2_msignal"]) - (data["minbatch_slices2_msignal"]))))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch_msignal"]) - ((-((((data["abs_minbatch_msignal"]) - (data["stdbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(((data["abs_avgbatch_slices2_msignal"]) + (np.cos((np.sin((data["abs_avgbatch_msignal"]))))))) > np.abs(np.where(np.abs(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(np.conjugate(complex(5.0))),complex(1.), complex(0.) )) > np.abs((((((data["abs_avgbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)) * 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)) + (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch_slices2"]) * (((complex(3.0)) + (((data["minbatch_msignal"]) * 2.0)))))) + (np.where(np.abs((((((data["abs_maxbatch_slices2"]) + (np.conjugate(complex(1.0))))) + (((((data["minbatch_msignal"]) * 2.0)) * 2.0)))/2.0)) <= np.abs(((complex(3.0)) + (np.conjugate((((((complex(3.0)) * 2.0)) + (data["minbatch_msignal"]))/2.0))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((((np.conjugate((((((data["minbatch_slices2_msignal"]) + (np.tanh((((np.conjugate(data["minbatch_slices2_msignal"])) + (complex(3.0)))))))/2.0)) * (complex(3.0))))) + (((np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(np.where(np.abs(np.sin((data["maxbatch_slices2"]))) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)))) * (complex(3.0))))) + (complex(3.0))))) +

                            0.100000*np.tanh(np.real(((((((data["minbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))) / (np.cos((np.cos((data["minbatch_slices2_msignal"]))))))) + ((((((((complex(9.0)) + (complex(9.0)))/2.0)) / (np.cos((data["minbatch_slices2_msignal"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) + (data["abs_minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) + (np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) > np.abs(((((data["mean_abs_chgbatch_msignal"]) / 2.0)) + ((((data["maxtominbatch_msignal"]) + (data["abs_minbatch_slices2_msignal"]))/2.0)))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((((((((((np.conjugate(data["minbatch_msignal"])) + (((complex(0,1)*np.conjugate(((np.cos((np.cos((complex(0.0)))))) * 2.0))) + (data["maxbatch_msignal"]))))) * 2.0)) * 2.0)) / 2.0)) + (data["minbatch_msignal"]))) + (complex(2.0))))) +

                            0.100000*np.tanh(np.real((((((data["medianbatch_msignal"]) + (np.tanh((data["abs_avgbatch_msignal"]))))/2.0)) + (((((np.cos((data["minbatch_slices2_msignal"]))) * 2.0)) - (np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch_slices2_msignal"]) + ((((data["minbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0)))) + ((((data["minbatch_slices2_msignal"]) + (np.sin((((data["abs_minbatch_slices2"]) * 2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((((data["abs_maxbatch_slices2_msignal"]) / (np.cos((data["minbatch_msignal"]))))) - (complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"])))) + (data["abs_maxbatch_slices2_msignal"]))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + (((complex(3.0)) - (((((np.where(np.abs(((complex(2.0)) / 2.0)) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) / 2.0)) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"]))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_slices2_msignal"]))) * (((((complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])) + (complex(3.0)))) + ((((((((data["abs_minbatch_slices2_msignal"]) + (np.cos(((-((complex(6.03218221664428711))))))))) * 2.0)) + (data["abs_minbatch_slices2_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(data["maxtominbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((data["minbatch_slices2_msignal"]) + (np.conjugate(((data["abs_minbatch_slices2"]) / 2.0))))) + (((np.where(np.abs(complex(9.0)) > np.abs(((data["minbatch_slices2_msignal"]) * (np.sin((complex(3.0)))))),complex(1.), complex(0.) )) / (np.cos((data["minbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2_msignal"]) + (((data["maxbatch_msignal"]) + (((((complex(2.0)) + ((((((data["abs_avgbatch_slices2"]) + (complex(0,1)*np.conjugate(data["signal"])))/2.0)) / 2.0)))) + (data["maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_msignal"]) * 2.0)) + (data["abs_minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((((((data["minbatch_msignal"]) + (np.cos((((data["minbatch_msignal"]) / 2.0)))))) / 2.0)) + (((np.cos((((data["minbatch_msignal"]) / 2.0)))) * 2.0)))) + (((((((np.cos((((data["minbatch_slices2_msignal"]) / 2.0)))) * 2.0)) + (data["minbatch_slices2_msignal"]))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((complex(8.0)) + ((((((((((-((((((data["stdbatch_slices2"]) / 2.0)) * 2.0))))) - (data["rangebatch_slices2"]))) - (data["rangebatch_slices2"]))) + (((complex(8.0)) + (complex(8.0)))))) + (data["mean_abs_chgbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_msignal"]) - (((np.sin((np.sin((complex(0.0)))))) - (data["minbatch"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) + (((data["signal_shift_+1_msignal"]) - ((-((((np.tanh((data["rangebatch_slices2"]))) + (data["minbatch_slices2"]))))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["abs_minbatch_slices2_msignal"]) + (((data["minbatch_msignal"]) + (((((((((((complex(1.0)) * 2.0)) + (((np.where(np.abs(np.where(np.abs(data["minbatch_slices2"]) > np.abs(((data["abs_minbatch_slices2_msignal"]) - (((data["abs_minbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) + (complex(1.0)))))/2.0)) + (complex(2.0)))/2.0)) + (data["meanbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((((((data["abs_maxbatch_slices2"]) * (np.conjugate(np.sin((data["abs_maxbatch_msignal"])))))) * 2.0)) + (((np.where(np.abs(complex(1.0)) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0))))) - ((((((((np.sin((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(np.tanh((complex(8.28935050964355469)))),complex(1.), complex(0.) )))) * 2.0)) / 2.0)) + (data["minbatch_slices2_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(data["maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((complex(2.90938210487365723)) + (((((((((complex(0,1)*np.conjugate(np.cos((np.cos((data["abs_minbatch_msignal"])))))) * (((data["abs_maxbatch"]) - (data["minbatch_msignal"]))))) + (data["minbatch_slices2_msignal"]))) / 2.0)) * (complex(3.0))))))) +

                            0.100000*np.tanh(np.real((((data["minbatch_slices2_msignal"]) + (np.conjugate(data["abs_maxbatch_msignal"])))/2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((data["minbatch_slices2_msignal"]) - (((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(complex(0.0)),complex(1.), complex(0.) )) + (data["minbatch_slices2_msignal"]))))) > np.abs(((np.cos((np.sin((np.conjugate(np.where(np.abs(complex(2.0)) > np.abs(((complex(12.27592658996582031)) - (data["stdbatch_slices2"]))),complex(1.), complex(0.) ))))))) / 2.0)),complex(1.), complex(0.) )) + (data["minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_msignal"]) + (((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((np.sin((((data["signal_shift_-1"]) - (data["minbatch_slices2"]))))) - (complex(3.0)))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) + ((-((np.sin((((data["abs_maxbatch_slices2_msignal"]) - (complex(3.0)))))))))))) +

                            0.100000*np.tanh(np.real((((((np.where(np.abs(np.where(np.abs(complex(-1.0)) <= np.abs(np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs((-((np.cos((np.cos((data["abs_minbatch_slices2_msignal"])))))))),complex(1.), complex(0.) )) / (((np.cos((data["minbatch_slices2_msignal"]))) / 2.0)))) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["minbatch_slices2_msignal"]) + (np.conjugate(((data["minbatch_slices2_msignal"]) + (complex(1.0))))))))) + (np.conjugate(((data["minbatch_slices2_msignal"]) + (complex(1.0))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((data["rangebatch_slices2"]))) * ((((((np.tanh((np.where(np.abs(complex(5.0)) <= np.abs(complex(5.0)),complex(1.), complex(0.) )))) + (complex(5.51956892013549805)))/2.0)) + ((((data["abs_avgbatch_slices2"]) + (np.conjugate(((complex(8.0)) + (complex(10.0))))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((((np.cos((data["minbatch_msignal"]))) * (((((data["rangebatch_slices2"]) * 2.0)) * 2.0)))) - ((-((((((complex(0,1)*np.conjugate(np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))) * (np.cos((data["minbatch_msignal"]))))) / 2.0))))))) + ((-((complex(-3.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(((np.cos((data["rangebatch_slices2"]))) + (((((data["abs_avgbatch_slices2"]) / 2.0)) / (np.cos((complex(0,1)*np.conjugate(np.cos((((data["abs_avgbatch_slices2"]) / (data["abs_avgbatch_slices2"])))))))))))),complex(1.), complex(0.) )) + (((data["minbatch_slices2_msignal"]) / (np.cos((data["rangebatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) + ((((np.tanh((((np.where(np.abs(data["minbatch_msignal"]) <= np.abs(complex(3.0)),complex(1.), complex(0.) )) / 2.0)))) + (((np.conjugate(data["meanbatch_slices2_msignal"])) + (((np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )) + ((((((data["meanbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))) + (data["abs_avgbatch_slices2"]))/2.0)))))))/2.0))))) +

                            0.100000*np.tanh(np.real((((data["abs_maxbatch_slices2_msignal"]) + ((-(((((data["maxbatch_slices2"]) + (np.sin((((np.sin((data["abs_maxbatch_msignal"]))) + (data["abs_minbatch_msignal"]))))))/2.0))))))/2.0))) +

                            0.100000*np.tanh(np.real((((data["maxtominbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(((np.where(np.abs(complex(1.0)) <= np.abs(np.tanh((data["minbatch_slices2_msignal"]))),complex(1.), complex(0.) )) * 2.0))))/2.0))) +

                            0.100000*np.tanh(np.real(((((complex(2.0)) + (np.cos((np.where(np.abs(data["minbatch_msignal"]) <= np.abs((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(np.sin((((data["abs_avgbatch_slices2"]) - (((data["minbatch"]) + (np.cos((data["minbatch_slices2"]))))))))),complex(1.), complex(0.) )) + (((complex(2.0)) / 2.0)))/2.0)),complex(1.), complex(0.) )))))) + (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) / (np.where(np.abs(data["abs_maxbatch"]) > np.abs((((np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(((np.where(np.abs((((complex(-2.0)) + (((data["maxtominbatch_slices2"]) - (data["abs_maxbatch_slices2_msignal"]))))/2.0)) > np.abs(np.conjugate(data["maxbatch_slices2_msignal"])),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) )) + (np.cos((np.tanh((complex(3.0)))))))/2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) - (np.cos(((((data["abs_maxbatch"]) + (data["maxbatch_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((complex(3.0)) + (((np.conjugate(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) ))) - (((data["rangebatch_slices2"]) - (np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(complex(2.0)),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((complex(6.0)) * (np.sin((((np.sin((data["rangebatch_slices2"]))) - (((np.sin((np.sin((np.conjugate(complex(0.0))))))) / 2.0))))))))) +

                            0.100000*np.tanh(np.real((((((-(((-((np.conjugate(np.cos((np.cos((data["maxbatch_slices2"])))))))))))) - ((-((complex(0,1)*np.conjugate(((data["medianbatch_msignal"]) * 2.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + (((np.conjugate(np.cos((np.tanh((((data["abs_maxbatch"]) - (np.tanh((((data["rangebatch_slices2"]) - (data["minbatch"])))))))))))) + (((complex(4.0)) + (np.cos((data["minbatch"])))))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) / (np.cos((((((data["minbatch_msignal"]) - ((((data["abs_maxbatch_slices2"]) + (np.cos((data["minbatch_slices2_msignal"]))))/2.0)))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((complex(1.0)) - (((((data["meanbatch_msignal"]) - (((data["abs_avgbatch_slices2"]) - (((data["stdbatch_slices2_msignal"]) + (((data["signal_shift_-1"]) - (np.sin((data["abs_maxbatch_slices2"]))))))))))) - (((data["minbatch_msignal"]) + (np.cos((data["minbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(data["maxbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((((np.sin((data["abs_maxbatch_msignal"]))) * (((data["rangebatch_slices2"]) * (np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(np.where(np.abs(data["stdbatch_slices2"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.tanh((data["stdbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((data["minbatch"]))) > np.abs((((((-((data["abs_minbatch_msignal"])))) * (np.where(np.abs(complex(7.74514865875244141)) <= np.abs(np.where(np.abs(data["stdbatch_slices2"]) <= np.abs(np.conjugate(((complex(3.0)) + (np.sin((complex(0,1)*np.conjugate(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )))))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((np.tanh((((((((np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )) * (np.tanh((complex(2.0)))))) / 2.0)) * 2.0)))) + (np.cos((((((data["minbatch"]) * 2.0)) - (((data["maxtominbatch_slices2_msignal"]) / 2.0)))))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + ((((complex(6.56695032119750977)) + (((data["minbatch_msignal"]) - ((((data["maxbatch_slices2_msignal"]) + (np.where(np.abs(((data["minbatch_msignal"]) + (np.where(np.abs(data["maxtominbatch"]) <= np.abs((((np.where(np.abs(data["rangebatch_msignal"]) > np.abs(((data["minbatch_msignal"]) - ((((complex(-3.0)) + (data["minbatch_msignal"]))/2.0)))),complex(1.), complex(0.) )) + (data["rangebatch_msignal"]))/2.0)),complex(1.), complex(0.) )))) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((data["maxbatch_msignal"]) - ((((np.cos((data["minbatch_msignal"]))) + (((np.cos((data["minbatch_msignal"]))) / (np.cos((data["minbatch_msignal"]))))))/2.0))))) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) - (((((complex(0.0)) + ((((((np.conjugate(((complex(13.34256935119628906)) * 2.0))) + ((((data["minbatch_msignal"]) + (((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) * 2.0)))/2.0)))) + (data["abs_avgbatch_slices2"]))/2.0)))) * (data["signal_shift_-1_msignal"])))))) +

                            0.100000*np.tanh(np.real((((complex(0,1)*np.conjugate((((-((data["signal_shift_-1"])))) - ((-((((np.cos((data["signal_shift_-1_msignal"]))) - (data["minbatch_slices2_msignal"]))))))))) + (((np.conjugate(((data["signal_shift_-1_msignal"]) - (np.cos((data["minbatch_slices2_msignal"])))))) * (complex(-3.0)))))/2.0))) +

                            0.100000*np.tanh(np.real((((((complex(0,1)*np.conjugate(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(((((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )) / (complex(3.0)))) / 2.0)),complex(1.), complex(0.) ))) + (data["meanbatch_slices2_msignal"]))/2.0)) + (data["minbatch"])))) +

                            0.100000*np.tanh(np.real((((data["minbatch_msignal"]) + (((complex(2.0)) + (np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(((data["meanbatch_slices2_msignal"]) + (data["minbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.tanh((((np.tanh((np.tanh((np.sin((np.sin((data["rangebatch_slices2_msignal"]))))))))) + (data["rangebatch_slices2"]))))) + (np.tanh((np.sin((data["rangebatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["rangebatch_slices2"]) + (complex(0,1)*np.conjugate(data["rangebatch_slices2"])))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["minbatch_msignal"]) + (np.sin((np.cos((np.cos((np.cos((np.cos((data["signal_shift_-1"]))))))))))))) + ((((data["signal_shift_+1_msignal"]) + (np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(complex(3.0)),complex(1.), complex(0.) )))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((np.where(np.abs(np.where(np.abs(np.cos((data["meanbatch_slices2_msignal"]))) > np.abs(np.conjugate(((complex(0.01358986180275679)) - (((data["abs_avgbatch_slices2_msignal"]) * 2.0))))),complex(1.), complex(0.) )) > np.abs(((data["abs_minbatch_slices2"]) / 2.0)),complex(1.), complex(0.) )) + (complex(-3.0)))/2.0)) - ((-((data["abs_maxbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((data["signal_shift_-1"]) * (np.sin(((((((complex(1.0)) + (data["maxtominbatch_slices2_msignal"]))) + (data["abs_avgbatch_slices2_msignal"]))/2.0)))))) - (np.conjugate((-((((data["signal_shift_+1_msignal"]) + (complex(-2.0)))))))))) <= np.abs(complex(0,1)*np.conjugate(np.where(np.abs(complex(1.0)) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["rangebatch_slices2"]) + (np.where(np.abs(np.sin((data["stdbatch_slices2"]))) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((data["abs_avgbatch_msignal"]))) * (complex(2.0))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + ((((complex(3.0)) + (((data["minbatch_slices2_msignal"]) + ((((complex(3.0)) + ((((complex(3.0)) + (np.where(np.abs(((complex(-3.0)) / (complex(2.0)))) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )))/2.0)))/2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.sin(((((((((((complex(12.87454509735107422)) / 2.0)) / 2.0)) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) * (((data["meanbatch_slices2_msignal"]) - (data["rangebatch_slices2"]))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(2.0)) / (np.cos((((np.where(np.abs(np.cos((np.tanh((data["mean_abs_chgbatch_slices2"]))))) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )) - ((((data["rangebatch_slices2"]) + ((-((np.where(np.abs(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.sin((np.tanh((np.where(np.abs(data["maxbatch_msignal"]) > np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )) <= np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))/2.0))))))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_slices2"]) / (((((np.cos((data["abs_avgbatch_slices2_msignal"]))) + (((complex(0,1)*np.conjugate(data["signal"])) + (((data["medianbatch_msignal"]) * (data["signal"]))))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(((complex(3.0)) + (((data["minbatch_msignal"]) + (np.tanh((np.where(np.abs((-((np.where(np.abs(((data["abs_avgbatch_msignal"]) / 2.0)) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))))) > np.abs(complex(3.0)),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + (((data["maxbatch_msignal"]) + (((data["minbatch_slices2_msignal"]) + (((np.sin((((data["maxbatch_msignal"]) + (((np.cos((data["mean_abs_chgbatch_msignal"]))) + (((data["signal_shift_+1_msignal"]) * (data["signal"]))))))))) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(complex(4.0))) +

                            0.100000*np.tanh(np.real((((((((((np.cos((data["abs_avgbatch_slices2_msignal"]))) + (np.cos((data["abs_avgbatch_slices2_msignal"]))))) * 2.0)) + (((data["minbatch"]) + (np.conjugate(data["medianbatch_msignal"])))))/2.0)) - (data["signal_shift_+1"])))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(((((np.where(np.abs(np.sin((((data["signal_shift_-1_msignal"]) * 2.0)))) > np.abs(np.conjugate(np.tanh(((-((data["minbatch_slices2"]))))))),complex(1.), complex(0.) )) - (complex(0,1)*np.conjugate(((data["medianbatch_msignal"]) / 2.0))))) * 2.0))) - (data["signal_shift_-1_msignal"]))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((np.tanh((complex(0,1)*np.conjugate(complex(-2.0))))) - (((((np.conjugate(data["abs_avgbatch_slices2"])) + ((((np.cos((data["signal_shift_+1_msignal"]))) + (np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(((data["minbatch_slices2"]) * (data["signal_shift_+1"]))),complex(1.), complex(0.) )))/2.0)))) * (data["signal_shift_+1_msignal"]))))) - (((data["abs_maxbatch_slices2_msignal"]) * (data["signal_shift_+1_msignal"]))))))) +

                            0.100000*np.tanh(np.real((((((data["abs_avgbatch_slices2"]) + (np.where(np.abs(np.tanh((data["abs_maxbatch_slices2"]))) <= np.abs((((data["rangebatch_msignal"]) + (((data["abs_avgbatch_slices2"]) * 2.0)))/2.0)),complex(1.), complex(0.) )))) + (data["rangebatch_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(((((((((data["minbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2"]))) - (np.cos((((((data["minbatch_slices2_msignal"]) - (np.conjugate(np.tanh((np.tanh((data["medianbatch_slices2"])))))))) * 2.0)))))) * 2.0)) + (((complex(3.96889781951904297)) + (data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.cos((np.conjugate(complex(1.0)))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["maxbatch_msignal"]))) * (((data["abs_avgbatch_slices2_msignal"]) + (np.conjugate(np.cos((np.where(np.abs(((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])) + (data["maxtominbatch_slices2_msignal"]))) <= np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((np.tanh((data["abs_minbatch_slices2"]))))) - (((data["meanbatch_slices2_msignal"]) / 2.0)))) - ((((complex(-3.0)) + ((((complex(-3.0)) + (((np.conjugate(complex(-2.0))) * (((np.sin((data["abs_minbatch_slices2_msignal"]))) / 2.0)))))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(((data["minbatch"]) - (data["mean_abs_chgbatch_slices2"]))) <= np.abs(data["signal"]),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["stdbatch_slices2"]) + (np.cos((np.where(np.abs(data["stdbatch_slices2"]) <= np.abs(np.cos((np.tanh((np.conjugate(data["medianbatch_slices2_msignal"])))))),complex(1.), complex(0.) )))))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((np.sin(((((complex(3.0)) + (data["maxtominbatch_slices2_msignal"]))/2.0)))) + (np.cos((np.cos((((data["maxbatch_msignal"]) + (data["maxbatch_msignal"]))))))))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos(((-((data["minbatch_msignal"])))))) + (((data["signal_shift_-1_msignal"]) * (((np.where(np.abs(complex(0.0)) <= np.abs((((((((-((data["minbatch_msignal"])))) - (data["meanbatch_msignal"]))) * 2.0)) * (np.where(np.abs(complex(0.0)) <= np.abs(complex(-3.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) + (data["minbatch_msignal"]))))))) * (complex(5.02947568893432617))))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2_msignal"]) * (np.sin((data["meanbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((((-((data["signal_shift_+1_msignal"])))) / (((((np.cos((np.where(np.abs(data["maxtominbatch"]) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))) / 2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(((np.tanh((((data["meanbatch_slices2_msignal"]) / 2.0)))) + (((((((((data["abs_avgbatch_msignal"]) - (np.tanh((((data["abs_avgbatch_msignal"]) / 2.0)))))) * 2.0)) * (data["signal_shift_+1_msignal"]))) * (((complex(-3.0)) / 2.0))))))) +

                            0.100000*np.tanh(np.real((((((((((data["abs_maxbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))/2.0)) / 2.0)) + (data["minbatch_msignal"]))) + (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((complex(9.0)) - (np.conjugate(((((data["meanbatch_slices2_msignal"]) * 2.0)) - (((((data["minbatch_slices2_msignal"]) * 2.0)) * (complex(3.0)))))))))) +

                            0.100000*np.tanh(np.real((((((((-((((complex(11.27023792266845703)) - (((((np.sin((data["maxbatch_slices2"]))) * 2.0)) * 2.0))))))) * 2.0)) + (((data["meanbatch_slices2"]) * (((data["meanbatch_slices2"]) * ((((((complex(3.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) * (((((np.sin((data["medianbatch_msignal"]))) * 2.0)) * 2.0)))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((((np.cos((((data["stdbatch_slices2"]) * 2.0)))) * 2.0)) + (((data["abs_maxbatch_msignal"]) + (data["minbatch_slices2_msignal"]))))) + (data["abs_maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((((((np.tanh((data["abs_avgbatch_slices2"]))) + (((((np.conjugate(complex(1.0))) * 2.0)) - (data["stdbatch_slices2_msignal"]))))/2.0)) + (np.tanh((complex(2.0))))))) +

                            0.100000*np.tanh(np.real((((((((data["abs_avgbatch_msignal"]) / 2.0)) * 2.0)) + (((data["abs_maxbatch"]) + (data["rangebatch_slices2"]))))/2.0))) +

                            0.100000*np.tanh(np.real((((((np.conjugate(np.sin((data["signal"])))) * (data["medianbatch_msignal"]))) + (np.where(np.abs((-(((-((np.cos((((data["minbatch_msignal"]) * (np.conjugate(complex(0.0))))))))))))) <= np.abs(((data["medianbatch_msignal"]) - (((data["minbatch_slices2"]) * 2.0)))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2_msignal"]) * ((-((((data["abs_minbatch_msignal"]) - (data["abs_minbatch_slices2"])))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2"]) + (np.sin((data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(data["abs_avgbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin(((((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate((((np.tanh((data["rangebatch_msignal"]))) + (complex(3.0)))/2.0))))/2.0)) + (data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((np.cos((data["stdbatch_slices2"]))) - (np.conjugate(np.tanh((np.cos((((((np.tanh((np.tanh((np.conjugate(data["abs_maxbatch_slices2_msignal"])))))) * 2.0)) - (data["minbatch_slices2_msignal"])))))))))) - (((data["mean_abs_chgbatch_slices2"]) * (np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["signal_shift_+1_msignal"]))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) * (((data["minbatch"]) - (((((data["maxbatch_msignal"]) + (data["maxtominbatch"]))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + (data["maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + ((((((np.cos((((complex(0,1)*np.conjugate(data["maxbatch_msignal"])) + (data["stdbatch_slices2"]))))) * 2.0)) + (np.cos((data["abs_minbatch_msignal"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((data["abs_maxbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(complex(-1.0))) > np.abs(complex(2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (np.conjugate(np.tanh((np.cos((data["abs_avgbatch_slices2_msignal"])))))))/2.0)) + (((np.conjugate(np.cos((((data["stdbatch_slices2"]) + (data["minbatch_slices2_msignal"])))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos(((((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((((complex(0,1)*np.conjugate(np.cos((complex(-1.0))))) + (complex(0,1)*np.conjugate(np.cos((((np.cos((data["signal_shift_+1"]))) * 2.0))))))/2.0)))) + (complex(0,1)*np.conjugate(np.tanh((data["medianbatch_msignal"])))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_slices2_msignal"]) + (((((((-((((data["maxbatch_slices2_msignal"]) - (data["maxbatch_slices2_msignal"])))))) + (np.tanh((((data["maxbatch_slices2_msignal"]) + (((data["maxbatch_slices2_msignal"]) + (data["rangebatch_slices2"]))))))))/2.0)) - (np.tanh((np.tanh((data["rangebatch_slices2"]))))))))) / (np.cos((data["stdbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real((((data["minbatch_slices2"]) + (((np.cos((data["minbatch_slices2"]))) - (data["stdbatch_slices2"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((((((np.conjugate(((np.cos(((-((data["stdbatch_slices2_msignal"])))))) * 2.0))) * 2.0)) * (((data["abs_avgbatch_msignal"]) + (((data["rangebatch_slices2_msignal"]) + (data["meanbatch_msignal"]))))))) * (np.conjugate(data["abs_avgbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_maxbatch"]) + (np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs((((data["stdbatch_slices2_msignal"]) + (np.where(np.abs(data["maxbatch_msignal"]) > np.abs(np.where(np.abs(data["abs_maxbatch"]) <= np.abs(np.conjugate(((np.conjugate(((data["abs_avgbatch_slices2_msignal"]) / 2.0))) * 2.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )))/2.0)) + (((data["abs_maxbatch"]) + (((data["minbatch_slices2_msignal"]) * (complex(3.0)))))))/2.0))) +

                            0.100000*np.tanh(np.real(((((data["stdbatch_slices2_msignal"]) / (complex(6.12609148025512695)))) - (((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1_msignal"]))) * (np.conjugate(complex(3.0)))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["abs_maxbatch_msignal"]))) * ((((data["maxbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos(((((((data["maxbatch_slices2"]) * 2.0)) + (data["rangebatch_msignal"]))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((((complex(3.0)) * (np.sin((data["maxbatch_msignal"]))))) * (((np.sin((((((((complex(3.0)) * 2.0)) * (((complex(3.0)) + (((complex(3.0)) * (data["maxbatch_msignal"]))))))) * (data["maxbatch_msignal"]))))) + (data["maxbatch_msignal"]))))) * (data["maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate(data["rangebatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(((((((((((np.sin((((((data["abs_maxbatch_msignal"]) + (np.conjugate(np.sin((np.cos((complex(6.0))))))))) / 2.0)))) - (((data["signal_shift_+1_msignal"]) * (np.conjugate(((data["abs_maxbatch_msignal"]) * 2.0))))))) + (data["maxbatch_slices2_msignal"]))/2.0)) * 2.0)) + (data["minbatch_slices2"]))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos(((-((((((data["minbatch_slices2_msignal"]) + (np.where(np.abs(np.conjugate(data["signal_shift_+1"])) > np.abs(data["stdbatch_msignal"]),complex(1.), complex(0.) )))) - (complex(0,1)*np.conjugate(np.where(np.abs(((data["minbatch_slices2_msignal"]) + (np.sin((data["abs_avgbatch_slices2"]))))) > np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) + (np.cos((complex(0,1)*np.conjugate(np.cos(((((data["maxtominbatch"]) + (((np.cos((data["medianbatch_slices2_msignal"]))) - (complex(3.0)))))/2.0))))))))) - (data["maxtominbatch"])))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((-((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (((np.sin((data["abs_maxbatch_slices2_msignal"]))) - (((data["abs_maxbatch_slices2_msignal"]) * (np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))))))))))))) +

                            0.100000*np.tanh(np.real(((((((complex(0,1)*np.conjugate(np.tanh((complex(2.0))))) + (data["medianbatch_msignal"]))) / 2.0)) + (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((((-((np.cos((data["maxbatch_slices2_msignal"])))))) + (((complex(4.0)) + (((data["abs_maxbatch_slices2"]) * (np.sin((np.cos((data["maxbatch_slices2_msignal"])))))))))))) +

                            0.100000*np.tanh(np.real((((-((((((data["abs_maxbatch"]) + (((((((data["maxtominbatch_msignal"]) + (((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1"]))))) * 2.0)) + (data["signal_shift_-1"]))))) * 2.0))))) - ((((data["medianbatch_slices2"]) + (np.tanh(((-((data["signal_shift_-1_msignal"])))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(((np.where(np.abs(complex(7.0)) <= np.abs((((data["abs_maxbatch"]) + (np.cos((complex(7.0)))))/2.0)),complex(1.), complex(0.) )) + (np.cos((np.conjugate(data["mean_abs_chgbatch_msignal"])))))),complex(1.), complex(0.) )) / 2.0)) <= np.abs(((complex(7.0)) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((-((data["signal_shift_-1"])))) - ((((np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )) + ((((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(np.sin(((((data["signal_shift_-1"]) + ((-((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))))))/2.0)))),complex(1.), complex(0.) )) + (data["maxtominbatch_slices2"]))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["maxbatch_msignal"]) - (((np.sin((((((data["maxbatch_msignal"]) - (((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (np.sin((data["abs_maxbatch_slices2_msignal"]))))))) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))))) + (np.sin((data["abs_maxbatch_slices2_msignal"]))))))) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (np.cos((data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((((np.where(np.abs(((data["meanbatch_msignal"]) * (((np.where(np.abs(np.tanh((complex(1.0)))) <= np.abs(((np.where(np.abs(np.tanh((np.cos((data["meanbatch_msignal"]))))) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) * (data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) )) * (((data["meanbatch_slices2_msignal"]) * 2.0)))))) > np.abs(data["minbatch"]),complex(1.), complex(0.) )) * 2.0)) - (data["signal_shift_+1_msignal"]))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2_msignal"]) - (((data["mean_abs_chgbatch_slices2"]) * ((((data["abs_maxbatch_slices2_msignal"]) + (np.tanh((np.tanh((data["signal"]))))))/2.0))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((((np.sin((complex(3.0)))) / 2.0)) / 2.0)) > np.abs(np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(complex(0.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (np.sin((((data["mean_abs_chgbatch_slices2"]) - (np.sin((data["rangebatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real((((((data["rangebatch_msignal"]) * (((((data["maxbatch_slices2_msignal"]) * 2.0)) * (np.sin((data["abs_maxbatch_msignal"]))))))) + (np.where(np.abs(((np.sin((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) <= np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))))) * (((data["medianbatch_slices2_msignal"]) * (np.sin((np.sin((data["abs_maxbatch_msignal"]))))))))) <= np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_msignal"]) + (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_slices2"]) * (data["meanbatch_slices2_msignal"]))) + (data["abs_maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((((((data["stdbatch_msignal"]) + (complex(0,1)*np.conjugate(np.cos((data["maxtominbatch"])))))/2.0)) / (np.sin((data["abs_avgbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) * (((np.conjugate(complex(-3.0))) - (((((data["maxtominbatch_slices2_msignal"]) - (np.sin((((data["medianbatch_msignal"]) * (((((data["minbatch_slices2"]) - (data["stdbatch_slices2_msignal"]))) - (((((data["maxtominbatch_slices2_msignal"]) - (((((((data["minbatch_slices2"]) - (data["stdbatch_slices2_msignal"]))) * 2.0)) * 2.0)))) / 2.0)))))))))) / 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((np.cos(((((((complex(0,1)*np.conjugate(((((np.tanh((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0))) * 2.0)) + (data["meanbatch_msignal"]))/2.0)))) + (data["meanbatch_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_maxbatch_msignal"]) + (np.cos((np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.sin((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((complex(2.0)) + (((np.conjugate(data["minbatch_msignal"])) + (np.where(np.abs(np.where(np.abs(((((data["rangebatch_slices2_msignal"]) + (((complex(2.0)) + (complex(2.0)))))) + (complex(2.0)))) <= np.abs(complex(2.0)),complex(1.), complex(0.) )) <= np.abs(np.conjugate(np.cos((complex(2.0))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )) * 2.0)) + (((data["signal_shift_+1_msignal"]) * (((data["medianbatch_slices2"]) - (((((data["signal_shift_+1"]) + (data["medianbatch_slices2_msignal"]))) + (((np.sin((np.sin((np.conjugate(np.cos((complex(7.58047533035278320))))))))) * 2.0))))))))))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(((((data["maxbatch_msignal"]) - (np.sin((data["meanbatch_msignal"]))))) + (((((data["meanbatch_msignal"]) * 2.0)) * 2.0))))) / 2.0)) / (((np.sin((data["meanbatch_slices2_msignal"]))) - (np.sin((np.sin((data["meanbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(data["minbatch_slices2"]) > np.abs(((np.where(np.abs((((data["maxbatch_msignal"]) + (((data["maxbatch_msignal"]) - (data["abs_maxbatch_slices2_msignal"]))))/2.0)) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )) - (np.where(np.abs(data["minbatch_slices2"]) > np.abs((((complex(0.0)) + (np.where(np.abs(((data["maxbatch_msignal"]) * 2.0)) <= np.abs(((data["meanbatch_slices2"]) / 2.0)),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(((((np.sin((((np.cos((data["minbatch_slices2_msignal"]))) * 2.0)))) * (np.cos((((((data["abs_maxbatch_slices2"]) / 2.0)) / 2.0)))))) / (data["abs_maxbatch_slices2"])))) + (np.cos((((complex(0,1)*np.conjugate(complex(8.88508415222167969))) - (((data["medianbatch_msignal"]) / 2.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + (((np.tanh((((data["minbatch_slices2_msignal"]) + (((complex(0,1)*np.conjugate((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))) + (np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.conjugate(np.cos((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )))))))) + (((data["maxbatch_msignal"]) + (data["minbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real((((((data["maxbatch_msignal"]) + ((((((((((((np.sin(((((data["minbatch_msignal"]) + (data["minbatch_msignal"]))/2.0)))) / 2.0)) / 2.0)) + (((((data["rangebatch_slices2"]) / (np.sin(((((data["rangebatch_slices2"]) + (data["maxbatch_msignal"]))/2.0)))))) + (((complex(1.0)) - ((((data["rangebatch_slices2"]) + (data["maxbatch_msignal"]))/2.0)))))))/2.0)) * 2.0)) / 2.0)))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_msignal"]) - (np.where(np.abs(data["abs_maxbatch"]) > np.abs((((data["abs_avgbatch_msignal"]) + (np.where(np.abs(np.cos((np.tanh((data["abs_avgbatch_slices2_msignal"]))))) <= np.abs(np.sin((data["abs_avgbatch_msignal"]))),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(complex(1.0))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((complex(4.0)) - (((data["signal_shift_+1"]) - ((((((((((-((complex(4.0))))) - (((data["signal_shift_+1"]) - (np.where(np.abs(complex(4.0)) > np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )))))) - (((data["signal_shift_+1"]) - (((data["signal_shift_+1"]) - (np.tanh((data["signal_shift_+1_msignal"]))))))))) * 2.0)) - (((data["signal_shift_+1"]) - (data["minbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(((complex(10.67986202239990234)) - (((np.where(np.abs(complex(10.46769905090332031)) > np.abs(((complex(6.66605615615844727)) + (np.cos(((((-((((((np.tanh((np.tanh((np.sin(((((((np.cos((complex(-3.0)))) + (((complex(10.46769905090332031)) / 2.0)))/2.0)) / 2.0)))))))) * 2.0)) / 2.0))))) * 2.0)))))),complex(1.), complex(0.) )) + (data["rangebatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((data["rangebatch_msignal"]) / (np.cos((data["stdbatch_slices2"]))))) + (((((((data["rangebatch_msignal"]) - (((np.cos((data["stdbatch_slices2"]))) * (((complex(2.0)) * (data["stdbatch_slices2"]))))))) / (np.cos((((((data["medianbatch_msignal"]) + (complex(3.0)))) * (data["stdbatch_slices2"]))))))) + (data["mean_abs_chgbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((((((((data["mean_abs_chgbatch_slices2"]) + (((complex(1.0)) * 2.0)))) + (((((((complex(1.0)) + (data["maxbatch_msignal"]))/2.0)) + (data["abs_avgbatch_slices2"]))/2.0)))) + (((data["rangebatch_slices2"]) * 2.0)))/2.0)) + (data["meanbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real((((complex(-1.0)) + ((((((np.cos((data["rangebatch_slices2"]))) - (data["maxtominbatch_slices2_msignal"]))) + ((-((data["rangebatch_slices2"])))))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_slices2_msignal"]))) + (np.cos((((((np.tanh((np.where(np.abs(((data["stdbatch_slices2"]) / 2.0)) > np.abs(((np.conjugate(np.tanh((data["abs_maxbatch_slices2_msignal"])))) + (((np.tanh((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.sin((((data["stdbatch_slices2"]) * 2.0)))))))) * 2.0)))),complex(1.), complex(0.) )))) / 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_maxbatch"]) > np.abs(((np.where(np.abs(((((np.conjugate(complex(0,1)*np.conjugate(complex(2.0)))) - (data["meanbatch_slices2"]))) - (((((data["rangebatch_slices2_msignal"]) * (np.tanh((data["stdbatch_slices2"]))))) - (np.where(np.abs(complex(-3.0)) <= np.abs(complex(2.0)),complex(1.), complex(0.) )))))) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.conjugate(((np.where(np.abs(np.cos((((data["abs_maxbatch_msignal"]) * 2.0)))) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) + ((-((np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(np.tanh((data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))))) + (((data["minbatch_slices2_msignal"]) + (np.conjugate(((data["maxbatch_msignal"]) - (data["signal_shift_-1_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((((np.cos((np.where(np.abs(((data["meanbatch_slices2"]) + (data["minbatch_msignal"]))) > np.abs(((data["meanbatch_slices2"]) + (data["minbatch_msignal"]))),complex(1.), complex(0.) )))) / 2.0)) + (((data["minbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))))/2.0)) - (data["stdbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.cos((data["abs_avgbatch_slices2_msignal"])))) + (((((np.sin((np.sin((np.sin((((np.sin((((complex(-2.0)) * (data["meanbatch_msignal"]))))) * 2.0)))))))) * 2.0)) + (((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["medianbatch_msignal"]))) + (complex(0,1)*np.conjugate(((((((data["medianbatch_slices2_msignal"]) + (data["rangebatch_msignal"]))) * (((((((complex(5.0)) * (data["rangebatch_msignal"]))) + (data["minbatch_msignal"]))) + (np.sin((data["stdbatch_slices2"]))))))) / 2.0)))))) +

                            0.100000*np.tanh(np.real(((((((((data["medianbatch_msignal"]) / (np.cos((((data["minbatch_msignal"]) * (data["minbatch_msignal"]))))))) + (complex(0,1)*np.conjugate(np.cos((np.tanh((data["stdbatch_slices2"])))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.sin((((complex(12.76208591461181641)) + (data["minbatch_slices2"]))))) <= np.abs(((data["abs_minbatch_msignal"]) + (complex(3.0)))),complex(1.), complex(0.) )) + (((((data["abs_avgbatch_msignal"]) + ((-((((complex(1.0)) * (data["signal_shift_-1_msignal"])))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.sin(((-((data["abs_maxbatch_slices2"])))))) * (data["medianbatch_slices2"]))) * (((((((np.sin(((-((data["abs_maxbatch_slices2"])))))) * (data["medianbatch_slices2"]))) + (((np.where(np.abs(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(((data["maxbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) * (data["rangebatch_slices2"]))))) + (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_msignal"]) - (((data["signal_shift_+1_msignal"]) - (((data["abs_maxbatch_slices2_msignal"]) + (((np.where(np.abs(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) <= np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )) > np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )) - (data["signal_shift_+1_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_msignal"]) / (np.cos((data["maxbatch_msignal"]))))) - ((((((data["meanbatch_slices2_msignal"]) * 2.0)) + (np.conjugate(((complex(13.78195095062255859)) + (data["minbatch_slices2_msignal"])))))/2.0))))) +

                            0.100000*np.tanh(np.real((((((((data["minbatch_msignal"]) + (complex(1.0)))) + (data["abs_avgbatch_msignal"]))) + (np.where(np.abs(complex(9.0)) > np.abs(((data["maxbatch_msignal"]) * (((data["abs_avgbatch_slices2"]) * 2.0)))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real((((((((data["abs_maxbatch_msignal"]) + (complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])))/2.0)) + (((np.where(np.abs(np.cos((np.cos((data["abs_maxbatch_slices2_msignal"]))))) > np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)))) + (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((-((np.sin((((data["signal_shift_+1_msignal"]) / (((np.cos((np.cos((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((complex(0,1)*np.conjugate(((((data["rangebatch_slices2_msignal"]) - (((data["abs_maxbatch_slices2"]) * (data["minbatch_slices2_msignal"]))))) / 2.0))) + ((((((complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"])) - (data["medianbatch_slices2_msignal"]))) + (((data["abs_maxbatch_slices2"]) * (data["minbatch_slices2_msignal"]))))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["rangebatch_slices2"]) - (((data["meanbatch_slices2_msignal"]) + ((-(((-((complex(0,1)*np.conjugate((((data["rangebatch_slices2"]) + ((((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["medianbatch_slices2_msignal"]) + (np.conjugate((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0))))))/2.0)))/2.0)))))))))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((np.tanh((((np.tanh(((((data["minbatch_slices2_msignal"]) + (((data["signal_shift_-1_msignal"]) * 2.0)))/2.0)))) / 2.0)))) / 2.0)) - (np.conjugate(((((data["signal_shift_-1_msignal"]) * 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((complex(1.0)) / 2.0)) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.sin((data["meanbatch_slices2_msignal"]))) * (((data["meanbatch_slices2_msignal"]) * (((((np.conjugate(data["meanbatch_slices2_msignal"])) * 2.0)) + (((np.sin((np.conjugate(data["rangebatch_slices2"])))) * (np.where(np.abs(complex(1.0)) > np.abs(np.cos((np.sin((np.sin(((-((((np.tanh((data["meanbatch_slices2_msignal"]))) / 2.0))))))))))),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((data["minbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(((data["rangebatch_msignal"]) - (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.tanh((np.tanh((data["minbatch_slices2_msignal"]))))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((complex(3.43792390823364258)) * (data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + ((((complex(1.0)) + (((((((((data["abs_avgbatch_slices2_msignal"]) + (complex(2.0)))) - (np.conjugate(data["mean_abs_chgbatch_msignal"])))) + (np.where(np.abs(np.sin((np.where(np.abs(complex(1.63407003879547119)) <= np.abs((((data["medianbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))) <= np.abs(((data["minbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) )))) - (complex(1.63407003879547119)))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(data["rangebatch_msignal"])) + (((complex(3.0)) / (((complex(3.0)) * (complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"]))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.sin((np.conjugate(np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) * (complex(3.0))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((((data["minbatch"]) * (data["medianbatch_msignal"]))) / 2.0)))) * (((((data["minbatch"]) * (data["medianbatch_msignal"]))) + (((((data["minbatch"]) * (data["medianbatch_msignal"]))) + (data["meanbatch_slices2_msignal"]))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.cos((((complex(0,1)*np.conjugate((((((data["medianbatch_slices2_msignal"]) - (data["medianbatch_msignal"]))) + (np.tanh((((data["medianbatch_slices2_msignal"]) - (data["medianbatch_msignal"]))))))/2.0))) - (((((data["medianbatch_slices2_msignal"]) - (data["medianbatch_msignal"]))) - (complex(0,1)*np.conjugate((((complex(5.0)) + (((data["medianbatch_slices2_msignal"]) - (data["medianbatch_msignal"]))))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(((((complex(3.0)) - (((((data["abs_minbatch_slices2_msignal"]) / 2.0)) + (((data["signal_shift_+1_msignal"]) * 2.0)))))) - (((data["signal_shift_-1_msignal"]) + (np.cos((data["meanbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch_slices2"]) / (data["maxtominbatch"]))) + ((((((((((np.where(np.abs(np.where(np.abs((-(((((np.sin((((data["minbatch_slices2_msignal"]) - (((data["maxtominbatch_slices2"]) * 2.0)))))) + (data["minbatch_slices2_msignal"]))/2.0))))) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )) + (data["minbatch_slices2_msignal"]))/2.0)) + (data["minbatch_slices2_msignal"]))/2.0)) + (data["minbatch_slices2_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["minbatch_slices2"]) <= np.abs(np.cos((data["minbatch_slices2"]))),complex(1.), complex(0.) )) > np.abs(((np.where(np.abs((-((((np.conjugate(np.tanh((data["medianbatch_msignal"])))) / ((((((((data["mean_abs_chgbatch_msignal"]) - (complex(14.35059833526611328)))) + (data["abs_maxbatch_slices2"]))/2.0)) / 2.0))))))) > np.abs(((data["maxbatch_slices2"]) / 2.0)),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.conjugate(((complex(0,1)*np.conjugate(np.cos((np.where(np.abs(data["signal_shift_+1"]) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) ))))) - (((complex(0.0)) + ((((-((data["maxtominbatch"])))) + (data["maxtominbatch"]))))))))) - (((data["medianbatch_slices2"]) + (data["maxtominbatch"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((complex(0.0)) * 2.0)) > np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) ))))  

    

    def GP_class_1(self,data):

        return self.Output( -1.623856 +

                            0.100000*np.tanh(np.real(((((((complex(13.70524692535400391)) / 2.0)) + (data["maxtominbatch_slices2_msignal"]))) - (((data["abs_minbatch_slices2_msignal"]) * (data["maxtominbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((data["maxtominbatch_msignal"]) * 2.0)) + (data["maxtominbatch_slices2_msignal"]))) + (data["maxtominbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((((((((((((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["abs_minbatch_msignal"]))/2.0)) - (((((np.cos(((((data["abs_minbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)))) + (data["abs_minbatch_msignal"]))) * 2.0)))) * (data["abs_minbatch_msignal"]))) + (((data["abs_minbatch_msignal"]) / 2.0)))) + (((data["abs_minbatch_msignal"]) + (data["maxtominbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["maxtominbatch_slices2_msignal"]) * 2.0)))) + (((np.sin((np.sin((data["maxbatch_msignal"]))))) + (((((np.cos((data["maxbatch_msignal"]))) * 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real((((data["minbatch_msignal"]) + (((((((((data["minbatch_msignal"]) + (((((-((data["minbatch_msignal"])))) + ((((-((np.cos((data["minbatch_msignal"])))))) / ((-(((-((np.cos((data["minbatch_msignal"]))))))))))))/2.0)))/2.0)) / 2.0)) + (((complex(3.0)) / ((((-((np.cos((data["minbatch_msignal"])))))) / 2.0)))))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos(((((((np.sin(((((((data["signal_shift_+1"]) + (((((data["mean_abs_chgbatch_msignal"]) * 2.0)) / 2.0)))/2.0)) * (np.tanh((complex(-3.0)))))))) + (data["abs_minbatch_slices2_msignal"]))/2.0)) * (np.tanh((complex(-3.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.sin((data["abs_minbatch_slices2_msignal"]))) * ((((np.cos(((((data["abs_avgbatch_msignal"]) + (complex(2.0)))/2.0)))) + ((((((((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) + (((data["abs_minbatch_slices2_msignal"]) * (data["abs_minbatch_slices2_msignal"]))))) + (complex(2.0)))/2.0)) * (data["abs_minbatch_slices2_msignal"]))))/2.0)))))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(0,1)*np.conjugate(np.conjugate(((((np.where(np.abs(((np.conjugate(((np.cos((((complex(0,1)*np.conjugate(np.conjugate(data["rangebatch_slices2_msignal"]))) + (data["maxbatch_msignal"]))))) / 2.0))) * 2.0)) > np.abs(np.where(np.abs(((complex(-3.0)) * 2.0)) <= np.abs(((complex(2.0)) - (data["medianbatch_slices2_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)) * 2.0)))) + (data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((np.conjugate((((((np.conjugate(((np.cos((data["maxtominbatch_slices2_msignal"]))) - (((np.cos((data["mean_abs_chgbatch_slices2"]))) * 2.0))))) * (complex(0,1)*np.conjugate(data["minbatch_msignal"])))) + (np.conjugate(data["minbatch_msignal"])))/2.0))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((((((data["minbatch_msignal"]) + (((((complex(6.0)) + (np.cos(((((data["minbatch_msignal"]) + (np.cos((data["abs_avgbatch_slices2"]))))/2.0)))))) / 2.0)))/2.0)) + (np.cos((data["maxtominbatch_slices2_msignal"]))))) + (np.cos((data["maxbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((((data["rangebatch_slices2"]) + ((-((complex(0,1)*np.conjugate(data["rangebatch_slices2"]))))))/2.0)) + ((((-((data["rangebatch_slices2"])))) + (complex(0,1)*np.conjugate(np.cos(((((data["rangebatch_slices2"]) + ((-((complex(0,1)*np.conjugate(((data["rangebatch_slices2"]) * 2.0)))))))/2.0))))))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["minbatch_msignal"]) * (((np.where(np.abs(complex(3.0)) > np.abs(np.where(np.abs(((data["minbatch_msignal"]) * (((data["medianbatch_msignal"]) * 2.0)))) > np.abs(((data["minbatch_msignal"]) * (((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((((data["maxtominbatch_slices2_msignal"]) + (data["signal_shift_-1"]))/2.0)) + (((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) / (np.cos((data["minbatch_msignal"]))))) + (((np.cos((np.cos((((data["minbatch_msignal"]) / (np.cos(((((data["minbatch_msignal"]) + (np.cos((data["minbatch_msignal"]))))/2.0)))))))))) + (((((data["minbatch_msignal"]) / (np.cos((data["minbatch_msignal"]))))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((((((((((complex(3.56757497787475586)) * (np.cos((data["maxbatch_msignal"]))))) * 2.0)) * 2.0)) * (complex(0,1)*np.conjugate(((np.conjugate(((np.conjugate(data["signal_shift_-1_msignal"])) - (complex(0,1)*np.conjugate(((data["maxbatch_msignal"]) * 2.0)))))) / 2.0))))) * 2.0)) + (((complex(3.56757497787475586)) * (np.cos((data["maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((complex(9.0)) - (((((data["rangebatch_slices2_msignal"]) + (((((data["abs_maxbatch"]) / 2.0)) / 2.0)))) * (((data["maxbatch_slices2"]) - (data["abs_minbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real((((((np.sin(((-((data["minbatch_slices2_msignal"])))))) + ((((((np.sin(((-((data["minbatch_slices2_msignal"])))))) * 2.0)) + (np.sin((np.sin(((((-((data["minbatch_slices2_msignal"])))) * 2.0)))))))/2.0)))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos(((((data["minbatch_msignal"]) + (np.cos((np.cos((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(((data["minbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )))))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((complex(3.0)) + (((data["minbatch_slices2_msignal"]) + (((complex(14.46821117401123047)) * (np.cos((((((data["abs_minbatch_msignal"]) + (np.cos((np.cos((((np.cos((data["maxbatch_slices2_msignal"]))) * (((((complex(-2.0)) + (((((((data["abs_minbatch_msignal"]) / 2.0)) * (data["abs_minbatch_msignal"]))) / 2.0)))) / 2.0)))))))))) / 2.0))))))))))) +

                            0.100000*np.tanh(np.real(((((complex(9.0)) - (data["abs_maxbatch_slices2_msignal"]))) + (((((data["rangebatch_msignal"]) * (data["abs_minbatch_msignal"]))) + (((np.where(np.abs(np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(np.sin((((complex(-3.0)) * (data["rangebatch_slices2"]))))),complex(1.), complex(0.) )) * (np.conjugate(((complex(9.0)) - (data["rangebatch_slices2"]))))))))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) / 2.0)) - (((((np.sin((((data["minbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(((((np.sin((data["minbatch_msignal"]))) * 2.0)) * 2.0)) <= np.abs(((np.cos((((((data["minbatch_slices2_msignal"]) / 2.0)) * 2.0)))) / 2.0)),complex(1.), complex(0.) )) <= np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )))))) * 2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real((((((((((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))) * 2.0)) * 2.0)) + (((((np.cos((data["maxtominbatch_slices2_msignal"]))) / 2.0)) - ((-((((data["mean_abs_chgbatch_slices2"]) + (((((complex(1.0)) * 2.0)) / (data["maxtominbatch_slices2"])))))))))))/2.0))) +

                            0.100000*np.tanh(np.real((((complex(3.0)) + (((((complex(12.01205062866210938)) * (((data["medianbatch_msignal"]) + (np.cos((data["maxbatch_msignal"]))))))) - (np.tanh((np.cos((complex(12.01205062866210938)))))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(((complex(3.0)) + (np.where(np.abs(((data["maxbatch_slices2"]) * (data["abs_maxbatch_slices2_msignal"]))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))) > np.abs(((complex(-1.0)) * (((data["minbatch"]) * (((complex(0,1)*np.conjugate(((data["signal"]) * (np.cos((data["abs_maxbatch_slices2_msignal"])))))) / 2.0)))))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + (((complex(3.0)) - (((((((np.cos((((data["minbatch_slices2_msignal"]) * 2.0)))) * 2.0)) / 2.0)) * (complex(4.28790187835693359))))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((np.sin((np.cos(((((data["minbatch_slices2_msignal"]) + ((((data["medianbatch_msignal"]) + (np.where(np.abs(np.conjugate(data["minbatch_msignal"])) <= np.abs(np.where(np.abs(data["medianbatch_slices2"]) > np.abs(((complex(2.0)) / 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)))/2.0)))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real((-((np.sin((((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(((np.sin((data["minbatch_msignal"]))) + ((-((data["minbatch_msignal"]))))))))) + (np.where(np.abs(np.where(np.abs(np.sin((np.where(np.abs((-((np.sin((data["meanbatch_slices2"])))))) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))) <= np.abs(((data["minbatch_msignal"]) + (data["minbatch_msignal"]))),complex(1.), complex(0.) )) <= np.abs(complex(0,1)*np.conjugate(data["maxtominbatch_slices2_msignal"])),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2"]) + ((((data["abs_minbatch_msignal"]) + (((data["medianbatch_slices2_msignal"]) + (((((complex(0.96318149566650391)) / 2.0)) - (data["abs_maxbatch_slices2_msignal"]))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.where(np.abs(data["stdbatch_slices2"]) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )))) + (np.sin((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((np.cos((data["medianbatch_slices2_msignal"]))) / (np.cos((np.conjugate(np.cos((np.conjugate(np.where(np.abs(((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) + (complex(2.0)))) > np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))))))) + (((data["maxbatch_slices2_msignal"]) / ((((data["medianbatch_slices2_msignal"]) + (np.cos((data["maxbatch_msignal"]))))/2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(((data["minbatch_msignal"]) + (data["rangebatch_slices2_msignal"]))),complex(1.), complex(0.) )) / (np.cos((((data["minbatch_slices2_msignal"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch"]) / (complex(0,1)*np.conjugate(((((((data["signal_shift_+1_msignal"]) + (np.cos((np.where(np.abs((((((-((complex(0,1)*np.conjugate(np.cos((data["minbatch_slices2_msignal"]))))))) / 2.0)) * 2.0)) <= np.abs((((-((complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"]))))) / 2.0)),complex(1.), complex(0.) )))))/2.0)) + ((-((complex(0,1)*np.conjugate(np.cos((data["minbatch_slices2_msignal"]))))))))/2.0)))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_minbatch_slices2"]) - (data["rangebatch_slices2"]))) - (((complex(-2.0)) * 2.0)))) - (np.sin(((((((-((data["medianbatch_msignal"])))) / 2.0)) - ((((((data["rangebatch_slices2"]) + (complex(0.0)))/2.0)) + ((((data["mean_abs_chgbatch_slices2"]) + (data["mean_abs_chgbatch_slices2"]))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.cos(((((((data["abs_minbatch_slices2_msignal"]) - (np.where(np.abs(np.tanh((data["mean_abs_chgbatch_msignal"]))) > np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(((((data["meanbatch_slices2_msignal"]) / 2.0)) - (np.cos((data["abs_minbatch_slices2_msignal"]))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) + (np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )))/2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(2.0)) > np.abs((((complex(1.0)) + (data["maxtominbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["maxbatch_slices2"]) > np.abs(np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(data["meanbatch_msignal"])) > np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) )) / (data["maxtominbatch_slices2_msignal"]))) - (np.cos((((data["rangebatch_slices2"]) * (np.where(np.abs(np.cos((((data["abs_minbatch_slices2_msignal"]) / 2.0)))) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((((((((np.sin((np.sin((((data["minbatch_msignal"]) * 2.0)))))) * 2.0)) + (complex(0,1)*np.conjugate(np.sin((np.cos((data["stdbatch_slices2_msignal"])))))))) * 2.0)) + (complex(0,1)*np.conjugate(((np.sin((((data["minbatch_msignal"]) * 2.0)))) + (complex(-2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((((data["abs_minbatch_msignal"]) + (data["maxtominbatch_msignal"]))/2.0)) <= np.abs(((((data["abs_minbatch_slices2_msignal"]) * (np.where(np.abs(((data["stdbatch_msignal"]) * 2.0)) > np.abs(np.tanh(((-((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )))) * 2.0)),complex(1.), complex(0.) )) + (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(np.tanh((((complex(7.68996667861938477)) - (((((np.where(np.abs((-((complex(3.0))))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (((data["medianbatch_slices2"]) * (data["maxbatch_msignal"]))))) / 2.0)))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(1.0)) <= np.abs((((np.sin((((((data["abs_maxbatch"]) - (np.cos(((((((data["stdbatch_slices2_msignal"]) * 2.0)) + ((((((np.cos((((data["abs_minbatch_msignal"]) * (np.conjugate(data["signal_shift_+1_msignal"])))))) + (complex(1.0)))/2.0)) + (np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))))/2.0)))))) * 2.0)))) + (complex(2.0)))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.sin((np.tanh((((np.sin((((data["minbatch_msignal"]) * 2.0)))) - (np.where(np.abs(np.tanh((((data["minbatch_msignal"]) * 2.0)))) > np.abs(((((np.tanh((((np.sin((((data["minbatch_msignal"]) * 2.0)))) - (data["minbatch_msignal"]))))) + (complex(9.0)))) * 2.0)),complex(1.), complex(0.) )))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((((((((np.tanh(((-((np.cos((complex(-2.0))))))))) + ((((((np.cos((data["maxbatch_msignal"]))) * 2.0)) + (np.cos((data["abs_minbatch_msignal"]))))/2.0)))) + (((((np.sin(((((-(((-((data["maxbatch_msignal"]))))))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) + (np.tanh((data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["minbatch_msignal"]) + (((np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])) > np.abs(((((((((np.where(np.abs(((complex(-2.0)) - (data["abs_avgbatch_slices2_msignal"]))) <= np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0)) / 2.0)) / 2.0)) * 2.0)),complex(1.), complex(0.) )) <= np.abs(((np.sin(((-((data["meanbatch_slices2"])))))) * 2.0)),complex(1.), complex(0.) )) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real((((((np.cos((((np.conjugate(data["mean_abs_chgbatch_slices2"])) * 2.0)))) + (data["maxbatch_slices2_msignal"]))/2.0)) / (np.cos((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((-((np.sin(((((data["abs_minbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["abs_maxbatch_slices2"]) + (np.where(np.abs(complex(4.0)) <= np.abs((((data["mean_abs_chgbatch_slices2"]) + ((((((np.sin((((data["minbatch_slices2_msignal"]) * 2.0)))) * 2.0)) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))/2.0)),complex(1.), complex(0.) )))/2.0)))) * (((complex(14.63929367065429688)) * (np.sin((np.sin((((np.sin((((data["minbatch_slices2_msignal"]) * 2.0)))) * 2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((complex(0,1)*np.conjugate(((data["medianbatch_msignal"]) / (np.where(np.abs((-((data["stdbatch_slices2_msignal"])))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) + (np.cos((complex(0,1)*np.conjugate(((np.where(np.abs(complex(2.0)) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )) * (np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs((-((data["mean_abs_chgbatch_slices2_msignal"])))),complex(1.), complex(0.) ))))))))/2.0)) + (data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((complex(4.09455394744873047)) + (((data["abs_maxbatch_slices2"]) * (((data["minbatch_msignal"]) / 2.0)))))) * 2.0)))) + (((((((complex(4.09455394744873047)) + (data["minbatch_msignal"]))) * 2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real((((-((data["abs_maxbatch_slices2_msignal"])))) * (np.sin((((data["minbatch_msignal"]) + (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) * (np.sin(((((-(((-((data["abs_maxbatch_slices2_msignal"]))))))) * (((data["minbatch_msignal"]) + (np.where(np.abs(((data["maxbatch_slices2_msignal"]) * 2.0)) > np.abs(np.conjugate(data["abs_minbatch_msignal"])),complex(1.), complex(0.) )))))))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((((data["meanbatch_msignal"]) + (((data["rangebatch_slices2"]) * 2.0)))) + (np.where(np.abs(((((data["meanbatch_msignal"]) + (((data["rangebatch_slices2"]) * 2.0)))) * (((data["meanbatch_msignal"]) + (((((data["stdbatch_slices2_msignal"]) * 2.0)) * 2.0)))))) <= np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((((data["maxtominbatch"]) + (data["maxtominbatch"]))/2.0)) > np.abs((((complex(-3.0)) + (data["medianbatch_msignal"]))/2.0)),complex(1.), complex(0.) )) - (((data["maxbatch_slices2"]) - (((np.sin((np.conjugate(data["medianbatch_slices2_msignal"])))) - ((((complex(-3.0)) + (data["maxtominbatch"]))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((((np.cos((data["signal_shift_-1_msignal"]))) + (data["minbatch_msignal"]))/2.0)) + (np.cos((np.cos((data["abs_avgbatch_slices2"]))))))) + (((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_msignal"])) + (np.sin((np.cos((data["signal_shift_-1_msignal"])))))))))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_minbatch_msignal"]) + (complex(1.0)))/2.0)) + (data["minbatch"]))/2.0))) +

                            0.100000*np.tanh(np.real((((((-((np.cos((((data["minbatch_msignal"]) / (np.cos((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs((((-((np.cos((np.tanh(((-((data["minbatch_msignal"]))))))))))) * 2.0)),complex(1.), complex(0.) ))))))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch_slices2_msignal"]) + ((((data["rangebatch_slices2"]) + (complex(0,1)*np.conjugate(np.cos((data["maxbatch_msignal"])))))/2.0)))) - ((-((np.where(np.abs(data["maxtominbatch_slices2"]) <= np.abs(((data["medianbatch_msignal"]) * ((-((data["abs_maxbatch_slices2_msignal"])))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate(((data["maxbatch_msignal"]) * 2.0))) - ((((-((((((np.conjugate(data["maxbatch_msignal"])) * 2.0)) * 2.0))))) + (data["minbatch"]))))))))) +

                            0.100000*np.tanh(np.real(((((((((data["mean_abs_chgbatch_msignal"]) + (complex(2.61388492584228516)))/2.0)) + (data["abs_minbatch_msignal"]))/2.0)) + (((complex(2.0)) + (((data["abs_minbatch_slices2"]) - ((-((np.cos((((((((data["minbatch_msignal"]) / 2.0)) + (((((complex(2.0)) / 2.0)) + (((complex(2.0)) - (data["abs_avgbatch_msignal"]))))))) * 2.0)))))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch_msignal"]) + (complex(0,1)*np.conjugate(np.conjugate(data["rangebatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((complex(3.0)) + (((data["minbatch_msignal"]) - (((((np.conjugate(np.tanh((((((data["abs_minbatch_msignal"]) - (((data["abs_minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"]))))) + (((data["minbatch_msignal"]) - (((np.tanh((complex(3.0)))) / 2.0))))))))) / 2.0)) / 2.0))))))) +

                            0.100000*np.tanh(np.real((((((np.where(np.abs(complex(4.0)) <= np.abs(np.where(np.abs(np.where(np.abs(complex(4.0)) <= np.abs((-((np.cos((data["rangebatch_slices2"])))))),complex(1.), complex(0.) )) > np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (np.where(np.abs(((np.conjugate(data["rangebatch_msignal"])) / 2.0)) > np.abs(complex(-2.0)),complex(1.), complex(0.) )))) + (((data["maxtominbatch_msignal"]) * (data["abs_minbatch_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((data["maxbatch_msignal"]))) + (((((data["medianbatch_msignal"]) + (data["medianbatch_slices2_msignal"]))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(((complex(0,1)*np.conjugate(((data["signal_shift_+1_msignal"]) * (((data["abs_minbatch_msignal"]) / 2.0))))) + (((data["minbatch_slices2_msignal"]) / 2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((((((np.where(np.abs(np.cos((data["abs_maxbatch_msignal"]))) > np.abs(np.sin((np.cos((data["abs_maxbatch_msignal"]))))),complex(1.), complex(0.) )) + (np.cos((np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) )))))/2.0)) * (((complex(1.0)) * 2.0)))) * 2.0)) * (np.cos((data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((data["rangebatch_slices2"]) / (np.cos(((((((data["maxbatch_slices2_msignal"]) + (np.cos((data["maxbatch_slices2_msignal"]))))/2.0)) * 2.0)))))) + (((((data["maxbatch_slices2_msignal"]) * 2.0)) * 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) - (((((((np.sin((np.sin((np.conjugate(data["signal_shift_-1"])))))) + (data["maxtominbatch"]))/2.0)) + ((((((data["mean_abs_chgbatch_msignal"]) + ((((((((data["signal_shift_-1"]) * 2.0)) * 2.0)) + (((data["signal_shift_+1"]) * 2.0)))/2.0)))/2.0)) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["maxtominbatch_msignal"]) / (np.sin((data["minbatch"]))))) + (complex(0,1)*np.conjugate(np.conjugate(data["minbatch"]))))) - (((np.where(np.abs(((data["signal_shift_-1_msignal"]) * (data["rangebatch_slices2"]))) <= np.abs(data["signal"]),complex(1.), complex(0.) )) / 2.0))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1"]) * (((((data["minbatch_msignal"]) + (((np.cos((((complex(0,1)*np.conjugate(data["abs_avgbatch_msignal"])) + (data["meanbatch_msignal"]))))) / 2.0)))) - (data["abs_avgbatch_msignal"]))))) - (data["maxtominbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0)))))) * ((((((complex(10.0)) + (np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(((np.sin(((((((data["signal_shift_+1_msignal"]) + (((data["maxbatch_msignal"]) * 2.0)))/2.0)) * 2.0)))) * (data["medianbatch_slices2"]))),complex(1.), complex(0.) )))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) <= np.abs(((data["medianbatch_slices2_msignal"]) * (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) * (data["minbatch_slices2"]))) + (((((((complex(3.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))) - (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))) - (complex(0,1)*np.conjugate(np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(np.sin((data["minbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((((((((data["minbatch_slices2_msignal"]) + (np.cos(((((np.tanh((np.tanh((data["meanbatch_slices2_msignal"]))))) + (np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )))/2.0)))))) + (((((np.cos((data["stdbatch_slices2"]))) * 2.0)) * 2.0)))) * 2.0)) + ((((data["meanbatch_msignal"]) + (np.cos(((((data["minbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"])))/2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.tanh((data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(((data["meanbatch_slices2_msignal"]) + (data["abs_maxbatch"]))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real((((((-((data["mean_abs_chgbatch_slices2"])))) + ((-((np.cos((data["mean_abs_chgbatch_slices2_msignal"])))))))) + (np.where(np.abs(((complex(9.0)) + (complex(2.0)))) > np.abs(((data["signal"]) / 2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(np.cos(((-((data["maxbatch_slices2"])))))),complex(1.), complex(0.) )) - (np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(np.where(np.abs(data["rangebatch_slices2_msignal"]) <= np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((complex(3.0)) + (((data["minbatch_msignal"]) + (np.where(np.abs((((data["rangebatch_msignal"]) + (np.where(np.abs(np.conjugate(data["minbatch_msignal"])) > np.abs(complex(3.0)),complex(1.), complex(0.) )))/2.0)) > np.abs((((complex(3.0)) + (data["minbatch_msignal"]))/2.0)),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(data["abs_maxbatch_slices2"])) * (((((((((((data["rangebatch_slices2"]) / 2.0)) + (((complex(2.0)) / 2.0)))) * (np.cos((data["maxbatch_slices2_msignal"]))))) + (complex(2.12250757217407227)))) * (np.cos((data["maxbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["abs_avgbatch_slices2_msignal"]) + (((data["medianbatch_msignal"]) + (np.conjugate(data["medianbatch_msignal"])))))/2.0)))) + (((complex(0,1)*np.conjugate(data["medianbatch_msignal"])) - (np.cos((((data["minbatch_slices2_msignal"]) * ((((data["abs_maxbatch_slices2"]) + (data["signal_shift_+1_msignal"]))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch"]) - (np.conjugate(np.conjugate(((((((((data["abs_maxbatch_msignal"]) + (((data["medianbatch_slices2"]) * (data["medianbatch_msignal"]))))/2.0)) + ((((((data["maxtominbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"]))/2.0)) * (data["medianbatch_slices2"]))))/2.0)) * (np.conjugate(data["medianbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.cos(((((data["signal_shift_+1_msignal"]) + (np.sin((data["signal_shift_+1_msignal"]))))/2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["signal"]) * (((np.cos((data["medianbatch_msignal"]))) * (((data["signal"]) + (data["signal_shift_+1_msignal"]))))))) + (data["medianbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) + (np.sin((((np.cos((data["signal_shift_-1_msignal"]))) - (data["medianbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real((-(((((data["maxtominbatch_slices2"]) + (((data["rangebatch_slices2"]) + ((((((np.conjugate(complex(-1.0))) * 2.0)) + ((-(((((data["meanbatch_msignal"]) + (((data["rangebatch_msignal"]) + (data["rangebatch_slices2"]))))/2.0))))))/2.0)))))/2.0)))))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(data["abs_maxbatch"]) > np.abs(((((np.cos((((np.cos((np.cos((((data["maxbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))))))) + (data["maxbatch_slices2_msignal"]))))) * 2.0)) * 2.0)),complex(1.), complex(0.) ))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(complex(2.0))) + (((np.conjugate(complex(3.0))) * (((((np.conjugate(data["medianbatch_msignal"])) + (np.conjugate(data["medianbatch_msignal"])))) - (np.where(np.abs(data["medianbatch_msignal"]) > np.abs(((np.conjugate(data["abs_maxbatch"])) / (np.cos((data["stdbatch_slices2"]))))),complex(1.), complex(0.) )))))))) / (np.cos((data["stdbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((np.cos(((((complex(0.0)) + (np.where(np.abs(np.sin((np.where(np.abs(data["minbatch"]) <= np.abs((((complex(0.0)) + (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )))) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0)))) + (((data["minbatch"]) + (complex(2.0)))))) + (data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(np.cos((data["maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_msignal"]) + (np.sin((data["abs_avgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_msignal"]) * (np.sin((((((data["medianbatch_msignal"]) * 2.0)) + (np.sin((((((data["medianbatch_msignal"]) * 2.0)) + (np.tanh(((((np.cos((data["medianbatch_msignal"]))) + (np.where(np.abs(data["signal"]) <= np.abs(((data["medianbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )))/2.0))))))))))))))) +

                            0.100000*np.tanh(np.real(((((data["rangebatch_slices2"]) + (np.where(np.abs(((np.sin((complex(1.56153595447540283)))) - (((data["maxbatch_slices2_msignal"]) * 2.0)))) <= np.abs(np.where(np.abs(((((data["rangebatch_slices2"]) + (data["signal_shift_-1"]))) * (complex(1.0)))) <= np.abs(np.where(np.abs(data["maxtominbatch_msignal"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) * (np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.cos(((-(((((data["minbatch_msignal"]) + (np.cos((data["maxbatch_slices2_msignal"]))))/2.0))))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((((np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) / 2.0)) - (data["medianbatch_msignal"]))) + (((((data["abs_maxbatch_slices2"]) * (np.cos((data["maxbatch_slices2_msignal"]))))) * 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_avgbatch_slices2_msignal"]) + (((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) - (complex(0,1)*np.conjugate(data["minbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["medianbatch_msignal"]))) + (((((((data["medianbatch_msignal"]) / 2.0)) + (((np.sin((((data["medianbatch_slices2_msignal"]) * 2.0)))) * 2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((complex(3.0)) - ((-(((((((data["abs_avgbatch_msignal"]) - (data["rangebatch_msignal"]))) + (np.cos((((data["abs_maxbatch_slices2_msignal"]) - (np.where(np.abs(((((data["rangebatch_msignal"]) - ((-((np.where(np.abs(data["rangebatch_msignal"]) > np.abs(((complex(3.0)) * (data["medianbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))) * 2.0)) > np.abs(complex(0,1)*np.conjugate(data["rangebatch_msignal"])),complex(1.), complex(0.) )))))))/2.0))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((complex(5.0)) - (((data["signal"]) * (((((((data["mean_abs_chgbatch_slices2_msignal"]) + ((((((((data["abs_maxbatch"]) * 2.0)) * 2.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))/2.0)) + ((((((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["abs_maxbatch_msignal"]) * (data["rangebatch_slices2"]))))/2.0)) + (np.conjugate(np.cos((data["meanbatch_slices2"])))))))/2.0))))))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0))))))) * 2.0)) + (np.tanh((np.conjugate(np.sin((np.conjugate(data["maxbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real((((((complex(6.84004926681518555)) * 2.0)) + (((np.tanh((data["medianbatch_msignal"]))) * 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["medianbatch_slices2_msignal"]) / (data["abs_maxbatch_slices2"]))))) + (((data["rangebatch_msignal"]) / (np.cos((((((data["abs_avgbatch_slices2_msignal"]) * (complex(6.0)))) / (data["abs_maxbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(-2.0)) + (((((((np.cos((data["rangebatch_slices2"]))) * 2.0)) * 2.0)) - (np.tanh((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["maxbatch_msignal"]))) + (np.sin((complex(-3.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(np.where(np.abs(np.sin((((data["abs_maxbatch"]) + ((((data["rangebatch_slices2"]) + (data["minbatch_msignal"]))/2.0)))))) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * ((((np.cos((((data["signal_shift_+1"]) * 2.0)))) + (data["abs_minbatch_slices2_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["signal_shift_-1_msignal"]) - (((np.cos((data["minbatch_msignal"]))) * 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.tanh((((((np.cos((data["maxbatch_msignal"]))) * 2.0)) + ((((data["minbatch_msignal"]) + (complex(3.0)))/2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])) + (((complex(3.0)) * 2.0)))) / (np.cos((np.conjugate(((((data["maxbatch_slices2_msignal"]) * 2.0)) - (np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(((complex(3.0)) / (np.cos((np.cos((complex(-2.0)))))))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["medianbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.tanh((np.where(np.abs(np.conjugate(((np.cos((np.where(np.abs(((data["medianbatch_slices2_msignal"]) / 2.0)) <= np.abs(complex(0.0)),complex(1.), complex(0.) )))) - (np.cos((complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"]))))))) <= np.abs(((((data["abs_maxbatch_msignal"]) * (complex(-2.0)))) * (data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((((data["abs_maxbatch_slices2"]) - (((data["abs_minbatch_slices2_msignal"]) * (np.where(np.abs(((np.sin((((complex(0.0)) * 2.0)))) / (complex(4.0)))) <= np.abs((((complex(0,1)*np.conjugate(data["minbatch"])) + (data["abs_maxbatch_slices2"]))/2.0)),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((((((data["rangebatch_slices2"]) * 2.0)) + (complex(0,1)*np.conjugate(((complex(3.0)) - (np.where(np.abs(np.cos((((data["rangebatch_slices2"]) * 2.0)))) <= np.abs(complex(3.0)),complex(1.), complex(0.) ))))))))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((data["maxbatch_slices2"]) * (np.sin(((-((data["stdbatch_slices2_msignal"])))))))) * (np.sin((data["abs_avgbatch_slices2_msignal"]))))) <= np.abs(np.cos((np.conjugate(((data["medianbatch_msignal"]) + (data["medianbatch_msignal"])))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.cos((((((data["maxbatch_slices2_msignal"]) - (np.where(np.abs(((((data["maxbatch_slices2_msignal"]) - ((((data["abs_minbatch_msignal"]) + (np.tanh((data["abs_maxbatch_slices2"]))))/2.0)))) * 2.0)) <= np.abs(np.tanh((((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((((((((np.sin((np.sin((np.sin((((data["medianbatch_msignal"]) * 2.0)))))))) * 2.0)) + (np.sin((((((np.sin((((data["medianbatch_msignal"]) * 2.0)))) + (np.sin((((data["maxbatch_slices2_msignal"]) * 2.0)))))) * 2.0)))))) + (((np.sin((((data["maxbatch_slices2_msignal"]) * 2.0)))) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) * (((np.where(np.abs(np.cos((data["signal"]))) <= np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) )) + (((data["rangebatch_msignal"]) * 2.0)))))) / 2.0))) +

                            0.100000*np.tanh(np.real(((((((np.conjugate(data["minbatch_msignal"])) + (data["abs_minbatch_slices2"]))) * (data["maxbatch_slices2_msignal"]))) - (((complex(-3.0)) + ((((-((data["maxbatch_msignal"])))) - (data["abs_minbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2"]) - (((((((data["rangebatch_slices2"]) / (((data["mean_abs_chgbatch_slices2"]) - (np.cos((np.sin((data["maxbatch_slices2_msignal"]))))))))) / 2.0)) - ((((-((np.cos((data["maxbatch_slices2_msignal"])))))) * (np.tanh((((data["mean_abs_chgbatch_slices2"]) - (((complex(13.99666023254394531)) + (data["rangebatch_slices2"])))))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((((data["abs_avgbatch_msignal"]) - (np.cos((np.tanh((np.sin((((((data["abs_avgbatch_msignal"]) - (((data["abs_avgbatch_msignal"]) / 2.0)))) - (np.tanh((np.where(np.abs(complex(4.0)) <= np.abs(np.cos((data["abs_avgbatch_msignal"]))),complex(1.), complex(0.) )))))))))))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real((((np.conjugate(((((((data["abs_avgbatch_msignal"]) + (np.sin((np.sin((data["abs_maxbatch"]))))))/2.0)) + (((((complex(-2.0)) / (np.sin((data["abs_maxbatch"]))))) + ((((((data["abs_maxbatch_slices2_msignal"]) / (np.cos((data["stdbatch_slices2"]))))) + (data["mean_abs_chgbatch_slices2"]))/2.0)))))/2.0))) + (data["minbatch"]))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))) * 2.0)) - (np.sin(((-((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["minbatch_slices2"]) * (((np.sin((np.cos((data["minbatch"]))))) - (complex(0,1)*np.conjugate(data["meanbatch_msignal"])))))))/2.0)))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_slices2"]) + (np.tanh((np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(np.cos((((data["minbatch_slices2"]) + (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["maxtominbatch_slices2"]))))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real((((((((data["abs_avgbatch_msignal"]) + (data["rangebatch_msignal"]))) - (complex(-2.0)))) + (((((-((np.where(np.abs(complex(10.0)) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))) + (np.cos(((-((((((data["stdbatch_msignal"]) + (data["abs_avgbatch_msignal"]))) - (complex(0,1)*np.conjugate(data["rangebatch_msignal"]))))))))))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((((((data["stdbatch_slices2"]) / (np.cos((data["stdbatch_slices2"]))))) / 2.0)) / (((((np.cos((((data["maxbatch_slices2_msignal"]) / (np.cos((np.where(np.abs((-((data["stdbatch_slices2"])))) > np.abs(np.cos((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0)))),complex(1.), complex(0.) )))))))) / 2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(((((((np.conjugate(((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * 2.0))) + (np.where(np.abs(data["abs_avgbatch_slices2"]) <= np.abs(np.cos((np.conjugate(np.cos((((np.cos((((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * 2.0)))) * 2.0))))))),complex(1.), complex(0.) )))) * 2.0)) + (((np.sin((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_slices2_msignal"]) * 2.0)) * ((((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.cos((np.cos((np.where(np.abs(np.cos((np.sin((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.tanh((data["abs_maxbatch_slices2"]))),complex(1.), complex(0.) )))))) <= np.abs(complex(2.0)),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )) + (np.where(np.abs(np.sin((data["abs_maxbatch_slices2_msignal"]))) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["stdbatch_slices2"]))) * (((data["maxbatch_slices2"]) + (((np.cos((np.cos((data["stdbatch_slices2"]))))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.cos((np.conjugate(data["maxbatch_slices2_msignal"])))) + (((np.cos((data["maxbatch_slices2_msignal"]))) + (np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.cos((((np.sin((np.sin((np.cos((((data["abs_avgbatch_slices2"]) + ((((data["minbatch_msignal"]) + (((data["abs_avgbatch_msignal"]) - (data["minbatch_msignal"]))))/2.0)))))))))) / 2.0)))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((((data["maxtominbatch_slices2_msignal"]) + (((data["rangebatch_slices2"]) / (np.sin((np.sin(((((data["signal_shift_+1"]) + (((data["abs_avgbatch_msignal"]) - (data["maxtominbatch_slices2_msignal"]))))/2.0)))))))))/2.0)) + (((data["rangebatch_slices2"]) / (np.sin((np.sin(((((data["signal_shift_+1"]) + (((data["abs_avgbatch_msignal"]) - (data["maxtominbatch_slices2_msignal"]))))/2.0)))))))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos(((((((((data["medianbatch_msignal"]) + (((data["medianbatch_msignal"]) * (data["minbatch_msignal"]))))/2.0)) * 2.0)) * (np.cos((data["medianbatch_msignal"]))))))) * (np.conjugate((((((((((data["stdbatch_slices2"]) * 2.0)) * (np.cos((np.sin((data["rangebatch_slices2"]))))))) + (((data["medianbatch_msignal"]) * (np.conjugate(data["rangebatch_slices2"])))))/2.0)) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((data["minbatch"]) + (complex(2.0))))) - (np.where(np.abs(data["minbatch"]) <= np.abs(np.where(np.abs(((np.cos((((((data["medianbatch_msignal"]) / 2.0)) * 2.0)))) + (np.where(np.abs(data["minbatch"]) <= np.abs(np.where(np.abs(np.tanh((data["abs_avgbatch_slices2_msignal"]))) <= np.abs(complex(2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) <= np.abs(np.sin((((((np.cos((data["maxtominbatch_msignal"]))) * 2.0)) * 2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.where(np.abs(np.sin((np.cos((data["medianbatch_msignal"]))))) > np.abs(np.conjugate((((data["abs_minbatch_slices2_msignal"]) + (np.conjugate(((np.sin((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))) / 2.0))))/2.0))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.sin((np.conjugate(((data["minbatch_msignal"]) / ((((complex(-2.0)) + (complex(0,1)*np.conjugate(np.where(np.abs(np.conjugate(data["minbatch_msignal"])) > np.abs(((np.sin((np.conjugate(((((data["minbatch_msignal"]) * 2.0)) / ((((complex(-2.0)) + (complex(0,1)*np.conjugate(np.sin((data["maxbatch_msignal"])))))/2.0))))))) / 2.0)),complex(1.), complex(0.) ))))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_msignal"]) / (((data["maxtominbatch"]) + (complex(0,1)*np.conjugate(data["maxtominbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((((data["medianbatch_slices2_msignal"]) / 2.0)) + (complex(0,1)*np.conjugate(np.sin((data["mean_abs_chgbatch_slices2"])))))/2.0)) * (((data["rangebatch_msignal"]) - (np.sin((np.cos((((np.sin(((((((((((data["maxbatch_slices2_msignal"]) / 2.0)) + (data["rangebatch_msignal"]))/2.0)) * (data["medianbatch_msignal"]))) / 2.0)))) * (complex(0.0))))))))))))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_-1_msignal"]) - (((data["minbatch_msignal"]) * (((data["signal_shift_-1_msignal"]) - (((data["minbatch_slices2_msignal"]) * (((((((data["medianbatch_msignal"]) / 2.0)) - (np.cos((data["minbatch_slices2_msignal"]))))) * 2.0)))))))))) - (np.sin(((((data["meanbatch_slices2_msignal"]) + ((-((((data["minbatch_slices2_msignal"]) - (np.cos((data["minbatch_slices2_msignal"])))))))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.cos((data["signal_shift_+1"]))) <= np.abs(np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((((data["maxtominbatch_slices2"]) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((data["minbatch"]) + (np.cos(((-((np.where(np.abs(np.sin((np.sin(((-((data["minbatch_slices2"])))))))) <= np.abs(np.where(np.abs(np.cos((complex(2.0)))) > np.abs(np.sin(((((np.sin((data["maxbatch_msignal"]))) + (np.cos((data["minbatch"]))))/2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["stdbatch_slices2"]))) * (data["maxbatch_slices2"]))) + (np.where(np.abs(np.where(np.abs(data["stdbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["stdbatch_slices2"])),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(data["stdbatch_slices2"])),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["maxbatch_slices2_msignal"]) * 2.0)))) * 2.0)) + (((((np.cos((((np.cos((((data["maxbatch_slices2_msignal"]) * 2.0)))) + (data["maxbatch_slices2_msignal"]))))) * 2.0)) + (np.conjugate(np.tanh((((np.tanh((data["stdbatch_slices2_msignal"]))) + (np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) > np.abs(((((data["maxbatch_slices2_msignal"]) * 2.0)) - (data["maxtominbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((-((complex(-1.0)))))))) +

                            0.100000*np.tanh(np.real((((np.cos((data["maxbatch_msignal"]))) + (np.where(np.abs(np.tanh((np.sin((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((((-((data["signal"])))) + (((data["mean_abs_chgbatch_slices2_msignal"]) - (np.where(np.abs(np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) <= np.abs(np.sin((np.sin((data["mean_abs_chgbatch_slices2"]))))),complex(1.), complex(0.) ))) > np.abs(((data["stdbatch_slices2"]) + (((((complex(0,1)*np.conjugate((((-((data["signal"])))) * 2.0))) * 2.0)) + (((((data["maxbatch_slices2"]) * 2.0)) / 2.0)))))),complex(1.), complex(0.) )))))/2.0))) +

                            0.100000*np.tanh(np.real((((((np.cos((np.cos(((((((np.where(np.abs(complex(3.0)) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )) + (((((((((((complex(3.0)) + (((data["abs_maxbatch_slices2"]) * (np.tanh((data["abs_maxbatch_slices2"]))))))/2.0)) + (np.cos((data["minbatch"]))))/2.0)) * 2.0)) - (data["minbatch"]))))/2.0)) / 2.0)))))) + (complex(3.0)))) + (data["minbatch"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.conjugate(((data["signal_shift_-1"]) * (data["abs_maxbatch_msignal"])))) / 2.0)) <= np.abs(np.cos((np.cos((((np.where(np.abs(complex(0,1)*np.conjugate(np.conjugate(((data["abs_avgbatch_slices2"]) / 2.0)))) > np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(np.sin((np.tanh((np.cos(((((complex(10.0)) + (np.conjugate(complex(0,1)*np.conjugate((((data["minbatch"]) + (((data["minbatch"]) + (np.where(np.abs(complex(2.0)) <= np.abs(np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(data["signal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))/2.0)))))/2.0)))))))) <= np.abs(data["minbatch"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((np.sin((complex(0,1)*np.conjugate(((complex(-1.0)) / 2.0))))))) <= np.abs(np.where(np.abs(complex(0.0)) > np.abs((((((data["abs_maxbatch_msignal"]) + (complex(0,1)*np.conjugate(data["abs_maxbatch_msignal"])))/2.0)) - (((data["mean_abs_chgbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((((data["signal_shift_-1_msignal"]) * (((((((data["signal_shift_+1"]) + (np.where(np.abs(np.where(np.abs(np.cos((((data["maxtominbatch"]) * 2.0)))) > np.abs(((np.sin((np.conjugate(complex(0.0))))) * 2.0)),complex(1.), complex(0.) )) <= np.abs(((data["maxtominbatch"]) - (np.conjugate(data["medianbatch_slices2_msignal"])))),complex(1.), complex(0.) )))) - (((((data["maxtominbatch"]) * 2.0)) * 2.0)))) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(((complex(3.10881686210632324)) - (((data["meanbatch_msignal"]) - ((-((((((np.conjugate(data["meanbatch_msignal"])) - (np.where(np.abs(((data["abs_maxbatch_slices2"]) - (complex(3.0)))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )))) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["maxbatch_slices2_msignal"]))) / (np.cos((((data["medianbatch_msignal"]) / (((np.conjugate(np.sin((np.cos((np.cos((data["medianbatch_msignal"])))))))) * 2.0)))))))) / (np.cos((((data["rangebatch_slices2"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((data["abs_avgbatch_msignal"]))) * 2.0)) - ((((complex(1.0)) + (((data["signal"]) * (((((((-((((data["maxbatch_slices2"]) / (np.sin((data["maxbatch_slices2"])))))))) + (((((complex(-1.0)) / 2.0)) * (((complex(-1.0)) * (data["maxbatch_msignal"]))))))/2.0)) * (data["maxbatch_msignal"]))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos((((np.cos((((data["minbatch_msignal"]) * 2.0)))) * 2.0)))) - (np.conjugate(np.conjugate(((np.conjugate(((data["medianbatch_slices2_msignal"]) / (((np.cos((((data["minbatch_msignal"]) * 2.0)))) / 2.0))))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_msignal"]) * (np.sin(((((data["abs_avgbatch_slices2_msignal"]) + ((((data["abs_avgbatch_slices2_msignal"]) + ((((np.where(np.abs(np.sin((data["abs_maxbatch"]))) > np.abs(((((data["medianbatch_slices2_msignal"]) / 2.0)) + (data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) )) + (data["abs_avgbatch_slices2_msignal"]))/2.0)))/2.0)))/2.0))))))) +

                            0.100000*np.tanh(np.real(data["abs_avgbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.cos((((((np.where(np.abs(np.where(np.abs(((complex(-1.0)) * (data["medianbatch_slices2_msignal"]))) <= np.abs(np.conjugate(((((np.where(np.abs(np.where(np.abs(((data["maxtominbatch_slices2"]) * 2.0)) <= np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["rangebatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"])))),complex(1.), complex(0.) )) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["rangebatch_slices2_msignal"]))) * (((data["abs_avgbatch_slices2"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((np.conjugate(data["signal_shift_+1"])) + ((-(((((np.where(np.abs(data["minbatch"]) <= np.abs(np.where(np.abs(data["minbatch"]) <= np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (np.where(np.abs(np.cos((np.cos((complex(0,1)*np.conjugate(np.sin((complex(8.11821556091308594))))))))) > np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )))/2.0))))))) + (np.cos((data["abs_avgbatch_msignal"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(np.sin((np.where(np.abs(((complex(1.0)) / 2.0)) <= np.abs((-((((complex(0,1)*np.conjugate(np.conjugate(np.conjugate(np.cos((np.where(np.abs(complex(-1.0)) <= np.abs(((data["abs_minbatch_msignal"]) / (complex(2.0)))),complex(1.), complex(0.) ))))))) * 2.0))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["abs_minbatch_msignal"]) - (np.cos((((data["abs_maxbatch"]) - (np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )))))))))) + (np.where(np.abs(data["abs_minbatch_msignal"]) <= np.abs((-(((((-((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(((data["maxtominbatch_slices2"]) - (data["abs_maxbatch_slices2"]))),complex(1.), complex(0.) ))))) - (data["maxtominbatch"])))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.where(np.abs(data["maxbatch_msignal"]) <= np.abs((((((data["maxtominbatch_msignal"]) + (data["maxbatch_msignal"]))/2.0)) + ((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.conjugate(((np.sin((data["signal_shift_-1_msignal"]))) * 2.0))))/2.0)))),complex(1.), complex(0.) )))) - (np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(complex(-2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["minbatch_msignal"]) + (np.where(np.abs(complex(8.66680240631103516)) > np.abs(np.conjugate(np.tanh((np.where(np.abs(np.tanh((((data["stdbatch_slices2"]) * (data["abs_minbatch_slices2"]))))) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((((np.cos((np.conjugate(((((np.cos((np.cos((np.conjugate(data["mean_abs_chgbatch_slices2"])))))) / 2.0)) + ((-((np.cos((((((np.cos((data["medianbatch_msignal"]))) / 2.0)) * (np.cos((np.conjugate(data["minbatch_msignal"])))))))))))))))) + (np.conjugate(data["minbatch_msignal"])))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.where(np.abs(np.cos((data["maxbatch_slices2"]))) > np.abs(((data["signal_shift_-1_msignal"]) / 2.0)),complex(1.), complex(0.) )))) > np.abs(np.tanh((((data["mean_abs_chgbatch_slices2"]) + (np.cos((data["signal_shift_+1"]))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((data["abs_minbatch_msignal"]) + (np.cos((np.where(np.abs(complex(3.0)) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )))))/2.0)) + (np.cos((data["maxbatch_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs((((-((data["medianbatch_msignal"])))) * 2.0)),complex(1.), complex(0.) )) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["maxtominbatch_slices2_msignal"]) * 2.0)) > np.abs(((((data["medianbatch_msignal"]) - (np.tanh((((np.where(np.abs((((data["minbatch_slices2_msignal"]) + (complex(1.0)))/2.0)) > np.abs(np.tanh((np.where(np.abs(((data["abs_avgbatch_slices2"]) / (data["maxtominbatch_slices2_msignal"]))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) - (complex(-3.0)))))))) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["abs_avgbatch_slices2_msignal"]) + ((((data["signal_shift_-1"]) + (np.cos((((complex(3.0)) - (np.where(np.abs(data["meanbatch_slices2"]) > np.abs(np.cos(((-((complex(0,1)*np.conjugate((((-((data["abs_avgbatch_slices2_msignal"])))) + ((-((((data["stdbatch_slices2"]) * 2.0))))))))))))),complex(1.), complex(0.) )))))))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["maxbatch_msignal"]) - (np.where(np.abs(np.cos((data["maxbatch_msignal"]))) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["stdbatch_msignal"]) <= np.abs((((data["minbatch"]) + (np.conjugate(data["medianbatch_slices2_msignal"])))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.cos((np.cos((np.cos((np.sin((((np.conjugate(np.where(np.abs(data["rangebatch_msignal"]) > np.abs(np.where(np.abs(np.where(np.abs(((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0)) > np.abs((((data["rangebatch_slices2"]) + (complex(6.0)))/2.0)),complex(1.), complex(0.) )) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) * 2.0)))))))))))) > np.abs(data["stdbatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1"])) +

                            0.100000*np.tanh(np.real(((((data["abs_minbatch_slices2_msignal"]) * (data["stdbatch_slices2"]))) + (np.where(np.abs(np.where(np.abs(data["rangebatch_slices2_msignal"]) > np.abs(complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])),complex(1.), complex(0.) )) <= np.abs(complex(1.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((data["abs_maxbatch_slices2_msignal"]))) > np.abs(np.sin((np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.sin((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) <= np.abs(np.where(np.abs(data["meanbatch_slices2"]) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["abs_maxbatch_msignal"]) - (((data["abs_minbatch_slices2"]) * (((np.where(np.abs(np.where(np.abs(np.sin((np.cos((np.where(np.abs(complex(4.0)) <= np.abs(complex(4.0)),complex(1.), complex(0.) )))))) > np.abs(np.where(np.abs(data["signal_shift_+1"]) > np.abs(((np.sin((complex(1.0)))) * 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)))))) + (np.conjugate(((data["rangebatch_slices2"]) * 2.0)))))))) +

                            0.100000*np.tanh(np.real((((((complex(4.63188266754150391)) + (((data["mean_abs_chgbatch_slices2"]) - (data["maxbatch_slices2_msignal"]))))) + (np.where(np.abs(np.sin((data["mean_abs_chgbatch_msignal"]))) <= np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos(((((-((np.sin(((((data["maxbatch_msignal"]) + (((np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(((data["maxbatch_msignal"]) * 2.0))),complex(1.), complex(0.) )) + (np.tanh((data["abs_avgbatch_slices2"]))))))/2.0))))))) + (((data["maxbatch_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs((-((complex(3.0))))),complex(1.), complex(0.) ))) + (((data["abs_minbatch_msignal"]) + (((((((data["mean_abs_chgbatch_slices2_msignal"]) - (np.cos((((complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])) + (((data["minbatch_msignal"]) + (data["abs_maxbatch_msignal"]))))))))) + (data["abs_maxbatch_slices2_msignal"]))) * (complex(1.0))))))))) +

                            0.100000*np.tanh(np.real((((((((((((data["minbatch_msignal"]) * (data["signal_shift_+1"]))) + (((((((complex(10.38442897796630859)) + (((complex(3.0)) + (data["minbatch_msignal"]))))) / 2.0)) * 2.0)))/2.0)) + (data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin(((((np.conjugate(((data["maxbatch_slices2_msignal"]) / 2.0))) + (np.where(np.abs((((((np.sin((data["signal_shift_+1"]))) + (np.where(np.abs(data["minbatch"]) <= np.abs(np.cos((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )))/2.0)) - (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(np.cos(((-((np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) ))))))),complex(1.), complex(0.) )))) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1_msignal"]) * (np.conjugate(((((data["minbatch_msignal"]) * (data["medianbatch_slices2"]))) + (complex(0,1)*np.conjugate(((((complex(-1.0)) * (data["medianbatch_slices2"]))) * (((data["signal_shift_+1_msignal"]) * (np.conjugate(complex(2.0)))))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) + (np.where(np.abs(np.tanh((data["medianbatch_msignal"]))) > np.abs(np.conjugate(data["minbatch"])),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["meanbatch_msignal"]) * (np.where(np.abs(np.where(np.abs(np.where(np.abs(data["meanbatch_slices2"]) > np.abs(((data["stdbatch_slices2_msignal"]) - (data["abs_minbatch_slices2"]))),complex(1.), complex(0.) )) <= np.abs(np.sin((data["meanbatch_slices2_msignal"]))),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["maxtominbatch"]) > np.abs(np.sin((data["stdbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) > np.abs(np.where(np.abs(np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(complex(14.44972515106201172)),complex(1.), complex(0.) )) <= np.abs(np.conjugate(data["maxbatch_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["medianbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((data["maxtominbatch"]) * (data["mean_abs_chgbatch_slices2"]))) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) / (data["stdbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((-((((np.where(np.abs(data["abs_maxbatch"]) > np.abs(np.tanh((complex(3.22034668922424316)))),complex(1.), complex(0.) )) + (data["abs_maxbatch"]))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.sin((np.conjugate((((((((data["maxbatch_msignal"]) + (((complex(0,1)*np.conjugate(np.tanh((np.conjugate(data["abs_minbatch_msignal"]))))) - (complex(0,1)*np.conjugate((((data["maxbatch_slices2"]) + (np.where(np.abs(((complex(0.0)) * 2.0)) > np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )))/2.0))))))/2.0)) * 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real(data["mean_abs_chgbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin((((((((((data["minbatch_slices2"]) + (data["stdbatch_slices2"]))/2.0)) * 2.0)) + (((data["minbatch_slices2"]) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch_slices2"]) - (np.cos((data["signal_shift_+1_msignal"]))))) - ((-((((data["abs_maxbatch_slices2"]) - (complex(0,1)*np.conjugate(data["abs_maxbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"]))) * (np.cos((((((data["maxbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"]))) + (np.cos((((data["rangebatch_slices2"]) + (complex(5.61049222946166992))))))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["abs_avgbatch_slices2_msignal"]) + (((complex(3.0)) / 2.0)))/2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["abs_maxbatch"]) * 2.0)) + (np.where(np.abs(((data["meanbatch_slices2"]) * 2.0)) <= np.abs(np.where(np.abs((-((((data["stdbatch_slices2"]) * 2.0))))) <= np.abs(((data["abs_maxbatch"]) - (((complex(3.0)) - (np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1"]) + (data["signal_shift_+1_msignal"]))) / ((((np.cos((data["minbatch"]))) + (np.conjugate(np.tanh((np.cos((np.sin((data["minbatch"])))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(data["maxbatch_slices2"])) <= np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(((((data["maxbatch_msignal"]) * (((complex(13.32183074951171875)) - (np.where(np.abs(((complex(0,1)*np.conjugate(np.tanh((data["signal_shift_+1_msignal"])))) / 2.0)) > np.abs(np.cos((np.tanh((complex(-1.0)))))),complex(1.), complex(0.) )))))) * (data["maxbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.cos((np.cos((np.cos((data["signal_shift_+1"]))))))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(0,1)*np.conjugate(((data["mean_abs_chgbatch_slices2_msignal"]) + (np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )) / (data["mean_abs_chgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(-1.0)) * (((np.conjugate(np.conjugate(np.conjugate(np.conjugate(((complex(0,1)*np.conjugate(data["stdbatch_slices2"])) + (data["stdbatch_slices2"]))))))) * (((np.sin((data["meanbatch_slices2_msignal"]))) + (((((data["medianbatch_slices2_msignal"]) + (((data["stdbatch_slices2"]) * (np.conjugate(data["stdbatch_slices2"])))))) * 2.0))))))))))) +

                            0.100000*np.tanh(np.real((((-(((((((((((((data["maxbatch_slices2"]) + (data["maxbatch_slices2"]))) * 2.0)) + (np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs((-((data["signal_shift_-1_msignal"])))),complex(1.), complex(0.) )))) * 2.0)) + (data["meanbatch_slices2"]))/2.0))))) * (data["signal_shift_-1_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) + (np.tanh((np.tanh((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["abs_avgbatch_slices2_msignal"]) + (data["minbatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["minbatch_msignal"]) + (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(np.cos(((((data["meanbatch_msignal"]) + (np.cos((data["minbatch_msignal"]))))/2.0)))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.where(np.abs((-((complex(3.0))))) <= np.abs(((data["signal"]) * 2.0)),complex(1.), complex(0.) )))) * ((((complex(0,1)*np.conjugate(data["meanbatch_msignal"])) + (complex(-2.0)))/2.0)))) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["minbatch"]) - (((data["maxtominbatch_slices2"]) / 2.0)))) + ((((data["minbatch"]) + (((data["minbatch"]) - (np.sin((((((data["minbatch"]) - (data["minbatch"]))) + (data["minbatch"]))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["maxbatch_msignal"]) + (np.cos((np.where(np.abs(np.tanh((data["abs_maxbatch_msignal"]))) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )))))/2.0)) / 2.0))))))  

    

    def GP_class_2(self,data):

        return self.Output( -2.199090 +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2"]) * (((((data["abs_maxbatch_msignal"]) + (data["meanbatch_msignal"]))) + (((((((data["abs_avgbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2"]))) + (data["meanbatch_msignal"]))) + (np.tanh((np.tanh((np.sin((np.conjugate(np.cos(((((((((data["meanbatch_slices2_msignal"]) - (data["minbatch_slices2"]))) + (data["medianbatch_slices2_msignal"]))/2.0)) * (data["rangebatch_slices2_msignal"]))))))))))))))))))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2_msignal"]) * (data["maxbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((((np.where(np.abs(((complex(0,1)*np.conjugate(data["maxtominbatch_msignal"])) * (((data["medianbatch_slices2"]) * 2.0)))) <= np.abs(((complex(-2.0)) / (data["mean_abs_chgbatch_slices2_msignal"]))),complex(1.), complex(0.) )) / 2.0)) + (np.conjugate(((data["maxbatch_slices2"]) * (((((data["abs_maxbatch_msignal"]) + (data["minbatch_msignal"]))) * 2.0))))))) / 2.0))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_slices2"]) + ((((((data["medianbatch_msignal"]) * ((((((((((data["maxbatch_slices2"]) * 2.0)) * 2.0)) + (((data["meanbatch_slices2"]) + (data["meanbatch_slices2"]))))) + (((((((data["maxbatch_slices2"]) * 2.0)) * 2.0)) * (data["maxbatch_slices2"]))))/2.0)))) + (data["maxbatch_slices2"]))/2.0)))) + (data["medianbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_slices2"]) / (np.cos((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) <= np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) / (np.cos((((data["abs_minbatch_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_msignal"]) + (((data["stdbatch_msignal"]) + (((data["meanbatch_slices2"]) + (((((data["meanbatch_slices2"]) * (data["abs_avgbatch_msignal"]))) + (((((data["meanbatch_slices2"]) + (((((data["meanbatch_slices2"]) * (data["abs_avgbatch_msignal"]))) + (np.cos((data["rangebatch_slices2"]))))))) * 2.0))))))))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2"]) * ((((data["abs_avgbatch_msignal"]) + ((((((((((((((data["maxbatch_slices2"]) + (complex(-2.0)))/2.0)) * (data["meanbatch_slices2_msignal"]))) * (np.cos((((complex(1.0)) / 2.0)))))) * (np.cos((np.cos((((((data["meanbatch_slices2_msignal"]) / 2.0)) / 2.0)))))))) / 2.0)) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real((((((((data["signal_shift_+1_msignal"]) + ((((np.cos((data["signal_shift_+1"]))) + (data["abs_avgbatch_slices2"]))/2.0)))/2.0)) + (data["maxbatch_slices2"]))) + ((((np.cos((data["maxbatch_slices2"]))) + ((((data["meanbatch_slices2_msignal"]) + (data["maxbatch_slices2"]))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["abs_avgbatch_slices2_msignal"]))) * 2.0)) + (data["stdbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_slices2"]) * (((((complex(0.0)) * (complex(13.99614143371582031)))) + ((((data["medianbatch_msignal"]) + ((((-((((data["maxbatch_slices2"]) - (((((np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )) * 2.0)) * 2.0))))))) / 2.0)))/2.0))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) * (((np.cos((((data["abs_maxbatch"]) * (((np.cos((data["minbatch_msignal"]))) * (((((np.cos((data["minbatch_msignal"]))) - (np.conjugate(data["minbatch_msignal"])))) - (data["abs_avgbatch_slices2_msignal"]))))))))) - (data["abs_maxbatch"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["minbatch_slices2_msignal"]) - (np.cos((complex(0,1)*np.conjugate(np.cos((((data["abs_avgbatch_msignal"]) - (data["signal_shift_+1"])))))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["abs_avgbatch_slices2_msignal"]) * (np.conjugate(((data["maxbatch_slices2"]) + (np.sin((np.sin(((((((-((data["medianbatch_msignal"])))) / 2.0)) - (((data["abs_maxbatch_msignal"]) + (((data["meanbatch_msignal"]) - (np.sin((data["mean_abs_chgbatch_slices2_msignal"])))))))))))))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch_msignal"]) * ((-((data["maxtominbatch_msignal"])))))) - (((data["stdbatch_msignal"]) / (data["maxtominbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(complex(1.0))) * ((((np.conjugate((-((((data["maxbatch_msignal"]) * ((((data["medianbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)))))))) + (((data["maxtominbatch_slices2_msignal"]) * (((data["signal"]) * 2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["meanbatch_slices2"]) / 2.0)))) + (((((data["abs_maxbatch_msignal"]) * (np.cos((((((np.conjugate(data["minbatch_msignal"])) * 2.0)) + (np.cos((((data["medianbatch_slices2_msignal"]) * 2.0)))))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2_msignal"]) / (np.cos((data["mean_abs_chgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["stdbatch_slices2"]) - (np.cos((data["stdbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(((np.cos((np.sin((data["abs_avgbatch_slices2_msignal"]))))) + (((np.cos((data["rangebatch_slices2"]))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.where(np.abs(((np.sin((np.where(np.abs(((np.sin((np.sin((np.cos((data["mean_abs_chgbatch_msignal"]))))))) * 2.0)) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )))) - (data["signal_shift_+1"]))) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((complex(3.0)) * (((np.cos((((((np.cos(((-((data["stdbatch_slices2"])))))) * (data["maxbatch_slices2"]))) / 2.0)))) / (np.conjugate(np.cos((((data["minbatch_msignal"]) * 2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((-((((complex(0.0)) / (((complex(-3.0)) * ((((np.cos((data["medianbatch_slices2_msignal"]))) + (((((data["medianbatch_msignal"]) / (complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"])))) * (data["abs_minbatch_slices2"]))))/2.0)))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["abs_maxbatch"]) - (complex(0,1)*np.conjugate(np.tanh((np.where(np.abs(np.sin((np.conjugate(((complex(0,1)*np.conjugate(np.tanh(((-((np.conjugate(((complex(-2.0)) + (data["abs_avgbatch_slices2"])))))))))) * 2.0))))) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) ))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real((((complex(0.0)) + (((((data["medianbatch_msignal"]) * 2.0)) * 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2"]) * ((-((((data["rangebatch_slices2_msignal"]) - ((((((((((complex(5.0)) * 2.0)) * 2.0)) * ((-((np.cos((data["minbatch_msignal"])))))))) + ((((((complex(5.0)) + (data["stdbatch_slices2"]))/2.0)) + (np.conjugate(data["minbatch_msignal"])))))/2.0)))))))))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_slices2"]) * (np.sin(((((data["rangebatch_msignal"]) + (complex(0,1)*np.conjugate((((data["maxbatch_slices2_msignal"]) + ((((data["rangebatch_msignal"]) + (np.where(np.abs((-((np.where(np.abs(complex(0,1)*np.conjugate(np.cos((data["maxtominbatch_slices2_msignal"])))) > np.abs(np.sin((data["meanbatch_slices2"]))),complex(1.), complex(0.) ))))) > np.abs(np.where(np.abs(data["rangebatch_slices2"]) > np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)))/2.0))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((data["meanbatch_slices2_msignal"]) - (data["minbatch_msignal"]))) + (complex(0,1)*np.conjugate(data["rangebatch_slices2"])))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["rangebatch_slices2"]))) / (((data["mean_abs_chgbatch_slices2"]) - (np.sin((np.sin((np.cos((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))))))))) +

                            0.100000*np.tanh(np.real((-((((np.cos((((complex(3.0)) * (((np.sin((data["meanbatch_msignal"]))) + (np.tanh(((((complex(3.0)) + (np.where(np.abs(((complex(0,1)*np.conjugate(complex(-1.0))) / 2.0)) <= np.abs((((-((((complex(6.0)) * 2.0))))) / 2.0)),complex(1.), complex(0.) )))/2.0)))))))))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) - (((data["maxtominbatch"]) / (((complex(1.0)) + (np.where(np.abs(((data["abs_avgbatch_slices2_msignal"]) / (((complex(1.0)) + (complex(0,1)*np.conjugate(((data["signal_shift_+1"]) + (((complex(1.0)) + (data["maxbatch_msignal"])))))))))) > np.abs(complex(1.0)),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real((-((((((np.sin((np.cos((np.cos(((-((data["abs_maxbatch_slices2"])))))))))) + (np.cos((data["minbatch_slices2_msignal"]))))) + ((((((((np.sin((data["signal"]))) * 2.0)) + (np.where(np.abs(complex(-2.0)) <= np.abs(((np.sin((((np.cos((data["minbatch_slices2_msignal"]))) + (data["minbatch_slices2_msignal"]))))) + (data["abs_maxbatch_slices2"]))),complex(1.), complex(0.) )))/2.0)) / 2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["minbatch_slices2_msignal"]) - (((np.cos((np.where(np.abs(((((data["minbatch_slices2_msignal"]) / (np.cos((((data["minbatch_slices2_msignal"]) - (((np.cos((np.cos((np.sin((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )))))))) * 2.0)))))))) * 2.0)) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )))) * 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((np.sin(((-((((np.conjugate(data["maxbatch_msignal"])) * 2.0))))))) / (np.cos((np.cos((np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(((data["rangebatch_slices2"]) * 2.0)),complex(1.), complex(0.) )))))))))) / (np.cos((np.where(np.abs(np.where(np.abs(data["maxbatch_msignal"]) > np.abs(data["minbatch"]),complex(1.), complex(0.) )) > np.abs(np.sin((((data["abs_minbatch_msignal"]) - (np.cos((data["abs_minbatch_slices2_msignal"]))))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.cos((np.where(np.abs(((data["medianbatch_msignal"]) * (np.where(np.abs(data["rangebatch_slices2_msignal"]) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))) <= np.abs(np.where(np.abs(((data["signal_shift_-1_msignal"]) / (data["medianbatch_slices2"]))) <= np.abs(np.cos((data["maxbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) + (np.cos((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["stdbatch_slices2"]))) * ((((((((data["stdbatch_slices2"]) + (complex(5.0)))/2.0)) * (((((data["stdbatch_slices2"]) * (complex(10.0)))) * 2.0)))) * ((((((data["mean_abs_chgbatch_slices2"]) + (((((((((complex(5.0)) + (data["stdbatch_slices2"]))/2.0)) + (data["stdbatch_slices2"]))/2.0)) * (((data["stdbatch_slices2"]) / 2.0)))))/2.0)) * (complex(5.0))))))))) +

                            0.100000*np.tanh(np.real(((np.cos(((((-((complex(0,1)*np.conjugate(np.conjugate(complex(2.55630683898925781))))))) + (((complex(-3.0)) * (data["medianbatch_msignal"]))))))) + (complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate((-((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(complex(2.0)),complex(1.), complex(0.) )))))) <= np.abs(complex(-3.0)),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real((((((-((((np.where(np.abs(data["maxtominbatch"]) <= np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (((data["mean_abs_chgbatch_msignal"]) - (data["minbatch"])))))))) + (complex(0,1)*np.conjugate(np.tanh((data["stdbatch_msignal"])))))) * (np.cos((((data["abs_avgbatch_msignal"]) - (np.where(np.abs(complex(-3.0)) <= np.abs(complex(-1.0)),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((-((np.where(np.abs(np.conjugate(data["rangebatch_slices2"])) <= np.abs(np.tanh((np.sin(((-((data["rangebatch_slices2"])))))))),complex(1.), complex(0.) ))))) <= np.abs(np.sin((data["rangebatch_slices2"]))),complex(1.), complex(0.) )) * (((complex(7.0)) * (((np.sin((np.cos((data["medianbatch_msignal"]))))) * (np.sin(((-((data["rangebatch_slices2"]))))))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((((np.conjugate(np.cos((np.conjugate(np.cos((data["signal"]))))))) + (data["abs_avgbatch_msignal"]))) / (np.cos((data["stdbatch_slices2"]))))))) + (((data["stdbatch_slices2"]) + (data["mean_abs_chgbatch_slices2"]))))) / (np.cos((data["stdbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real((((((((data["abs_minbatch_slices2_msignal"]) * (((data["meanbatch_slices2"]) + (data["rangebatch_slices2_msignal"]))))) + (data["abs_maxbatch"]))/2.0)) + (((data["abs_minbatch_slices2_msignal"]) * (((((data["abs_maxbatch"]) + (data["maxtominbatch"]))) + (data["medianbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((((data["mean_abs_chgbatch_slices2"]) * (data["minbatch"]))) - (((data["abs_minbatch_slices2_msignal"]) + (np.where(np.abs(((data["stdbatch_slices2"]) + (data["minbatch"]))) <= np.abs(data["minbatch"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.cos((np.sin(((((data["medianbatch_msignal"]) + (complex(0,1)*np.conjugate(complex(5.0))))/2.0)))))) + ((((data["medianbatch_msignal"]) + (complex(0,1)*np.conjugate((((-((complex(0,1)*np.conjugate(complex(5.0)))))) + ((-((complex(0,1)*np.conjugate(complex(5.0))))))))))/2.0))))) +

                            0.100000*np.tanh(np.real((((data["rangebatch_msignal"]) + (np.cos((data["minbatch"]))))/2.0))) +

                            0.100000*np.tanh(np.real((((((((((complex(-1.0)) * 2.0)) + (np.conjugate(((np.cos((((complex(1.0)) / (np.cos((data["rangebatch_msignal"]))))))) / (np.cos((data["stdbatch_slices2"])))))))) + (data["maxbatch_slices2"]))/2.0)) / (np.cos((data["stdbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch"]) + (data["maxbatch_slices2_msignal"]))) * (np.sin((((data["minbatch_slices2_msignal"]) - (np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(np.tanh((np.where(np.abs(((data["abs_maxbatch"]) * (((np.sin((((complex(-2.0)) / 2.0)))) / 2.0)))) > np.abs(np.sin((((data["abs_avgbatch_msignal"]) * (data["medianbatch_slices2"]))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) - (((np.where(np.abs(np.sin((np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(np.cos((data["minbatch_msignal"]))) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(((data["rangebatch_slices2"]) * 2.0)),complex(1.), complex(0.) ))) > np.abs(np.where(np.abs(np.cos((data["minbatch_msignal"]))) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) > np.abs(np.where(np.abs(np.cos((data["minbatch_msignal"]))) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * (data["abs_maxbatch"])))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch"]) * (data["maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((data["minbatch"]) + (np.cos((data["meanbatch_msignal"]))))/2.0)) <= np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(np.cos(((((data["meanbatch_slices2_msignal"]) + (np.conjugate(np.cos((data["minbatch"])))))/2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) - (complex(-1.0))))) +

                            0.100000*np.tanh(np.real(((((((((np.sin((data["abs_maxbatch"]))) * (((((data["maxtominbatch_slices2_msignal"]) + (data["minbatch_slices2"]))) / 2.0)))) * 2.0)) * 2.0)) + (np.tanh((np.where(np.abs(data["maxtominbatch"]) > np.abs(np.sin((((complex(-3.0)) * (np.cos((np.sin((np.where(np.abs(data["rangebatch_slices2_msignal"]) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )))))))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) - ((((data["maxtominbatch_slices2_msignal"]) + (np.where(np.abs((((complex(-3.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) <= np.abs((((((data["minbatch_slices2"]) + (((np.where(np.abs(complex(2.0)) <= np.abs(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(np.cos((((((data["minbatch_slices2_msignal"]) - (data["maxtominbatch_slices2_msignal"]))) - (data["maxtominbatch_slices2_msignal"]))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)))/2.0)) * 2.0)),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) - (((((np.cos((data["minbatch_msignal"]))) + (np.where(np.abs(((np.cos((data["minbatch_msignal"]))) / 2.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) )))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos((complex(3.0)))) - (((((data["abs_maxbatch"]) / 2.0)) - (((data["abs_maxbatch"]) * (np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.tanh((np.cos((data["abs_maxbatch"]))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((((((((data["minbatch_slices2"]) - (((complex(4.0)) + (data["minbatch_slices2"]))))) / 2.0)) + (np.where(np.abs(np.sin((data["mean_abs_chgbatch_msignal"]))) > np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_msignal"]) * (np.tanh((np.sin((((data["minbatch_slices2_msignal"]) - (((data["meanbatch_slices2"]) + (((data["abs_minbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.sin((np.where(np.abs(((data["medianbatch_msignal"]) * 2.0)) <= np.abs(((complex(0,1)*np.conjugate(((data["minbatch_slices2_msignal"]) * (((data["abs_minbatch_slices2_msignal"]) + (data["maxbatch_msignal"])))))) / 2.0)),complex(1.), complex(0.) )))))))))))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.sin((np.sin((((data["minbatch_msignal"]) - (data["meanbatch_slices2_msignal"]))))))))) * (((np.sin((np.sin((((np.sin((np.sin((np.cos((data["abs_avgbatch_slices2"]))))))) * (data["maxbatch_slices2"]))))))) + (data["maxbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((complex(0,1)*np.conjugate(data["stdbatch_msignal"])) - (((((data["minbatch_msignal"]) / 2.0)) - (complex(2.0)))))) * 2.0)))) + (((complex(0,1)*np.conjugate(data["stdbatch_msignal"])) - (data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_slices2_msignal"]) * 2.0)) + (((((((((data["abs_avgbatch_slices2"]) * (((np.sin((((((data["maxbatch_slices2_msignal"]) * 2.0)) - (complex(2.0)))))) * 2.0)))) * 2.0)) - (np.cos((data["abs_avgbatch_slices2"]))))) + ((((-((((data["maxbatch_slices2_msignal"]) * 2.0))))) / 2.0))))))) +

                            0.100000*np.tanh(np.real((((np.cos((data["maxbatch_slices2_msignal"]))) + (data["maxbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((((np.sin((np.tanh((complex(3.0)))))) + (complex(0,1)*np.conjugate((((complex(12.72160243988037109)) + ((((-((np.conjugate(((((data["abs_maxbatch_slices2"]) / 2.0)) - (data["abs_maxbatch_msignal"]))))))) / 2.0)))/2.0))))/2.0)))) + (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.cos(((((((((data["maxbatch_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))))/2.0)) - (complex(0,1)*np.conjugate(((data["abs_maxbatch_msignal"]) - (((data["mean_abs_chgbatch_slices2_msignal"]) - (np.cos((((data["abs_avgbatch_slices2_msignal"]) / 2.0))))))))))) + (data["maxbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-((np.cos((data["maxtominbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(((data["maxbatch_msignal"]) / (np.cos((np.cos((complex(1.0))))))))) - (np.sin((data["maxbatch_msignal"]))))) - (np.sin((((data["maxbatch_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real((((((data["rangebatch_msignal"]) / (np.cos((data["stdbatch_slices2"]))))) + (np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs((((np.sin((np.sin((data["stdbatch_slices2"]))))) + (data["stdbatch_slices2"]))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real((((((data["rangebatch_msignal"]) + (data["maxbatch_slices2"]))/2.0)) * (np.sin((((data["rangebatch_msignal"]) + (data["maxbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((((data["minbatch_msignal"]) - (np.conjugate(data["meanbatch_msignal"])))))) - ((-((np.cos((np.sin((data["minbatch_msignal"]))))))))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(((((data["rangebatch_slices2"]) * (data["meanbatch_msignal"]))) * (data["rangebatch_slices2"]))) <= np.abs(np.sin(((-((np.sin((np.sin((data["maxbatch_msignal"])))))))))),complex(1.), complex(0.) )) + (data["rangebatch_msignal"]))) * ((-((np.sin((data["abs_maxbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_avgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.cos(((-((((complex(0,1)*np.conjugate((-(((-((((data["meanbatch_msignal"]) - (((((complex(-2.0)) + (data["maxbatch_slices2_msignal"]))) * 2.0))))))))))) - ((((-((((data["meanbatch_msignal"]) - (((((complex(-2.0)) + (data["maxbatch_slices2_msignal"]))) * 2.0))))))) * 2.0))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))) + (((data["rangebatch_slices2"]) * (np.sin((((data["medianbatch_msignal"]) + ((-((data["rangebatch_slices2"]))))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.conjugate(np.sin((np.cos((np.sin((data["minbatch_slices2"])))))))),complex(1.), complex(0.) )) + (((data["minbatch"]) + (np.where(np.abs(np.conjugate((-((np.where(np.abs(data["minbatch_slices2"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )))))) > np.abs((-((data["maxbatch_msignal"])))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((((complex(8.34829139709472656)) * ((-((np.sin((data["abs_maxbatch_msignal"])))))))) + (np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs((-((np.sin((data["abs_maxbatch_msignal"])))))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["minbatch_msignal"]) * (((np.where(np.abs(((((data["medianbatch_slices2_msignal"]) * 2.0)) + (np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(((data["minbatch_msignal"]) / 2.0)),complex(1.), complex(0.) )))) > np.abs(((complex(1.0)) + (complex(0.0)))),complex(1.), complex(0.) )) * 2.0)))))) + (np.cos((((data["medianbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real((-((np.cos((((data["minbatch_msignal"]) + ((-((np.cos((((data["minbatch_msignal"]) + (np.cos((((data["minbatch_msignal"]) + (np.sin((np.cos((data["abs_avgbatch_slices2_msignal"])))))))))))))))))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((((data["maxbatch_msignal"]) - ((-((((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate(np.where(np.abs((-((((data["abs_minbatch_msignal"]) - ((-((np.sin(((-((((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate(data["abs_avgbatch_msignal"]))))))))))))))))) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))))))))))))))) +

                            0.100000*np.tanh(np.real(((((((np.tanh((data["maxbatch_slices2_msignal"]))) - (data["abs_minbatch_slices2_msignal"]))) + (np.where(np.abs((((data["abs_minbatch_slices2_msignal"]) + (np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs((((((((data["rangebatch_slices2"]) + (data["rangebatch_slices2"]))/2.0)) - (data["abs_minbatch_slices2_msignal"]))) / (data["abs_maxbatch_slices2"]))),complex(1.), complex(0.) )))/2.0)) <= np.abs(data["abs_minbatch_msignal"]),complex(1.), complex(0.) )))) / (np.conjugate(np.conjugate(np.cos((data["stdbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.tanh((np.sin((((((data["maxbatch_slices2_msignal"]) + (complex(-1.0)))) * 2.0)))))) * (((data["maxbatch_slices2"]) - (((np.sin((np.conjugate(complex(-1.0))))) * ((((np.tanh((np.sin((((((data["maxbatch_slices2"]) - (data["maxbatch_slices2"]))) * 2.0)))))) + (data["medianbatch_msignal"]))/2.0)))))))))) +

                            0.100000*np.tanh(np.real(((((data["abs_minbatch_slices2"]) + (data["stdbatch_msignal"]))) * (np.conjugate(data["mean_abs_chgbatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch_msignal"]) + (np.cos((np.cos(((((complex(-2.0)) + (complex(0,1)*np.conjugate(np.where(np.abs(((data["signal_shift_+1"]) * (np.sin(((((-((complex(1.0))))) * 2.0)))))) <= np.abs(((data["abs_avgbatch_slices2"]) / 2.0)),complex(1.), complex(0.) ))))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["maxbatch_msignal"]) - (((complex(14.87356090545654297)) - (np.where(np.abs(np.cos((data["minbatch"]))) > np.abs(np.where(np.abs(((((data["abs_minbatch_msignal"]) * 2.0)) / 2.0)) <= np.abs(np.where(np.abs(((((data["maxbatch_msignal"]) - (np.conjugate(complex(14.87356090545654297))))) * 2.0)) > np.abs(complex(1.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((-((np.sin((data["maxbatch_msignal"])))))) + (np.cos((np.where(np.abs(data["meanbatch_slices2"]) > np.abs(((np.where(np.abs(complex(4.0)) > np.abs(np.where(np.abs((((data["meanbatch_slices2"]) + (np.tanh((data["stdbatch_slices2_msignal"]))))/2.0)) > np.abs((-((np.sin((np.sin(((-((data["abs_maxbatch"]))))))))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) )))))/2.0))) +

                            0.100000*np.tanh(np.real((((np.cos(((((data["medianbatch_msignal"]) + (((((np.cos(((((data["medianbatch_msignal"]) + (complex(-2.0)))/2.0)))) - (data["rangebatch_slices2"]))) - ((-((data["meanbatch_slices2_msignal"])))))))/2.0)))) + (((complex(3.0)) + (data["minbatch"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((((complex(-1.0)) - (np.cos((data["minbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(data["maxbatch_msignal"])) +

                            0.100000*np.tanh(np.real((-((((np.tanh((((complex(2.0)) * (np.cos((((np.sin((((np.where(np.abs(np.cos((data["meanbatch_msignal"]))) <= np.abs(np.where(np.abs(np.where(np.abs(((((data["abs_maxbatch_slices2"]) + ((-((data["minbatch_msignal"])))))) / 2.0)) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(complex(2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)))) + (data["minbatch_msignal"]))))))))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((np.cos((((((data["medianbatch_msignal"]) / (np.tanh((np.cos((((((((((data["medianbatch_msignal"]) / 2.0)) * 2.0)) / 2.0)) / 2.0)))))))) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["abs_avgbatch_msignal"]))) + (((data["abs_avgbatch_msignal"]) * ((-(((-((complex(0,1)*np.conjugate(((((np.sin((data["abs_avgbatch_msignal"]))) + (np.cos((data["maxtominbatch_slices2"]))))) * 2.0))))))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(((((data["rangebatch_slices2_msignal"]) / 2.0)) * 2.0)) <= np.abs(((np.conjugate(data["maxbatch_msignal"])) + (np.where(np.abs((((data["maxbatch_slices2"]) + ((((np.cos((((data["rangebatch_msignal"]) * 2.0)))) + (np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(np.cos((data["abs_minbatch_msignal"]))),complex(1.), complex(0.) )))/2.0)))/2.0)) > np.abs((((data["meanbatch_msignal"]) + (complex(0.0)))/2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["abs_avgbatch_slices2"]) - (data["minbatch"]))) <= np.abs(np.where(np.abs(data["rangebatch_msignal"]) > np.abs(np.tanh((np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(((np.where(np.abs(data["signal_shift_-1"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) - (((data["rangebatch_msignal"]) - (data["abs_maxbatch"]))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_slices2"]) + ((((((data["maxbatch_msignal"]) + (data["rangebatch_msignal"]))/2.0)) - (np.cos((np.where(np.abs((((((data["abs_avgbatch_slices2"]) + (data["rangebatch_msignal"]))/2.0)) + (data["maxbatch_slices2_msignal"]))) <= np.abs(((data["maxbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))))) * (np.sin(((((((data["maxbatch_msignal"]) + (data["rangebatch_msignal"]))/2.0)) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((((np.conjugate(data["medianbatch_slices2"])) + (((((np.sin((data["meanbatch_slices2"]))) + (data["minbatch_slices2"]))) + ((((np.conjugate(complex(2.0))) + (data["mean_abs_chgbatch_slices2"]))/2.0)))))/2.0)) + (((data["meanbatch_slices2"]) * (((data["abs_avgbatch_msignal"]) / (np.cos((((data["maxtominbatch_msignal"]) / 2.0)))))))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1"]) * (np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs((((((np.where(np.abs(data["abs_avgbatch_slices2"]) <= np.abs(((data["maxtominbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )) * 2.0)) + (((data["abs_minbatch_slices2"]) - (((data["abs_avgbatch_slices2"]) / 2.0)))))/2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["maxtominbatch_slices2_msignal"]) + (np.where(np.abs((((((np.tanh((np.sin((((data["signal_shift_-1"]) * 2.0)))))) * (data["maxtominbatch_slices2_msignal"]))) + ((((((np.tanh((data["rangebatch_slices2_msignal"]))) - (data["mean_abs_chgbatch_msignal"]))) + (data["abs_minbatch_slices2_msignal"]))/2.0)))/2.0)) <= np.abs(complex(0,1)*np.conjugate(np.sin((((np.cos((complex(1.0)))) / 2.0))))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["abs_maxbatch"]) / 2.0)) / (np.cos((((np.where(np.abs(np.conjugate(((np.sin((complex(2.0)))) - (data["abs_avgbatch_slices2_msignal"])))) > np.abs(np.where(np.abs(((data["signal_shift_+1_msignal"]) / (data["abs_avgbatch_slices2"]))) > np.abs(((data["stdbatch_slices2_msignal"]) - (np.where(np.abs(complex(0,1)*np.conjugate(data["medianbatch_msignal"])) > np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(4.0)) <= np.abs(((complex(0,1)*np.conjugate(np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(((data["abs_maxbatch_slices2"]) - (data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) ))) + (data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((np.sin((data["abs_maxbatch"]))))) + (data["abs_avgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((((complex(0,1)*np.conjugate(np.conjugate(data["minbatch_msignal"]))) + (np.conjugate((-((np.sin((complex(-3.0)))))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["minbatch"]) + (np.cos((np.tanh(((((np.cos((np.where(np.abs((((np.cos(((((data["mean_abs_chgbatch_msignal"]) + (data["abs_minbatch_msignal"]))/2.0)))) + (data["signal_shift_-1"]))/2.0)) > np.abs(np.where(np.abs(data["minbatch"]) > np.abs(np.cos((complex(-1.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) + (data["abs_minbatch_msignal"]))/2.0)))))))) * (np.where(np.abs(data["signal"]) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["minbatch_slices2_msignal"]) * 2.0)))) + (((np.conjugate(((np.cos((((data["minbatch_slices2_msignal"]) * 2.0)))) + (((np.cos((((data["minbatch_slices2_msignal"]) * 2.0)))) + (np.cos((data["minbatch_slices2_msignal"])))))))) * 2.0)))) + (np.cos((((data["minbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) + (np.sin((((np.cos((np.where(np.abs(data["mean_abs_chgbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_msignal"])),complex(1.), complex(0.) )))) + (np.sin((data["minbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((((((((((data["rangebatch_slices2"]) * (complex(1.0)))) * (((data["rangebatch_slices2"]) * 2.0)))) + (data["rangebatch_slices2"]))) * (np.cos((((data["maxbatch_msignal"]) + (((data["rangebatch_slices2"]) * 2.0)))))))) + (np.cos((((data["maxbatch_msignal"]) + (((data["rangebatch_slices2"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((((((np.sin((data["abs_maxbatch_msignal"]))) + (np.where(np.abs(np.conjugate(np.cos((data["abs_maxbatch_msignal"])))) > np.abs(np.sin((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs((((((data["meanbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0)) * (data["minbatch_slices2"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.tanh((np.where(np.abs(np.where(np.abs(data["stdbatch_msignal"]) <= np.abs(complex(0.0)),complex(1.), complex(0.) )) <= np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )))) > np.abs(((np.conjugate(data["abs_maxbatch_slices2"])) + (((data["abs_maxbatch_slices2"]) * 2.0)))),complex(1.), complex(0.) )) + (np.cos((((((data["stdbatch_msignal"]) * 2.0)) + (((data["abs_maxbatch_slices2"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(np.where(np.abs(np.conjugate(np.conjugate(complex(0,1)*np.conjugate(((data["abs_avgbatch_slices2"]) * (complex(1.0))))))) > np.abs(np.conjugate(complex(2.0))),complex(1.), complex(0.) )) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) * (complex(1.0)))) - (data["maxtominbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.sin(((((np.where(np.abs(np.where(np.abs(data["minbatch_msignal"]) <= np.abs(((np.conjugate(data["minbatch_msignal"])) * 2.0)),complex(1.), complex(0.) )) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )) + (np.cos((((np.conjugate(data["minbatch_msignal"])) * 2.0)))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(((((((data["maxtominbatch_slices2_msignal"]) - (np.cos((np.conjugate(np.conjugate(np.tanh((np.cos((data["meanbatch_slices2_msignal"]))))))))))) / 2.0)) * (np.cos((np.sin((np.sin((data["mean_abs_chgbatch_slices2"]))))))))),complex(1.), complex(0.) )) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos(((((((data["abs_maxbatch"]) * 2.0)) + ((((np.where(np.abs(((data["abs_maxbatch"]) * 2.0)) > np.abs(np.tanh((complex(2.0)))),complex(1.), complex(0.) )) + (data["abs_maxbatch"]))/2.0)))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) + (np.sin((np.tanh((data["minbatch_msignal"]))))))) * (np.cos((((data["minbatch_msignal"]) + (np.sin((np.cos((((data["abs_maxbatch_msignal"]) + (((np.sin((((data["minbatch_slices2_msignal"]) + (np.sin((data["minbatch_slices2_msignal"]))))))) / 2.0))))))))))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) * (((((data["mean_abs_chgbatch_slices2_msignal"]) - (np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(np.where(np.abs(np.where(np.abs(data["rangebatch_slices2_msignal"]) <= np.abs((-((data["mean_abs_chgbatch_slices2_msignal"])))),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) * (data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["minbatch"]) - (data["meanbatch_slices2"]))) + (np.where(np.abs((((((np.where(np.abs(np.conjugate(((np.conjugate(complex(-1.0))) + (data["minbatch"])))) > np.abs(complex(1.0)),complex(1.), complex(0.) )) / 2.0)) + ((((data["medianbatch_slices2_msignal"]) + (np.tanh((data["minbatch"]))))/2.0)))/2.0)) > np.abs(data["minbatch"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(7.0)) > np.abs(np.cos((((((np.where(np.abs(((np.where(np.abs((((data["abs_maxbatch"]) + ((((data["abs_maxbatch"]) + ((-((np.cos((complex(1.0))))))))/2.0)))/2.0)) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) - (data["minbatch_msignal"]))) <= np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0)) * (data["abs_avgbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real((((((((complex(2.0)) + (np.conjugate(data["minbatch"])))/2.0)) / 2.0)) + (((data["minbatch"]) + (((data["abs_maxbatch_msignal"]) - (data["signal_shift_-1"])))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((((((data["minbatch_msignal"]) + ((-((data["meanbatch_msignal"])))))) + ((-((complex(0,1)*np.conjugate((-((np.conjugate(np.conjugate(data["minbatch_msignal"])))))))))))))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1_msignal"]) * (((data["abs_minbatch_msignal"]) - ((((-((((((data["rangebatch_slices2"]) + (np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(((((data["signal_shift_+1_msignal"]) * (data["maxbatch_slices2_msignal"]))) * (((data["stdbatch_slices2_msignal"]) - ((((-((data["abs_maxbatch_msignal"])))) * (data["meanbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )))) * 2.0))))) * (data["meanbatch_slices2_msignal"]))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((data["mean_abs_chgbatch_slices2"]) / (data["mean_abs_chgbatch_slices2"]))) > np.abs(((data["abs_minbatch_msignal"]) - (data["mean_abs_chgbatch_slices2_msignal"]))),complex(1.), complex(0.) )) - (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1_msignal"]) * ((((data["minbatch_slices2_msignal"]) + ((((data["abs_minbatch_slices2_msignal"]) + (np.sin((((np.conjugate(np.where(np.abs(np.sin((np.where(np.abs(((((data["signal_shift_+1_msignal"]) / 2.0)) * 2.0)) > np.abs(((((data["signal_shift_+1_msignal"]) * (data["abs_minbatch_slices2_msignal"]))) * 2.0)),complex(1.), complex(0.) )))) > np.abs(np.where(np.abs(data["meanbatch_msignal"]) > np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) / 2.0)))))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1"]) / ((((((complex(1.0)) + ((((np.where(np.abs((((-((complex(1.0))))) / 2.0)) > np.abs(((data["abs_minbatch_slices2"]) * ((((np.where(np.abs(((data["abs_maxbatch"]) * 2.0)) > np.abs(complex(1.0)),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_slices2"]))/2.0)))),complex(1.), complex(0.) )) + (data["minbatch"]))/2.0)))/2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) > np.abs(np.cos((complex(0.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(np.sin(((-((((data["signal_shift_-1"]) + (complex(2.0))))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2_msignal"]) + (np.sin(((((data["rangebatch_msignal"]) + (np.cos((np.where(np.abs(np.cos((np.conjugate(np.sin(((((data["rangebatch_msignal"]) + (((data["medianbatch_msignal"]) / 2.0)))/2.0))))))) <= np.abs(((data["abs_maxbatch_slices2"]) * 2.0)),complex(1.), complex(0.) )))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["minbatch_slices2_msignal"]) + (((data["minbatch_slices2_msignal"]) + (np.tanh((complex(0,1)*np.conjugate((-((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(((data["minbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))))))))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.where(np.abs(data["minbatch_slices2"]) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((((data["medianbatch_msignal"]) * ((((((((((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1_msignal"]))) * (complex(0.0)))) + (np.sin(((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0)))))) * 2.0)) + (data["signal_shift_-1_msignal"]))/2.0)))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) / (((((((((data["meanbatch_slices2_msignal"]) + (np.cos((complex(0,1)*np.conjugate(np.where(np.abs(np.conjugate(data["abs_avgbatch_slices2_msignal"])) > np.abs(np.cos((np.cos((((data["meanbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"]))))))),complex(1.), complex(0.) ))))))/2.0)) * (data["meanbatch_slices2_msignal"]))) + (np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(complex(3.0)) <= np.abs(data["minbatch"]),complex(1.), complex(0.) )) > np.abs(((data["rangebatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((data["meanbatch_slices2_msignal"]) + (np.where(np.abs(data["rangebatch_slices2"]) > np.abs(complex(1.0)),complex(1.), complex(0.) )))/2.0)) / (np.cos((((data["maxbatch_slices2_msignal"]) + (data["rangebatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) + (((((((data["maxbatch_slices2"]) * (data["mean_abs_chgbatch_slices2"]))) - (np.where(np.abs(np.sin((((data["mean_abs_chgbatch_slices2"]) * (data["maxbatch_slices2"]))))) > np.abs((-((((((complex(0,1)*np.conjugate(data["rangebatch_slices2_msignal"])) / 2.0)) * (data["meanbatch_slices2_msignal"])))))),complex(1.), complex(0.) )))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((-((complex(0,1)*np.conjugate(((data["minbatch_slices2"]) * (data["mean_abs_chgbatch_slices2"]))))))) - (((data["minbatch_slices2"]) * (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["meanbatch_slices2_msignal"]) + (((((((((data["meanbatch_slices2_msignal"]) / 2.0)) - (np.sin((data["rangebatch_slices2"]))))) / 2.0)) / 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["meanbatch_slices2_msignal"]) - (np.where(np.abs(((data["mean_abs_chgbatch_msignal"]) * 2.0)) > np.abs(((data["meanbatch_slices2_msignal"]) - (np.where(np.abs(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(np.cos((np.conjugate(np.tanh(((((-((data["meanbatch_slices2_msignal"])))) / 2.0))))))),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((np.sin((((data["medianbatch_slices2_msignal"]) / 2.0)))) / (((data["stdbatch_slices2_msignal"]) - (((data["mean_abs_chgbatch_slices2"]) - (data["minbatch_slices2"]))))))) + (np.tanh((complex(0,1)*np.conjugate((-(((((((data["medianbatch_slices2"]) - (np.tanh((data["medianbatch_slices2_msignal"]))))) + (((data["mean_abs_chgbatch_slices2"]) - (data["minbatch_slices2"]))))/2.0)))))))))/2.0)))) +

                            0.100000*np.tanh(np.real(((((((np.sin(((((-((((data["maxbatch_slices2_msignal"]) + (((data["stdbatch_slices2"]) - (complex(10.0))))))))) * 2.0)))) * 2.0)) * 2.0)) - (((np.sin(((((-((((data["maxbatch_slices2_msignal"]) + (((data["stdbatch_slices2"]) - (((data["stdbatch_slices2"]) - (complex(10.0))))))))))) * 2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2"]) + (((((data["mean_abs_chgbatch_slices2"]) + ((((-((np.where(np.abs(((((data["minbatch_slices2_msignal"]) - (((data["mean_abs_chgbatch_slices2"]) + (((data["mean_abs_chgbatch_slices2"]) * 2.0)))))) / 2.0)) <= np.abs(((((((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) * 2.0)) * 2.0)) * 2.0)),complex(1.), complex(0.) ))))) * 2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(np.sin((data["abs_maxbatch_slices2"]))) <= np.abs(np.sin((np.sin((data["maxbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) * 2.0)) * ((((data["abs_maxbatch_slices2"]) + (((((complex(0,1)*np.conjugate(data["meanbatch_msignal"])) + (np.tanh((np.sin((np.sin((data["abs_maxbatch_slices2"]))))))))) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(((data["signal_shift_-1_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(data["signal_shift_-1_msignal"]) <= np.abs(complex(12.13010692596435547)),complex(1.), complex(0.) )) <= np.abs((((data["signal_shift_-1_msignal"]) + (np.tanh((np.conjugate((-((((data["signal_shift_-1_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(complex(0,1)*np.conjugate(np.sin((np.where(np.abs(data["signal_shift_-1"]) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) ))))))))))))/2.0)),complex(1.), complex(0.) ))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["rangebatch_slices2_msignal"]) <= np.abs(((((np.conjugate(data["maxtominbatch"])) + (np.sin(((-((((np.conjugate(((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) / 2.0))) * 2.0))))))))) - (((np.where(np.abs(data["abs_maxbatch"]) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["maxtominbatch"]) + (np.where(np.abs(complex(0.0)) <= np.abs(np.where(np.abs(complex(0.0)) <= np.abs(np.tanh((complex(0,1)*np.conjugate((((-((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((complex(2.0)) * 2.0))))))) + (data["stdbatch_slices2"])))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)))) * (data["stdbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2"]) <= np.abs((((data["maxtominbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(complex(1.0)),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(((((((data["maxtominbatch_msignal"]) + (((complex(-2.0)) - (data["abs_maxbatch_msignal"]))))/2.0)) + (data["meanbatch_slices2"]))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["mean_abs_chgbatch_slices2"])) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(np.sin((complex(0.0)))) > np.abs((((data["stdbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0)),complex(1.), complex(0.) )) + (((((((((data["stdbatch_slices2_msignal"]) + (data["rangebatch_msignal"]))) - (data["signal"]))) + (data["stdbatch_slices2_msignal"]))) * (data["stdbatch_slices2_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2_msignal"]) * (np.where(np.abs(np.where(np.abs(np.tanh((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(((((data["stdbatch_slices2_msignal"]) / 2.0)) * 2.0)),complex(1.), complex(0.) )))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs((((np.cos((data["maxtominbatch_slices2_msignal"]))) + (((np.where(np.abs(data["meanbatch_slices2"]) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) / 2.0)))/2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((np.conjugate(((data["minbatch"]) / 2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.cos(((-((data["abs_avgbatch_slices2_msignal"])))))))) - (((np.cos((data["maxbatch_slices2_msignal"]))) / (np.tanh((np.cos(((-((np.cos((data["rangebatch_slices2"]))))))))))))))) +

                            0.100000*np.tanh(np.real((((((-((np.cos((((data["abs_maxbatch_slices2"]) / 2.0))))))) / (data["maxtominbatch"]))) + (((np.conjugate((((np.cos((data["medianbatch_msignal"]))) + (data["rangebatch_slices2"]))/2.0))) - (((((data["maxtominbatch_slices2_msignal"]) * 2.0)) * (np.sin((data["abs_maxbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.tanh((np.where(np.abs(((np.cos((((np.conjugate(((np.tanh((np.where(np.abs(np.cos((np.where(np.abs(complex(-3.0)) > np.abs(np.tanh((data["stdbatch_slices2"]))),complex(1.), complex(0.) )))) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))) + (data["medianbatch_msignal"])))) / 2.0)))) / 2.0)) > np.abs(np.where(np.abs(np.cos((np.cos(((-((((data["signal_shift_+1_msignal"]) / 2.0))))))))) > np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(complex(2.0))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["mean_abs_chgbatch_slices2_msignal"]) + ((-((np.where(np.abs(((data["stdbatch_slices2_msignal"]) - (np.where(np.abs((((np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["stdbatch_msignal"]))))) + (data["stdbatch_msignal"]))/2.0)) <= np.abs(data["minbatch"]),complex(1.), complex(0.) )))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))/2.0)) + (data["stdbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["abs_avgbatch_msignal"]) + (np.sin((np.where(np.abs(data["signal_shift_+1"]) <= np.abs(np.sin(((((((data["signal_shift_+1"]) + (np.conjugate(np.cos((data["abs_avgbatch_msignal"])))))) + (data["abs_avgbatch_msignal"]))/2.0)))),complex(1.), complex(0.) )))))))) * 2.0)) * (np.tanh((np.cos((((data["minbatch"]) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((((data["meanbatch_slices2_msignal"]) * (complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"])))) - (((complex(3.0)) + (np.where(np.abs(((data["meanbatch_slices2_msignal"]) / 2.0)) > np.abs(complex(5.13052034378051758)),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) - (((data["maxtominbatch"]) + (np.where(np.abs(data["maxtominbatch"]) > np.abs(np.where(np.abs(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(((data["maxtominbatch"]) + (complex(0,1)*np.conjugate(np.where(np.abs(((complex(-1.0)) + (data["abs_minbatch_msignal"]))) <= np.abs(complex(0,1)*np.conjugate(((np.where(np.abs(complex(2.0)) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) / 2.0))),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )) > np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) * (((data["signal_shift_+1_msignal"]) * (((((data["maxbatch_slices2"]) * 2.0)) + (np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["maxtominbatch_slices2"]) <= np.abs(((data["rangebatch_msignal"]) + (data["signal_shift_+1"]))),complex(1.), complex(0.) )) + ((-(((((np.cos((data["maxbatch_slices2"]))) + (np.cos(((((((complex(1.0)) + (data["medianbatch_slices2"]))/2.0)) + (complex(0,1)*np.conjugate(np.cos((data["signal_shift_+1"])))))))))/2.0)))))))) +

                            0.100000*np.tanh(np.real((((np.cos((((np.where(np.abs(complex(0,1)*np.conjugate(complex(1.0))) > np.abs((((((complex(0,1)*np.conjugate(data["abs_avgbatch_msignal"])) * 2.0)) + (((np.tanh((((data["stdbatch_slices2_msignal"]) / 2.0)))) + (data["mean_abs_chgbatch_slices2"]))))/2.0)),complex(1.), complex(0.) )) * (((((((data["minbatch_msignal"]) * (complex(1.0)))) / 2.0)) / 2.0)))))) + (np.cos((complex(1.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(np.tanh((np.cos((data["abs_avgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((data["signal_shift_+1_msignal"]) + (((((np.tanh(((((data["signal_shift_+1_msignal"]) + (data["stdbatch_msignal"]))/2.0)))) + (((data["signal_shift_+1_msignal"]) / (((data["meanbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(data["medianbatch_msignal"])))))))) * 2.0)))) * 2.0)) + (data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((((((((((np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )) + ((((((data["maxbatch_msignal"]) * 2.0)) + ((((np.tanh((data["rangebatch_slices2"]))) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))/2.0)))/2.0)) * 2.0)) + (data["signal_shift_+1"]))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["signal_shift_-1"]) * 2.0)) > np.abs(((data["signal_shift_+1"]) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch"]) + (np.conjugate(np.conjugate(((data["medianbatch_msignal"]) + (data["signal_shift_+1_msignal"]))))))) / (data["maxtominbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((((((complex(-2.0)) / 2.0)) / 2.0)) + (np.where(np.abs(np.where(np.abs(np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs((-((np.where(np.abs(data["signal"]) > np.abs(complex(1.0)),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )) <= np.abs(data["signal"]),complex(1.), complex(0.) )) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))/2.0)) > np.abs(complex(0,1)*np.conjugate(data["signal_shift_-1"])),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch"]) <= np.abs((((data["medianbatch_slices2"]) + (((data["meanbatch_msignal"]) + (((np.where(np.abs(data["minbatch"]) <= np.abs((((data["meanbatch_msignal"]) + (((data["minbatch_slices2_msignal"]) + (((complex(0.0)) + (data["mean_abs_chgbatch_slices2"]))))))/2.0)),complex(1.), complex(0.) )) - (data["medianbatch_slices2"]))))))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((((np.cos((((data["signal_shift_-1"]) / 2.0)))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((data["abs_maxbatch_slices2_msignal"]))) <= np.abs(np.cos((((data["meanbatch_msignal"]) * (np.cos((np.cos((np.where(np.abs(((data["stdbatch_msignal"]) + (data["stdbatch_msignal"]))) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["rangebatch_msignal"]) + ((((-((np.conjugate((((data["abs_avgbatch_slices2_msignal"]) + (np.where(np.abs(complex(-3.0)) <= np.abs(np.conjugate(data["rangebatch_msignal"])),complex(1.), complex(0.) )))/2.0)))))) * 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["rangebatch_slices2"]) > np.abs(complex(2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((np.cos((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(np.sin((((np.where(np.abs(np.cos((((data["stdbatch_msignal"]) - (((data["stdbatch_slices2"]) * 2.0)))))) <= np.abs((-((data["stdbatch_msignal"])))),complex(1.), complex(0.) )) / (data["stdbatch_msignal"]))))) > np.abs(np.sin((((data["stdbatch_msignal"]) - (data["stdbatch_msignal"]))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((((((np.where(np.abs(((data["signal_shift_+1_msignal"]) - (np.conjugate(np.sin((((data["meanbatch_msignal"]) * 2.0))))))) > np.abs(complex(0.0)),complex(1.), complex(0.) )) - (data["minbatch_msignal"]))) + (np.sin((complex(0,1)*np.conjugate(complex(-1.0)))))))))) - (((data["meanbatch_msignal"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((data["signal_shift_-1_msignal"]) - (np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(((data["signal_shift_-1_msignal"]) * 2.0)),complex(1.), complex(0.) )))))) * (data["abs_minbatch_msignal"]))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((complex(2.0)) + (((((data["minbatch_msignal"]) + (np.cos((((complex(1.0)) - (((((np.where(np.abs(complex(3.0)) <= np.abs(((data["meanbatch_slices2_msignal"]) - (((((data["minbatch_msignal"]) / 2.0)) / 2.0)))),complex(1.), complex(0.) )) * (((data["abs_minbatch_slices2_msignal"]) / 2.0)))) / 2.0)))))))) / 2.0)))/2.0)) * (data["signal_shift_-1"])))) +

                            0.100000*np.tanh(np.real(np.conjugate((((np.where(np.abs(((np.cos((data["medianbatch_slices2_msignal"]))) / 2.0)) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (((np.where(np.abs(np.cos((data["stdbatch_slices2_msignal"]))) <= np.abs(((data["rangebatch_slices2"]) - (data["stdbatch_msignal"]))),complex(1.), complex(0.) )) * 2.0)))/2.0)))) +

                            0.100000*np.tanh(np.real(np.sin(((((((complex(5.0)) / (data["stdbatch_msignal"]))) + (np.where(np.abs(((complex(-2.0)) * (((complex(6.12609148025512695)) / (data["stdbatch_msignal"]))))) <= np.abs((((((complex(5.0)) / (((complex(5.0)) / (data["stdbatch_msignal"]))))) + (((complex(5.0)) * (complex(5.0)))))/2.0)),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_avgbatch_slices2_msignal"]))) * (np.cos((((data["rangebatch_slices2"]) * (np.where(np.abs(np.tanh((data["meanbatch_msignal"]))) > np.abs(((data["maxtominbatch"]) - (np.where(np.abs(((((data["signal_shift_-1"]) / 2.0)) / 2.0)) > np.abs(np.cos((np.conjugate(((np.cos(((-((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))) / 2.0))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) * (((np.where(np.abs((((data["meanbatch_slices2"]) + (data["abs_minbatch_slices2_msignal"]))/2.0)) <= np.abs(((data["meanbatch_slices2"]) * 2.0)),complex(1.), complex(0.) )) + (data["stdbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((((np.sin(((((np.conjugate(((((data["maxtominbatch"]) * 2.0)) * 2.0))) + ((((np.sin((data["stdbatch_msignal"]))) + (data["maxtominbatch"]))/2.0)))/2.0)))) / 2.0)) - (((((data["maxtominbatch_msignal"]) * 2.0)) * (((np.sin(((((np.sin((data["stdbatch_msignal"]))) + (data["maxtominbatch"]))/2.0)))) * 2.0)))))) / 2.0)) - (complex(-1.0))))) +

                            0.100000*np.tanh(np.real(data["abs_avgbatch_msignal"])) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(data["signal"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) <= np.abs(np.conjugate((((((complex(0,1)*np.conjugate(data["medianbatch_msignal"])) / 2.0)) + (np.where(np.abs(np.where(np.abs(np.conjugate(data["maxtominbatch_slices2"])) <= np.abs(complex(0.0)),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_avgbatch_msignal"]) - ((((-((np.cos((np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(((((data["mean_abs_chgbatch_slices2_msignal"]) / (np.cos(((((data["maxtominbatch"]) + (np.conjugate(data["rangebatch_slices2"])))/2.0)))))) / 2.0)),complex(1.), complex(0.) ))))))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.tanh((((np.sin(((((data["maxtominbatch_slices2"]) + (((np.where(np.abs((((data["maxtominbatch_slices2"]) + (((((np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )) * 2.0)) * 2.0)))/2.0)) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )) * 2.0)))/2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxtominbatch"]) <= np.abs((((((((((((((np.tanh((data["medianbatch_slices2"]))) + (data["maxbatch_slices2"]))/2.0)) + (data["medianbatch_slices2"]))/2.0)) * (complex(4.33852767944335938)))) + (np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(complex(2.0)),complex(1.), complex(0.) )))/2.0)) * ((((((data["meanbatch_msignal"]) * (complex(4.33852767944335938)))) + (np.where(np.abs(complex(6.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) )))/2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((complex(8.0)) * (np.tanh((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(complex(3.0)),complex(1.), complex(0.) )))))) > np.abs(np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(np.where(np.abs(np.sin((((complex(1.0)) - (np.tanh((data["abs_maxbatch_msignal"]))))))) > np.abs(((complex(3.0)) * 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_minbatch_msignal"]) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) * (((data["signal_shift_+1_msignal"]) + (np.sin((np.cos((np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(((data["maxbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"]))) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.tanh((np.tanh((data["abs_avgbatch_slices2"]))))) <= np.abs(np.cos((np.tanh((np.tanh((complex(3.0)))))))),complex(1.), complex(0.) )) + (np.cos((data["minbatch"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((np.where(np.abs(np.cos((data["minbatch"]))) > np.abs(np.where(np.abs(np.cos((data["abs_avgbatch_slices2_msignal"]))) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) + (np.where(np.abs(np.sin(((-((((np.conjugate(data["maxtominbatch_slices2_msignal"])) + (((data["mean_abs_chgbatch_msignal"]) * 2.0))))))))) <= np.abs(np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.where(np.abs(complex(0,1)*np.conjugate(data["signal_shift_-1"])) > np.abs(((data["mean_abs_chgbatch_slices2"]) * 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((-((np.sin((data["maxbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) <= np.abs(((np.where(np.abs(np.where(np.abs(np.sin(((-((np.sin((data["abs_maxbatch"])))))))) <= np.abs(np.sin((np.sin((np.conjugate((((data["stdbatch_slices2_msignal"]) + (((complex(3.0)) + (complex(1.0)))))/2.0))))))),complex(1.), complex(0.) )) <= np.abs(complex(1.0)),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((np.sin((((data["minbatch_msignal"]) - (data["meanbatch_msignal"]))))) - (np.sin((((((data["minbatch_msignal"]) * 2.0)) - (data["meanbatch_msignal"]))))))) * 2.0)) - (np.sin((((((data["minbatch_msignal"]) * 2.0)) - (np.cos((np.sin((((((data["minbatch_msignal"]) * 2.0)) - (data["meanbatch_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(np.conjugate(np.cos(((((data["signal_shift_+1"]) + (complex(2.0)))/2.0))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs((((np.conjugate(data["abs_avgbatch_slices2_msignal"])) + ((-((complex(0,1)*np.conjugate(complex(0.0)))))))/2.0)) > np.abs(np.sin((complex(1.0)))),complex(1.), complex(0.) )) / 2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2"]) + ((-((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(np.tanh(((-((((np.sin((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])))) * 2.0))))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((complex(0,1)*np.conjugate((((-((np.conjugate(np.where(np.abs(np.sin((complex(0,1)*np.conjugate((-((((((np.tanh((np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["minbatch"])),complex(1.), complex(0.) )))) / 2.0)) * (data["medianbatch_slices2_msignal"]))))))))) <= np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )))))) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((np.where(np.abs(np.where(np.abs(np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"])),complex(1.), complex(0.) )) <= np.abs((((data["abs_minbatch_msignal"]) + (np.tanh((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["maxbatch_msignal"]))))))/2.0)),complex(1.), complex(0.) )) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) * 2.0)) + (data["maxbatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_msignal"]) > np.abs(np.sin((np.cos((np.conjugate(np.where(np.abs(np.where(np.abs(data["maxbatch_slices2"]) > np.abs(np.cos((np.conjugate(data["maxbatch_slices2"])))),complex(1.), complex(0.) )) <= np.abs(((np.cos((np.conjugate(np.where(np.abs(np.tanh((data["maxbatch_slices2"]))) <= np.abs(((np.where(np.abs(np.cos((complex(3.20007634162902832)))) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) ))))) * 2.0)),complex(1.), complex(0.) ))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((((data["mean_abs_chgbatch_slices2"]) - (data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(((np.where(np.abs(np.where(np.abs(((np.conjugate(data["abs_avgbatch_slices2_msignal"])) * 2.0)) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) > np.abs((((complex(0.0)) + (complex(0,1)*np.conjugate(data["signal_shift_-1"])))/2.0)),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) ))))  

    

    def GP_class_3(self,data):

        return self.Output( -2.012666 +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) * ((((((complex(-3.0)) - (data["meanbatch_slices2_msignal"]))) + (((((complex(-3.0)) - (data["abs_maxbatch_msignal"]))) + (np.where(np.abs(complex(-3.0)) > np.abs(((((complex(-3.0)) - (np.conjugate(complex(1.0))))) * (data["abs_minbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["minbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(((((np.conjugate((((-((np.conjugate(((np.sin((data["minbatch_slices2_msignal"]))) - (np.tanh((((data["abs_avgbatch_slices2"]) + (data["signal_shift_+1_msignal"]))))))))))) / 2.0))) + (((data["meanbatch_msignal"]) * 2.0)))) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch"]) * (np.sin((np.tanh((np.sin((data["minbatch_msignal"]))))))))) - (np.sin((np.cos((data["abs_maxbatch"])))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) + (data["signal"])))) +

                            0.100000*np.tanh(np.real((((((((complex(0,1)*np.conjugate(data["signal_shift_-1"])) - (((((np.cos((data["stdbatch_slices2_msignal"]))) * (data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0)))) + (data["signal_shift_+1"]))/2.0)) - (np.cos((((data["signal_shift_-1"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch_slices2_msignal"]) * (((((np.where(np.abs(data["abs_maxbatch"]) > np.abs(np.where(np.abs(complex(-3.0)) <= np.abs(np.where(np.abs(complex(-3.0)) <= np.abs(data["minbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (data["rangebatch_slices2"]))) + (((np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(((np.sin((((data["signal"]) * (complex(-2.0)))))) - (complex(9.85528564453125000)))),complex(1.), complex(0.) )) - (complex(9.85528564453125000))))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["minbatch_msignal"]))) * 2.0)) - (((np.where(np.abs(((data["minbatch_slices2_msignal"]) + (((data["signal"]) - (((((((((((np.tanh((np.sin((data["minbatch_msignal"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))) > np.abs(np.sin((data["signal"]))),complex(1.), complex(0.) )) * (complex(0.0))))))) +

                            0.100000*np.tanh(np.real((((np.sin((((complex(9.0)) - ((((-((data["minbatch_msignal"])))) * (np.sin((data["minbatch_msignal"]))))))))) + (((complex(9.0)) * ((((((complex(9.0)) * (np.sin((np.sin((data["minbatch_msignal"]))))))) + (data["minbatch_msignal"]))/2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) + (np.tanh((np.sin((((np.conjugate(np.sin((((np.sin((data["minbatch_msignal"]))) / 2.0))))) - (np.conjugate(np.sin((((np.cos((np.sin((np.cos((np.sin((data["minbatch_msignal"]))))))))) - (((np.cos((data["minbatch_msignal"]))) / 2.0)))))))))))))))) +

                            0.100000*np.tanh(np.real(((((((data["rangebatch_msignal"]) * (((((((complex(-1.0)) - (complex(-1.0)))) * (data["signal"]))) + (complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"])))))) + (data["signal"]))) + (data["signal_shift_-1"])))) +

                            0.100000*np.tanh(np.real(((((((np.sin((data["minbatch_msignal"]))) * 2.0)) * 2.0)) - (np.tanh((np.cos((((np.sin((((np.sin((np.cos((np.sin((data["minbatch_msignal"]))))))) * 2.0)))) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.conjugate(data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(((((np.sin((((complex(9.37986183166503906)) * 2.0)))) * 2.0)) + (((((np.sin((np.sin((data["minbatch_msignal"]))))) * 2.0)) + (np.conjugate(np.sin((np.sin((data["minbatch_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.sin((data["minbatch_msignal"]))) * 2.0))) + (np.tanh((((data["minbatch_msignal"]) + (np.sin((((data["minbatch_msignal"]) + (np.sin((((data["minbatch_msignal"]) + (np.tanh((np.sin((data["minbatch_msignal"])))))))))))))))))))) +

                            0.100000*np.tanh(np.real(((((((((data["stdbatch_slices2_msignal"]) * 2.0)) - (((data["mean_abs_chgbatch_msignal"]) - (data["stdbatch_slices2_msignal"]))))) - (data["stdbatch_slices2_msignal"]))) * (data["signal"])))) +

                            0.100000*np.tanh(np.real((((((data["minbatch_msignal"]) + (np.cos((((((((((data["signal_shift_-1"]) + (np.cos((data["abs_maxbatch_msignal"]))))) * 2.0)) + (np.where(np.abs(np.cos((np.conjugate(np.conjugate(np.cos((np.cos((data["abs_maxbatch_msignal"]))))))))) > np.abs(data["signal"]),complex(1.), complex(0.) )))) + (data["minbatch_msignal"]))))))/2.0)) / (np.cos((data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((-((np.cos((data["abs_maxbatch_slices2_msignal"])))))) * (data["abs_maxbatch_msignal"]))) - (np.where(np.abs(np.cos((np.cos((data["abs_maxbatch_slices2_msignal"]))))) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) + (((np.where(np.abs(data["stdbatch_slices2_msignal"]) <= np.abs(complex(0,1)*np.conjugate(np.tanh((data["signal_shift_-1_msignal"])))),complex(1.), complex(0.) )) * (np.sin((np.sin((((data["minbatch_msignal"]) - (np.sin((np.sin((data["minbatch_msignal"])))))))))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((np.tanh(((((np.sin((np.tanh((np.where(np.abs(np.sin((data["signal_shift_+1"]))) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )))))) + (((np.tanh((data["abs_maxbatch_slices2"]))) / 2.0)))/2.0)))))) * (data["abs_maxbatch_slices2"]))) * (np.tanh((np.sin((np.sin((((((data["maxbatch_msignal"]) / 2.0)) * (data["abs_maxbatch_slices2"])))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((complex(-3.0)) - (((((np.where(np.abs(data["abs_maxbatch"]) > np.abs(np.where(np.abs((-((((complex(1.0)) / 2.0))))) > np.abs(np.where(np.abs(data["stdbatch_slices2_msignal"]) <= np.abs(np.conjugate(data["maxbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (((data["abs_maxbatch_msignal"]) * 2.0)))) + (data["maxbatch_slices2_msignal"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_+1"]) + (np.cos((data["meanbatch_slices2"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((((complex(4.33285236358642578)) * 2.0)) - (((data["rangebatch_slices2"]) + (np.where(np.abs(complex(4.33285236358642578)) <= np.abs(((complex(4.33285236358642578)) / 2.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch"]) / (((np.cos(((((-((np.cos((((data["medianbatch_msignal"]) + ((((data["medianbatch_msignal"]) + (((((complex(0.0)) * ((-((np.cos((data["medianbatch_msignal"])))))))) * (np.sin((((np.cos((complex(0.0)))) / 2.0)))))))/2.0))))))))) * 2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(np.cos((((data["abs_minbatch_msignal"]) * (np.cos((np.cos((((data["abs_minbatch_msignal"]) * (((data["abs_maxbatch_msignal"]) + ((-((data["abs_maxbatch_slices2"])))))))))))))))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_msignal"]) * (((((np.sin((((((data["abs_maxbatch_msignal"]) * (np.sin((data["minbatch_msignal"]))))) - (data["meanbatch_slices2"]))))) * 2.0)) * 2.0)))) - (((np.conjugate(np.cos((np.conjugate(data["abs_maxbatch_slices2_msignal"]))))) * (np.cos(((-((data["abs_avgbatch_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["abs_avgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.sin(((-((data["abs_maxbatch_msignal"])))))) * (((data["maxtominbatch_slices2_msignal"]) - ((((((-((np.where(np.abs(complex(3.0)) > np.abs(complex(3.0)),complex(1.), complex(0.) ))))) + (data["mean_abs_chgbatch_slices2_msignal"]))) - (((data["rangebatch_msignal"]) / (np.cos((data["rangebatch_msignal"])))))))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["minbatch_msignal"]))) * 2.0)) - (np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(np.sin((((np.cos((np.sin((((np.sin((complex(2.0)))) + (complex(0,1)*np.conjugate(np.sin((data["minbatch_msignal"])))))))))) - (((np.sin((data["minbatch_msignal"]))) * 2.0)))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((((((np.sin(((-((np.conjugate(((np.sin((np.cos((np.cos((data["signal"]))))))) * 2.0)))))))) + (((np.cos((data["abs_avgbatch_msignal"]))) * 2.0)))/2.0)) + (data["signal"]))) / (np.cos((data["signal"])))))) +

                            0.100000*np.tanh(np.real((-(((((np.cos((((complex(-3.0)) / (np.cos((data["stdbatch_slices2"]))))))) + (((complex(-3.0)) / (np.cos((data["stdbatch_slices2"]))))))/2.0)))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate(data["minbatch_msignal"])))) + (((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate(data["maxbatch_msignal"])))))))) - ((-((np.where(np.abs(((np.sin((((data["abs_minbatch_msignal"]) + (((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate(data["stdbatch_msignal"])))))))) - (data["signal_shift_+1_msignal"]))) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["minbatch_slices2_msignal"]) + (np.tanh(((-((np.sin((((data["minbatch_slices2_msignal"]) + ((-((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(((data["minbatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) ))))))))))))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0)))))) * 2.0)))) + (np.sin((((data["maxbatch_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_slices2"]) + (complex(-3.0)))) / (np.cos((((((data["maxbatch_msignal"]) - (data["abs_maxbatch"]))) * (np.tanh((((complex(-3.0)) + (np.cos(((-((np.where(np.abs(data["abs_maxbatch"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))))))))))))))))) +

                            0.100000*np.tanh(np.real(((((((((np.sin((np.sin((np.sin((((data["maxbatch_slices2_msignal"]) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) - (np.sin((np.where(np.abs(((data["maxbatch_slices2_msignal"]) * 2.0)) <= np.abs((((-(((((np.sin((((data["maxbatch_slices2_msignal"]) * 2.0)))) + (data["minbatch_slices2"]))/2.0))))) * 2.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((-((((data["maxbatch_msignal"]) - (np.where(np.abs(((data["maxbatch_msignal"]) * 2.0)) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.cos(((((-((((data["maxbatch_msignal"]) - (np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) > np.abs(np.sin((((((complex(0.0)) + (data["rangebatch_slices2"]))) * 2.0)))),complex(1.), complex(0.) ))))))) * 2.0)))) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.sin((complex(0,1)*np.conjugate(complex(8.38843536376953125)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((((-((np.where(np.abs(data["minbatch_slices2"]) <= np.abs(np.where(np.abs(data["medianbatch_msignal"]) > np.abs(data["minbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) + (((data["signal_shift_-1"]) + (((complex(0,1)*np.conjugate(np.where(np.abs(((complex(0,1)*np.conjugate(data["minbatch"])) / 2.0)) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) ))) - (data["minbatch"]))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs((-((np.where(np.abs(np.where(np.abs(((data["signal"]) * (np.cos((data["mean_abs_chgbatch_slices2"]))))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )) * (data["minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real((((((((data["medianbatch_slices2_msignal"]) + (np.sin((data["minbatch_msignal"]))))) + (np.sin((((np.sin((complex(0,1)*np.conjugate(np.where(np.abs(np.tanh((data["abs_avgbatch_slices2_msignal"]))) > np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) ))))) + (((data["minbatch_msignal"]) + (data["medianbatch_slices2"]))))))))/2.0)) + ((((-(((-((np.sin((data["minbatch_msignal"]))))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["rangebatch_slices2"]))) * (((np.conjugate(((complex(12.86748600006103516)) * (data["meanbatch_slices2_msignal"])))) - (((((complex(0.0)) * (np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(np.where(np.abs(data["rangebatch_slices2"]) > np.abs(np.sin((data["medianbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["meanbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(((((data["signal_shift_-1"]) - (complex(0,1)*np.conjugate(data["rangebatch_slices2"])))) + ((((-((complex(-3.0))))) + (np.where(np.abs(complex(0,1)*np.conjugate(((data["rangebatch_slices2"]) + (np.cos((data["abs_avgbatch_slices2_msignal"])))))) <= np.abs(np.tanh((((np.where(np.abs(complex(-3.0)) <= np.abs(np.conjugate(data["maxtominbatch"])),complex(1.), complex(0.) )) * 2.0)))),complex(1.), complex(0.) )))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((data["minbatch"]) + (data["maxbatch_msignal"]))) > np.abs(((data["mean_abs_chgbatch_slices2_msignal"]) + (np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(((data["abs_avgbatch_slices2_msignal"]) * (((data["abs_avgbatch_slices2_msignal"]) / (np.sin((data["rangebatch_slices2"]))))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) + (((data["abs_avgbatch_slices2_msignal"]) / (np.sin((data["minbatch"])))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(6.34004259109497070)) > np.abs(np.where(np.abs(((complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(np.sin((((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) ))) / 2.0)) <= np.abs(np.tanh((((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / (np.cos((((data["mean_abs_chgbatch_slices2"]) + (((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(np.sin((data["medianbatch_msignal"]))),complex(1.), complex(0.) )) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["mean_abs_chgbatch_slices2"]) * (((data["mean_abs_chgbatch_slices2"]) + (np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(np.where(np.abs(data["abs_maxbatch"]) <= np.abs(np.tanh((data["mean_abs_chgbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) * (((((np.cos((((data["abs_maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))))) * (np.sin((data["medianbatch_slices2"]))))) + (np.conjugate(((data["abs_maxbatch_slices2_msignal"]) * (np.sin(((-((((data["abs_maxbatch_slices2_msignal"]) + (np.tanh((data["meanbatch_msignal"])))))))))))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((np.sin((np.conjugate(((np.conjugate(((data["minbatch_msignal"]) - (np.sin((((data["minbatch_msignal"]) - (np.conjugate(np.conjugate(data["minbatch_msignal"])))))))))) - (data["rangebatch_slices2"])))))) * 2.0))))) +

                            0.100000*np.tanh(np.real((-((((((data["rangebatch_slices2"]) - (((complex(-3.0)) / (((data["mean_abs_chgbatch_slices2"]) - (np.where(np.abs(np.sin((np.sin((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(np.sin((np.conjugate(np.where(np.abs(data["rangebatch_slices2_msignal"]) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )) > np.abs(((data["medianbatch_slices2_msignal"]) - (data["stdbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )))))))) / 2.0)))))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["minbatch_msignal"]) + (np.sin((((data["stdbatch_slices2"]) - (np.sin((((np.where(np.abs(data["meanbatch_slices2"]) > np.abs(np.where(np.abs(((np.cos((data["meanbatch_slices2_msignal"]))) * (data["meanbatch_slices2"]))) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)))))))))) - (complex(0,1)*np.conjugate(((data["minbatch_msignal"]) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_msignal"]) * (np.cos((data["abs_avgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate((-((((((data["rangebatch_msignal"]) / 2.0)) + (complex(0,1)*np.conjugate((((((data["maxtominbatch_slices2_msignal"]) / (data["abs_maxbatch_slices2"]))) + (data["signal_shift_+1_msignal"]))/2.0))))))))) - (data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2"]) / (np.cos((((data["meanbatch_msignal"]) + (np.where(np.abs(np.conjugate(np.where(np.abs(complex(14.72847461700439453)) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))) <= np.abs(np.where(np.abs(np.sin((np.cos((np.conjugate(np.where(np.abs(data["medianbatch_msignal"]) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) ))))))) <= np.abs(((np.conjugate(data["stdbatch_slices2"])) / 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real((-((((((data["meanbatch_msignal"]) / (((np.cos((((complex(0.0)) + (data["abs_maxbatch_msignal"]))))) / 2.0)))) + (np.cos((np.where(np.abs(np.cos((data["maxtominbatch"]))) > np.abs(((data["meanbatch_msignal"]) / (((np.cos(((((-((np.conjugate(data["maxtominbatch"]))))) * 2.0)))) / 2.0)))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2_msignal"]) / (np.sin((data["abs_maxbatch"]))))) + (((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) + (np.cos((((data["stdbatch_slices2"]) + (((complex(-2.0)) / (np.sin((data["abs_maxbatch"])))))))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.sin(((((((((((((data["maxbatch_slices2_msignal"]) * 2.0)) / 2.0)) * 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0)) * 2.0)))) * 2.0))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) * (np.tanh((np.sin(((((data["minbatch_msignal"]) + (np.tanh((np.where(np.abs(np.conjugate(np.cos((data["minbatch_msignal"])))) > np.abs(np.where(np.abs((((data["minbatch_msignal"]) + (np.sin(((((data["minbatch_msignal"]) + (np.sin((((np.where(np.abs(data["minbatch_msignal"]) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) * (data["minbatch_msignal"]))))))/2.0)))))/2.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["minbatch_msignal"]) - (((np.tanh((np.where(np.abs(np.where(np.abs(((data["maxbatch_slices2_msignal"]) * 2.0)) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))) * (data["medianbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((((np.tanh((np.sin((((np.where(np.abs((((-((((np.where(np.abs(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) > np.abs(((data["minbatch_msignal"]) / 2.0)),complex(1.), complex(0.) )) / 2.0))))) * 2.0)) > np.abs(((data["minbatch_msignal"]) + (data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )) - (((np.where(np.abs(complex(-3.0)) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )) + (data["abs_maxbatch_msignal"]))))))))) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((np.where(np.abs((((-(((((data["minbatch"]) + (np.cos(((-((((complex(3.0)) / (data["mean_abs_chgbatch_slices2"])))))))))/2.0))))) + ((((np.cos((complex(3.0)))) + (data["mean_abs_chgbatch_slices2"]))/2.0)))) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (((complex(3.0)) / (data["mean_abs_chgbatch_slices2"]))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch"]) + (((complex(1.0)) + ((((((data["minbatch"]) + (((data["minbatch_msignal"]) + (complex(1.0)))))/2.0)) / (data["mean_abs_chgbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((((((((((data["abs_maxbatch"]) + (data["stdbatch_msignal"]))/2.0)) * 2.0)) + (complex(-2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2_msignal"]) - ((((data["maxtominbatch"]) + (((data["abs_avgbatch_slices2_msignal"]) + ((((np.conjugate(np.tanh((data["stdbatch_slices2_msignal"])))) + (((((data["maxtominbatch"]) - (data["maxtominbatch"]))) / 2.0)))/2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch_msignal"]) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((np.where(np.abs(((np.cos((np.where(np.abs(np.cos((((data["abs_maxbatch_slices2"]) * 2.0)))) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )))) - (((np.sin((data["mean_abs_chgbatch_slices2"]))) / 2.0)))) <= np.abs(np.where(np.abs(np.cos((((data["maxbatch_msignal"]) / 2.0)))) > np.abs((-((((data["medianbatch_slices2"]) * (data["signal"])))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(np.conjugate(complex(0,1)*np.conjugate(complex(10.29189300537109375)))) <= np.abs(np.cos(((((data["mean_abs_chgbatch_msignal"]) + (complex(0.0)))/2.0)))),complex(1.), complex(0.) )) + (data["stdbatch_slices2"]))) + (np.tanh((((data["abs_avgbatch_msignal"]) - (data["mean_abs_chgbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.conjugate(data["maxtominbatch_msignal"])) / 2.0)) <= np.abs((((data["minbatch_slices2"]) + (data["stdbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + (((np.where(np.abs(np.sin(((-((np.conjugate(((((complex(1.0)) + ((-(((((np.where(np.abs((-((data["maxbatch_slices2_msignal"])))) > np.abs(((data["stdbatch_msignal"]) + (data["minbatch_msignal"]))),complex(1.), complex(0.) )) + (data["minbatch_msignal"]))/2.0))))))) + (data["maxbatch_slices2_msignal"]))))))))) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) + (complex(5.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(((complex(0.0)) * (np.cos((data["signal_shift_-1_msignal"]))))),complex(1.), complex(0.) )) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) + (((np.sin((((data["maxbatch_msignal"]) * 2.0)))) + (np.sin((((data["maxbatch_msignal"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + ((-((((data["maxtominbatch"]) - (np.where(np.abs(((np.sin((data["maxbatch_slices2"]))) * (data["abs_avgbatch_slices2"]))) > np.abs(((data["minbatch_msignal"]) - (data["minbatch_msignal"]))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((np.sin((data["minbatch_slices2_msignal"]))))) / ((-((data["maxtominbatch"]))))))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(((np.cos((((data["stdbatch_slices2"]) * 2.0)))) + (((np.sin((((complex(0,1)*np.conjugate(complex(2.0))) + (complex(0,1)*np.conjugate((((complex(-2.0)) + (data["maxbatch_slices2_msignal"]))/2.0))))))) * 2.0)))) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) ))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((((((data["minbatch_slices2_msignal"]) / (((np.tanh((np.sin((data["maxbatch_slices2"]))))) + (data["maxbatch_msignal"]))))) - (complex(7.36214351654052734)))) / (np.cos((data["abs_maxbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.cos((np.conjugate(((data["mean_abs_chgbatch_slices2"]) - (((complex(0.0)) - (np.sin((np.conjugate((-((((data["mean_abs_chgbatch_slices2"]) + (data["maxbatch_msignal"])))))))))))))))))) +

                            0.100000*np.tanh(np.real((-(((((-((((((-((data["abs_avgbatch_msignal"])))) + (data["minbatch"]))/2.0))))) / 2.0)))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((data["minbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.real((((data["meanbatch_slices2_msignal"]) + (np.conjugate(np.conjugate(data["abs_avgbatch_slices2_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["meanbatch_slices2_msignal"]) + (np.cos((np.cos((np.cos((np.where(np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.cos((data["meanbatch_slices2_msignal"]))))) <= np.abs(complex(0,1)*np.conjugate(data["abs_maxbatch"])),complex(1.), complex(0.) )))))))))))) * (((((((np.sin(((-((complex(0.0))))))) + (np.sin((complex(2.0)))))/2.0)) + (data["abs_maxbatch"]))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["abs_maxbatch"]) + (((data["medianbatch_slices2_msignal"]) + (data["rangebatch_msignal"]))))) - (((complex(0,1)*np.conjugate((-((np.where(np.abs(np.sin((((data["abs_maxbatch"]) + (((data["medianbatch_slices2_msignal"]) + (data["rangebatch_msignal"]))))))) > np.abs(complex(0,1)*np.conjugate(complex(0.0))),complex(1.), complex(0.) )))))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((complex(0,1)*np.conjugate(((data["signal_shift_-1_msignal"]) + ((((-((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"]))))) * 2.0))))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((((((((((data["abs_maxbatch_msignal"]) * 2.0)) + (np.sin((np.sin((((np.where(np.abs(complex(-3.0)) > np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0)))))))/2.0)) * 2.0)) + (np.where(np.abs(data["stdbatch_msignal"]) > np.abs(((data["abs_maxbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )))/2.0)) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos(((((-((np.conjugate(data["rangebatch_slices2"]))))) + (((((data["abs_avgbatch_msignal"]) - (complex(0,1)*np.conjugate(((((np.conjugate(data["rangebatch_slices2"])) + (data["signal_shift_+1_msignal"]))) + (((data["abs_avgbatch_msignal"]) + (np.cos((data["rangebatch_slices2"])))))))))) - (np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(((data["maxbatch_slices2"]) + ((-((data["medianbatch_slices2"])))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.sin((data["signal_shift_-1_msignal"]))))) - (((data["signal"]) - (((complex(0.0)) - (np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(((((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )) * (((complex(3.0)) - (complex(-3.0)))))) - ((((data["maxbatch_slices2_msignal"]) + (data["abs_maxbatch"]))/2.0)))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((np.cos((data["medianbatch_slices2_msignal"]))))) * ((((((complex(1.0)) + (((np.cos((np.cos((data["medianbatch_slices2_msignal"]))))) * 2.0)))/2.0)) + (data["meanbatch_slices2"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["minbatch"]) * (complex(0,1)*np.conjugate(np.cos(((((((np.tanh((data["minbatch"]))) + (np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["minbatch"]),complex(1.), complex(0.) )))/2.0)) + (data["abs_avgbatch_msignal"])))))))) + (data["abs_avgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((((data["stdbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))) + (data["abs_minbatch_msignal"]))) * (np.sin(((((data["abs_minbatch_msignal"]) + (data["rangebatch_msignal"]))/2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(data["mean_abs_chgbatch_msignal"])) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.tanh((np.cos((data["stdbatch_slices2"]))))) <= np.abs(np.where(np.abs(np.tanh((((np.cos((complex(9.0)))) / 2.0)))) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / (np.cos((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.sin((((data["minbatch_msignal"]) * ((((complex(-1.0)) + (np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(((data["maxbatch_msignal"]) - (np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(((((data["minbatch_msignal"]) * (np.conjugate(data["abs_minbatch_slices2"])))) - (((complex(9.0)) * (np.conjugate(np.sin((data["rangebatch_slices2"])))))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))/2.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((data["abs_minbatch_msignal"]) - (data["maxtominbatch"])))) + (np.where(np.abs(np.conjugate(((data["abs_minbatch_msignal"]) - (data["maxtominbatch"])))) <= np.abs(np.conjugate(data["maxtominbatch"])),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((((data["abs_avgbatch_msignal"]) - (np.where(np.abs(((data["minbatch_slices2"]) + (data["signal_shift_+1_msignal"]))) <= np.abs(((data["minbatch_slices2"]) + (data["signal_shift_+1_msignal"]))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(np.where(np.abs(((complex(11.09273624420166016)) - (complex(0,1)*np.conjugate(data["rangebatch_slices2"])))) <= np.abs((-((np.where(np.abs(((((data["signal_shift_+1_msignal"]) / 2.0)) - (np.conjugate(data["maxtominbatch_slices2"])))) <= np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"]))))) + (complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"]))))))) + (np.where(np.abs(np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.tanh((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.tanh((data["maxbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )) <= np.abs(np.cos((np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.cos((data["maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1_msignal"])) +

                            0.100000*np.tanh(np.real(np.cos((((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) - (np.cos((np.where(np.abs(((data["abs_maxbatch_slices2_msignal"]) / 2.0)) <= np.abs(((data["abs_maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["rangebatch_msignal"]) + (np.where(np.abs((((((((data["stdbatch_msignal"]) * (data["rangebatch_msignal"]))) - ((((complex(-2.0)) + ((((complex(0,1)*np.conjugate(complex(2.0))) + (np.tanh((data["stdbatch_msignal"]))))/2.0)))/2.0)))) + (complex(0,1)*np.conjugate(np.sin((data["minbatch_slices2_msignal"])))))/2.0)) > np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1"]) / (np.conjugate(np.cos((np.conjugate(np.conjugate(((((data["maxbatch_msignal"]) - (np.cos((((data["signal_shift_+1"]) - (data["medianbatch_msignal"]))))))) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_slices2"]) + (((data["meanbatch_slices2_msignal"]) - (data["stdbatch_msignal"]))))) * (np.sin((data["mean_abs_chgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )) + (data["abs_maxbatch"]))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(np.sin((((data["minbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(complex(1.0)))))))) + (((np.conjugate(np.sin((data["signal_shift_-1_msignal"])))) + (data["signal_shift_-1_msignal"]))))) + (data["signal_shift_-1_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxbatch_slices2_msignal"]) * (((np.tanh((np.where(np.abs((-((complex(3.0))))) > np.abs(((np.where(np.abs(complex(1.0)) > np.abs(complex(5.0)),complex(1.), complex(0.) )) + (np.sin((((((((data["meanbatch_slices2_msignal"]) + (np.conjugate(data["signal_shift_+1"])))/2.0)) + ((-((complex(3.0))))))/2.0)))))),complex(1.), complex(0.) )))) - (complex(0,1)*np.conjugate(np.cos((complex(3.0)))))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["abs_avgbatch_msignal"]) + (((((data["stdbatch_msignal"]) * 2.0)) + (np.where(np.abs((((complex(3.0)) + (((np.where(np.abs(complex(-3.0)) > np.abs(((data["minbatch_slices2"]) / 2.0)),complex(1.), complex(0.) )) + (((data["abs_avgbatch_msignal"]) - (data["meanbatch_msignal"]))))))/2.0)) <= np.abs(np.sin((data["rangebatch_slices2_msignal"]))),complex(1.), complex(0.) )))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((((((complex(0.0)) - (((data["stdbatch_slices2"]) * (data["abs_maxbatch_slices2_msignal"]))))) + ((-((np.sin((complex(1.0))))))))/2.0)) / 2.0)) + ((((-((np.sin((((data["stdbatch_slices2"]) * (np.sin((((data["stdbatch_slices2"]) * (data["abs_maxbatch_slices2_msignal"])))))))))))) * (data["maxbatch_slices2"]))))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((((((((complex(0,1)*np.conjugate(data["minbatch_msignal"])) * 2.0)) * 2.0)) + (np.cos((data["minbatch_msignal"]))))) - ((((data["maxbatch_msignal"]) + (data["minbatch_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(data["meanbatch_msignal"])) <= np.abs(((data["medianbatch_msignal"]) * (np.where(np.abs(((np.where(np.abs(data["signal_shift_+1"]) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )) * (complex(0,1)*np.conjugate(((np.where(np.abs(data["medianbatch_msignal"]) > np.abs(np.where(np.abs(np.cos((data["abs_minbatch_slices2"]))) <= np.abs(((data["medianbatch_slices2_msignal"]) * (complex(0,1)*np.conjugate(data["meanbatch_msignal"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0))))) > np.abs(np.where(np.abs(complex(1.0)) > np.abs(complex(-3.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((((((((complex(-3.0)) * (complex(0,1)*np.conjugate(data["rangebatch_slices2"])))) + (data["rangebatch_slices2"]))) - (data["meanbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_avgbatch_msignal"]) + (complex(7.91564416885375977)))) + (data["minbatch_msignal"]))) - ((((((data["maxtominbatch_slices2"]) + (data["rangebatch_slices2"]))/2.0)) - (complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate((((complex(-2.0)) + (complex(7.91564416885375977)))/2.0))) + (data["minbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_avgbatch_msignal"]) / (((data["meanbatch_msignal"]) - (((np.where(np.abs(((data["medianbatch_slices2"]) - (data["rangebatch_msignal"]))) > np.abs(np.cos((((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(np.where(np.abs(np.conjugate(((np.tanh((np.sin((((data["maxbatch_msignal"]) * 2.0)))))) / 2.0))) <= np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (data["minbatch_slices2"]))))),complex(1.), complex(0.) )) * 2.0)))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin(((((((-(((((np.where(np.abs(np.sin((data["abs_maxbatch"]))) > np.abs(((data["medianbatch_msignal"]) / 2.0)),complex(1.), complex(0.) )) + (data["medianbatch_slices2_msignal"]))/2.0))))) + (data["minbatch_msignal"]))) + (np.conjugate(np.where(np.abs(((data["signal_shift_-1_msignal"]) + (np.cos((data["signal_shift_-1"]))))) > np.abs(((data["medianbatch_msignal"]) * (np.tanh((np.cos((data["medianbatch_msignal"]))))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * ((((((((complex(2.0)) * (np.conjugate(data["medianbatch_slices2"])))) + (((complex(11.19394779205322266)) * (((complex(0.0)) + (data["abs_maxbatch_slices2_msignal"]))))))/2.0)) + (((data["medianbatch_slices2"]) + (complex(11.19394779205322266))))))))) +

                            0.100000*np.tanh(np.real((((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["maxtominbatch"]))/2.0)) - (((data["maxtominbatch"]) + (np.where(np.abs(np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(((np.where(np.abs(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )) - (((data["maxtominbatch"]) + (np.where(np.abs(complex(-1.0)) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(((data["medianbatch_msignal"]) * (((data["stdbatch_slices2_msignal"]) + (complex(13.05723571777343750)))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2_msignal"]) - (data["signal_shift_-1_msignal"]))) / ((((((np.sin((complex(0,1)*np.conjugate(((((data["meanbatch_slices2_msignal"]) / 2.0)) * (complex(-1.0))))))) * (np.where(np.abs(((complex(-1.0)) * 2.0)) > np.abs(np.cos((data["meanbatch_slices2_msignal"]))),complex(1.), complex(0.) )))) + (np.sin((data["rangebatch_slices2"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((((np.where(np.abs(((np.conjugate(data["meanbatch_slices2_msignal"])) / 2.0)) > np.abs(np.cos((data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )) * (((data["abs_avgbatch_slices2_msignal"]) / 2.0)))) + (np.conjugate(data["minbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["maxtominbatch_msignal"]) <= np.abs(np.tanh((np.where(np.abs(((data["meanbatch_slices2"]) * 2.0)) <= np.abs(complex(9.76048851013183594)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(3.41459226608276367))) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(data["minbatch"]) > np.abs(complex(0,1)*np.conjugate((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.conjugate((((((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) + (complex(0,1)*np.conjugate(np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))))/2.0)) * (data["meanbatch_slices2"]))))) +

                            0.100000*np.tanh(np.real((((((data["rangebatch_slices2_msignal"]) * (complex(0,1)*np.conjugate((-((((data["signal_shift_+1"]) * (np.sin(((((np.cos((data["signal_shift_+1"]))) + (((((np.cos((data["maxbatch_slices2"]))) + (np.where(np.abs(data["maxbatch_slices2"]) > np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )))) * 2.0)))/2.0)))))))))))) + ((((((data["medianbatch_slices2"]) * 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1_msignal"]) * (((data["maxbatch_slices2_msignal"]) / (data["mean_abs_chgbatch_msignal"]))))) - (((data["meanbatch_slices2_msignal"]) * (np.cos((np.tanh((data["signal_shift_+1_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((complex(0,1)*np.conjugate(data["minbatch_slices2"])) + (((np.sin((data["signal_shift_+1"]))) + ((((data["meanbatch_msignal"]) + (np.sin((((((((data["abs_avgbatch_msignal"]) * ((((((data["medianbatch_slices2"]) + (((data["abs_avgbatch_msignal"]) / 2.0)))/2.0)) - (data["mean_abs_chgbatch_slices2"]))))) - (((data["maxtominbatch_msignal"]) * 2.0)))) + (data["minbatch_slices2"]))))))/2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_+1_msignal"]) + (data["minbatch"]))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(((data["stdbatch_slices2"]) + (((complex(3.0)) + (((complex(10.48393917083740234)) * 2.0)))))),complex(1.), complex(0.) )))) / 2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((-(((((data["minbatch_msignal"]) + (np.cos((np.where(np.abs((-((complex(7.0))))) <= np.abs(np.sin((np.sin(((((data["minbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0)))))),complex(1.), complex(0.) )))))/2.0))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_msignal"]) - ((((((data["maxbatch_slices2"]) + (complex(-2.0)))) + (((np.cos((np.where(np.abs(np.sin((data["maxtominbatch_slices2"]))) > np.abs(np.where(np.abs(((np.conjugate((((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0))) * 2.0)) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) * 2.0)))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((np.sin((data["minbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((np.where(np.abs(np.tanh((((data["rangebatch_msignal"]) + (data["rangebatch_slices2"]))))) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )) * (complex(0,1)*np.conjugate(data["abs_maxbatch"])))) + (((np.cos((data["rangebatch_slices2"]))) - (data["minbatch_msignal"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["abs_minbatch_msignal"]) - (((data["maxtominbatch_slices2"]) + (np.tanh((((((data["abs_maxbatch"]) / 2.0)) + ((((((data["signal_shift_-1_msignal"]) + (data["meanbatch_slices2"]))/2.0)) / 2.0)))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(((np.tanh(((((-((((data["medianbatch_slices2_msignal"]) * 2.0))))) * 2.0)))) * (((data["abs_maxbatch_msignal"]) / 2.0)))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) + ((((complex(11.88483142852783203)) + (np.where(np.abs(data["meanbatch_msignal"]) > np.abs(complex(11.88483142852783203)),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs((-((np.conjugate(np.cos((data["stdbatch_msignal"]))))))) <= np.abs(np.cos((np.tanh((((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(np.cos((((complex(2.0)) * (np.tanh((data["maxbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )) * (data["minbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["stdbatch_slices2"]) <= np.abs(((((data["stdbatch_slices2_msignal"]) * (np.conjugate(data["abs_avgbatch_slices2"])))) / 2.0)),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["signal_shift_-1"]) - ((((np.where(np.abs((((data["abs_avgbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate((((data["medianbatch_slices2_msignal"]) + (np.conjugate(np.sin((((data["signal_shift_-1"]) - (data["rangebatch_slices2"])))))))/2.0))))/2.0)) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((((np.sin((data["rangebatch_msignal"]))) - (np.conjugate(np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(np.cos((((data["rangebatch_msignal"]) + (data["maxtominbatch_slices2"]))))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(((data["rangebatch_msignal"]) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((((((data["maxbatch_slices2_msignal"]) - (np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs((((data["maxtominbatch_msignal"]) + ((((data["maxtominbatch_msignal"]) + (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(((np.cos((((((data["maxbatch_slices2_msignal"]) - (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs((((data["maxtominbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))) * 2.0)))) * 2.0)),complex(1.), complex(0.) )))/2.0)))/2.0)),complex(1.), complex(0.) )))) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(((((np.sin((data["minbatch_slices2"]))) * (((np.cos((data["maxbatch_msignal"]))) * 2.0)))) / 2.0)) > np.abs(np.cos((np.cos((((data["medianbatch_slices2"]) * (np.cos((((complex(3.0)) / 2.0)))))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((((((np.cos((data["meanbatch_msignal"]))) * (np.sin((data["minbatch_msignal"]))))) - (complex(0,1)*np.conjugate(np.sin((((np.cos((data["meanbatch_msignal"]))) * 2.0))))))) * (data["abs_maxbatch"]))) - (np.cos((data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(data["stdbatch_slices2"])) <= np.abs(np.where(np.abs(np.conjugate(complex(1.0))) <= np.abs(np.sin((data["mean_abs_chgbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["meanbatch_slices2_msignal"]) - (data["stdbatch_slices2"]))) <= np.abs(((data["meanbatch_slices2_msignal"]) * (((((((data["stdbatch_slices2"]) * ((-((np.cos((data["meanbatch_slices2_msignal"])))))))) * (((((data["meanbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))) * (data["abs_minbatch_slices2_msignal"]))))) * (data["medianbatch_slices2_msignal"]))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(10.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(((data["signal_shift_+1"]) * (((data["maxtominbatch"]) / 2.0)))),complex(1.), complex(0.) )) <= np.abs((-((np.where(np.abs(np.sin((data["minbatch_slices2_msignal"]))) > np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_-1"]) + (((data["signal_shift_+1_msignal"]) - (data["meanbatch_slices2"]))))) + (((data["signal_shift_-1"]) + (((np.conjugate(((data["signal_shift_+1_msignal"]) - (data["meanbatch_slices2"])))) / ((-((np.cos((((data["maxbatch_slices2_msignal"]) * 2.0)))))))))))))) +

                            0.100000*np.tanh(np.real(((((((((((np.sin((((data["maxtominbatch_msignal"]) + (data["abs_maxbatch_slices2"]))))) * 2.0)) + (data["maxtominbatch_msignal"]))/2.0)) + (data["stdbatch_slices2_msignal"]))) + (data["maxtominbatch_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((((((data["abs_avgbatch_msignal"]) + (complex(0,1)*np.conjugate(((np.where(np.abs((-((data["signal_shift_-1"])))) > np.abs(((data["stdbatch_slices2_msignal"]) - (data["abs_avgbatch_msignal"]))),complex(1.), complex(0.) )) + (data["abs_avgbatch_msignal"])))))) + (complex(0,1)*np.conjugate(np.where(np.abs(np.cos((complex(0,1)*np.conjugate(data["signal_shift_-1"])))) <= np.abs(((data["stdbatch_slices2_msignal"]) - (data["signal_shift_-1"]))),complex(1.), complex(0.) ))))) + ((-((data["signal_shift_-1"]))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(np.tanh((data["mean_abs_chgbatch_slices2"]))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(data["rangebatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(np.cos((((data["signal_shift_+1"]) - (((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["abs_avgbatch_slices2_msignal"]))) - (complex(0,1)*np.conjugate(((data["signal_shift_+1"]) - ((((np.cos((((data["signal_shift_+1"]) - (data["abs_avgbatch_slices2_msignal"]))))) + ((-((data["abs_avgbatch_slices2_msignal"])))))/2.0)))))))))))) +

                            0.100000*np.tanh(np.real((((((((np.where(np.abs(complex(7.0)) <= np.abs(((np.cos((data["maxtominbatch_slices2"]))) + (((np.conjugate(((complex(-1.0)) / 2.0))) + (((np.conjugate(((np.where(np.abs(complex(7.0)) <= np.abs(((data["minbatch_msignal"]) + (data["minbatch_msignal"]))),complex(1.), complex(0.) )) / 2.0))) + (data["minbatch_msignal"]))))))),complex(1.), complex(0.) )) / 2.0)) + (data["minbatch_msignal"]))) + (complex(7.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(complex(9.54168128967285156)),complex(1.), complex(0.) )) > np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((data["maxtominbatch_msignal"]))) * ((((complex(0,1)*np.conjugate((-((data["abs_avgbatch_msignal"]))))) + (np.where(np.abs(data["rangebatch_msignal"]) > np.abs(((data["signal_shift_+1_msignal"]) - (np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["minbatch"]) <= np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.where(np.abs((((-(((-((((((np.where(np.abs(data["minbatch"]) > np.abs((-((complex(0,1)*np.conjugate(((data["abs_maxbatch"]) + (complex(2.0)))))))),complex(1.), complex(0.) )) / 2.0)) + (np.cos((complex(2.0)))))))))))) - (data["abs_maxbatch_msignal"]))) <= np.abs(np.tanh((complex(-2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((np.sin((data["maxbatch_msignal"]))))) > np.abs((((np.where(np.abs(data["signal_shift_-1"]) <= np.abs(data["stdbatch_msignal"]),complex(1.), complex(0.) )) + (np.where(np.abs(np.tanh((np.sin(((-((data["medianbatch_slices2_msignal"])))))))) > np.abs((((np.where(np.abs(((np.conjugate(((complex(-3.0)) * (np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) / 2.0)) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["medianbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((complex(0,1)*np.conjugate(((np.conjugate((-((data["rangebatch_msignal"]))))) / 2.0))) + (((np.conjugate(data["rangebatch_slices2"])) + (((data["abs_avgbatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) + (((((data["maxbatch_slices2"]) + (np.sin((np.where(np.abs(data["rangebatch_msignal"]) > np.abs(np.cos(((-((data["rangebatch_msignal"])))))),complex(1.), complex(0.) )))))) / 2.0))))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["rangebatch_msignal"]) - (((data["meanbatch_slices2_msignal"]) + (((np.sin((data["abs_minbatch_slices2"]))) * 2.0)))))) - (((np.sin((data["abs_minbatch_slices2"]))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["minbatch_msignal"]) + (np.where(np.abs(data["minbatch_msignal"]) > np.abs(np.where(np.abs(np.where(np.abs((-((data["minbatch_slices2"])))) > np.abs((((data["minbatch_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )) > np.abs(np.cos((data["minbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((data["abs_maxbatch"]) + (data["maxbatch_msignal"]))/2.0)) <= np.abs((((data["maxtominbatch_slices2_msignal"]) + (np.where(np.abs(((data["abs_minbatch_slices2_msignal"]) * (np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(np.sin((data["medianbatch_msignal"]))),complex(1.), complex(0.) )))) > np.abs(complex(2.0)),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["rangebatch_slices2"]) * 2.0)) * ((((((((np.cos((data["stdbatch_slices2"]))) / (np.cos(((-((data["mean_abs_chgbatch_slices2"])))))))) * (data["signal"]))) + (complex(0,1)*np.conjugate(np.cos((((data["signal"]) - (np.cos((data["mean_abs_chgbatch_slices2"])))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) * (np.conjugate(data["minbatch_msignal"])))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(((np.cos(((-((np.where(np.abs(np.cos((((data["rangebatch_slices2"]) + (complex(-3.0)))))) <= np.abs(np.conjugate(((((data["medianbatch_slices2"]) * 2.0)) * 2.0))),complex(1.), complex(0.) ))))))) - (((data["meanbatch_slices2"]) * 2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["signal"]) + ((-((np.cos((np.conjugate(data["abs_avgbatch_slices2"]))))))))) + (((data["rangebatch_slices2"]) + (np.where(np.abs(np.cos((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )))) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.tanh(((((data["meanbatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))) * ((((((data["maxtominbatch_slices2_msignal"]) + (np.cos((np.tanh(((((data["meanbatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))))))/2.0)) + (data["mean_abs_chgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["abs_avgbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) > np.abs(((complex(0,1)*np.conjugate((((-((complex(0,1)*np.conjugate(data["meanbatch_slices2"]))))) + (data["abs_minbatch_slices2_msignal"])))) + ((((data["abs_avgbatch_slices2"]) + (np.cos((((complex(-1.0)) / 2.0)))))/2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * (((np.where(np.abs(((np.conjugate(data["abs_maxbatch_slices2"])) - (data["maxbatch_msignal"]))) <= np.abs(np.tanh((np.cos((np.where(np.abs(((np.conjugate(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * (data["abs_maxbatch_slices2"])))) - (data["abs_maxbatch_slices2_msignal"]))) <= np.abs(np.cos((complex(12.93422603607177734)))),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )) - (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((np.cos((data["meanbatch_msignal"]))) + (np.where(np.abs(np.where(np.abs(np.cos((data["meanbatch_msignal"]))) <= np.abs((((np.cos((data["meanbatch_slices2"]))) + (data["meanbatch_msignal"]))/2.0)),complex(1.), complex(0.) )) > np.abs((-((np.conjugate(data["medianbatch_slices2"]))))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(-3.0)) * (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((data["abs_maxbatch_slices2"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.cos((data["meanbatch_msignal"]))) > np.abs((((data["meanbatch_msignal"]) + (complex(0.0)))/2.0)),complex(1.), complex(0.) )) <= np.abs(complex(5.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(complex(6.49832868576049805))) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.sin((data["maxbatch_slices2_msignal"]))) > np.abs(np.cos((np.where(np.abs(np.where(np.abs(np.sin((np.sin((np.sin((np.sin((data["abs_minbatch_slices2"]))))))))) > np.abs(np.cos((complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])))),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(np.conjugate(complex(0,1)*np.conjugate(data["rangebatch_slices2_msignal"]))) > np.abs(complex(0.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((data["maxtominbatch"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) > np.abs(((((-((((((-((data["medianbatch_slices2"])))) + (data["abs_avgbatch_msignal"]))/2.0))))) + ((((np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )) + ((((complex(8.45835113525390625)) + (((data["abs_minbatch_slices2"]) + (data["medianbatch_slices2"]))))/2.0)))/2.0)))/2.0)),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(np.conjugate(data["signal"])),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.cos((((complex(-2.0)) - (data["mean_abs_chgbatch_msignal"]))))) * ((((data["medianbatch_msignal"]) + (np.cos((((np.cos((((((complex(-2.0)) - (data["mean_abs_chgbatch_msignal"]))) - (data["mean_abs_chgbatch_msignal"]))))) * ((((data["medianbatch_msignal"]) + (np.cos((np.cos((np.cos((((complex(-2.0)) - (data["mean_abs_chgbatch_msignal"]))))))))))/2.0)))))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(data["maxbatch_msignal"])) <= np.abs(((data["meanbatch_msignal"]) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["mean_abs_chgbatch_msignal"]) * ((((data["medianbatch_slices2"]) + (((data["maxtominbatch"]) * (((data["abs_avgbatch_msignal"]) - (data["medianbatch_msignal"]))))))/2.0)))))) +

                            0.100000*np.tanh(np.real(data["medianbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2_msignal"]) * (np.cos((data["abs_maxbatch_slices2"]))))) - (np.where(np.abs(((data["meanbatch_slices2_msignal"]) * ((((data["medianbatch_msignal"]) + (((np.cos((data["rangebatch_slices2_msignal"]))) * (data["meanbatch_slices2_msignal"]))))/2.0)))) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_msignal"]) * (np.sin((((np.tanh((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) - (((data["meanbatch_msignal"]) + (np.where(np.abs(complex(3.0)) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(np.where(np.abs(data["signal_shift_+1"]) <= np.abs(np.conjugate(np.cos((((((data["rangebatch_slices2_msignal"]) * (complex(2.0)))) * (data["minbatch_slices2_msignal"])))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((data["abs_maxbatch_slices2_msignal"]))) <= np.abs(((((-((np.cos((data["abs_maxbatch_slices2"])))))) + (np.tanh((((np.cos((data["meanbatch_slices2_msignal"]))) + (np.where(np.abs(np.sin((complex(0,1)*np.conjugate(data["meanbatch_msignal"])))) <= np.abs(np.cos((data["abs_maxbatch_slices2"]))),complex(1.), complex(0.) )))))))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(7.56590557098388672)) - (((data["minbatch_msignal"]) * (complex(0,1)*np.conjugate(((np.where(np.abs(np.conjugate(((data["abs_maxbatch"]) - (complex(0,1)*np.conjugate(data["stdbatch_msignal"]))))) <= np.abs(((np.where(np.abs(np.where(np.abs(np.where(np.abs(complex(7.56590557098388672)) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) - (complex(0,1)*np.conjugate(data["stdbatch_slices2"])))),complex(1.), complex(0.) )) - (complex(0,1)*np.conjugate(data["stdbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(0,1)*np.conjugate(data["minbatch_slices2"])) + (data["minbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.sin((np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )))) + (data["signal_shift_+1_msignal"]))) > np.abs(np.sin((((data["medianbatch_msignal"]) + (data["signal_shift_+1_msignal"]))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos(((((-((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )) + (((data["abs_avgbatch_slices2_msignal"]) - (data["rangebatch_slices2"])))))))) + (np.where(np.abs(((data["meanbatch_slices2"]) / 2.0)) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(np.sin((np.where(np.abs(data["signal_shift_+1"]) > np.abs(np.sin((np.conjugate((((data["signal_shift_+1_msignal"]) + (data["medianbatch_slices2"]))/2.0))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((complex(0,1)*np.conjugate(np.cos((np.cos((np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(((complex(-2.0)) - (data["rangebatch_msignal"]))),complex(1.), complex(0.) ))))))) - (((data["abs_minbatch_msignal"]) / (np.cos((((data["abs_maxbatch_slices2"]) - (complex(-2.0)))))))))) - (((data["mean_abs_chgbatch_msignal"]) / (np.cos((((data["abs_maxbatch_slices2"]) + (data["rangebatch_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0.86772698163986206)) <= np.abs(np.conjugate(np.conjugate(((data["signal"]) + (data["meanbatch_slices2"]))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((data["signal_shift_-1_msignal"]) + (complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])))))) - (((((((((((((data["stdbatch_slices2"]) / 2.0)) + (data["minbatch"]))/2.0)) + (complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])))) / 2.0)) + (np.sin((np.cos((data["maxbatch_slices2_msignal"]))))))/2.0)))) - (np.sin((data["signal_shift_-1_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.where(np.abs(((data["abs_avgbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))) <= np.abs(np.conjugate(np.where(np.abs(data["minbatch_msignal"]) <= np.abs(data["maxtominbatch"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) )) - (complex(-3.0)))) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate((((-((data["meanbatch_slices2_msignal"])))) / 2.0))) > np.abs(np.sin((((complex(1.42630255222320557)) + (np.where(np.abs(data["minbatch"]) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((np.conjugate(((data["rangebatch_slices2_msignal"]) - (np.where(np.abs((-((data["stdbatch_slices2"])))) > np.abs(np.where(np.abs(np.sin((np.cos((data["abs_minbatch_slices2"]))))) > np.abs(complex(0,1)*np.conjugate(((data["rangebatch_slices2_msignal"]) / 2.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((((((data["rangebatch_slices2_msignal"]) + (((data["abs_maxbatch_msignal"]) * (np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["minbatch"]),complex(1.), complex(0.) )))))/2.0)) + (np.where(np.abs(np.sin((np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )))) <= np.abs(np.tanh((np.where(np.abs(np.sin((data["signal"]))) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.sin(((((complex(5.0)) + ((((((np.where(np.abs(np.where(np.abs(data["maxbatch_slices2"]) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) + (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(complex(1.0)),complex(1.), complex(0.) )))/2.0)) / 2.0)))/2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2_msignal"]) / ((((data["meanbatch_slices2"]) + (((data["maxtominbatch"]) + (np.cos((np.tanh((((np.sin((complex(3.0)))) + (complex(1.0)))))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["maxtominbatch_slices2"]) <= np.abs(np.sin((np.where(np.abs(data["maxtominbatch"]) <= np.abs(complex(-3.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) + (np.sin(((((data["maxtominbatch"]) + (np.where(np.abs(data["medianbatch_slices2"]) <= np.abs((((((data["maxtominbatch"]) + (data["abs_maxbatch"]))/2.0)) / 2.0)),complex(1.), complex(0.) )))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.cos((complex(0,1)*np.conjugate((-((complex(-3.0)))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos((data["stdbatch_slices2_msignal"])))))    

      

    def GP_class_4(self,data):

        return self.Output( -2.516509 +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) - (np.sin((((data["medianbatch_slices2"]) - (np.conjugate(((((data["meanbatch_slices2"]) - (np.sin((np.where(np.abs(complex(0.0)) <= np.abs(np.where(np.abs(data["maxbatch_slices2"]) > np.abs(complex(1.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) - (data["medianbatch_slices2"]))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(0,1)*np.conjugate(((data["stdbatch_slices2_msignal"]) * (data["medianbatch_slices2"])))) > np.abs(((((np.where(np.abs(((np.conjugate(data["signal"])) * 2.0)) <= np.abs(data["signal"]),complex(1.), complex(0.) )) / 2.0)) + (((np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["minbatch"]))),complex(1.), complex(0.) )) - (data["stdbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) - (data["stdbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["minbatch_slices2_msignal"]))) * 2.0)) + (np.where(np.abs(((np.tanh((data["minbatch_slices2_msignal"]))) * 2.0)) > np.abs((-((((data["medianbatch_slices2"]) + (np.sin((((np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(((((((data["minbatch_slices2_msignal"]) * 2.0)) + (((np.sin((data["minbatch_slices2_msignal"]))) * 2.0)))) * 2.0)),complex(1.), complex(0.) )) * 2.0))))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) - ((((data["rangebatch_msignal"]) + (data["maxtominbatch_slices2"]))/2.0)))) * (np.conjugate((((data["abs_avgbatch_msignal"]) + (np.sin((np.cos((data["signal_shift_+1"]))))))/2.0)))))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(data["maxbatch_slices2"])) - (data["signal_shift_+1"]))) - (data["abs_avgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_slices2_msignal"]))) * (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((((np.sin((data["minbatch_slices2_msignal"]))) * 2.0)) * 2.0)) - (np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(np.conjugate((((((((data["meanbatch_slices2"]) + (data["meanbatch_slices2"]))) + (np.sin((data["meanbatch_slices2_msignal"]))))/2.0)) * 2.0))))))))) +

                            0.100000*np.tanh(np.real((((((-((np.sin((complex(0,1)*np.conjugate(((data["signal_shift_-1"]) - (((complex(-1.0)) + (data["minbatch_msignal"]))))))))))) + (complex(0,1)*np.conjugate(complex(2.0))))) + (np.cos((((data["minbatch_msignal"]) - (complex(0,1)*np.conjugate((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(complex(2.0))))/2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.conjugate(((((np.where(np.abs(np.cos((np.where(np.abs(np.conjugate(data["abs_maxbatch_msignal"])) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )))) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) - (complex(0,1)*np.conjugate((-((((data["rangebatch_slices2"]) - (np.where(np.abs(np.conjugate(data["abs_maxbatch_msignal"])) <= np.abs(((data["rangebatch_slices2"]) - (complex(5.0)))),complex(1.), complex(0.) )))))))))) - (data["rangebatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((((data["minbatch_msignal"]) + (np.where(np.abs((-((data["stdbatch_slices2_msignal"])))) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.sin((((complex(0,1)*np.conjugate(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(np.sin((((data["abs_maxbatch"]) + (np.where(np.abs(data["rangebatch_slices2"]) > np.abs(np.sin((data["maxbatch_slices2"]))),complex(1.), complex(0.) )))))),complex(1.), complex(0.) ))) - (data["maxbatch_slices2"]))))))) - (((np.conjugate(data["abs_avgbatch_slices2_msignal"])) - (((np.sin((data["rangebatch_slices2"]))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((complex(0,1)*np.conjugate(((data["medianbatch_slices2"]) * 2.0))) - ((-((data["medianbatch_slices2"])))))) > np.abs(np.cos((((data["meanbatch_msignal"]) * (((data["maxtominbatch_slices2"]) * (data["minbatch_slices2_msignal"]))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch"]) - (((np.where(np.abs((((data["abs_maxbatch_msignal"]) + (complex(1.0)))/2.0)) > np.abs(np.cos((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )) / 2.0)))) * (np.tanh((np.tanh(((-((np.cos((data["abs_maxbatch_msignal"]))))))))))))) +

                            0.100000*np.tanh(np.real((((((data["maxbatch_msignal"]) + (np.tanh(((((((data["maxbatch_msignal"]) + ((((((((data["maxbatch_msignal"]) * ((((data["medianbatch_slices2"]) + (data["abs_maxbatch"]))/2.0)))) / 2.0)) + (np.sin((data["abs_maxbatch_slices2_msignal"]))))/2.0)))/2.0)) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))))))/2.0)) * (((data["abs_maxbatch_slices2_msignal"]) * (np.sin((data["abs_maxbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) * ((((((((data["minbatch_msignal"]) * 2.0)) * (((complex(9.0)) * (data["minbatch_msignal"]))))) + (((np.conjugate(np.tanh((complex(-2.0))))) - (data["maxbatch_slices2"]))))/2.0))))) +

                            0.100000*np.tanh(np.real((((((-(((((np.sin((data["stdbatch_slices2"]))) + (np.where(np.abs(data["maxtominbatch_msignal"]) <= np.abs((((((-((complex(-1.0))))) * (complex(0,1)*np.conjugate(np.conjugate(data["stdbatch_slices2"]))))) - (np.cos((data["minbatch_msignal"]))))),complex(1.), complex(0.) )))/2.0))))) - (((data["abs_minbatch_msignal"]) / ((((data["abs_maxbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0)))))) - (data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.sin((data["maxbatch_slices2_msignal"]))) / 2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["minbatch_msignal"]) * (np.where(np.abs(complex(7.03310775756835938)) > np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) * 2.0)) - (np.where(np.abs(data["rangebatch_slices2_msignal"]) > np.abs((-(((((data["abs_minbatch_msignal"]) + (((np.where(np.abs(((complex(3.0)) + (((data["abs_avgbatch_slices2"]) + (complex(-3.0)))))) <= np.abs(np.where(np.abs(complex(7.03310775756835938)) > np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (data["minbatch_msignal"]))))/2.0))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["rangebatch_slices2"]) * (np.where(np.abs(data["rangebatch_slices2"]) > np.abs(np.where(np.abs(np.sin((data["maxbatch_msignal"]))) <= np.abs(np.sin((np.sin((data["minbatch_msignal"]))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["rangebatch_slices2"]))) - (np.sin((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(np.conjugate(np.sin((complex(0,1)*np.conjugate(((((((data["mean_abs_chgbatch_msignal"]) * 2.0)) / 2.0)) + (complex(1.0)))))))),complex(1.), complex(0.) )))))) + (((np.sin((np.tanh((np.cos((data["minbatch_msignal"]))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["meanbatch_msignal"]) / (complex(0,1)*np.conjugate((((data["meanbatch_msignal"]) + (np.where(np.abs(np.tanh((data["abs_avgbatch_slices2_msignal"]))) <= np.abs((((-((((np.cos((data["maxbatch_slices2_msignal"]))) - (np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(complex(1.0)),complex(1.), complex(0.) ))))))) - ((((data["mean_abs_chgbatch_slices2"]) + (data["meanbatch_msignal"]))/2.0)))),complex(1.), complex(0.) )))/2.0))))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) * ((-(((((np.sin((data["maxbatch_slices2_msignal"]))) + (np.where(np.abs((-((complex(-3.0))))) > np.abs(data["stdbatch_msignal"]),complex(1.), complex(0.) )))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) - ((-((np.cos((np.cos((((((np.sin((data["maxbatch_slices2_msignal"]))) * 2.0)) - (data["minbatch_msignal"])))))))))))) * (np.cos((data["maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((((data["minbatch_slices2_msignal"]) + (np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.sin((data["rangebatch_slices2"]))),complex(1.), complex(0.) )))/2.0)) / (np.cos(((-((((np.sin((data["rangebatch_slices2"]))) - (complex(-2.0))))))))))) + ((-((data["meanbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real(((((((np.sin((np.sin((((data["minbatch_msignal"]) * 2.0)))))) * 2.0)) - (np.cos((np.where(np.abs((((((((np.tanh((complex(-1.0)))) + (data["rangebatch_msignal"]))/2.0)) * (np.tanh((np.tanh((np.sin((np.cos((data["minbatch_msignal"]))))))))))) * 2.0)) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((complex(0,1)*np.conjugate(np.cos((((((data["maxbatch_slices2_msignal"]) * 2.0)) + ((((complex(1.0)) + (complex(0,1)*np.conjugate(np.cos((np.tanh((data["minbatch_msignal"])))))))/2.0))))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((((complex(0,1)*np.conjugate(((((data["minbatch_slices2_msignal"]) * 2.0)) - (((data["minbatch_slices2_msignal"]) + (data["signal_shift_-1_msignal"])))))) + (data["abs_maxbatch_msignal"]))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(np.where(np.abs(np.tanh((complex(-3.0)))) <= np.abs((-((complex(3.17876172065734863))))),complex(1.), complex(0.) )) > np.abs((-((data["stdbatch_slices2"])))),complex(1.), complex(0.) )) <= np.abs((-((((data["abs_maxbatch_msignal"]) - (np.sin((((data["maxbatch_msignal"]) - (data["mean_abs_chgbatch_slices2"])))))))))),complex(1.), complex(0.) ))) - (((data["maxbatch_msignal"]) * 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-((data["stdbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.cos((((((data["maxbatch_msignal"]) * 2.0)) - (((complex(0,1)*np.conjugate(complex(-3.0))) - (np.tanh((np.conjugate(np.sin((data["medianbatch_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(3.0)) - (((np.cos((data["signal_shift_-1"]))) + (data["maxbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin((complex(-2.0))))) +

                            0.100000*np.tanh(np.real((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (data["rangebatch_slices2"]))) + ((-((np.cos((np.sin((((((data["rangebatch_slices2"]) * (complex(0.0)))) + (((np.sin((np.sin((data["rangebatch_slices2"]))))) / 2.0))))))))))))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((np.sin((np.sin((data["rangebatch_slices2"]))))))) * (((data["signal"]) - (np.sin((((data["rangebatch_slices2"]) * (((data["mean_abs_chgbatch_slices2"]) * 2.0)))))))))) * (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((((data["abs_avgbatch_slices2_msignal"]) - (((np.tanh((((data["abs_maxbatch_slices2_msignal"]) + (data["meanbatch_slices2"]))))) + (((data["abs_avgbatch_slices2_msignal"]) + (np.tanh((complex(0.0)))))))))) * (np.cos((((data["rangebatch_slices2"]) + ((((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) + (data["maxbatch_msignal"]))/2.0)))))))) * (complex(9.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((((np.cos((data["stdbatch_slices2_msignal"]))) + (data["signal"]))) + (complex(0,1)*np.conjugate(((data["abs_maxbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(data["maxbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) * 2.0))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(data["abs_maxbatch"])) + (((((complex(2.0)) * (((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["minbatch_msignal"]) / (((np.cos((data["abs_maxbatch_msignal"]))) * 2.0)))))))) + (((data["minbatch_msignal"]) / ((-((np.cos((data["minbatch_msignal"]))))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((data["abs_maxbatch_slices2_msignal"]))) - (complex(0,1)*np.conjugate(np.where(np.abs(((complex(-1.0)) * (data["medianbatch_msignal"]))) <= np.abs(((data["rangebatch_slices2"]) - (((complex(0,1)*np.conjugate(data["maxbatch_slices2"])) - (data["signal_shift_+1_msignal"]))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real((((((data["maxtominbatch_msignal"]) / (np.cos((data["maxbatch_slices2_msignal"]))))) + (((((((np.sin((data["stdbatch_slices2_msignal"]))) - (np.cos((np.cos((data["maxbatch_slices2_msignal"]))))))) - ((((((data["abs_minbatch_msignal"]) + (np.sin((data["maxtominbatch_msignal"]))))/2.0)) * (data["abs_minbatch_slices2_msignal"]))))) - (data["mean_abs_chgbatch_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) + (np.sin((((data["abs_maxbatch_slices2_msignal"]) + ((((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))) + (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(complex(0,1)*np.conjugate(((data["abs_maxbatch_slices2_msignal"]) + (np.conjugate(data["medianbatch_msignal"]))))),complex(1.), complex(0.) )))/2.0)))))))) + (np.sin((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((np.where(np.abs(np.conjugate(data["meanbatch_slices2"])) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )) * 2.0)))) +

                            0.100000*np.tanh(np.real(((((((((((data["medianbatch_slices2_msignal"]) - (((((np.cos((((data["medianbatch_msignal"]) * 2.0)))) / 2.0)) / 2.0)))) + (np.tanh((((np.cos((data["minbatch_msignal"]))) - (complex(1.0)))))))/2.0)) + (complex(1.0)))/2.0)) + (data["medianbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_slices2_msignal"]) - (complex(-1.0)))) * (((data["meanbatch_msignal"]) * (((complex(0,1)*np.conjugate(complex(-1.0))) + (((((data["rangebatch_msignal"]) * (np.conjugate(np.conjugate(np.conjugate(((data["minbatch_slices2_msignal"]) + (((((data["medianbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"]))) / 2.0))))))))) / 2.0))))))))) +

                            0.100000*np.tanh(np.real((((((complex(-3.0)) + (data["meanbatch_msignal"]))/2.0)) * (np.sin((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(np.conjugate(np.cos((np.cos(((-((((((((((np.sin((np.conjugate(data["abs_avgbatch_slices2"])))) + (np.conjugate(data["rangebatch_msignal"])))/2.0)) + (data["signal_shift_-1_msignal"]))/2.0)) * 2.0)))))))))) > np.abs(data["signal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["maxbatch_slices2_msignal"]))) + (np.sin((data["maxbatch_slices2_msignal"]))))) + (((((np.conjugate(((np.sin((data["maxbatch_slices2_msignal"]))) * 2.0))) / (((data["maxtominbatch"]) + (data["abs_maxbatch_slices2_msignal"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((complex(13.92804527282714844)) / (((data["stdbatch_slices2"]) - (np.where(np.abs(complex(13.92804527282714844)) > np.abs(np.where(np.abs(np.sin((data["abs_maxbatch"]))) <= np.abs(((data["stdbatch_slices2"]) - (((data["rangebatch_msignal"]) / (np.where(np.abs(complex(13.92804527282714844)) > np.abs(np.where(np.abs(data["abs_maxbatch"]) <= np.abs(np.where(np.abs(data["stdbatch_slices2"]) <= np.abs((((data["minbatch_msignal"]) + (complex(-2.0)))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(data["minbatch"])) +

                            0.100000*np.tanh(np.real(((((((np.sin((np.sin((data["maxbatch_slices2_msignal"]))))) * (data["maxbatch_slices2_msignal"]))) * 2.0)) * ((((complex(3.0)) + (((np.sin((np.where(np.abs(complex(1.0)) <= np.abs((((complex(3.0)) + (((data["maxbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))))/2.0)),complex(1.), complex(0.) )))) + (np.conjugate((-((np.tanh((complex(3.0)))))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(-3.0))) +

                            0.100000*np.tanh(np.real((((-((((((((((np.sin((np.cos(((((np.tanh((np.conjugate(data["rangebatch_slices2"])))) + (((data["rangebatch_slices2"]) * 2.0)))/2.0)))))) + (data["rangebatch_slices2"]))/2.0)) + (((((data["rangebatch_slices2"]) + (((data["maxbatch_slices2_msignal"]) * 2.0)))) / 2.0)))/2.0)) * (((data["maxbatch_msignal"]) * (np.cos((data["maxbatch_slices2_msignal"])))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(data["stdbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) + (complex(-3.0)))) + (((np.where(np.abs((-((complex(-3.0))))) > np.abs(((data["medianbatch_msignal"]) + (((((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))) + (((((np.sin((data["maxbatch_msignal"]))) * 2.0)) / 2.0)))) / 2.0)))),complex(1.), complex(0.) )) / 2.0))))) +

                            0.100000*np.tanh(np.real((((((data["abs_maxbatch_msignal"]) * (np.conjugate(((((((np.cos((data["signal_shift_+1_msignal"]))) + (data["medianbatch_msignal"]))) * 2.0)) * 2.0))))) + (((((((np.cos((data["minbatch_msignal"]))) + (data["medianbatch_msignal"]))) * 2.0)) * 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(data["maxbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((np.cos((data["meanbatch_msignal"]))) * (((data["meanbatch_msignal"]) * (((data["minbatch_slices2_msignal"]) - (np.cos((((((((data["medianbatch_msignal"]) + ((((data["rangebatch_slices2"]) + (complex(11.75808429718017578)))/2.0)))/2.0)) + (((data["abs_maxbatch_slices2_msignal"]) - (np.sin((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )))))))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(((((((np.sin((((complex(0,1)*np.conjugate(((data["maxbatch_msignal"]) - (np.where(np.abs(np.sin((complex(0.0)))) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))) - (data["meanbatch_msignal"]))))) / 2.0)) * (np.where(np.abs(complex(0.0)) <= np.abs(np.sin((((data["meanbatch_msignal"]) + (data["minbatch_slices2_msignal"]))))),complex(1.), complex(0.) )))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(complex(-1.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["minbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real((((((np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )) + ((((data["medianbatch_slices2_msignal"]) + ((((((data["signal_shift_-1"]) + (((data["maxbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))))) + (data["maxbatch_slices2_msignal"]))/2.0)))/2.0)))/2.0)) + (((((np.sin((data["maxbatch_slices2_msignal"]))) + (np.conjugate(data["medianbatch_msignal"])))) + (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["abs_maxbatch_msignal"]) + (((data["abs_maxbatch_msignal"]) + (((np.tanh((((((data["abs_maxbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))) * 2.0)))) * 2.0)))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((complex(0,1)*np.conjugate(np.where(np.abs((((data["meanbatch_msignal"]) + (data["signal_shift_+1"]))/2.0)) > np.abs(np.sin((((data["meanbatch_msignal"]) - (((((np.sin((data["meanbatch_msignal"]))) / 2.0)) + (np.sin((data["meanbatch_msignal"]))))))))),complex(1.), complex(0.) ))) - (((data["meanbatch_msignal"]) + (data["meanbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["medianbatch_msignal"]) + (((np.cos((((((data["abs_avgbatch_slices2_msignal"]) * (data["medianbatch_msignal"]))) + (data["meanbatch_msignal"]))))) + (((np.sin((((np.sin((data["rangebatch_slices2"]))) * 2.0)))) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_slices2"]) - (data["minbatch_msignal"]))) * ((((((np.sin((((data["minbatch_msignal"]) - (((complex(-1.0)) - (np.tanh(((-((np.conjugate(data["signal_shift_+1_msignal"]))))))))))))) * 2.0)) + (((data["meanbatch_slices2_msignal"]) - (np.where(np.abs(data["abs_minbatch_msignal"]) <= np.abs((((-((complex(-1.0))))) * 2.0)),complex(1.), complex(0.) )))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["stdbatch_msignal"]) - (((data["stdbatch_msignal"]) / (np.cos((data["stdbatch_slices2"]))))))) - ((((-((np.sin((((np.cos((((data["stdbatch_slices2"]) + (data["minbatch_slices2"]))))) * (data["stdbatch_slices2"])))))))) / (np.cos((data["stdbatch_slices2"]))))))) - (((data["stdbatch_msignal"]) / (data["stdbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate((((-((np.sin((data["meanbatch_msignal"])))))) * (np.conjugate((-((data["meanbatch_slices2_msignal"])))))))) + ((((-((np.sin((data["meanbatch_msignal"])))))) * (((data["rangebatch_slices2"]) - (complex(-2.0))))))))) +

                            0.100000*np.tanh(np.real((((data["abs_minbatch_slices2_msignal"]) + (((np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(np.sin((np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.conjugate(((data["abs_minbatch_slices2_msignal"]) + ((((data["stdbatch_slices2_msignal"]) + (np.where(np.abs(np.cos((((data["medianbatch_slices2_msignal"]) * 2.0)))) > np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) + (data["medianbatch_slices2_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (np.where(np.abs(np.conjugate(np.sin((((np.sin(((((data["abs_avgbatch_slices2_msignal"]) + (((((((-((data["meanbatch_slices2_msignal"])))) + (data["medianbatch_msignal"]))/2.0)) * 2.0)))/2.0)))) / 2.0))))) > np.abs(complex(1.0)),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((((((complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])) * 2.0)) + (((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((((((((data["medianbatch_msignal"]) + (((data["signal_shift_-1_msignal"]) + (((data["abs_minbatch_msignal"]) - (np.where(np.abs(data["maxtominbatch"]) > np.abs((((data["abs_avgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))/2.0)),complex(1.), complex(0.) )))))))/2.0)) / 2.0)) + (complex(-3.0)))) + (((data["medianbatch_msignal"]) - (np.cos((data["stdbatch_slices2_msignal"]))))))) + (np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["meanbatch_slices2_msignal"]) - (np.cos((np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((((((data["medianbatch_msignal"]) + (((data["medianbatch_slices2"]) + (data["meanbatch_msignal"]))))/2.0)) + (((((complex(0,1)*np.conjugate((((np.cos((((np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["meanbatch_msignal"])),complex(1.), complex(0.) )) + (((data["abs_maxbatch_slices2"]) / 2.0)))))) + (data["abs_avgbatch_slices2"]))/2.0))) / 2.0)) + ((-(((-((complex(0.0)))))))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["medianbatch_msignal"]) + (data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((data["minbatch_msignal"]) + (np.conjugate((-((np.sin((data["rangebatch_slices2"]))))))))) + (np.conjugate(complex(8.0))))) / 2.0))) +

                            0.100000*np.tanh(np.real((-((np.cos((((data["abs_maxbatch_msignal"]) + (((((np.where(np.abs(((data["abs_maxbatch_msignal"]) / 2.0)) > np.abs(((data["abs_minbatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) )) / 2.0)) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(complex(-1.0))) +

                            0.100000*np.tanh(np.real(np.cos((np.conjugate(((complex(1.0)) + ((((-((data["rangebatch_slices2"])))) + (complex(0,1)*np.conjugate(((((data["medianbatch_slices2_msignal"]) - (complex(9.0)))) - (((data["rangebatch_slices2"]) + (complex(1.0))))))))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((((((np.cos((data["abs_maxbatch_slices2_msignal"]))) * (data["abs_minbatch_msignal"]))) + (data["abs_minbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(np.where(np.abs(data["signal_shift_-1_msignal"]) <= np.abs(complex(0,1)*np.conjugate(((data["abs_avgbatch_slices2_msignal"]) * (data["signal_shift_-1_msignal"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (data["meanbatch_msignal"]))/2.0)) + (((data["meanbatch_msignal"]) + (np.conjugate(data["signal_shift_-1_msignal"])))))/2.0))) +

                            0.100000*np.tanh(np.real((((((data["abs_maxbatch_msignal"]) + (np.sin((data["maxtominbatch"]))))/2.0)) * ((-((((((np.sin((((data["meanbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])))))) * 2.0)) + (np.sin((data["maxtominbatch_slices2"]))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["maxbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.tanh((((data["medianbatch_msignal"]) + (data["signal_shift_-1_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.cos((((data["minbatch_msignal"]) / 2.0)))) * (np.sin((complex(-1.0)))))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (((data["medianbatch_msignal"]) + (((np.cos((((((data["maxtominbatch_slices2"]) - (data["signal_shift_-1_msignal"]))) + (data["maxtominbatch"]))))) + (data["minbatch_slices2_msignal"]))))))/2.0))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(data["minbatch_msignal"]) > np.abs((((data["medianbatch_slices2"]) + (((np.cos((data["minbatch_msignal"]))) * 2.0)))/2.0)),complex(1.), complex(0.) )) + (((data["medianbatch_slices2_msignal"]) + (((np.cos((data["minbatch_msignal"]))) * 2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(data["abs_avgbatch_slices2"])) +

                            0.100000*np.tanh(np.real((((np.tanh((np.where(np.abs(data["stdbatch_msignal"]) <= np.abs(((((data["stdbatch_msignal"]) * (data["maxbatch_slices2_msignal"]))) / 2.0)),complex(1.), complex(0.) )))) + (np.conjugate(((complex(2.56862711906433105)) * (np.conjugate(data["stdbatch_slices2_msignal"]))))))/2.0))) +

                            0.100000*np.tanh(np.real((((-((((((data["abs_maxbatch_slices2"]) * (np.sin((data["meanbatch_slices2_msignal"]))))) - (np.where(np.abs((-((np.sin((data["abs_avgbatch_slices2"])))))) <= np.abs(np.sin((data["meanbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxbatch_msignal"]) - (np.where(np.abs(np.sin((((((((data["signal_shift_+1_msignal"]) / 2.0)) / 2.0)) - (((data["medianbatch_msignal"]) + (data["maxbatch_msignal"]))))))) <= np.abs(np.conjugate(data["mean_abs_chgbatch_slices2"])),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["minbatch_msignal"]) * (np.conjugate(((data["mean_abs_chgbatch_slices2"]) + (complex(0,1)*np.conjugate(np.sin((np.tanh((data["minbatch_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) + (data["abs_minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.conjugate((-((((data["mean_abs_chgbatch_slices2_msignal"]) - (((np.sin((np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(((data["maxbatch_msignal"]) - (data["medianbatch_msignal"]))),complex(1.), complex(0.) )))) / 2.0))))))))) + (((data["minbatch_slices2"]) * (((np.sin((data["stdbatch_slices2_msignal"]))) - (np.tanh((data["mean_abs_chgbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((data["meanbatch_msignal"]) + (np.conjugate(complex(9.0))))) + (np.conjugate(complex(0,1)*np.conjugate(((np.conjugate(np.sin((np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.tanh((data["abs_minbatch_slices2"])))) > np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )))))) + (np.sin((data["maxbatch_msignal"]))))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) - ((((((data["rangebatch_msignal"]) + (np.tanh((np.where(np.abs(((np.where(np.abs(data["rangebatch_msignal"]) > np.abs((-((data["maxtominbatch_slices2_msignal"])))),complex(1.), complex(0.) )) - (data["abs_minbatch_slices2_msignal"]))) > np.abs(complex(-3.0)),complex(1.), complex(0.) )))))/2.0)) / (np.sin((np.sin((((data["stdbatch_slices2"]) * (((np.sin((complex(-3.0)))) - (complex(-3.0))))))))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((complex(-3.0)))) + (np.cos((np.conjugate(((data["abs_maxbatch_slices2_msignal"]) * (complex(-3.0)))))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((data["abs_avgbatch_slices2"]))) / (np.cos((((data["abs_avgbatch_slices2"]) + ((((data["rangebatch_msignal"]) + (np.sin((np.cos((np.where(np.abs((((((data["rangebatch_msignal"]) + (np.cos((data["rangebatch_slices2_msignal"]))))/2.0)) / (np.cos((data["rangebatch_msignal"]))))) > np.abs(complex(0.0)),complex(1.), complex(0.) )))))))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((data["rangebatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(np.sin((((complex(-2.0)) * (((np.conjugate(data["abs_maxbatch_slices2_msignal"])) - (complex(0,1)*np.conjugate(np.where(np.abs(np.tanh((np.sin((((data["abs_maxbatch"]) - (data["abs_maxbatch_slices2_msignal"]))))))) <= np.abs(((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.cos((data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )) - (data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.sin((((data["maxbatch_slices2_msignal"]) - (data["rangebatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real((((((((np.where(np.abs((-((data["maxbatch_slices2"])))) > np.abs(complex(3.0)),complex(1.), complex(0.) )) * 2.0)) + (((data["meanbatch_slices2"]) * 2.0)))/2.0)) + (((data["abs_minbatch_slices2"]) / 2.0))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) - (((((data["abs_minbatch_msignal"]) + ((-((((data["minbatch"]) / (complex(3.0))))))))) * (((data["minbatch"]) + (np.conjugate(data["abs_avgbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.sin((((((data["abs_minbatch_msignal"]) * 2.0)) + (complex(0,1)*np.conjugate(((np.where(np.abs(data["stdbatch_slices2_msignal"]) <= np.abs(complex(1.0)),complex(1.), complex(0.) )) - (((data["abs_maxbatch_msignal"]) + (np.where(np.abs((-((np.conjugate(np.sin((np.tanh((complex(1.0)))))))))) > np.abs(np.conjugate(data["medianbatch_msignal"])),complex(1.), complex(0.) )))))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((complex(-2.0)) * (((data["stdbatch_slices2_msignal"]) / ((((np.sin((data["rangebatch_slices2"]))) + (complex(0,1)*np.conjugate(np.conjugate(np.where(np.abs(((np.cos((data["rangebatch_slices2"]))) + (np.conjugate(np.sin((data["rangebatch_slices2"])))))) <= np.abs(((np.sin((data["rangebatch_slices2"]))) / 2.0)),complex(1.), complex(0.) )))))/2.0))))))) +

                            0.100000*np.tanh(np.real((((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (np.sin((((((data["mean_abs_chgbatch_slices2"]) * (data["mean_abs_chgbatch_slices2"]))) + (data["medianbatch_msignal"]))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.sin((data["minbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["abs_minbatch_msignal"]) - (np.where(np.abs(np.where(np.abs(np.sin(((-((np.conjugate(np.sin((np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(complex(-3.0)),complex(1.), complex(0.) )))))))))) > np.abs(np.sin((complex(2.0)))),complex(1.), complex(0.) )) <= np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )))))) + (((((((np.conjugate(data["minbatch_slices2_msignal"])) / (np.cos((data["maxtominbatch_slices2_msignal"]))))) * 2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(data["minbatch_slices2"])) +

                            0.100000*np.tanh(np.real(np.cos((((data["rangebatch_slices2"]) - (np.where(np.abs(((data["meanbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) <= np.abs(((((np.sin((np.cos((((np.cos((np.conjugate(data["meanbatch_msignal"])))) - (data["medianbatch_msignal"]))))))) - (np.where(np.abs(np.cos((np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) )))) > np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )))) - (data["rangebatch_slices2"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (np.where(np.abs(data["medianbatch_msignal"]) > np.abs((((data["medianbatch_msignal"]) + (np.where(np.abs(data["maxbatch_msignal"]) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.where(np.abs(np.where(np.abs(np.where(np.abs(complex(3.0)) > np.abs(((complex(3.0)) - (np.where(np.abs(data["signal_shift_+1"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) > np.abs(np.where(np.abs(np.where(np.abs(data["signal_shift_+1"]) <= np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)) <= np.abs(np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.where(np.abs((-((data["medianbatch_msignal"])))) <= np.abs(((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0)),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((np.cos((data["meanbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(((data["signal_shift_-1_msignal"]) / 2.0)),complex(1.), complex(0.) )) <= np.abs(((data["medianbatch_slices2_msignal"]) - (data["rangebatch_msignal"]))),complex(1.), complex(0.) )))) - (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((complex(0,1)*np.conjugate(np.cos((np.where(np.abs(complex(2.0)) <= np.abs(((data["medianbatch_slices2_msignal"]) - (data["maxtominbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))) - (((data["minbatch_slices2_msignal"]) * (((data["medianbatch_slices2_msignal"]) - (complex(3.51712083816528320)))))))) - (complex(0,1)*np.conjugate(((data["medianbatch_slices2_msignal"]) - (complex(3.51712083816528320))))))) / 2.0))) +

                            0.100000*np.tanh(np.real(((((((((((data["maxbatch_slices2_msignal"]) * (((((data["abs_maxbatch"]) * 2.0)) * (((np.sin((data["maxbatch_slices2_msignal"]))) / 2.0)))))) - (data["abs_avgbatch_slices2"]))) - (data["signal"]))) - (complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"])))) + (np.cos((complex(-1.0))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((np.conjugate(np.conjugate(np.sin((complex(1.0)))))) + (np.where(np.abs((((((complex(0,1)*np.conjugate(((np.cos((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.conjugate(data["rangebatch_slices2"])))))) / 2.0))) + (np.sin((data["rangebatch_slices2"]))))/2.0)) / 2.0)) <= np.abs(np.where(np.abs(data["abs_minbatch_msignal"]) <= np.abs(np.tanh((data["abs_minbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((-((np.conjugate(complex(-2.0)))))) <= np.abs(np.sin((np.where(np.abs(((np.sin((data["mean_abs_chgbatch_msignal"]))) / 2.0)) <= np.abs(np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.cos((np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(np.sin(((-((np.sin(((-((data["abs_avgbatch_msignal"]))))))))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.tanh((np.where(np.abs(data["signal_shift_+1"]) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))) / 2.0))) + (((complex(9.20465660095214844)) / (((np.cos((data["medianbatch_msignal"]))) / (np.cos((data["minbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) * (((data["signal"]) + (((((data["signal"]) * (complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"])))) * (((data["signal"]) + (((np.sin((((np.where(np.abs(data["stdbatch_msignal"]) <= np.abs(np.sin((((data["signal"]) + (np.cos((data["minbatch_msignal"]))))))),complex(1.), complex(0.) )) * 2.0)))) - (data["minbatch_msignal"])))))))))))) +

                            0.100000*np.tanh(np.real((((((data["meanbatch_slices2_msignal"]) / 2.0)) + (((data["minbatch_slices2"]) - (((np.cos((data["minbatch"]))) - (np.tanh(((((np.conjugate(((data["minbatch_slices2_msignal"]) * 2.0))) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((((np.sin((((np.sin((data["rangebatch_slices2"]))) * 2.0)))) * 2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["minbatch_msignal"]) + (((((((data["mean_abs_chgbatch_slices2"]) + (((np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )) + ((((((data["mean_abs_chgbatch_slices2"]) / 2.0)) + (((data["mean_abs_chgbatch_slices2"]) + (np.where(np.abs(complex(0,1)*np.conjugate(((data["mean_abs_chgbatch_slices2"]) / 2.0))) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))))/2.0)))))/2.0)) + (data["maxbatch_msignal"]))/2.0)))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.conjugate(((np.conjugate(((data["meanbatch_slices2_msignal"]) / (complex(6.0))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2"]) / (np.sin((((((np.cos((np.cos((((np.cos((np.cos((data["rangebatch_slices2"]))))) / 2.0)))))) / 2.0)) + (((((data["medianbatch_slices2_msignal"]) / 2.0)) + (data["rangebatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(((np.cos((data["medianbatch_msignal"]))) - (data["stdbatch_slices2"]))),complex(1.), complex(0.) )) + (((((data["abs_maxbatch"]) / 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((data["abs_maxbatch"]))) <= np.abs(np.where(np.abs(data["meanbatch_slices2"]) > np.abs(np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(complex(2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((np.conjugate(np.cos((complex(0.0))))) + (data["meanbatch_msignal"]))/2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_slices2"]) * (((np.cos((((data["rangebatch_slices2"]) * (np.cos((np.cos((np.cos((np.cos((np.cos((np.cos((np.cos((((data["rangebatch_slices2"]) / 2.0)))))))))))))))))))) * (np.cos((np.conjugate(np.sin((np.where(np.abs(np.cos((data["rangebatch_slices2"]))) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )))))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["maxbatch_msignal"]) + (np.cos((np.where(np.abs(((data["maxbatch_msignal"]) + (np.sin((complex(0.0)))))) > np.abs(np.tanh(((-(((((data["abs_maxbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0))))))),complex(1.), complex(0.) )))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["maxbatch_slices2_msignal"]) * 2.0)) <= np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((np.conjugate(np.tanh((np.tanh((data["abs_minbatch_slices2"])))))) + (np.tanh((np.conjugate(np.conjugate(complex(0,1)*np.conjugate(data["meanbatch_slices2_msignal"])))))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((data["maxbatch_slices2_msignal"]))) + (((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.sin(((((data["stdbatch_slices2_msignal"]) + (np.conjugate((((((np.sin((data["medianbatch_slices2_msignal"]))) + ((-((data["medianbatch_slices2_msignal"])))))) + (data["signal_shift_-1"]))/2.0))))/2.0)))),complex(1.), complex(0.) )) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["rangebatch_slices2"]) + (((((np.where(np.abs((((((data["rangebatch_slices2"]) + (complex(0,1)*np.conjugate((((((((((data["rangebatch_slices2"]) + (((data["abs_maxbatch"]) * 2.0)))/2.0)) / 2.0)) / 2.0)) * (data["stdbatch_msignal"])))))/2.0)) * 2.0)) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) / 2.0)) * (data["meanbatch_slices2_msignal"]))))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((((((np.tanh((data["signal_shift_-1_msignal"]))) * (np.where(np.abs(((data["meanbatch_msignal"]) - (np.where(np.abs(np.conjugate((((data["medianbatch_msignal"]) + (complex(0.0)))/2.0))) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))) <= np.abs(((complex(2.0)) + (np.where(np.abs(complex(2.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))) / 2.0)))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0)))) + (np.tanh((np.sin((data["maxbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["signal_shift_+1_msignal"]) / 2.0)) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.conjugate(complex(2.0))) * (np.cos((np.conjugate(data["abs_avgbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((np.tanh(((-((data["mean_abs_chgbatch_msignal"])))))) - (((np.sin((((data["maxbatch_msignal"]) - (((np.sin((np.conjugate(data["maxbatch_msignal"])))) * (data["signal_shift_-1_msignal"]))))))) * 2.0)))))) + (((((((np.sin((((data["maxbatch_msignal"]) - (np.tanh((data["signal_shift_-1_msignal"]))))))) * 2.0)) * 2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) + (((np.where(np.abs(np.where(np.abs(np.cos((((data["medianbatch_msignal"]) - (complex(0.0)))))) > np.abs((((np.sin((data["maxbatch_slices2_msignal"]))) + (((((data["medianbatch_msignal"]) + (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))) * 2.0)))/2.0)),complex(1.), complex(0.) )) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) * (np.sin((data["maxbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (np.cos((np.tanh((complex(3.0)))))))/2.0))) +

                            0.100000*np.tanh(np.real(complex(1.0))) +

                            0.100000*np.tanh(np.real((((data["minbatch_slices2"]) + (complex(0,1)*np.conjugate((((complex(-1.0)) + (data["minbatch_slices2"]))/2.0))))/2.0))) +

                            0.100000*np.tanh(np.real(data["abs_minbatch_slices2"])) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((((data["rangebatch_slices2_msignal"]) - (((((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) * (data["signal_shift_+1_msignal"]))) / 2.0)))) - (data["minbatch"]))))) +

                            0.100000*np.tanh(np.real(np.sin((data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["mean_abs_chgbatch_msignal"]) / ((((((np.cos((np.cos((((np.cos((np.sin((np.cos(((-((np.sin((data["stdbatch_slices2"])))))))))))) / 2.0)))))) / 2.0)) + (complex(10.0)))/2.0)))) - (((np.cos((complex(2.0)))) / 2.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (np.sin((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) + (((((data["medianbatch_slices2_msignal"]) + (np.where(np.abs(data["abs_minbatch_msignal"]) <= np.abs(np.tanh((np.conjugate(np.cos((complex(2.0))))))),complex(1.), complex(0.) )))) + (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs((((-((((complex(2.0)) / 2.0))))) + (data["signal_shift_+1"]))),complex(1.), complex(0.) )))))) * (((((data["signal_shift_-1_msignal"]) * 2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((((data["meanbatch_slices2_msignal"]) + (((complex(-3.0)) * (complex(0,1)*np.conjugate(np.where(np.abs(data["minbatch"]) <= np.abs(np.sin((data["meanbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(((complex(0,1)*np.conjugate(complex(-2.0))) - (complex(1.0)))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(np.tanh((((complex(2.0)) * (data["mean_abs_chgbatch_slices2"]))))) > np.abs((((complex(0,1)*np.conjugate(np.where(np.abs(complex(3.0)) <= np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) ))) + (complex(0,1)*np.conjugate(np.cos((data["meanbatch_msignal"])))))/2.0)),complex(1.), complex(0.) )) > np.abs(np.conjugate(complex(10.0))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((complex(-3.0)))) > np.abs(np.sin(((((((np.tanh((data["rangebatch_slices2_msignal"]))) + (np.conjugate(complex(-3.0))))) + (((complex(-2.0)) * (complex(7.0)))))/2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (complex(0.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((complex(-1.0)) - ((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(data["minbatch_msignal"])))/2.0)))))) + (((complex(-1.0)) - ((((complex(0,1)*np.conjugate(data["minbatch_msignal"])) + (complex(0,1)*np.conjugate(np.where(np.abs(complex(4.06565761566162109)) > np.abs(complex(-1.0)),complex(1.), complex(0.) ))))/2.0))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.where(np.abs(np.cos((np.cos((data["maxtominbatch"]))))) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))) + (np.cos((data["rangebatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch"]))) - (np.where(np.abs(np.sin((np.conjugate(data["rangebatch_slices2"])))) > np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.cos((data["rangebatch_slices2"]))) * 2.0)))) +

                            0.100000*np.tanh(np.real((((((((((((((((((((data["signal_shift_+1_msignal"]) + (data["meanbatch_msignal"]))) * 2.0)) + (((data["medianbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))))) * 2.0)) + (data["medianbatch_msignal"]))) + (data["medianbatch_msignal"]))) / 2.0)) + (data["meanbatch_msignal"]))/2.0)) + (data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["maxbatch_msignal"]) - (data["signal_shift_-1_msignal"]))) - (np.where(np.abs(data["maxtominbatch"]) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(complex(0.0))) +

                            0.100000*np.tanh(np.real(np.sin((np.tanh((np.where(np.abs(data["maxbatch_slices2"]) > np.abs(np.cos((data["rangebatch_slices2"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.tanh((np.sin(((-((((data["meanbatch_msignal"]) - (np.where(np.abs(((data["meanbatch_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(np.sin((complex(2.0)))),complex(1.), complex(0.) ))))) <= np.abs(np.conjugate(((np.sin((data["stdbatch_slices2"]))) + (data["medianbatch_slices2"])))),complex(1.), complex(0.) )))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.sin((((np.conjugate(complex(0,1)*np.conjugate(np.sin((np.tanh((complex(0.0)))))))) + (complex(2.0)))))) > np.abs((((-((data["abs_maxbatch_slices2"])))) - (data["signal"]))),complex(1.), complex(0.) )) > np.abs(np.conjugate(data["mean_abs_chgbatch_slices2_msignal"])),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(np.sin((np.where(np.abs(data["stdbatch_slices2"]) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch"]) <= np.abs(((((complex(3.0)) * (np.tanh((data["maxbatch_msignal"]))))) * (data["meanbatch_slices2"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((np.sin((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))) + ((((((data["abs_minbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) + (((data["abs_minbatch_msignal"]) + (np.cos((data["minbatch_msignal"]))))))))/2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(np.sin((((data["abs_avgbatch_slices2"]) + (data["stdbatch_slices2"])))))) * (((((((((((data["stdbatch_slices2"]) + (((data["maxbatch_slices2"]) / 2.0)))/2.0)) + (complex(1.0)))) + (complex(1.0)))) + (np.tanh((complex(4.0)))))/2.0)))) * (complex(0,1)*np.conjugate(data["rangebatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.sin((((data["abs_maxbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"])))))) + (data["mean_abs_chgbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.cos((np.where(np.abs(((((np.where(np.abs(np.cos((data["medianbatch_slices2"]))) <= np.abs(np.cos((complex(3.0)))),complex(1.), complex(0.) )) / 2.0)) / 2.0)) <= np.abs(np.tanh((complex(3.0)))),complex(1.), complex(0.) ))))) * (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.sin((data["abs_maxbatch"])))) * (((np.tanh((complex(0,1)*np.conjugate(((complex(10.0)) + (np.sin((complex(10.0))))))))) + (np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(((((complex(10.0)) + (np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["abs_minbatch_msignal"]),complex(1.), complex(0.) )))) + (np.where(np.abs(data["meanbatch_slices2"]) > np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) / 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(np.sin((data["mean_abs_chgbatch_slices2"]))),complex(1.), complex(0.) )) * 2.0)))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) + (np.sin((((data["abs_minbatch_slices2_msignal"]) + (((data["abs_minbatch_slices2_msignal"]) * (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(np.sin((((data["maxbatch_slices2"]) / 2.0)))),complex(1.), complex(0.) ))) <= np.abs((((((-((data["meanbatch_msignal"])))) * (data["stdbatch_msignal"]))) * (complex(0,1)*np.conjugate(np.where(np.abs(np.tanh((data["stdbatch_slices2"]))) <= np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(complex(0,1)*np.conjugate(((np.cos((complex(-1.0)))) + (data["maxtominbatch_slices2_msignal"]))))) <= np.abs(((data["medianbatch_slices2"]) + (np.sin((((((data["abs_minbatch_slices2"]) + (((data["maxtominbatch_slices2_msignal"]) / 2.0)))) - (complex(1.0)))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((((data["signal_shift_-1_msignal"]) - (data["maxbatch_msignal"])))))) - (complex(0,1)*np.conjugate(((np.where(np.abs((-((data["medianbatch_slices2_msignal"])))) <= np.abs((((-((data["meanbatch_msignal"])))) - (data["maxbatch_msignal"]))),complex(1.), complex(0.) )) - (np.tanh((((data["signal_shift_-1_msignal"]) - (data["maxbatch_msignal"]))))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.conjugate((((np.where(np.abs(np.cos((data["signal_shift_+1_msignal"]))) <= np.abs(((data["medianbatch_slices2"]) * (data["rangebatch_msignal"]))),complex(1.), complex(0.) )) + (((data["rangebatch_msignal"]) * (data["medianbatch_slices2"]))))/2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["maxbatch_msignal"]) - ((((-((complex(3.0))))) * 2.0)))) > np.abs(np.tanh((data["maxbatch_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["abs_minbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.real(np.conjugate(data["medianbatch_slices2"]))) +

                            0.100000*np.tanh(np.real((-((data["rangebatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) * (np.where(np.abs(np.sin((data["maxbatch_msignal"]))) <= np.abs(np.tanh((data["signal_shift_-1_msignal"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate(((((np.cos((np.conjugate(data["maxbatch_slices2_msignal"])))) * 2.0)) * 2.0))))/2.0))) / (((np.tanh((complex(0,1)*np.conjugate(((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0)) * 2.0))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_avgbatch_slices2"]) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["meanbatch_slices2_msignal"]) / (np.cos((np.sin((np.where(np.abs(complex(1.0)) <= np.abs(((np.sin((data["maxbatch_slices2_msignal"]))) - (data["stdbatch_slices2"]))),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.tanh((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((complex(0,1)*np.conjugate(complex(3.0))))) * ((((-((((data["meanbatch_slices2_msignal"]) * (complex(0,1)*np.conjugate(np.where(np.abs((-((((np.cos((complex(0,1)*np.conjugate(((data["stdbatch_slices2"]) + (complex(0,1)*np.conjugate(data["stdbatch_slices2"]))))))) / 2.0))))) > np.abs(((data["abs_minbatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) )))))))) + (((data["stdbatch_slices2"]) + (complex(-1.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((((data["maxtominbatch_slices2"]) + (((((data["meanbatch_msignal"]) + (data["abs_minbatch_msignal"]))) + (np.cos((((complex(0.0)) + (np.where(np.abs(((data["meanbatch_msignal"]) - (complex(3.0)))) <= np.abs(np.tanh((data["minbatch"]))),complex(1.), complex(0.) )))))))))) + (np.sin((np.sin((data["abs_minbatch_slices2_msignal"])))))))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["meanbatch_slices2_msignal"]) + (((data["rangebatch_slices2"]) - (np.cos((data["medianbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((((complex(-2.0)) * (((((data["signal_shift_-1_msignal"]) * (np.where(np.abs(np.tanh((((((np.tanh((complex(0,1)*np.conjugate(((data["signal_shift_-1_msignal"]) / 2.0))))) / 2.0)) - (complex(0,1)*np.conjugate(((((complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])) / 2.0)) - (data["abs_minbatch_slices2_msignal"])))))))) <= np.abs(np.tanh((complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])))),complex(1.), complex(0.) )))) * 2.0)))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["signal_shift_+1"]) + (data["rangebatch_slices2"]))) + (np.cos((np.cos((data["minbatch_slices2_msignal"]))))))/2.0))))))   

   

    def GP_class_5(self,data):

        return self.Output( -2.889212 +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) - (np.cos((((np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) - (np.where(np.abs(((data["abs_avgbatch_slices2"]) - (complex(-3.0)))) <= np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_slices2"]) - (((np.where(np.abs(np.sin((np.tanh((np.conjugate(np.cos((((data["meanbatch_slices2_msignal"]) / 2.0))))))))) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) / 2.0)))) / 2.0))) +

                            0.100000*np.tanh(np.real(((((data["stdbatch_msignal"]) + ((-((((data["minbatch_msignal"]) - (np.where(np.abs((((complex(-2.0)) + ((((data["abs_minbatch_msignal"]) + (((np.sin((data["minbatch_slices2"]))) + (np.tanh((((data["signal_shift_+1"]) * 2.0)))))))/2.0)))/2.0)) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) ))))))))) * (((data["stdbatch_msignal"]) - (complex(3.35434627532958984))))))) +

                            0.100000*np.tanh(np.real((((((((data["maxbatch_slices2"]) - (data["maxtominbatch_msignal"]))) + (np.tanh(((((((data["signal"]) * 2.0)) + (data["maxbatch_slices2"]))/2.0)))))/2.0)) + (((data["meanbatch_slices2"]) - (np.cos((data["signal_shift_+1_msignal"])))))))) +

                            0.100000*np.tanh(np.real((((data["signal"]) + (data["maxtominbatch"]))/2.0))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2"]) + (np.sin((np.sin((((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(complex(0,1)*np.conjugate(data["medianbatch_slices2"])),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(data["signal"]) > np.abs(((data["signal_shift_-1"]) - (np.where(np.abs(((((np.conjugate(data["signal"])) * 2.0)) * 2.0)) > np.abs(data["signal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) + (data["signal"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["medianbatch_slices2"]) + (complex(-1.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((data["medianbatch_slices2"]))) + (np.tanh((np.sin((data["abs_minbatch_slices2_msignal"]))))))) + (((data["meanbatch_slices2"]) - (data["maxtominbatch"])))))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1"])) +

                            0.100000*np.tanh(np.real(((((np.cos((data["abs_maxbatch_slices2_msignal"]))) - ((((complex(-2.0)) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))) - (np.where(np.abs(((np.cos((data["abs_maxbatch_slices2_msignal"]))) / 2.0)) <= np.abs(((data["abs_minbatch_slices2_msignal"]) - (data["abs_minbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(complex(-1.0))) +

                            0.100000*np.tanh(np.real(((np.cos((np.where(np.abs(np.sin((complex(3.0)))) <= np.abs(data["abs_minbatch_msignal"]),complex(1.), complex(0.) )))) - (np.where(np.abs(data["signal_shift_-1"]) <= np.abs((-((((data["mean_abs_chgbatch_slices2"]) * (np.tanh((data["meanbatch_slices2_msignal"])))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((-((data["abs_avgbatch_slices2_msignal"])))) + ((((((np.cos((np.conjugate(data["signal_shift_+1"])))) + (((complex(0.0)) * (np.tanh((((data["meanbatch_slices2"]) * 2.0)))))))/2.0)) - (((np.cos((data["abs_avgbatch_slices2_msignal"]))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.conjugate(((np.conjugate(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.cos((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) ))) + (data["abs_maxbatch_msignal"])))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["rangebatch_slices2"]) * (np.cos((data["maxbatch_slices2_msignal"]))))) + (np.cos((data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) - (np.tanh((data["medianbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(np.cos((data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["abs_maxbatch_msignal"]) + (np.cos((np.sin((np.where(np.abs(np.sin((((data["meanbatch_slices2_msignal"]) + (((((((np.sin((np.sin((data["abs_maxbatch_msignal"]))))) * 2.0)) * 2.0)) + (complex(-3.0)))))))) <= np.abs(((np.where(np.abs(data["maxbatch_slices2"]) > np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) )))))))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(0,1)*np.conjugate((((-((((data["maxbatch_slices2_msignal"]) + ((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(np.where(np.abs(complex(0,1)*np.conjugate(complex(-2.0))) <= np.abs(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(np.tanh((complex(1.0)))),complex(1.), complex(0.) )) > np.abs(np.tanh((data["minbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))))))) * 2.0))) + (data["maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((np.cos((((np.sin((complex(2.0)))) + ((((data["maxbatch_slices2_msignal"]) + (np.tanh((((np.sin((np.sin((data["maxbatch_msignal"]))))) / 2.0)))))/2.0)))))) * 2.0)) * 2.0)) + (np.sin((data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((((complex(-1.0)) + (data["signal_shift_+1"]))) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) * (data["abs_avgbatch_slices2"]))) - (((((np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"]))))) + (np.cos((data["abs_maxbatch_slices2_msignal"]))))) / 2.0))))) +

                            0.100000*np.tanh(np.real((-((data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real((((-((np.cos((((data["meanbatch_msignal"]) + ((((data["abs_avgbatch_slices2_msignal"]) + ((((((-((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(np.tanh((np.where(np.abs(((complex(-1.0)) * (data["abs_maxbatch"]))) > np.abs(np.conjugate(complex(-1.0))),complex(1.), complex(0.) )))) <= np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) / 2.0)) / 2.0)))/2.0))))))))) * (data["meanbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["abs_maxbatch_slices2"]) * (np.sin((((data["abs_maxbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(np.sin((((data["abs_maxbatch_msignal"]) / 2.0)))) <= np.abs(np.cos((data["abs_avgbatch_msignal"]))),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["abs_maxbatch"]) <= np.abs(((data["abs_maxbatch_slices2"]) + (data["meanbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.where(np.abs(np.tanh((np.cos((data["maxtominbatch_slices2_msignal"]))))) > np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((np.conjugate(((((((((complex(-2.0)) + (((data["maxbatch_slices2_msignal"]) - (np.cos((data["maxbatch_slices2_msignal"]))))))/2.0)) + (data["signal_shift_+1"]))/2.0)) * (np.cos((data["abs_maxbatch_msignal"])))))) + (np.where(np.abs(data["signal_shift_+1"]) > np.abs((-((data["abs_minbatch_slices2_msignal"])))),complex(1.), complex(0.) )))) * 2.0)))) +

                            0.100000*np.tanh(np.real((((np.sin((((((data["abs_maxbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.conjugate(data["abs_maxbatch_msignal"]))))) * 2.0)))) + (data["abs_maxbatch_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real((((((data["signal_shift_-1"]) + (np.cos((np.where(np.abs(np.where(np.abs((((data["abs_maxbatch_msignal"]) + (data["abs_minbatch_slices2_msignal"]))/2.0)) > np.abs((-((data["abs_avgbatch_msignal"])))),complex(1.), complex(0.) )) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))))/2.0)) / (np.cos((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["abs_avgbatch_msignal"]) + (((np.cos((np.conjugate(complex(0,1)*np.conjugate(data["abs_maxbatch_msignal"]))))) * (((((data["abs_maxbatch_msignal"]) * 2.0)) * (np.where(np.abs(complex(0.0)) > np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )))))))/2.0)) - (complex(0,1)*np.conjugate(((((np.cos((data["medianbatch_slices2_msignal"]))) * 2.0)) * (data["abs_maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((((((data["minbatch_msignal"]) / 2.0)) + (((((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"])) / (complex(2.0)))) / 2.0)))) + ((-((((data["minbatch_msignal"]) * (np.cos((data["minbatch_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((-((np.cos((((np.cos((data["medianbatch_slices2_msignal"]))) * 2.0))))))) > np.abs(np.where(np.abs(np.tanh((np.cos((np.cos((data["medianbatch_slices2_msignal"]))))))) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (np.cos((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * ((((complex(0,1)*np.conjugate(np.conjugate(data["rangebatch_slices2"]))) + (((np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))) * (data["rangebatch_slices2"]))))/2.0)))) * ((((((data["abs_maxbatch"]) + (data["signal_shift_+1"]))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_msignal"]) * (((data["meanbatch_msignal"]) + (np.where(np.abs(np.cos((data["medianbatch_slices2_msignal"]))) <= np.abs(np.cos(((((data["abs_avgbatch_slices2_msignal"]) + (((complex(7.0)) + (np.cos((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )))))))/2.0)))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(-2.0)) <= np.abs(((data["abs_avgbatch_msignal"]) - (np.where(np.abs(((complex(-2.0)) * (data["abs_maxbatch_slices2_msignal"]))) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) - (np.cos((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((((data["stdbatch_slices2"]) - (((np.conjugate(((np.conjugate(data["abs_maxbatch_slices2_msignal"])) - (data["maxtominbatch_slices2_msignal"])))) / 2.0)))))) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((data["abs_maxbatch_slices2"]) * (np.sin((data["maxbatch_slices2_msignal"]))))) * 2.0)) * (data["stdbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) * (((np.cos((((complex(0.0)) - (data["meanbatch_slices2_msignal"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real((((((((np.sin((data["maxbatch_msignal"]))) + (np.sin((((data["maxbatch_msignal"]) + (data["maxbatch_msignal"]))))))) + (((np.where(np.abs(((data["maxbatch_msignal"]) * 2.0)) <= np.abs(np.sin((np.sin((data["maxbatch_msignal"]))))),complex(1.), complex(0.) )) / 2.0)))/2.0)) + (((np.sin((data["maxbatch_msignal"]))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) + (complex(-2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["medianbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) * 2.0)) * (np.sin((np.sin((data["maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])))) + (complex(0,1)*np.conjugate(np.conjugate(complex(1.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["meanbatch_msignal"]) * (data["meanbatch_slices2_msignal"]))) > np.abs(np.tanh((complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"])))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((data["meanbatch_msignal"]))) * (((((((((((np.cos((data["meanbatch_msignal"]))) + (data["minbatch_msignal"]))/2.0)) + (np.cos((data["abs_maxbatch_slices2"]))))) + (data["minbatch_msignal"]))/2.0)) + (np.cos((complex(0,1)*np.conjugate(data["meanbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(2.87741017341613770)) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["maxbatch_msignal"]) - (np.tanh((((np.where(np.abs(np.cos((np.conjugate(((((np.tanh((np.cos((data["maxbatch_msignal"]))))) * 2.0)) * 2.0))))) > np.abs(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(((data["maxbatch_msignal"]) - (np.tanh((data["maxbatch_msignal"]))))) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_maxbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.where(np.abs(data["signal"]) > np.abs(((data["meanbatch_slices2_msignal"]) - ((-((data["mean_abs_chgbatch_slices2"])))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) + ((((data["medianbatch_msignal"]) + (((((((((data["meanbatch_slices2_msignal"]) + (((((-((complex(4.0))))) + (np.where(np.abs((-((data["medianbatch_slices2_msignal"])))) <= np.abs(np.conjugate(np.cos((data["stdbatch_slices2"])))),complex(1.), complex(0.) )))/2.0)))) + (complex(0.0)))) - (complex(1.0)))) * 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real((((((((np.conjugate(data["abs_maxbatch"])) + (np.conjugate(((np.where(np.abs(np.cos((complex(0,1)*np.conjugate(((((data["meanbatch_msignal"]) - (np.cos((data["abs_maxbatch_slices2"]))))) / 2.0))))) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )) + (data["abs_maxbatch_slices2"])))))/2.0)) * (np.sin((data["abs_maxbatch_msignal"]))))) / 2.0))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) - ((((((data["mean_abs_chgbatch_slices2"]) + (((complex(3.0)) + (np.where(np.abs(np.sin((((np.cos((data["meanbatch_slices2_msignal"]))) * 2.0)))) <= np.abs(((np.tanh((((np.cos((data["mean_abs_chgbatch_slices2"]))) * 2.0)))) * 2.0)),complex(1.), complex(0.) )))))/2.0)) / (np.cos((data["meanbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((((data["maxbatch_msignal"]) + (data["rangebatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.cos(((((-((data["minbatch_msignal"])))) + (complex(0,1)*np.conjugate(((np.cos((np.where(np.abs(np.cos((np.cos(((((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2_msignal"])) + (data["medianbatch_msignal"]))/2.0)))))) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))) + ((((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2_msignal"])) + ((-((data["minbatch_msignal"])))))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((((np.sin((np.cos((complex(0,1)*np.conjugate(np.tanh((data["medianbatch_slices2_msignal"])))))))) + (((((data["medianbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))) * 2.0)))) * (np.sin(((-((np.cos((data["stdbatch_slices2"]))))))))))) +

                            0.100000*np.tanh(np.real(data["signal_shift_-1_msignal"])) +

                            0.100000*np.tanh(np.real(((np.conjugate(((data["meanbatch_slices2"]) * (np.sin((data["maxbatch_msignal"])))))) - (((((np.where(np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(((data["meanbatch_slices2"]) * (np.tanh((np.tanh((complex(-2.0)))))))),complex(1.), complex(0.) )) > np.abs(((((data["abs_avgbatch_slices2"]) / 2.0)) - (np.conjugate(data["abs_avgbatch_slices2_msignal"])))),complex(1.), complex(0.) )) * 2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) / (((data["mean_abs_chgbatch_slices2"]) - (((((np.cos((np.conjugate((-((np.conjugate((-((((data["mean_abs_chgbatch_slices2"]) - (np.where(np.abs(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])) <= np.abs((-((((data["medianbatch_msignal"]) * 2.0))))),complex(1.), complex(0.) )))))))))))))) / 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real((((((((((data["medianbatch_msignal"]) - (data["medianbatch_slices2_msignal"]))) / 2.0)) + (data["medianbatch_msignal"]))/2.0)) + (((((complex(-3.0)) + (((data["medianbatch_msignal"]) + ((((((((data["medianbatch_msignal"]) + ((-((complex(0.0))))))/2.0)) * 2.0)) / 2.0)))))) + (((np.tanh((((data["medianbatch_msignal"]) + (data["medianbatch_slices2_msignal"]))))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_-1"]) * (((((data["rangebatch_slices2"]) * (np.cos(((((data["abs_maxbatch_msignal"]) + (data["stdbatch_slices2"]))/2.0)))))) * 2.0)))) + (((data["rangebatch_slices2"]) + (((complex(0,1)*np.conjugate((((data["abs_maxbatch_msignal"]) + (data["stdbatch_slices2"]))/2.0))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(((data["maxtominbatch_slices2_msignal"]) - (((data["rangebatch_msignal"]) + (data["medianbatch_msignal"]))))),complex(1.), complex(0.) )) + (((complex(2.0)) + (data["abs_avgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((np.sin(((((((data["abs_minbatch_slices2_msignal"]) + (((data["meanbatch_msignal"]) * 2.0)))/2.0)) + (np.where(np.abs(np.tanh((((data["meanbatch_msignal"]) - (((data["meanbatch_msignal"]) * 2.0)))))) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2_msignal"]) / (((np.cos((((data["medianbatch_msignal"]) * 2.0)))) * (np.sin((np.sin((np.cos((np.sin((data["abs_maxbatch_msignal"]))))))))))))) - (np.cos((np.where(np.abs(complex(0.0)) <= np.abs(((np.tanh((data["meanbatch_slices2_msignal"]))) / 2.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) - (np.where(np.abs(((data["signal_shift_-1_msignal"]) - (((np.conjugate(data["meanbatch_msignal"])) - (np.sin((np.cos(((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0)))))))))) <= np.abs((((-((((data["medianbatch_slices2_msignal"]) / 2.0))))) - (complex(3.0)))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["rangebatch_slices2"]) - (np.conjugate((((data["meanbatch_msignal"]) + (np.conjugate(np.tanh((np.conjugate(np.sin((((np.sin((np.tanh(((((((((data["rangebatch_slices2"]) - (data["meanbatch_msignal"]))) * 2.0)) + (data["minbatch_slices2_msignal"]))/2.0)))))) - (np.cos((data["maxtominbatch_slices2"]))))))))))))/2.0)))))))) +

                            0.100000*np.tanh(np.real((-((np.cos((((data["meanbatch_msignal"]) * (complex(1.0)))))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["minbatch_msignal"]))) * 2.0)) * (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) + (np.tanh(((((np.sin((((data["meanbatch_msignal"]) - (np.sin((((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(((data["abs_maxbatch_slices2"]) + (data["medianbatch_msignal"]))),complex(1.), complex(0.) )) + (data["rangebatch_slices2"]))))))))) + (data["stdbatch_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.where(np.abs(((np.cos(((((((data["abs_minbatch_slices2"]) + (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))/2.0)) * (data["abs_maxbatch_msignal"]))))) * (((complex(0,1)*np.conjugate(np.sin((complex(1.0))))) + (data["abs_maxbatch_msignal"]))))) > np.abs(complex(7.0)),complex(1.), complex(0.) )) + (data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.sin((data["minbatch_msignal"]))) <= np.abs(((data["medianbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )) - (np.where(np.abs(((np.sin((data["maxtominbatch_msignal"]))) / (np.sin((data["stdbatch_msignal"]))))) <= np.abs(np.sin(((-((((data["minbatch_msignal"]) / 2.0))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) - (np.cos((np.where(np.abs(complex(-1.0)) <= np.abs(((data["meanbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(((data["meanbatch_slices2_msignal"]) + (((data["meanbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"])))))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["rangebatch_slices2"]))) * (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch_slices2_msignal"]) + (((complex(7.11982917785644531)) + (np.where(np.abs(data["rangebatch_slices2_msignal"]) <= np.abs(((complex(0,1)*np.conjugate(data["maxtominbatch_slices2_msignal"])) * (((data["maxtominbatch_slices2_msignal"]) + (((complex(7.11982917785644531)) + (np.sin((((((complex(7.11982917785644531)) + (((np.cos(((-((data["minbatch_msignal"])))))) / 2.0)))) / 2.0)))))))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["abs_maxbatch_msignal"]))) - (((((complex(11.45155143737792969)) - (((((data["abs_maxbatch_slices2"]) - (complex(1.0)))) * (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.cos((np.cos((np.cos((data["abs_maxbatch_msignal"]))))))),complex(1.), complex(0.) )))))) * (data["meanbatch_slices2_msignal"]))))) * (np.cos((data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate((((data["rangebatch_slices2"]) + (np.sin((data["rangebatch_slices2"]))))/2.0))) > np.abs(((data["maxbatch_slices2_msignal"]) + ((-((np.sin((np.conjugate(((data["maxbatch_slices2"]) * 2.0)))))))))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) / (np.cos((data["maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((((-((complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"]))))) + (((((((data["meanbatch_slices2_msignal"]) * (complex(-2.0)))) / (np.sin((data["rangebatch_slices2"]))))) + (np.sin(((-((data["abs_maxbatch"]))))))))))) +

                            0.100000*np.tanh(np.real((((((((data["rangebatch_slices2"]) + ((((((data["rangebatch_slices2"]) + (np.sin((np.where(np.abs(data["rangebatch_slices2"]) > np.abs(((np.conjugate(data["maxbatch_slices2"])) * 2.0)),complex(1.), complex(0.) )))))/2.0)) * 2.0)))/2.0)) / (np.sin((data["rangebatch_slices2"]))))) - (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) - (np.where(np.abs(((data["meanbatch_msignal"]) - (np.tanh((np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))))) <= np.abs(np.where(np.abs(data["meanbatch_msignal"]) > np.abs(complex(3.81429886817932129)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["maxbatch_slices2_msignal"]))) * (((((((np.cos((data["rangebatch_slices2"]))) + (complex(0,1)*np.conjugate(complex(1.0))))) * 2.0)) + (data["maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["signal_shift_-1"]))) + (complex(0,1)*np.conjugate(np.sin((np.conjugate((-((np.where(np.abs(np.sin((np.tanh((complex(-1.0)))))) <= np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )))))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((np.sin((data["rangebatch_slices2"]))) - (complex(0,1)*np.conjugate((-((data["medianbatch_slices2"]))))))))) + (np.sin((np.sin((data["rangebatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((((complex(0,1)*np.conjugate(((((complex(0,1)*np.conjugate(complex(-2.0))) - (data["meanbatch_msignal"]))) * 2.0))) + (((data["mean_abs_chgbatch_slices2_msignal"]) / (np.cos((complex(-2.0))))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) * 2.0))) +

                            0.100000*np.tanh(np.real(((((complex(6.0)) - (np.tanh((((np.tanh((((data["meanbatch_msignal"]) / 2.0)))) / 2.0)))))) / (np.cos((((data["rangebatch_slices2"]) + (np.tanh(((((data["minbatch_msignal"]) + (np.cos((data["medianbatch_slices2"]))))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) - (complex(0,1)*np.conjugate(((np.conjugate((((((((((-(((((np.sin((data["meanbatch_msignal"]))) + (data["meanbatch_msignal"]))/2.0))))) + (np.conjugate(data["medianbatch_msignal"])))/2.0)) + (complex(-3.0)))/2.0)) / 2.0))) + (((((complex(0,1)*np.conjugate(np.sin((data["medianbatch_msignal"])))) * 2.0)) - (data["meanbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (np.where(np.abs(complex(0.0)) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(1.10663676261901855)) > np.abs(np.conjugate(((((((((complex(1.10664033889770508)) + (np.conjugate(np.sin((np.cos((data["abs_maxbatch_slices2_msignal"])))))))) / 2.0)) * 2.0)) * 2.0))),complex(1.), complex(0.) )) + (np.conjugate(((np.conjugate(np.sin((((np.conjugate(data["abs_maxbatch_slices2_msignal"])) * 2.0))))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) - (np.cos((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(((((complex(0,1)*np.conjugate(data["abs_maxbatch_slices2"])) * (data["abs_avgbatch_msignal"]))) / 2.0)),complex(1.), complex(0.) )))))) * (np.sin((data["minbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((np.tanh((np.tanh((((data["meanbatch_msignal"]) * 2.0)))))) + (data["meanbatch_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real((((((-((np.cos((data["meanbatch_msignal"])))))) / 2.0)) + ((((-((np.cos((data["meanbatch_msignal"])))))) / 2.0))))) +

                            0.100000*np.tanh(np.real((((np.sin((np.conjugate(complex(1.0))))) + (data["stdbatch_slices2"]))/2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2"]) / (((((complex(0,1)*np.conjugate(data["stdbatch_slices2"])) + (data["abs_avgbatch_msignal"]))) * (np.sin((data["mean_abs_chgbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((data["abs_avgbatch_slices2_msignal"]) * (np.sin((((data["minbatch_msignal"]) - (np.sin(((-((((np.where(np.abs(((data["abs_avgbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))) > np.abs(np.sin((((data["stdbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) * 2.0))))))))))))) * 2.0)))) +

                            0.100000*np.tanh(np.real(((((((((((np.sin((data["mean_abs_chgbatch_slices2"]))) / 2.0)) + (((np.sin((data["maxbatch_msignal"]))) - (np.where(np.abs(data["maxbatch_msignal"]) > np.abs(np.sin((data["signal"]))),complex(1.), complex(0.) )))))/2.0)) * (np.sin((np.sin((data["maxbatch_msignal"]))))))) + (((np.sin((data["abs_maxbatch_msignal"]))) * (complex(10.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) / (((complex(-2.0)) + (((data["medianbatch_msignal"]) - (((np.cos((complex(-2.0)))) / (complex(-2.0))))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.conjugate(np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(np.cos((data["abs_minbatch_slices2_msignal"]))),complex(1.), complex(0.) ))) - (data["meanbatch_slices2"])))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((data["medianbatch_msignal"])))) - (((data["meanbatch_slices2_msignal"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_msignal"]) + (np.conjugate(np.conjugate(complex(0,1)*np.conjugate(((((complex(-3.0)) + (np.where(np.abs(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_msignal"])) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )))) + (data["minbatch_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_slices2_msignal"]))) - (complex(0,1)*np.conjugate(((np.sin(((((((((complex(3.0)) - (np.sin((data["stdbatch_slices2"]))))) * 2.0)) + (np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"]))))))/2.0)))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["abs_avgbatch_msignal"]) + (((np.where(np.abs(data["stdbatch_msignal"]) > np.abs(np.cos((np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) * 2.0)))/2.0)) + (data["rangebatch_slices2"])))))) +

                            0.100000*np.tanh(np.real((((((-((np.conjugate(np.cos((((np.cos((data["rangebatch_slices2"]))) * (data["maxbatch_slices2"]))))))))) + (complex(0,1)*np.conjugate(np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(((((np.cos((((np.cos((data["rangebatch_slices2"]))) + (data["rangebatch_slices2"]))))) - (data["rangebatch_slices2"]))) / 2.0)),complex(1.), complex(0.) ))))) / 2.0))) +

                            0.100000*np.tanh(np.real((((((data["stdbatch_slices2"]) + ((((np.sin((data["maxbatch_msignal"]))) + (((((complex(1.0)) / 2.0)) * 2.0)))/2.0)))/2.0)) + (complex(0,1)*np.conjugate(np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(np.cos((complex(1.0)))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(((((data["signal_shift_+1"]) / 2.0)) * 2.0))) <= np.abs(complex(0.0)),complex(1.), complex(0.) )) <= np.abs(complex(0,1)*np.conjugate(np.where(np.abs(data["signal_shift_-1"]) > np.abs(np.cos((data["abs_maxbatch"]))),complex(1.), complex(0.) ))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) * (((((complex(0,1)*np.conjugate(((((np.sin(((-((complex(0.0))))))) * 2.0)) + (np.sin((data["rangebatch_slices2"])))))) - (np.sin((data["rangebatch_slices2"]))))) / 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real((-((np.cos(((((data["minbatch_msignal"]) + (((np.where(np.abs(complex(0,1)*np.conjugate(data["stdbatch_slices2_msignal"])) <= np.abs((((-((np.tanh((data["abs_avgbatch_slices2_msignal"])))))) * 2.0)),complex(1.), complex(0.) )) * 2.0)))/2.0)))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_msignal"]) + ((((((data["stdbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"]))) + (complex(0,1)*np.conjugate(((data["minbatch_msignal"]) + (complex(-3.0))))))/2.0))))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2_msignal"]) * (data["signal"])))) +

                            0.100000*np.tanh(np.real(((((np.cos(((-((data["rangebatch_slices2"])))))) * 2.0)) - (np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((np.sin((((data["mean_abs_chgbatch_msignal"]) + (np.tanh((np.conjugate(data["meanbatch_msignal"])))))))) + (((((((np.cos((np.cos((data["signal_shift_-1_msignal"]))))) + (((data["meanbatch_msignal"]) * (complex(2.0)))))) * (((data["mean_abs_chgbatch_msignal"]) * (data["signal_shift_-1_msignal"]))))) * 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs((((((data["meanbatch_msignal"]) + (np.cos((data["signal_shift_+1"]))))/2.0)) + (np.where(np.abs(np.conjugate(data["stdbatch_msignal"])) > np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (data["abs_maxbatch_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((np.where(np.abs(np.cos((((data["abs_maxbatch_slices2_msignal"]) + (np.sin((np.sin((np.sin((data["abs_minbatch_slices2"]))))))))))) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))) <= np.abs(complex(2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["minbatch_msignal"]))) / 2.0)) + (((np.cos((data["minbatch_slices2_msignal"]))) * (complex(3.0))))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch"]) * (np.sin((((np.sin((data["abs_maxbatch"]))) - (((data["signal_shift_-1"]) - (np.where(np.abs(((data["abs_maxbatch"]) - (((data["abs_maxbatch"]) - (((data["signal_shift_-1"]) - (np.where(np.abs(np.sin((data["abs_minbatch_slices2"]))) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )))))))) > np.abs(np.sin((data["abs_maxbatch"]))),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch_msignal"]) - (np.cos((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs((((data["stdbatch_slices2"]) + (np.cos((np.cos((np.cos((((np.where(np.abs(data["signal_shift_+1"]) > np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) * (data["abs_maxbatch_msignal"]))))))))))/2.0)),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(data["stdbatch_slices2"])) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(complex(-3.0)) > np.abs(((((np.conjugate(((data["signal"]) / 2.0))) * (data["maxbatch_msignal"]))) - (data["maxbatch_slices2"]))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.cos((((np.where(np.abs((-((np.cos((complex(0,1)*np.conjugate(((data["meanbatch_msignal"]) - (np.sin((data["maxbatch_slices2"]))))))))))) > np.abs((((data["abs_minbatch_msignal"]) + (data["maxbatch_msignal"]))/2.0)),complex(1.), complex(0.) )) + (data["maxbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["rangebatch_slices2"]) + (np.where(np.abs(np.sin(((-((np.where(np.abs(np.conjugate(np.conjugate(((np.tanh((data["medianbatch_slices2_msignal"]))) / 2.0)))) <= np.abs(complex(0.0)),complex(1.), complex(0.) ))))))) <= np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) - (np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.conjugate(data["medianbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((complex(0.0)) + (((data["meanbatch_slices2_msignal"]) * (np.sin((data["minbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch"]) + (((data["stdbatch_msignal"]) * (np.cos(((((((data["abs_maxbatch_msignal"]) + (np.cos((data["mean_abs_chgbatch_slices2"]))))/2.0)) / 2.0))))))))))) +

                            0.100000*np.tanh(np.real((((((data["rangebatch_slices2"]) + (data["signal_shift_+1_msignal"]))/2.0)) / (((complex(1.0)) / (np.cos(((-((((data["abs_maxbatch_slices2_msignal"]) - (np.where(np.abs(np.tanh((np.cos((np.cos(((-((((data["abs_maxbatch"]) - (complex(1.0))))))))))))) <= np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) )))))))))))))) +

                            0.100000*np.tanh(np.real(data["minbatch"])) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_msignal"]) - (np.where(np.abs(np.where(np.abs(data["rangebatch_slices2"]) > np.abs(np.where(np.abs(((np.cos((data["maxbatch_msignal"]))) * 2.0)) <= np.abs(((data["rangebatch_slices2"]) / 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(((((data["abs_maxbatch_msignal"]) / 2.0)) * (data["signal_shift_-1_msignal"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((((np.conjugate(np.conjugate(data["stdbatch_slices2"]))) * (data["medianbatch_msignal"]))) + (np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["maxtominbatch_slices2"]) <= np.abs(complex(12.64071369171142578)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) + (np.sin((np.conjugate(data["stdbatch_slices2"]))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((data["maxbatch_msignal"]) - (data["minbatch_msignal"]))) - (complex(0,1)*np.conjugate(data["minbatch_msignal"])))))) + (np.where(np.abs((((data["maxbatch_msignal"]) + (np.sin((complex(0,1)*np.conjugate(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(((data["minbatch_msignal"]) * (((data["abs_maxbatch_msignal"]) - (data["minbatch_msignal"]))))),complex(1.), complex(0.) ))))))/2.0)) > np.abs(((np.cos((data["mean_abs_chgbatch_slices2"]))) / 2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch"]) <= np.abs(((np.conjugate(data["rangebatch_msignal"])) + (np.tanh((complex(0,1)*np.conjugate(data["rangebatch_msignal"])))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1_msignal"]) - (np.where(np.abs(np.sin((((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(np.sin((((data["mean_abs_chgbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) * (((data["abs_minbatch_slices2"]) * (data["mean_abs_chgbatch_msignal"]))))))) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))) * ((-((data["meanbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["maxbatch_slices2"]) - (data["signal_shift_+1"]))) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_msignal"]) + (np.cos((np.where(np.abs(complex(-2.0)) <= np.abs(((data["abs_maxbatch_slices2"]) * 2.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((-((np.conjugate(data["signal_shift_+1_msignal"]))))) * ((((-((np.where(np.abs(np.tanh((data["meanbatch_slices2_msignal"]))) > np.abs(((np.cos(((-((data["meanbatch_slices2_msignal"])))))) / 2.0)),complex(1.), complex(0.) ))))) + (((data["meanbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((data["mean_abs_chgbatch_slices2"]))) > np.abs(np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(((np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.where(np.abs(np.cos((np.where(np.abs(np.sin((data["meanbatch_slices2_msignal"]))) > np.abs(np.where(np.abs(data["rangebatch_slices2_msignal"]) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) > np.abs(np.sin((data["maxtominbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((((np.tanh(((-((data["abs_maxbatch_msignal"])))))) + (data["abs_maxbatch"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) - (np.where(np.abs(complex(0,1)*np.conjugate(complex(2.0))) > np.abs(((data["signal_shift_-1_msignal"]) + (((data["signal_shift_-1_msignal"]) - (data["maxbatch_slices2"]))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["maxtominbatch_msignal"]))) * (((((-((((((data["maxtominbatch_msignal"]) - (((data["stdbatch_slices2_msignal"]) * (np.conjugate(data["signal_shift_-1"])))))) * ((((data["meanbatch_slices2"]) + (np.where(np.abs(data["rangebatch_slices2"]) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )))/2.0))))))) + (((((data["rangebatch_slices2"]) / 2.0)) / 2.0)))/2.0)))) * (data["minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_slices2"]))) * (np.where(np.abs(data["signal_shift_+1"]) > np.abs(np.conjugate(complex(0,1)*np.conjugate(data["abs_minbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((np.conjugate(data["signal"])))))))) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["abs_maxbatch"]) + (data["mean_abs_chgbatch_msignal"]))))) * 2.0)) + (complex(2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(np.cos((((data["medianbatch_slices2_msignal"]) / 2.0)))) <= np.abs(np.sin((data["maxbatch_msignal"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.cos(((((np.tanh((np.cos((((np.cos((((np.sin((data["stdbatch_slices2_msignal"]))) / 2.0)))) / 2.0)))))) + (data["abs_maxbatch"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["medianbatch_msignal"]))) / ((((((np.sin((np.conjugate(np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(np.sin((data["medianbatch_msignal"])))),complex(1.), complex(0.) ))))) - (data["signal_shift_-1_msignal"]))) + (np.conjugate((((((((data["stdbatch_slices2_msignal"]) - (data["signal_shift_-1_msignal"]))) + (np.conjugate(data["minbatch"])))/2.0)) / 2.0))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((data["medianbatch_slices2_msignal"]) + (np.tanh((np.cos(((((complex(2.0)) + (data["maxtominbatch"]))/2.0)))))))/2.0)) <= np.abs(complex(-2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["abs_maxbatch"]))) + (np.sin((data["abs_maxbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(data["signal_shift_+1_msignal"]))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((data["abs_minbatch_slices2_msignal"]))) > np.abs(((data["signal"]) * (np.cos(((-((np.conjugate((-((((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(np.cos((np.sin((np.cos((np.where(np.abs(((complex(3.0)) - (np.cos((data["rangebatch_slices2"]))))) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )))))))),complex(1.), complex(0.) )) * 2.0))))))))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((np.sin((np.cos((np.conjugate(complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate((-((data["maxbatch_slices2_msignal"]))))) - ((-((np.where(np.abs(complex(0,1)*np.conjugate(np.cos((complex(0.65111893415451050))))) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) )))))))))))))))) +

                            0.100000*np.tanh(np.real(((((((data["medianbatch_slices2_msignal"]) + (((data["minbatch_slices2"]) * ((-((np.cos(((((-((data["signal_shift_-1_msignal"])))) / 2.0))))))))))/2.0)) + (((data["medianbatch_slices2_msignal"]) * ((-((data["signal_shift_-1_msignal"])))))))/2.0))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_msignal"]) * (np.sin(((-((data["rangebatch_slices2"])))))))) - (np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(np.sin((((np.sin((np.sin((np.sin((np.where(np.abs(((data["signal_shift_+1_msignal"]) * (((data["abs_minbatch_slices2_msignal"]) / 2.0)))) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))))))) + (complex(2.0)))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((complex(9.0)) * (data["abs_maxbatch_msignal"]))) <= np.abs(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((data["minbatch_slices2_msignal"]))) + (np.tanh((np.sin((data["minbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(complex(0.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((complex(3.0)))) / (data["maxtominbatch_msignal"]))) + (((np.conjugate(np.cos((((complex(1.0)) * 2.0))))) + (((data["stdbatch_msignal"]) + (((np.cos((data["abs_maxbatch"]))) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(((complex(3.0)) + (np.conjugate(data["signal"]))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((data["medianbatch_slices2"]))) + (complex(0,1)*np.conjugate((-((np.tanh((np.where(np.abs(np.where(np.abs(data["signal"]) > np.abs(np.where(np.abs(complex(1.0)) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_-1"]) + (data["signal"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(((np.sin((((np.tanh((np.sin((np.conjugate(data["abs_avgbatch_slices2"])))))) * 2.0)))) * (np.tanh((((data["minbatch_slices2"]) / 2.0)))))),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(data["abs_maxbatch"])) <= np.abs(np.tanh((data["mean_abs_chgbatch_slices2"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_slices2"]) + (((((((data["meanbatch_slices2_msignal"]) * 2.0)) + (((np.cos((np.cos((((((data["meanbatch_slices2_msignal"]) * 2.0)) / 2.0)))))) * 2.0)))) * 2.0)))) - ((-(((((data["meanbatch_slices2_msignal"]) + (((data["medianbatch_slices2"]) + ((((-((complex(0,1)*np.conjugate((-((data["meanbatch_slices2"])))))))) * 2.0)))))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.conjugate(data["abs_maxbatch"])))) + (((np.sin((np.sin((np.where(np.abs(np.tanh((complex(2.0)))) <= np.abs(np.tanh((data["maxbatch_msignal"]))),complex(1.), complex(0.) )))))) + (((np.tanh((((data["abs_minbatch_msignal"]) / 2.0)))) + (np.cos((np.conjugate(np.conjugate(data["abs_maxbatch"])))))))))))) +

                            0.100000*np.tanh(np.real(((((data["rangebatch_slices2"]) + (((np.sin((complex(1.0)))) / 2.0)))) * (((data["rangebatch_slices2"]) * (np.sin((np.sin((((data["rangebatch_slices2"]) + (((data["abs_maxbatch_slices2_msignal"]) / 2.0))))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((complex(-1.0))))) +

                            0.100000*np.tanh(np.real(np.sin(((((np.sin((np.cos((((np.sin((np.cos(((((data["medianbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0)))))) + (complex(10.05275821685791016)))))))) + ((((((((data["medianbatch_msignal"]) + (complex(0,1)*np.conjugate(data["medianbatch_msignal"])))/2.0)) * 2.0)) - (data["rangebatch_slices2"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(complex(0,1)*np.conjugate(np.conjugate(np.tanh((((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])) + (data["abs_avgbatch_slices2"]))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["signal_shift_+1_msignal"]) / 2.0)))) +

                            0.100000*np.tanh(np.real(np.cos((((data["rangebatch_slices2"]) - (np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(np.cos((np.where(np.abs(complex(3.0)) <= np.abs(np.where(np.abs(np.cos((((data["rangebatch_slices2"]) - (np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(np.tanh((data["rangebatch_slices2"]))),complex(1.), complex(0.) )))))) > np.abs(complex(-3.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(complex(1.0))) > np.abs(np.sin((np.where(np.abs(((np.cos((data["abs_minbatch_slices2_msignal"]))) * 2.0)) <= np.abs(((data["signal"]) / 2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) <= np.abs(((data["stdbatch_slices2"]) * (data["signal_shift_-1"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) * (((((complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"])) - (((((((np.sin((data["meanbatch_slices2_msignal"]))) - (np.sin((complex(0.0)))))) - (np.sin((np.sin((data["abs_maxbatch_msignal"]))))))) * (data["maxbatch_slices2"]))))) - (np.conjugate(data["medianbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real((-((((data["signal_shift_+1_msignal"]) - (np.cos((((((np.sin((data["signal_shift_-1_msignal"]))) * 2.0)) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.sin((np.sin((data["maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["signal"]) / 2.0)) + (data["medianbatch_slices2_msignal"]))/2.0)))) +

                            0.100000*np.tanh(np.real((((np.conjugate(np.tanh((complex(0,1)*np.conjugate(data["signal_shift_-1"]))))) + ((((np.cos((data["abs_minbatch_slices2_msignal"]))) + (np.tanh((complex(0,1)*np.conjugate(data["maxtominbatch"])))))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(((np.where(np.abs(complex(0,1)*np.conjugate(np.cos((data["signal_shift_+1_msignal"])))) > np.abs(np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.where(np.abs(complex(-2.0)) <= np.abs((((data["signal_shift_+1_msignal"]) + (np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)) > np.abs(np.cos((complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((complex(14.49003028869628906)) + (np.tanh((data["abs_maxbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(((data["signal_shift_-1_msignal"]) * (complex(0,1)*np.conjugate((((((np.sin((complex(-2.0)))) + (data["abs_maxbatch"]))/2.0)) + (data["medianbatch_slices2"]))))))) / (np.cos((np.sin((((data["minbatch"]) / (data["meanbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((np.where(np.abs(((np.cos((data["rangebatch_slices2"]))) * (np.cos((complex(0,1)*np.conjugate(np.where(np.abs(data["maxbatch_msignal"]) > np.abs(np.where(np.abs(((data["maxbatch_msignal"]) / 2.0)) <= np.abs(np.cos((((data["maxtominbatch_slices2"]) * (((np.cos(((-((complex(5.72163248062133789))))))) / 2.0)))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) <= np.abs(complex(8.53381443023681641)),complex(1.), complex(0.) )))) / 2.0))) +

                            0.100000*np.tanh(np.real(((complex(-1.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((complex(4.0)) * (((data["signal_shift_-1_msignal"]) * (((np.where(np.abs(complex(0,1)*np.conjugate(((complex(1.0)) * (((data["signal_shift_-1_msignal"]) * (((data["minbatch"]) - (((np.cos((data["maxbatch_msignal"]))) * 2.0))))))))) <= np.abs(complex(-3.0)),complex(1.), complex(0.) )) - (((np.cos((data["maxbatch_msignal"]))) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["meanbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((((((np.cos((data["signal_shift_-1_msignal"]))) + (((data["meanbatch_msignal"]) / 2.0)))/2.0)) + (data["signal_shift_-1_msignal"]))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((complex(-1.0))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.tanh((np.tanh((np.where(np.abs(complex(-1.0)) <= np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((((complex(0,1)*np.conjugate(data["medianbatch_slices2"])) / 2.0)) / 2.0)) * 2.0)) > np.abs(complex(-2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.where(np.abs((((complex(-1.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)) > np.abs(complex(0,1)*np.conjugate(complex(3.87890076637268066))),complex(1.), complex(0.) ))) * (((data["abs_minbatch_slices2_msignal"]) / (((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) + (((data["stdbatch_slices2"]) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(((np.sin((np.where(np.abs(((data["stdbatch_slices2"]) / 2.0)) > np.abs(np.cos((np.sin((np.tanh((data["mean_abs_chgbatch_msignal"]))))))),complex(1.), complex(0.) )))) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.tanh((complex(0,1)*np.conjugate(data["abs_maxbatch_msignal"])))) / 2.0)) * (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs((((((data["abs_maxbatch_slices2_msignal"]) + (((np.tanh((complex(2.0)))) + (data["maxbatch_msignal"]))))/2.0)) - (np.tanh((data["signal_shift_-1"]))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.sin((data["maxbatch_msignal"]))) + (((np.sin((((((np.where(np.abs(complex(2.0)) > np.abs(np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(np.where(np.abs(np.sin((np.sin((complex(2.0)))))) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)) * 2.0)))) / 2.0)))))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(complex(3.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2_msignal"]) + (complex(-1.0))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(np.where(np.abs(np.where(np.abs(data["stdbatch_slices2"]) > np.abs(np.where(np.abs(data["minbatch_msignal"]) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate((-((data["meanbatch_slices2_msignal"]))))),complex(1.), complex(0.) )) + (complex(3.0)))/2.0))) +

                            0.100000*np.tanh(np.real(data["rangebatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((data["abs_avgbatch_msignal"]) + (np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(((complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate((((data["medianbatch_slices2_msignal"]) + (((data["maxbatch_slices2"]) * (np.conjugate(np.cos(((-((complex(-3.0)))))))))))/2.0))) > np.abs((-((data["stdbatch_slices2"])))),complex(1.), complex(0.) ))) / 2.0)),complex(1.), complex(0.) )))/2.0)) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((np.sin((np.conjugate(complex(0.90665841102600098))))) / 2.0)) / 2.0)) + (np.cos(((((((np.conjugate(data["rangebatch_slices2"])) - (data["abs_minbatch_slices2_msignal"]))) + (np.sin((((data["abs_avgbatch_msignal"]) / 2.0)))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["abs_minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.tanh((complex(0,1)*np.conjugate(((data["abs_minbatch_slices2"]) / 2.0)))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["rangebatch_slices2"]) + (np.sin((((data["rangebatch_slices2"]) + (np.sin((np.sin((data["minbatch_slices2_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(((np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) )))) / 2.0)) > np.abs(((data["minbatch_msignal"]) / 2.0)),complex(1.), complex(0.) ))))  

    

    def GP_class_6(self,data):

        return self.Output( -3.281070 +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) - ((((((((data["maxbatch_slices2_msignal"]) - (np.where(np.abs(data["maxbatch_slices2"]) > np.abs(((data["medianbatch_msignal"]) + ((((-((data["stdbatch_slices2"])))) - (np.tanh((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(((data["maxtominbatch"]) / 2.0)),complex(1.), complex(0.) )))))))),complex(1.), complex(0.) )))) + (data["medianbatch_slices2_msignal"]))/2.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.tanh((np.conjugate(((data["meanbatch_slices2_msignal"]) * 2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(data["signal"])) +

                            0.100000*np.tanh(np.real(data["mean_abs_chgbatch_slices2"])) +

                            0.100000*np.tanh(np.real(data["medianbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) * (data["medianbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch_slices2"]) <= np.abs(((data["stdbatch_msignal"]) * (np.sin((np.cos(((-((np.tanh(((-((np.tanh((((np.sin((np.tanh((((data["stdbatch_msignal"]) + ((-((data["signal_shift_+1"])))))))))) * 2.0)))))))))))))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2"]) - (((np.where(np.abs(((np.where(np.abs((((((data["signal_shift_+1_msignal"]) + (data["meanbatch_slices2"]))/2.0)) - (data["mean_abs_chgbatch_slices2_msignal"]))) <= np.abs(np.sin((np.sin((np.cos((((data["meanbatch_slices2_msignal"]) * (data["abs_maxbatch_msignal"]))))))))),complex(1.), complex(0.) )) / 2.0)) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) + (complex(-1.0))))) +

                            0.100000*np.tanh(np.real((((((np.sin((np.conjugate(np.sin((data["medianbatch_slices2_msignal"])))))) * 2.0)) + (data["medianbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real((((((data["medianbatch_msignal"]) - (np.tanh((data["maxbatch_slices2"]))))) + (data["meanbatch_slices2"]))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((np.cos((data["abs_maxbatch_msignal"]))))) * (data["rangebatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((complex(12.15232276916503906)) * (((data["meanbatch_slices2_msignal"]) - (np.where(np.abs(data["signal_shift_-1"]) <= np.abs((((((data["meanbatch_msignal"]) - (np.sin((data["signal_shift_-1_msignal"]))))) + (data["meanbatch_msignal"]))/2.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["maxbatch_msignal"]))) * ((((((((np.cos((data["maxbatch_msignal"]))) * (((data["maxbatch_msignal"]) + (((((data["signal_shift_+1"]) + (((np.cos((data["maxbatch_msignal"]))) / 2.0)))) + (((data["medianbatch_slices2_msignal"]) * 2.0)))))))) / 2.0)) + (data["maxbatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_msignal"]) - (np.tanh((data["signal"]))))) * (data["signal"])))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2_msignal"]) / 2.0)) + (np.cos((np.conjugate(np.tanh((data["signal_shift_+1_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((((data["signal"]) - (np.where(np.abs(((data["mean_abs_chgbatch_msignal"]) * 2.0)) > np.abs(((((((np.sin((data["signal_shift_+1"]))) * (complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"])))) * 2.0)) * (data["signal_shift_-1"]))),complex(1.), complex(0.) )))) - (data["mean_abs_chgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(data["signal"])) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) - (np.where(np.abs(complex(3.0)) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate(data["maxbatch_msignal"])))) - (((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate((-((((data["maxbatch_msignal"]) - (complex(0,1)*np.conjugate((-((data["maxbatch_msignal"])))))))))))))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) + (data["meanbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(data["medianbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) + (((((data["signal_shift_+1_msignal"]) * (np.tanh((((data["meanbatch_slices2"]) * (np.sin((data["signal_shift_+1"]))))))))) * (data["abs_avgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((-(((-((complex(0,1)*np.conjugate(((((complex(-1.0)) + (((np.conjugate(data["abs_avgbatch_slices2_msignal"])) * 2.0)))) / 2.0))))))))) * 2.0)) + ((((complex(-3.0)) + (data["stdbatch_slices2"]))/2.0))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch"])) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) - (((data["minbatch_slices2_msignal"]) / (((data["medianbatch_slices2_msignal"]) + (np.sin((complex(-2.0))))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) - (np.tanh((data["abs_maxbatch"])))))) +

                            0.100000*np.tanh(np.real((((((np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(complex(0,1)*np.conjugate(np.cos((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0))))),complex(1.), complex(0.) )) + (np.cos((data["abs_maxbatch_slices2_msignal"]))))/2.0)) * (data["abs_maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((np.cos((((data["abs_maxbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs((((complex(2.0)) + (np.cos((((data["rangebatch_slices2_msignal"]) + ((((complex(-1.0)) + (((np.cos((complex(-1.0)))) / 2.0)))/2.0)))))))/2.0)),complex(1.), complex(0.) )) > np.abs(np.conjugate(np.conjugate(data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )))))) * (complex(2.0))))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["maxbatch_slices2_msignal"]) + (np.cos((data["stdbatch_slices2"]))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((np.where(np.abs(((data["medianbatch_slices2_msignal"]) + (np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(data["abs_minbatch_msignal"]),complex(1.), complex(0.) )))) <= np.abs(np.sin((data["meanbatch_msignal"]))),complex(1.), complex(0.) )) + (np.sin((data["meanbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_msignal"]) * (((complex(-3.0)) * (np.cos((((data["maxbatch_msignal"]) + (((np.conjugate(complex(0,1)*np.conjugate(data["medianbatch_msignal"]))) - (complex(1.0))))))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_msignal"]))) * (((((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) + (((((data["medianbatch_slices2_msignal"]) * (data["signal_shift_-1"]))) + (np.where(np.abs(data["medianbatch_msignal"]) > np.abs(complex(1.0)),complex(1.), complex(0.) )))))) + (data["abs_maxbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["maxbatch_msignal"]))) * 2.0)) - ((-(((-((np.conjugate(np.where(np.abs(((data["signal"]) - (np.cos((data["maxbatch_slices2_msignal"]))))) <= np.abs(np.where(np.abs(complex(0.0)) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(np.cos((complex(0.0)))),complex(1.), complex(0.) ))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((data["minbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(data["minbatch_msignal"]) > np.abs(((np.where(np.abs(data["minbatch_msignal"]) > np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(complex(-1.0)) <= np.abs(np.where(np.abs(((np.cos((data["stdbatch_slices2_msignal"]))) * 2.0)) > np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["minbatch_msignal"]) - (np.sin((complex(0,1)*np.conjugate(np.cos(((-((np.sin((np.tanh((np.cos(((-((np.where(np.abs(data["rangebatch_msignal"]) > np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )))))))))))))))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((data["medianbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((((np.cos((data["minbatch_msignal"]))) * 2.0)) * 2.0)) + (((complex(-3.0)) * (np.where(np.abs(np.conjugate(((np.cos((data["maxbatch_msignal"]))) * 2.0))) <= np.abs(complex(-1.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_minbatch_slices2_msignal"]) + (complex(-3.0))))))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_msignal"]) * (((np.sin((np.where(np.abs(((np.where(np.abs(complex(2.0)) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) * (np.conjugate(data["medianbatch_msignal"])))) <= np.abs(np.conjugate(np.where(np.abs(data["stdbatch_msignal"]) > np.abs((((data["abs_minbatch_slices2_msignal"]) + ((((data["abs_minbatch_slices2_msignal"]) + (np.sin(((-((data["maxbatch_slices2"])))))))/2.0)))/2.0)),complex(1.), complex(0.) ))),complex(1.), complex(0.) )))) - (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((-((data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((data["meanbatch_slices2_msignal"]) - (((((((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])) / (np.cos((((((data["medianbatch_msignal"]) * 2.0)) - (data["meanbatch_msignal"]))))))) * 2.0)) - (((data["stdbatch_msignal"]) * 2.0)))))) > np.abs((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(data["meanbatch_slices2"]) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_msignal"]) * (np.cos(((-(((-((data["minbatch_msignal"]))))))))))) + ((((data["minbatch_slices2_msignal"]) + ((((np.conjugate(((np.where(np.abs(np.cos(((-((data["maxbatch_msignal"])))))) <= np.abs(np.cos((data["minbatch_msignal"]))),complex(1.), complex(0.) )) * (data["minbatch_msignal"])))) + ((((-(((-(((-((complex(1.0))))))))))) / 2.0)))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((complex(-2.0)) + (((np.sin((data["meanbatch_slices2_msignal"]))) - (np.where(np.abs(np.sin((np.cos((complex(-2.0)))))) > np.abs(np.sin((complex(-2.0)))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((complex(0,1)*np.conjugate(((data["abs_maxbatch"]) - (data["minbatch_msignal"])))) + (complex(0,1)*np.conjugate(((complex(-3.0)) - (data["minbatch_msignal"])))))) + (data["minbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((((((data["maxbatch_slices2_msignal"]) * (np.tanh(((-((((data["maxbatch_slices2_msignal"]) * 2.0))))))))) * 2.0)))) * 2.0)) + (np.tanh((np.where(np.abs(np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )) > np.abs(complex(3.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_msignal"]))) * (complex(0.0))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.cos((((data["meanbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2"]))))))) - (((np.where(np.abs(complex(-3.0)) > np.abs(((np.cos((complex(-3.0)))) * 2.0)),complex(1.), complex(0.) )) - (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.cos((data["maxbatch_msignal"]))))) / (np.conjugate(np.cos((data["minbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["minbatch_msignal"]))) * (data["maxbatch_slices2"]))) / 2.0))) +

                            0.100000*np.tanh(np.real((((-(((((((((-((((((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) / 2.0)) / 2.0))))) * 2.0)) / 2.0)) - (((data["minbatch"]) + (complex(-3.0))))))))) - (((((((complex(0.0)) * 2.0)) / ((((((complex(-3.0)) / 2.0)) + (complex(0.0)))/2.0)))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(((data["maxbatch_slices2_msignal"]) - (np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(((np.sin((np.where(np.abs(np.where(np.abs(data["abs_minbatch_msignal"]) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs((-((data["medianbatch_msignal"])))),complex(1.), complex(0.) )))) / 2.0)),complex(1.), complex(0.) )))) > np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((data["mean_abs_chgbatch_slices2"]) - (((((np.sin((((np.sin((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))) * 2.0)))) * 2.0)) * (np.cos((data["minbatch_msignal"]))))))) + (((np.cos((data["medianbatch_msignal"]))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_slices2"]) / 2.0)) / (np.sin((((data["maxbatch_slices2_msignal"]) - (((complex(-2.0)) + (((((np.sin((np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.cos((((data["maxbatch_slices2_msignal"]) - (data["maxbatch_slices2_msignal"]))))),complex(1.), complex(0.) )))) + (data["mean_abs_chgbatch_msignal"]))) - (((data["maxbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["stdbatch_slices2_msignal"]) - (np.where(np.abs(data["signal_shift_-1"]) > np.abs(((np.conjugate(((data["abs_avgbatch_slices2_msignal"]) * (np.sin((np.conjugate(((((np.cos((np.cos((data["minbatch_slices2"]))))) * 2.0)) / 2.0)))))))) * (complex(-3.0)))),complex(1.), complex(0.) )))))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(data["minbatch"]),complex(1.), complex(0.) )) - (np.where(np.abs(((data["abs_maxbatch"]) * 2.0)) > np.abs((((((((np.tanh((np.where(np.abs(np.cos((data["signal_shift_+1"]))) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))) - (complex(2.0)))) + (data["abs_avgbatch_msignal"]))/2.0)) + (complex(2.0)))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) - (np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(np.cos((data["maxbatch_slices2"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.sin((((complex(0,1)*np.conjugate(((data["signal_shift_+1"]) + ((-((((((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.tanh((np.tanh((np.cos((np.cos((data["stdbatch_slices2"]))))))))))) * (np.sin((data["signal_shift_+1"]))))) + (np.conjugate(data["minbatch_msignal"])))))))))) - (data["abs_maxbatch_slices2_msignal"])))))) * (data["abs_maxbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch_msignal"]) + (np.where(np.abs(np.tanh((((data["minbatch_slices2"]) - (np.conjugate(data["maxbatch_slices2"])))))) <= np.abs(np.where(np.abs(np.tanh((data["medianbatch_msignal"]))) > np.abs(np.where(np.abs(((((data["maxbatch_slices2_msignal"]) / 2.0)) * 2.0)) > np.abs(np.cos((data["minbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(data["minbatch"])) +

                            0.100000*np.tanh(np.real(np.conjugate(complex(0,1)*np.conjugate(complex(0.0))))) +

                            0.100000*np.tanh(np.real(((np.sin((((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"]))) / 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((np.cos((data["abs_maxbatch_msignal"]))))) + (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.cos((data["minbatch_msignal"]))) * (((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(np.tanh((((data["abs_avgbatch_slices2_msignal"]) - (((data["minbatch_msignal"]) + (((data["rangebatch_slices2"]) + (((data["minbatch_slices2_msignal"]) * (np.sin((data["rangebatch_slices2"]))))))))))))),complex(1.), complex(0.) )) <= np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )) - (data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((complex(-1.0)) * (np.cos((np.tanh(((((((((np.tanh((np.tanh((np.tanh((np.cos((((((data["abs_minbatch_msignal"]) / 2.0)) * ((((data["meanbatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"]))/2.0)))))))))))) + (data["abs_minbatch_msignal"]))/2.0)) / 2.0)) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(((((complex(0,1)*np.conjugate(np.sin((data["meanbatch_slices2"])))) - (data["abs_maxbatch"]))) + (np.where(np.abs(np.cos((data["maxbatch_slices2"]))) > np.abs((((complex(8.0)) + (((data["abs_maxbatch_msignal"]) - (np.cos((np.cos((data["abs_maxbatch_msignal"]))))))))/2.0)),complex(1.), complex(0.) )))) > np.abs(np.where(np.abs(data["minbatch_msignal"]) <= np.abs(((data["minbatch_msignal"]) + (np.conjugate(data["abs_avgbatch_slices2_msignal"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((data["medianbatch_msignal"]))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((np.sin((((np.sin((((data["meanbatch_msignal"]) + (data["maxbatch_msignal"]))))) * 2.0)))) * 2.0)) - (np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(((complex(8.0)) * (data["mean_abs_chgbatch_slices2"])))) > np.abs(((data["meanbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))),complex(1.), complex(0.) ))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["medianbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.cos((data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["meanbatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(((((((np.sin((((data["meanbatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"]))))),complex(1.), complex(0.) )))))))) * 2.0)) * 2.0)) * 2.0)) > np.abs(((data["abs_maxbatch_slices2_msignal"]) + (np.sin((data["meanbatch_slices2_msignal"]))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["signal_shift_-1"]) + ((-((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((((((complex(0,1)*np.conjugate(data["abs_maxbatch"])) * (np.cos((np.tanh((np.cos((data["meanbatch_msignal"]))))))))) + (((((data["abs_minbatch_slices2"]) - (np.sin((data["abs_avgbatch_msignal"]))))) * (((((data["abs_minbatch_slices2"]) - (data["stdbatch_slices2_msignal"]))) * (data["abs_maxbatch_slices2"]))))))/2.0))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((np.conjugate((-((data["minbatch_msignal"]))))) - (np.where(np.abs(data["minbatch"]) <= np.abs(np.sin(((((complex(-3.0)) + (data["minbatch_msignal"]))/2.0)))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(complex(-1.0))) +

                            0.100000*np.tanh(np.real(((data["signal"]) - (np.where(np.abs(((((data["medianbatch_msignal"]) * 2.0)) * 2.0)) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.cos((((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(np.where(np.abs(np.sin((complex(0.0)))) <= np.abs(np.sin(((((((((((np.cos((data["abs_avgbatch_msignal"]))) + (complex(0,1)*np.conjugate(complex(3.0))))) * 2.0)) * 2.0)) + (complex(0.0)))/2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(5.07730865478515625))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) + (np.conjugate((-((np.where(np.abs(((data["minbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))) <= np.abs(((((data["maxbatch_slices2_msignal"]) - (complex(0.0)))) - (np.tanh((((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["rangebatch_slices2"]))) / 2.0)))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) * (np.cos((data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.conjugate(((((-((np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(((((np.sin((((np.sin((data["medianbatch_msignal"]))) * 2.0)))) + (complex(-3.0)))) - (complex(7.69875335693359375))))) > np.abs(((((data["abs_minbatch_slices2"]) * 2.0)) / 2.0)),complex(1.), complex(0.) )) <= np.abs(complex(0,1)*np.conjugate(((data["rangebatch_slices2"]) + (data["abs_maxbatch_slices2"])))),complex(1.), complex(0.) ))))) + (data["stdbatch_slices2"]))/2.0)))))) +

                            0.100000*np.tanh(np.real(data["medianbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((np.cos((data["minbatch_msignal"]))) * 2.0)) + (((((np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(((complex(-3.0)) - (data["stdbatch_slices2_msignal"]))),complex(1.), complex(0.) )) * 2.0)) * (((complex(-3.0)) + ((((data["maxbatch_slices2_msignal"]) + (np.tanh((data["minbatch"]))))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.cos((np.where(np.abs(((data["minbatch"]) * 2.0)) <= np.abs((-((complex(1.0))))),complex(1.), complex(0.) )))) - (((data["maxbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((complex(-1.0)))) - ((((data["minbatch"]) + (complex(0,1)*np.conjugate(np.where(np.abs(complex(-1.0)) > np.abs((((-((complex(0,1)*np.conjugate(data["maxtominbatch"]))))) / 2.0)),complex(1.), complex(0.) ))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.sin((((data["signal_shift_+1"]) * (np.where(np.abs(np.cos((np.where(np.abs(data["rangebatch_msignal"]) <= np.abs((-((((data["minbatch_msignal"]) * (((complex(-2.0)) / 2.0))))))),complex(1.), complex(0.) )))) > np.abs(np.sin((data["stdbatch_msignal"]))),complex(1.), complex(0.) )))))))) + ((((complex(-2.0)) + (np.sin((data["maxtominbatch_slices2_msignal"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.cos((np.conjugate(((np.where(np.abs(data["stdbatch_slices2_msignal"]) <= np.abs(((data["minbatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) )) - (data["abs_maxbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.conjugate(data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.cos((data["maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["minbatch_msignal"]) + ((-((np.where(np.abs(np.cos((np.sin(((((data["minbatch_msignal"]) + (complex(-2.0)))/2.0)))))) <= np.abs((-(((((np.cos((np.sin(((((data["minbatch_msignal"]) + (np.where(np.abs(data["minbatch_msignal"]) <= np.abs(data["minbatch_slices2"]),complex(1.), complex(0.) )))/2.0)))))) + ((-((data["minbatch_msignal"])))))/2.0))))),complex(1.), complex(0.) ))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(((np.tanh((complex(-1.0)))) + ((-((np.conjugate(np.sin((data["stdbatch_slices2_msignal"]))))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(data["maxbatch_msignal"])) * 2.0)) * (np.where(np.abs(data["maxbatch_msignal"]) > np.abs(((((np.cos((np.tanh((data["meanbatch_slices2"]))))) * 2.0)) * (np.where(np.abs(data["maxbatch_msignal"]) <= np.abs((((data["signal_shift_+1"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((((complex(-3.0)) + (complex(-3.0)))/2.0)) - (data["minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real((((((((data["mean_abs_chgbatch_msignal"]) + (((data["signal"]) + (np.cos((data["signal"]))))))/2.0)) + (((np.cos((data["signal"]))) * (((((data["signal_shift_-1"]) + (data["minbatch_slices2"]))) * 2.0)))))) + (((np.conjugate(np.cos((data["signal_shift_+1"])))) * (np.cos((data["signal_shift_-1"])))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) - (np.where(np.abs(np.conjugate(((data["rangebatch_msignal"]) / 2.0))) > np.abs((((data["maxbatch_slices2"]) + (((np.tanh((np.cos((np.cos((data["meanbatch_slices2"]))))))) * (data["abs_minbatch_slices2"]))))/2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((data["maxtominbatch_msignal"]) + (((data["abs_maxbatch_msignal"]) * ((((data["signal"]) + (np.where(np.abs(data["stdbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(data["signal"]) <= np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(np.tanh((((complex(0,1)*np.conjugate(np.where(np.abs(np.sin((np.conjugate((-((np.conjugate(((data["abs_minbatch_slices2"]) - (((np.conjugate(((np.sin((np.tanh((data["minbatch_msignal"]))))) - (data["maxbatch_slices2_msignal"])))) - (data["abs_minbatch_slices2"])))))))))))) <= np.abs(((data["minbatch_slices2"]) + (complex(-1.0)))),complex(1.), complex(0.) ))) - (data["abs_avgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.cos((data["minbatch_msignal"]))))) * (np.tanh((data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.where(np.abs(((np.where(np.abs(((np.tanh((complex(0,1)*np.conjugate(data["rangebatch_slices2"])))) - (data["meanbatch_msignal"]))) <= np.abs(complex(0.0)),complex(1.), complex(0.) )) * 2.0)) <= np.abs(np.sin((np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) * (((data["abs_maxbatch"]) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.sin(((((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) + (complex(3.0)))/2.0)))) <= np.abs(np.sin((complex(3.0)))),complex(1.), complex(0.) ))) > np.abs(((complex(-3.0)) + (np.tanh((data["abs_maxbatch"]))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["stdbatch_slices2_msignal"]) / 2.0)))) - (complex(0,1)*np.conjugate(data["stdbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs((((data["signal"]) + (np.sin((complex(2.0)))))/2.0)) > np.abs((((((((complex(-3.0)) + ((-((((((np.where(np.abs(data["minbatch_slices2"]) > np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )) - (np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(((data["abs_avgbatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) )))) * 2.0))))))/2.0)) / 2.0)) / 2.0)),complex(1.), complex(0.) )) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((np.tanh((np.tanh((np.sin((np.conjugate(data["stdbatch_slices2"]))))))))))) +

                            0.100000*np.tanh(np.real(complex(-1.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.where(np.abs(complex(1.0)) <= np.abs(np.where(np.abs(np.tanh((((complex(1.0)) + (np.conjugate(data["minbatch_slices2_msignal"])))))) <= np.abs(complex(0,1)*np.conjugate(((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) / 2.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(np.cos((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )) * (np.cos((complex(-1.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(((((complex(2.0)) * 2.0)) - ((-((((data["medianbatch_slices2"]) + (((data["mean_abs_chgbatch_slices2"]) / 2.0))))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((((complex(2.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real(np.cos((((np.sin((np.sin((data["signal_shift_-1_msignal"]))))) - (np.cos((((complex(-2.0)) - (np.sin((np.cos((((data["maxbatch_slices2_msignal"]) - (data["abs_minbatch_slices2"])))))))))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.cos((np.where(np.abs(complex(0,1)*np.conjugate(complex(-1.0))) > np.abs(complex(0,1)*np.conjugate(((data["abs_minbatch_msignal"]) + (((complex(0,1)*np.conjugate(np.cos((data["meanbatch_msignal"])))) * (data["meanbatch_msignal"])))))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(((complex(-3.0)) - ((((data["minbatch_msignal"]) + (np.where(np.abs(complex(0,1)*np.conjugate((((complex(-3.0)) + (complex(0.0)))/2.0))) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(((data["signal"]) / 2.0))) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((data["mean_abs_chgbatch_msignal"]))) - (np.cos((data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((((((data["abs_avgbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0)) + (np.tanh((data["meanbatch_msignal"]))))/2.0)) * (data["abs_maxbatch_msignal"]))) * (((((data["meanbatch_slices2_msignal"]) * (((complex(3.0)) * 2.0)))) * (((data["signal_shift_+1_msignal"]) * (np.sin((data["maxbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((((np.tanh((np.conjugate((-((np.sin(((((-((np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) * 2.0)))))))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.tanh((((data["abs_avgbatch_slices2_msignal"]) * (np.tanh((data["abs_maxbatch"])))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["abs_avgbatch_slices2"]) + (((np.sin((data["mean_abs_chgbatch_msignal"]))) - ((-((((data["rangebatch_msignal"]) - (np.where(np.abs(complex(-2.0)) <= np.abs(np.where(np.abs((-((((data["mean_abs_chgbatch_msignal"]) - (complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"]))))))) <= np.abs(np.sin((((data["rangebatch_msignal"]) + (((data["signal_shift_+1"]) - (data["medianbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((-((complex(0,1)*np.conjugate(np.where(np.abs(data["signal"]) <= np.abs((((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((-((complex(-3.0))))))) + (data["signal"]))/2.0)),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(np.conjugate(complex(1.0))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((((data["maxbatch_slices2_msignal"]) + (((complex(0,1)*np.conjugate((((((((((complex(0,1)*np.conjugate(np.conjugate(data["abs_maxbatch_msignal"]))) / 2.0)) / 2.0)) + (np.conjugate(data["abs_maxbatch_msignal"])))/2.0)) / 2.0))) * (np.cos(((((data["abs_maxbatch_slices2_msignal"]) + (data["signal_shift_-1"]))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((complex(-3.0)) + (data["mean_abs_chgbatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin(((((complex(0,1)*np.conjugate((((data["maxbatch_slices2_msignal"]) + (((complex(0,1)*np.conjugate(complex(0.0))) + ((((np.sin(((((complex(-2.0)) + (data["mean_abs_chgbatch_msignal"]))/2.0)))) + (((data["minbatch_slices2_msignal"]) + (data["abs_minbatch_msignal"]))))/2.0)))))/2.0))) + (((data["minbatch_slices2_msignal"]) + ((((data["minbatch_slices2"]) + (((((complex(-2.0)) * 2.0)) / 2.0)))/2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((data["minbatch_msignal"]))) + (data["minbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real((((((data["medianbatch_msignal"]) + (np.conjugate(((data["meanbatch_msignal"]) / 2.0))))/2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real((((data["abs_avgbatch_msignal"]) + (complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])))/2.0))) +

                            0.100000*np.tanh(np.real((-((complex(0,1)*np.conjugate(np.where(np.abs(np.tanh((np.cos((data["rangebatch_slices2"]))))) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.tanh((np.cos((complex(5.0)))))) > np.abs(complex(-2.0)),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((-(((((((np.where(np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["signal_shift_+1"]))) > np.abs(np.conjugate(((data["maxbatch_slices2"]) - (np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(np.cos((np.tanh((((data["abs_maxbatch_msignal"]) - (data["maxbatch_msignal"]))))))),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )) + (complex(0,1)*np.conjugate(data["signal_shift_+1"])))/2.0)) - (np.sin((data["maxbatch_slices2"])))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((np.sin((data["meanbatch_slices2"]))) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0.0)) > np.abs(np.where(np.abs(complex(-3.0)) > np.abs((((np.conjugate(complex(2.0))) + (np.where(np.abs(np.conjugate(data["maxtominbatch_slices2_msignal"])) > np.abs(complex(0,1)*np.conjugate(data["meanbatch_slices2"])),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((data["rangebatch_msignal"])))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["meanbatch_msignal"]) - (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(np.sin((data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((data["mean_abs_chgbatch_slices2"]))) / (np.tanh((((data["mean_abs_chgbatch_slices2"]) - (np.where(np.abs(data["abs_maxbatch"]) > np.abs(np.where(np.abs(complex(1.39533555507659912)) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((((complex(-1.0)) - (data["abs_maxbatch"]))) + ((-((((((data["maxbatch_slices2"]) / 2.0)) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((-((data["abs_avgbatch_slices2_msignal"])))) + (np.conjugate(np.conjugate((((-((data["abs_minbatch_msignal"])))) + (np.where(np.abs((-((data["abs_minbatch_msignal"])))) > np.abs(np.sin((data["abs_maxbatch_msignal"]))),complex(1.), complex(0.) )))))))/2.0)))) +

                            0.100000*np.tanh(np.real((((-(((-((complex(0,1)*np.conjugate((-((((complex(1.0)) - (np.where(np.abs(complex(0,1)*np.conjugate((-((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) )))))) <= np.abs(complex(1.0)),complex(1.), complex(0.) )))))))))))))) - (np.sin((complex(1.61947524547576904))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) * ((((-((data["medianbatch_slices2_msignal"])))) * (((((np.where(np.abs(((((data["medianbatch_slices2_msignal"]) * 2.0)) * (data["maxbatch_slices2"]))) > np.abs((-((np.cos((data["abs_maxbatch_msignal"])))))),complex(1.), complex(0.) )) + (data["minbatch"]))) - (((data["signal_shift_-1"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(complex(3.0))) / 2.0)) - (data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_slices2"]) / 2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(data["medianbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((-((complex(0,1)*np.conjugate(data["stdbatch_msignal"]))))) * (data["meanbatch_slices2"]))) > np.abs((((-((data["minbatch"])))) * 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(((data["maxtominbatch"]) / (data["abs_maxbatch"]))) > np.abs(np.where(np.abs(np.where(np.abs(np.cos((((data["signal_shift_+1_msignal"]) - (np.conjugate(data["abs_avgbatch_msignal"])))))) > np.abs(((np.conjugate(((np.conjugate(data["maxbatch_slices2_msignal"])) * 2.0))) / 2.0)),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(data["abs_avgbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(np.conjugate(np.where(np.abs(np.tanh((np.where(np.abs((((np.tanh((((np.sin((((np.tanh((data["abs_minbatch_slices2"]))) / 2.0)))) - (data["maxbatch_slices2_msignal"]))))) + (data["abs_minbatch_slices2_msignal"]))/2.0)) <= np.abs(np.tanh((data["abs_minbatch_slices2"]))),complex(1.), complex(0.) )))) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )))) / 2.0)) * (complex(1.0))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(data["abs_maxbatch"])) * (((np.cos((data["maxbatch_slices2_msignal"]))) * (data["signal_shift_-1_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin((data["maxtominbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.conjugate(np.sin(((((data["minbatch_msignal"]) + (np.conjugate(np.sin((np.conjugate(np.sin((np.sin((data["minbatch_msignal"]))))))))))/2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_slices2"]) * (np.conjugate(((np.sin((np.conjugate(np.conjugate(((((data["minbatch_msignal"]) - (np.where(np.abs(np.cos((((data["rangebatch_slices2"]) + (data["signal_shift_-1"]))))) <= np.abs((((np.cos((complex(0,1)*np.conjugate(np.tanh((data["medianbatch_slices2_msignal"])))))) + (np.sin((data["meanbatch_msignal"]))))/2.0)),complex(1.), complex(0.) )))) / 2.0)))))) / 2.0)))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(complex(2.0)) > np.abs(np.where(np.abs(((data["abs_maxbatch_msignal"]) + (data["meanbatch_msignal"]))) <= np.abs((((np.where(np.abs(np.sin((data["abs_maxbatch_msignal"]))) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (complex(2.0)))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((data["maxbatch_slices2"]) + (((np.cos(((((((data["maxbatch_slices2"]) + (((data["rangebatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))))/2.0)) + (data["medianbatch_slices2"]))))) + (data["maxtominbatch_slices2_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(complex(0.0))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["medianbatch_slices2_msignal"]))) - (((data["minbatch_slices2_msignal"]) + (((((-((np.sin((((data["medianbatch_slices2_msignal"]) * 2.0))))))) + (data["signal_shift_-1"]))/2.0)))))) * (data["medianbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.conjugate(((data["signal_shift_+1_msignal"]) / 2.0))))) +

                            0.100000*np.tanh(np.real((((data["medianbatch_slices2_msignal"]) + (complex(2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["mean_abs_chgbatch_slices2"]) + (((((data["maxtominbatch_msignal"]) - (np.where(np.abs(np.where(np.abs(data["signal"]) > np.abs(np.cos((np.cos((np.where(np.abs(np.cos((((data["mean_abs_chgbatch_slices2"]) + (((data["maxtominbatch_msignal"]) - (data["mean_abs_chgbatch_slices2"]))))))) <= np.abs((-((data["signal"])))),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )) > np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) )))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxtominbatch_slices2_msignal"]) + (((np.sin((data["rangebatch_slices2"]))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin((np.sin((complex(0,1)*np.conjugate(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(((complex(1.0)) - (((complex(-3.0)) / 2.0)))) > np.abs(complex(-2.0)),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(np.where(np.abs(np.cos((((complex(-1.0)) * (np.cos((data["stdbatch_slices2_msignal"]))))))) > np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.cos((((np.tanh((np.where(np.abs(((complex(6.44996309280395508)) - (data["maxbatch_slices2_msignal"]))) <= np.abs(((((((data["maxtominbatch"]) - (data["maxbatch_slices2_msignal"]))) - (((data["maxbatch_slices2_msignal"]) / (data["abs_avgbatch_slices2"]))))) - (data["maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )))) - (data["maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["medianbatch_msignal"]))) * (((data["signal_shift_-1"]) - (np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.sin((np.where(np.abs(((np.sin((data["rangebatch_slices2"]))) / 2.0)) > np.abs(((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])) / 2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))) * (np.where(np.abs(np.where(np.abs(np.tanh((data["abs_minbatch_msignal"]))) > np.abs((((-((((data["signal_shift_+1_msignal"]) - (complex(3.0))))))) + (((np.sin((np.cos((np.where(np.abs(complex(2.0)) > np.abs(np.cos((data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))) + (np.tanh((data["abs_minbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )) > np.abs(((data["signal_shift_+1_msignal"]) * 2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["rangebatch_slices2"]) > np.abs(((data["signal_shift_+1"]) - (data["abs_avgbatch_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate(np.sin((np.conjugate(data["abs_avgbatch_slices2_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((np.where(np.abs(np.conjugate(data["meanbatch_slices2_msignal"])) <= np.abs(complex(1.0)),complex(1.), complex(0.) )))) > np.abs(np.sin((complex(1.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.conjugate(data["maxtominbatch"])))) <= np.abs(np.sin((data["minbatch_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(((complex(2.23900842666625977)) + (complex(0,1)*np.conjugate(data["signal"])))) <= np.abs(np.tanh((data["signal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["stdbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))) + (np.cos((data["maxbatch_slices2"]))))/2.0)))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((data["meanbatch_slices2_msignal"]))) <= np.abs(np.sin((np.sin((np.where(np.abs((((((((complex(0.0)) + (complex(-1.0)))) / 2.0)) + (complex(0,1)*np.conjugate(np.tanh(((-((data["abs_minbatch_msignal"]))))))))/2.0)) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["stdbatch_msignal"]))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(data["stdbatch_slices2"])) > np.abs(np.conjugate(np.cos((data["medianbatch_msignal"])))),complex(1.), complex(0.) ))) * (np.tanh((np.cos((data["maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((((((data["medianbatch_msignal"]) - (np.sin((((data["signal_shift_-1_msignal"]) - (np.cos((((data["stdbatch_slices2"]) / 2.0)))))))))) - (np.sin((((data["medianbatch_msignal"]) * 2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["abs_maxbatch_slices2_msignal"]) / 2.0)))) +

                            0.100000*np.tanh(np.real(((np.cos((np.where(np.abs(np.cos((data["maxbatch_slices2_msignal"]))) > np.abs(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(complex(0.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) - (complex(0,1)*np.conjugate(np.conjugate(data["abs_avgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(np.cos((np.tanh((data["medianbatch_msignal"])))))) <= np.abs(((data["stdbatch_slices2_msignal"]) * (np.tanh((data["abs_minbatch_slices2"]))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((((np.tanh((np.where(np.abs(data["signal_shift_+1"]) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(np.conjugate(data["medianbatch_slices2_msignal"]))) <= np.abs(np.where(np.abs(((data["maxtominbatch"]) + (np.where(np.abs(complex(2.0)) > np.abs(np.cos((((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )))) <= np.abs(np.where(np.abs(((data["medianbatch_msignal"]) / 2.0)) <= np.abs(((np.cos((np.conjugate(data["signal_shift_+1_msignal"])))) + (data["maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((data["abs_minbatch_slices2"]) + ((((np.where(np.abs((((data["abs_minbatch_slices2"]) + (data["minbatch"]))/2.0)) <= np.abs(((np.where(np.abs(((complex(-3.0)) * (data["signal_shift_+1_msignal"]))) > np.abs(np.where(np.abs(((np.cos(((-((data["signal_shift_-1_msignal"])))))) * 2.0)) > np.abs(data["minbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) )) + (data["minbatch_slices2"]))/2.0)))/2.0)) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(complex(-1.0))) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(((np.tanh((complex(0,1)*np.conjugate(data["medianbatch_slices2"])))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(complex(0,1)*np.conjugate(complex(1.0))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((((np.tanh((data["abs_minbatch_slices2"]))) + (np.where(np.abs(complex(-1.0)) > np.abs(np.conjugate(((data["signal_shift_-1_msignal"]) + (complex(9.0))))),complex(1.), complex(0.) )))))) > np.abs(((complex(-1.0)) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.sin((((((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(np.where(np.abs(np.conjugate(data["maxbatch_slices2_msignal"])) <= np.abs(data["signal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / (data["maxbatch_slices2"]))) / 2.0)))) * (((complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.tanh((data["meanbatch_slices2_msignal"]))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))) + (data["mean_abs_chgbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["maxbatch_slices2"]) + (((np.sin((np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(complex(0,1)*np.conjugate(np.cos((data["meanbatch_slices2"])))),complex(1.), complex(0.) )))) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((np.cos((((np.where(np.abs(((np.where(np.abs(complex(0,1)*np.conjugate(data["maxbatch_msignal"])) > np.abs(np.where(np.abs(((np.conjugate(complex(3.0))) * 2.0)) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (data["abs_minbatch_msignal"]))) <= np.abs(np.where(np.abs(complex(3.0)) > np.abs(np.sin(((((np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.cos((data["maxbatch_msignal"]))),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_slices2"]))/2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.tanh((((data["medianbatch_msignal"]) / 2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(((((((((np.tanh((np.cos((data["abs_maxbatch_msignal"]))))) / 2.0)) * (data["minbatch_msignal"]))) / 2.0)) * (complex(0,1)*np.conjugate(data["maxbatch_msignal"])))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1_msignal"]) * (np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) <= np.abs(np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) <= np.abs(np.cos((np.cos((data["signal_shift_+1_msignal"]))))),complex(1.), complex(0.) ))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((data["meanbatch_msignal"]) + (np.conjugate(((complex(0.0)) + (((data["meanbatch_msignal"]) * (np.where(np.abs(((np.sin((np.conjugate(((data["meanbatch_msignal"]) / 2.0))))) * (data["meanbatch_msignal"]))) > np.abs(((data["signal_shift_+1"]) * (((data["signal_shift_-1"]) * (complex(2.64151883125305176)))))),complex(1.), complex(0.) ))))))))/2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_msignal"]))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_maxbatch"]) <= np.abs(complex(0,1)*np.conjugate(complex(0.0))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(2.0)) - (data["medianbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs((-((np.sin(((-(((-(((((((((complex(-2.0)) + ((-(((-((np.conjugate(complex(-1.0))))))))))/2.0)) / 2.0)) / 2.0))))))))))))),complex(1.), complex(0.) )) > np.abs((((data["rangebatch_msignal"]) + ((((((complex(0,1)*np.conjugate(data["signal"])) * 2.0)) + (data["signal_shift_-1_msignal"]))/2.0)))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((-((np.conjugate(data["signal_shift_-1_msignal"]))))) / (np.conjugate(np.cos(((((((data["abs_maxbatch_msignal"]) + (np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs((-(((-((np.conjugate((((((data["abs_maxbatch_msignal"]) + (np.conjugate(data["abs_maxbatch_msignal"])))/2.0)) / 2.0))))))))),complex(1.), complex(0.) )))/2.0)) / 2.0)))))))))     

    

    def GP_class_7(self,data):

        return self.Output( -2.939938 +

                            0.100000*np.tanh(np.real((((data["meanbatch_slices2"]) + (((np.sin((data["abs_maxbatch_slices2"]))) - (data["abs_avgbatch_slices2_msignal"]))))/2.0))) +

                            0.100000*np.tanh(np.real(((complex(-3.0)) * (((((np.cos((data["stdbatch_slices2"]))) + (np.sin((np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(((complex(-3.0)) * (np.sin((data["maxbatch_slices2_msignal"]))))),complex(1.), complex(0.) )))))) - (np.where(np.abs(complex(-3.0)) <= np.abs(np.cos((((data["abs_minbatch_slices2"]) + (data["abs_minbatch_slices2"]))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["signal"]) + ((-((np.where(np.abs(np.tanh((complex(0,1)*np.conjugate((((data["abs_minbatch_msignal"]) + (((((data["abs_minbatch_slices2"]) / 2.0)) + (data["medianbatch_slices2"]))))/2.0))))) > np.abs(data["signal"]),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(((data["mean_abs_chgbatch_msignal"]) + (data["meanbatch_slices2"]))),complex(1.), complex(0.) )) + (((((data["mean_abs_chgbatch_slices2"]) - (np.where(np.abs(((data["mean_abs_chgbatch_msignal"]) + ((-((((data["medianbatch_slices2_msignal"]) + (data["signal"])))))))) > np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )))) + (((((data["meanbatch_msignal"]) + (data["medianbatch_slices2"]))) + (complex(-3.0))))))))) +

                            0.100000*np.tanh(np.real(((data["signal"]) - (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1"]) - (np.cos((data["signal_shift_+1"])))))) +

                            0.100000*np.tanh(np.real(((complex(-3.0)) + ((((complex(-3.0)) + (((data["medianbatch_slices2"]) * (((data["rangebatch_msignal"]) + (data["signal"]))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.conjugate(data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.cos(((((np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(complex(-1.0)),complex(1.), complex(0.) )) + (((((data["signal_shift_-1_msignal"]) + (((data["mean_abs_chgbatch_slices2_msignal"]) * ((((-((complex(-3.0))))) / 2.0)))))) * 2.0)))/2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) - (((np.where(np.abs(np.where(np.abs(complex(0.0)) <= np.abs(np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(((complex(3.0)) - (complex(0,1)*np.conjugate(data["signal_shift_-1"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) * 2.0))))) +

                            0.100000*np.tanh(np.real((((((((complex(0.0)) + (data["medianbatch_slices2_msignal"]))) + (data["abs_maxbatch_msignal"]))) + (complex(-2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) + ((-((complex(0,1)*np.conjugate(np.sin((data["signal_shift_-1_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((((((complex(1.0)) + (complex(3.0)))) * (np.sin((data["maxbatch_slices2_msignal"]))))) * ((-((((((complex(1.0)) + (np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(((complex(1.0)) + (((data["maxbatch_slices2_msignal"]) / 2.0)))),complex(1.), complex(0.) )))) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(-1.0)) - ((-((((data["meanbatch_msignal"]) * 2.0)))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["maxbatch_msignal"]) + (np.where(np.abs(np.sin((data["rangebatch_slices2_msignal"]))) <= np.abs(np.sin((data["maxbatch_msignal"]))),complex(1.), complex(0.) )))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) * (((((data["signal"]) - ((((data["signal"]) + (((data["medianbatch_slices2_msignal"]) * (data["signal"]))))/2.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["rangebatch_slices2"]) + (np.sin(((-((np.sin((np.tanh(((-((data["abs_minbatch_msignal"]))))))))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["signal"]) - (np.where(np.abs(np.conjugate(np.where(np.abs(np.conjugate(data["mean_abs_chgbatch_slices2"])) > np.abs(np.where(np.abs(data["meanbatch_msignal"]) > np.abs(np.where(np.abs(np.where(np.abs(complex(2.0)) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs((-((data["rangebatch_msignal"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) <= np.abs(data["signal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((((complex(-3.0)) + (((np.cos((complex(4.0)))) + (data["minbatch_slices2"]))))/2.0)) + (((np.conjugate(data["abs_maxbatch"])) - (complex(4.0))))))) +

                            0.100000*np.tanh(np.real((-(((((np.cos((np.where(np.abs(complex(3.0)) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs(((((np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(data["stdbatch_msignal"]),complex(1.), complex(0.) )) * (data["abs_avgbatch_slices2"]))) + (complex(0.0)))) > np.abs(data["signal"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) )))) + (complex(0.0)))/2.0)))))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((complex(0,1)*np.conjugate((((((data["maxbatch_slices2"]) + (np.cos((((np.sin((data["maxtominbatch_slices2"]))) + (((complex(-3.0)) + ((((complex(-1.0)) + (np.sin(((-((complex(1.0))))))))/2.0)))))))))/2.0)) / 2.0)))))) - (data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["maxbatch_msignal"]) + (complex(0.0)))))) + (((np.cos((data["maxbatch_msignal"]))) - (np.cos((((data["maxbatch_msignal"]) - (np.cos((np.cos((data["meanbatch_slices2"])))))))))))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((((((((data["meanbatch_slices2"]) * 2.0)) * (((data["mean_abs_chgbatch_slices2"]) + ((-((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(np.sin((np.where(np.abs(((complex(3.0)) + (data["abs_avgbatch_slices2"]))) <= np.abs(((data["mean_abs_chgbatch_slices2"]) * 2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))))) * 2.0)) * (((data["stdbatch_slices2"]) * 2.0))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) - (((np.sin((data["abs_maxbatch_slices2_msignal"]))) - (((data["medianbatch_slices2_msignal"]) + ((((((np.where(np.abs(np.where(np.abs(np.sin((np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (np.sin((data["meanbatch_msignal"]))))/2.0)) - (np.sin((data["abs_maxbatch_slices2_msignal"])))))))))))) +

                            0.100000*np.tanh(np.real(((((complex(1.0)) - ((((data["abs_minbatch_slices2"]) + (((data["abs_minbatch_msignal"]) / (((complex(-1.0)) + (data["stdbatch_slices2"]))))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["signal_shift_+1"]) * 2.0)) - (data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.where(np.abs(complex(9.99783229827880859)) > np.abs((((complex(9.99783611297607422)) + (np.where(np.abs(np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(np.cos((data["minbatch_slices2_msignal"]))),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(np.cos((data["minbatch_slices2_msignal"]))) <= np.abs(complex(9.99783611297607422)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(complex(9.99783229827880859)),complex(1.), complex(0.) )))/2.0)),complex(1.), complex(0.) )) <= np.abs(complex(9.99783229827880859)),complex(1.), complex(0.) )) / (np.cos((data["minbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((-((((((complex(3.0)) * 2.0)) * 2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((complex(3.0)) + (data["stdbatch_slices2"]))) * (data["meanbatch_msignal"]))) <= np.abs(complex(1.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_msignal"]) - (((((((data["signal_shift_-1"]) * 2.0)) * (((((data["abs_maxbatch_slices2"]) + (((((((complex(1.0)) * (complex(1.0)))) * 2.0)) - (np.conjugate(complex(1.0))))))) * (np.where(np.abs(data["medianbatch_msignal"]) > np.abs(complex(1.0)),complex(1.), complex(0.) )))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((np.cos((((complex(0,1)*np.conjugate(np.conjugate(data["maxbatch_slices2_msignal"]))) - (((data["mean_abs_chgbatch_slices2"]) * 2.0)))))))))) +

                            0.100000*np.tanh(np.real(((np.tanh(((-((np.conjugate(((data["medianbatch_msignal"]) * 2.0)))))))) * (np.conjugate(data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["maxbatch_msignal"]) + (np.tanh((np.cos((((data["abs_maxbatch_msignal"]) + (np.cos((((np.cos((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2"]) <= np.abs(np.where(np.abs(data["rangebatch_msignal"]) > np.abs(np.tanh((np.cos((np.tanh((((data["maxbatch_msignal"]) / 2.0)))))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) * 2.0))))))))))))))) +

                            0.100000*np.tanh(np.real((((((((((data["medianbatch_slices2_msignal"]) - (np.cos((np.sin((data["meanbatch_slices2_msignal"]))))))) / 2.0)) + (np.cos((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))))/2.0)) - (np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["rangebatch_msignal"]) - (complex(8.90248966217041016)))) - (((data["rangebatch_slices2_msignal"]) - (((((((complex(3.0)) / 2.0)) * ((((-((data["meanbatch_slices2_msignal"])))) + (data["minbatch_msignal"]))))) * (np.sin((data["minbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["stdbatch_slices2"]))) + (np.cos((data["meanbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2"]) * (((np.cos((data["meanbatch_slices2_msignal"]))) - (np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.conjugate(np.cos((data["meanbatch_msignal"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((-((np.conjugate(data["abs_avgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.conjugate(((data["maxbatch_msignal"]) * (np.conjugate(((data["maxbatch_msignal"]) + ((((data["maxbatch_msignal"]) + (((data["abs_maxbatch_slices2"]) * (data["mean_abs_chgbatch_slices2"]))))/2.0))))))))) * (np.conjugate(((complex(-2.0)) + ((((data["maxbatch_msignal"]) + (np.sin((data["meanbatch_slices2"]))))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(data["rangebatch_msignal"])) * (data["signal"])))) +

                            0.100000*np.tanh(np.real(((np.cos((((complex(0,1)*np.conjugate(data["maxbatch_msignal"])) + (data["maxbatch_msignal"]))))) - (np.tanh((complex(0,1)*np.conjugate(complex(-3.0)))))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((((((((complex(0,1)*np.conjugate((-((np.tanh((np.where(np.abs(data["signal_shift_-1"]) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))))))) / 2.0)) * (data["minbatch_msignal"]))) + (data["minbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2"]) * (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((np.sin((((np.sin((data["minbatch_slices2_msignal"]))) - (data["maxbatch_slices2_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["mean_abs_chgbatch_slices2"]) / 2.0)))) + (complex(-3.0)))) - (np.sin((data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate((((data["minbatch_msignal"]) + (np.where(np.abs(np.conjugate(((data["meanbatch_msignal"]) * (np.sin((data["minbatch_msignal"])))))) > np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.where(np.abs(data["minbatch_slices2"]) > np.abs(np.where(np.abs(np.sin((((((data["medianbatch_msignal"]) * (data["medianbatch_slices2_msignal"]))) * 2.0)))) > np.abs(np.tanh((complex(-2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"]))))) + (complex(0,1)*np.conjugate(data["signal"]))))) +

                            0.100000*np.tanh(np.real(((np.cos((((np.tanh((((((data["maxbatch_slices2"]) + (complex(0,1)*np.conjugate(np.where(np.abs(np.conjugate(np.where(np.abs(np.where(np.abs(((data["minbatch_slices2"]) * 2.0)) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(((complex(-1.0)) - (data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) ))) <= np.abs(np.sin((data["abs_minbatch_slices2"]))),complex(1.), complex(0.) ))))) * 2.0)))) + (data["minbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((data["meanbatch_msignal"]) * 2.0)) + (np.cos((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["abs_maxbatch"]))))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["minbatch_msignal"]))) * ((-((((((((np.sin((np.where(np.abs(data["minbatch_msignal"]) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))) * 2.0)) * 2.0)) + (data["medianbatch_slices2"]))))))))) +

                            0.100000*np.tanh(np.real(((complex(-1.0)) - (np.where(np.abs(data["medianbatch_slices2"]) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.sin(((((np.sin((data["maxbatch_msignal"]))) + (data["minbatch_msignal"]))/2.0)))) * (((np.cos((np.tanh((data["meanbatch_msignal"]))))) + (((data["maxbatch_slices2_msignal"]) - (((data["abs_maxbatch_msignal"]) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["minbatch_msignal"]) + (np.where(np.abs(data["stdbatch_msignal"]) <= np.abs((((((data["minbatch_msignal"]) * 2.0)) + (np.tanh((((((data["minbatch_msignal"]) + (((np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )) + (np.cos((data["abs_avgbatch_slices2"]))))))) / 2.0)))))/2.0)),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["minbatch_slices2"]) > np.abs(complex(11.51182270050048828)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((((data["minbatch_slices2_msignal"]) * (np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(np.cos((((((np.tanh((np.cos((data["minbatch_msignal"]))))) * 2.0)) + (np.sin((np.sin((complex(0.0)))))))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real((((((complex(4.0)) + (np.where(np.abs(((np.cos((((data["meanbatch_msignal"]) * 2.0)))) - (np.sin((complex(-2.0)))))) > np.abs(complex(0,1)*np.conjugate(data["maxbatch_msignal"])),complex(1.), complex(0.) )))/2.0)) / (np.cos((((data["meanbatch_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["medianbatch_msignal"]) - (((np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(((complex(3.0)) * 2.0)),complex(1.), complex(0.) )) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((data["stdbatch_slices2_msignal"])))) + (np.where(np.abs(((data["rangebatch_msignal"]) - (complex(2.0)))) > np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((((-((np.sin((data["abs_maxbatch_slices2_msignal"])))))) * 2.0)) + ((-((np.sin((np.conjugate(np.cos(((-((data["meanbatch_msignal"]))))))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(0,1)*np.conjugate(data["meanbatch_msignal"])) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(((complex(1.0)) * (np.cos(((-((np.cos((np.cos((np.cos((((((np.sin((complex(4.82916831970214844)))) + (np.cos((data["meanbatch_msignal"]))))) * 2.0))))))))))))))),complex(1.), complex(0.) )) * 2.0)))) * 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["minbatch_msignal"]) + (np.tanh((np.where(np.abs(np.sin((np.sin((data["abs_maxbatch_slices2"]))))) <= np.abs(np.sin((np.where(np.abs(np.tanh((np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_minbatch_slices2_msignal"]) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(1.0)) <= np.abs(complex(-2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(np.sin((((data["rangebatch_msignal"]) + (data["rangebatch_msignal"])))))) <= np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["maxtominbatch_slices2"]) <= np.abs(np.cos((complex(1.0)))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((data["stdbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(np.where(np.abs(np.where(np.abs(np.sin((((np.where(np.abs((-((data["meanbatch_msignal"])))) > np.abs(complex(-1.0)),complex(1.), complex(0.) )) / 2.0)))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs((-((np.where(np.abs(np.cos(((-((np.where(np.abs(complex(2.0)) > np.abs(np.tanh((np.tanh((complex(2.0)))))),complex(1.), complex(0.) ))))))) <= np.abs(((complex(2.0)) * 2.0)),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )) <= np.abs(((complex(0,1)*np.conjugate(((data["mean_abs_chgbatch_slices2"]) / 2.0))) / 2.0)),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((-((((np.where(np.abs(np.sin((data["maxbatch_slices2"]))) > np.abs(np.tanh((np.where(np.abs(((data["abs_maxbatch"]) + (data["stdbatch_slices2_msignal"]))) <= np.abs((((((data["stdbatch_slices2_msignal"]) + (np.sin((data["abs_maxbatch_msignal"]))))/2.0)) + (data["medianbatch_slices2_msignal"]))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) - (data["abs_maxbatch_slices2_msignal"])))))) / (np.sin((((data["abs_maxbatch"]) + (data["stdbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.sin(((-(((-((data["minbatch_msignal"]))))))))) > np.abs(np.where(np.abs(((data["maxtominbatch"]) - (((data["maxbatch_slices2"]) * (((np.cos((np.conjugate(complex(2.0))))) / 2.0)))))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (((data["maxbatch_slices2"]) * (((np.sin(((((-((data["minbatch_msignal"])))) / 2.0)))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch"]) - (np.where(np.abs(complex(-1.0)) > np.abs(((((data["signal"]) + ((-((((data["maxtominbatch"]) - ((((data["maxtominbatch"]) + (complex(-1.0)))/2.0))))))))) - (((np.where(np.abs(data["signal"]) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )) - (((np.sin((data["mean_abs_chgbatch_msignal"]))) / 2.0)))))),complex(1.), complex(0.) )))) - (data["abs_minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) - (np.sin((data["maxbatch_slices2_msignal"]))))) - (np.sin((((np.sin((np.sin((data["signal_shift_-1"]))))) + (data["maxbatch_msignal"]))))))) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) - ((((np.sin((data["minbatch_msignal"]))) + (np.conjugate(data["mean_abs_chgbatch_slices2"])))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(-2.0))) +

                            0.100000*np.tanh(np.real(((data["minbatch"]) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) + (np.where(np.abs((((np.conjugate(np.sin((np.cos((np.cos((data["maxbatch_msignal"])))))))) + (((data["meanbatch_msignal"]) / 2.0)))/2.0)) > np.abs(((data["medianbatch_msignal"]) * (np.cos((((np.cos((np.cos((data["rangebatch_slices2"]))))) * (((complex(2.0)) * 2.0)))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.sin((complex(11.64997005462646484)))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2"])) +

                            0.100000*np.tanh(np.real((-((((data["minbatch"]) / 2.0)))))) +

                            0.100000*np.tanh(np.real(((((complex(2.0)) - ((-((((data["meanbatch_slices2"]) - ((-((data["meanbatch_slices2"]))))))))))) * (np.cos((((((complex(0,1)*np.conjugate(np.where(np.abs(data["stdbatch_msignal"]) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))) * 2.0)) - (data["medianbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(data["stdbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_slices2_msignal"]) / 2.0))) +

                            0.100000*np.tanh(np.real(((complex(0.0)) + (complex(0,1)*np.conjugate(complex(8.18946552276611328)))))) +

                            0.100000*np.tanh(np.real(np.cos((((complex(0,1)*np.conjugate((-((((data["abs_avgbatch_slices2_msignal"]) - (data["maxbatch_msignal"]))))))) - (data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((((np.sin((np.conjugate(complex(-2.0))))) * 2.0)) + ((((((((np.sin((np.conjugate(complex(-2.0))))) * 2.0)) + (np.cos((complex(-2.0)))))) + (complex(-2.0)))/2.0)))) + (complex(-2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["maxbatch_msignal"]) + (np.where(np.abs(data["maxbatch_msignal"]) > np.abs(np.cos((((np.conjugate(complex(0.0))) + ((-((complex(5.0))))))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((np.cos((np.sin((data["abs_maxbatch"])))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((-((np.tanh((data["abs_avgbatch_msignal"])))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.conjugate((((np.conjugate((((data["abs_minbatch_slices2"]) + (np.conjugate(data["abs_avgbatch_slices2_msignal"])))/2.0))) + (data["minbatch_slices2_msignal"]))/2.0))) * (data["stdbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(data["mean_abs_chgbatch_msignal"]) <= np.abs((-((np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2_msignal"]) * (((np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(((np.conjugate(complex(0,1)*np.conjugate(np.tanh((data["abs_avgbatch_msignal"]))))) * 2.0)),complex(1.), complex(0.) )) > np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )) - (np.cos((((np.conjugate(data["abs_minbatch_slices2_msignal"])) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate(np.conjugate(((data["maxtominbatch_slices2_msignal"]) + (complex(7.80110645294189453)))))) + (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.sin((data["maxbatch_msignal"])))) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2"])) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(data["medianbatch_msignal"]) > np.abs((-((np.where(np.abs(((data["abs_maxbatch_msignal"]) * (complex(2.0)))) > np.abs(np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.sin(((((np.sin((data["abs_avgbatch_slices2"]))) + ((((data["minbatch_msignal"]) + (complex(-3.0)))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(data["rangebatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin(((((data["minbatch_msignal"]) + (((np.conjugate(data["meanbatch_msignal"])) * (data["meanbatch_msignal"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((((data["medianbatch_slices2"]) * (data["meanbatch_msignal"]))) * (((data["signal_shift_-1_msignal"]) * 2.0)))) + (np.where(np.abs(np.tanh((((np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs((-((((data["meanbatch_slices2"]) * (complex(11.80633640289306641))))))),complex(1.), complex(0.) )) / 2.0)))) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(np.where(np.abs((((data["maxbatch_slices2"]) + (complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])))/2.0)) > np.abs(np.where(np.abs(np.conjugate(complex(1.0))) > np.abs(((data["maxtominbatch_slices2"]) + (data["maxbatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((-(((-((np.sin((data["maxtominbatch_slices2"])))))))))))) +

                            0.100000*np.tanh(np.real((((-((complex(0,1)*np.conjugate(((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))))) * (complex(0.0))))) +

                            0.100000*np.tanh(np.real((-((np.tanh((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(np.where(np.abs(complex(3.0)) > np.abs(np.where(np.abs(np.cos((((data["meanbatch_msignal"]) * 2.0)))) <= np.abs(((np.where(np.abs(np.cos((complex(-2.0)))) <= np.abs(np.tanh((((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )) - (data["maxtominbatch_msignal"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.cos((((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.sin((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(np.sin((((data["maxbatch_msignal"]) + (((data["medianbatch_msignal"]) + (complex(0,1)*np.conjugate(data["maxbatch_msignal"])))))))),complex(1.), complex(0.) ))))) <= np.abs(np.sin((data["signal_shift_+1"]))),complex(1.), complex(0.) ))))) + (data["mean_abs_chgbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(complex(0,1)*np.conjugate(data["signal_shift_+1"])) > np.abs((((((np.sin((data["maxtominbatch"]))) + (complex(0,1)*np.conjugate(data["signal_shift_+1"])))/2.0)) * 2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((complex(-3.0)) - (((((data["minbatch_slices2_msignal"]) - (((np.sin((complex(0,1)*np.conjugate(np.sin((data["maxbatch_msignal"])))))) + (data["abs_minbatch_msignal"]))))) / (((complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])) + (data["maxtominbatch_slices2_msignal"]))))))) * (np.sin((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(((((complex(0,1)*np.conjugate(complex(1.0))) - (data["meanbatch_msignal"]))) * (((data["stdbatch_slices2_msignal"]) - (((data["abs_maxbatch_slices2_msignal"]) + (complex(1.73310792446136475))))))))) * (complex(0,1)*np.conjugate(((data["abs_avgbatch_slices2"]) - (data["signal_shift_-1"]))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.sin((complex(8.82361507415771484)))) <= np.abs(((np.where(np.abs(data["signal"]) <= np.abs((-((((data["abs_avgbatch_slices2_msignal"]) * (np.where(np.abs(((data["abs_avgbatch_slices2_msignal"]) - (complex(2.0)))) > np.abs(np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) )) * (((np.sin((data["mean_abs_chgbatch_msignal"]))) + (data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.cos((((np.sin((((data["medianbatch_msignal"]) - (np.where(np.abs((-((((data["maxbatch_slices2"]) / 2.0))))) <= np.abs(((((np.cos((((data["maxtominbatch_slices2_msignal"]) / 2.0)))) - (np.cos((np.cos((data["medianbatch_slices2"]))))))) / 2.0)),complex(1.), complex(0.) )))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["abs_maxbatch_slices2"]) + (np.tanh((data["abs_maxbatch_slices2"]))))/2.0)) + ((((data["abs_maxbatch_slices2"]) + (data["maxtominbatch"]))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["signal"]) - (((np.where(np.abs(data["signal"]) > np.abs(data["signal"]),complex(1.), complex(0.) )) * (np.cos((data["mean_abs_chgbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_-1_msignal"]) / (((data["meanbatch_msignal"]) - (((((data["signal_shift_-1_msignal"]) / (((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((data["abs_maxbatch"]) / (np.sin((((complex(11.25769901275634766)) / 2.0)))))))) - (((((data["maxtominbatch_msignal"]) * (complex(0,1)*np.conjugate(data["meanbatch_msignal"])))) * (complex(-3.0)))))))) * (complex(-3.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((np.conjugate(((data["maxbatch_msignal"]) * (np.cos((data["signal"]))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["maxbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((np.conjugate((-((data["abs_maxbatch_msignal"]))))) - (np.cos((data["meanbatch_slices2_msignal"]))))) + (np.tanh((complex(0,1)*np.conjugate(data["abs_minbatch_msignal"])))))/2.0)))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch"]) - (complex(0,1)*np.conjugate(np.tanh((((((data["minbatch_slices2"]) * (data["abs_maxbatch_msignal"]))) - (np.cos((((data["abs_maxbatch"]) - (np.sin((data["medianbatch_msignal"]))))))))))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.tanh((((data["mean_abs_chgbatch_slices2"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["abs_minbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(np.tanh((data["signal"])))))/2.0)) + (data["minbatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((-((data["abs_maxbatch_slices2"])))) * ((-((((data["abs_maxbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.sin((np.sin((data["signal_shift_-1"]))))) / 2.0))) * (data["minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["minbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate(np.sin((complex(0,1)*np.conjugate(np.tanh((complex(4.0))))))))))) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(((data["minbatch_msignal"]) / 2.0))),complex(1.), complex(0.) )) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(((data["rangebatch_msignal"]) + (np.cos((complex(1.0)))))) > np.abs(np.cos((((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (((data["stdbatch_slices2"]) * 2.0)))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.cos((data["stdbatch_slices2_msignal"])))) * (data["medianbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((np.cos(((((data["medianbatch_slices2_msignal"]) + (complex(9.10487461090087891)))/2.0)))) - (((np.where(np.abs(np.where(np.abs(((np.sin((data["medianbatch_slices2_msignal"]))) - (np.sin((np.tanh((np.conjugate(data["medianbatch_slices2_msignal"])))))))) <= np.abs(((data["abs_avgbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(np.conjugate(data["stdbatch_msignal"]))),complex(1.), complex(0.) )) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((np.tanh(((-((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(complex(4.0)),complex(1.), complex(0.) ))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["maxbatch_msignal"]) + (data["maxbatch_slices2"]))))) * 2.0)) * (data["maxbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_maxbatch"]) + (np.where(np.abs(data["abs_maxbatch"]) <= np.abs(((((complex(0.0)) + (data["abs_maxbatch"]))) + (np.where(np.abs(((data["mean_abs_chgbatch_slices2"]) / ((-((data["abs_maxbatch"])))))) <= np.abs(np.where(np.abs(np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(complex(0,1)*np.conjugate(data["abs_maxbatch"])),complex(1.), complex(0.) )) <= np.abs(np.conjugate(data["medianbatch_slices2"])),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((((data["medianbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.tanh((((((np.conjugate(data["rangebatch_slices2_msignal"])) * (np.sin((data["minbatch_slices2"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["maxbatch_slices2"]))) * (data["abs_avgbatch_slices2_msignal"]))) - (((data["medianbatch_msignal"]) / (((np.cos((data["maxbatch_slices2"]))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(data["signal_shift_-1"])) +

                            0.100000*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(np.cos((np.cos((np.tanh((data["signal_shift_+1_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((data["mean_abs_chgbatch_slices2"]) + (data["abs_avgbatch_msignal"]))) * (((((data["abs_maxbatch"]) + (data["maxtominbatch"]))) + (((((data["mean_abs_chgbatch_slices2"]) / (data["stdbatch_slices2_msignal"]))) - (np.where(np.abs(data["rangebatch_slices2"]) > np.abs(np.sin((np.conjugate(((complex(0.0)) / 2.0))))),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(((((((np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.sin((data["medianbatch_msignal"]))),complex(1.), complex(0.) )) * 2.0)) / 2.0)) / 2.0))) * (np.sin((((np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch"]) + (data["abs_maxbatch_slices2_msignal"]))) / ((((np.cos((data["stdbatch_msignal"]))) + (((complex(0,1)*np.conjugate(((((((data["abs_avgbatch_msignal"]) / 2.0)) / 2.0)) * 2.0))) * (np.cos((data["stdbatch_msignal"]))))))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((data["meanbatch_msignal"]) + (np.where(np.abs((((data["minbatch"]) + (complex(3.0)))/2.0)) <= np.abs(((complex(2.0)) - (data["mean_abs_chgbatch_slices2_msignal"]))),complex(1.), complex(0.) )))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(np.where(np.abs(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(complex(1.0)),complex(1.), complex(0.) )) * 2.0)) - (np.conjugate(np.tanh((((np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) / 2.0)))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((data["signal_shift_-1_msignal"]) + (complex(0,1)*np.conjugate(((np.cos(((-((data["abs_maxbatch_slices2_msignal"])))))) * (((data["signal_shift_-1_msignal"]) * (((data["abs_maxbatch_slices2_msignal"]) * (data["abs_maxbatch_slices2_msignal"])))))))))/2.0)))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.cos((data["medianbatch_slices2_msignal"])))) / 2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((((((data["signal"]) / 2.0)) / 2.0)))) - (np.cos((np.tanh((data["signal_shift_+1"]))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1"]) + (np.cos((((((((((((data["rangebatch_slices2"]) * (np.tanh((((((((data["meanbatch_msignal"]) / 2.0)) / 2.0)) / 2.0)))))) + (data["maxbatch_slices2"]))/2.0)) - (data["maxtominbatch_slices2_msignal"]))) + (data["rangebatch_slices2"]))/2.0)))))) * (np.sin((data["maxtominbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(np.real(np.tanh((np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(np.where(np.abs(((((np.where(np.abs(np.sin((complex(0,1)*np.conjugate(data["stdbatch_slices2_msignal"])))) > np.abs(complex(0,1)*np.conjugate((((data["abs_avgbatch_slices2_msignal"]) + (complex(3.0)))/2.0))),complex(1.), complex(0.) )) / 2.0)) * 2.0)) > np.abs(np.conjugate(((data["abs_avgbatch_msignal"]) * 2.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin((((((np.tanh((data["meanbatch_msignal"]))) - (((data["minbatch_msignal"]) - (np.sin((data["minbatch_slices2"]))))))) * (np.where(np.abs(np.cos((data["medianbatch_msignal"]))) <= np.abs(np.sin((complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"])) > np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) ))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(complex(3.20074152946472168)) <= np.abs(np.tanh((np.sin(((((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])) + (complex(0.0)))/2.0)))))),complex(1.), complex(0.) )) - (np.sin((data["maxbatch_slices2_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((data["rangebatch_msignal"]))) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.where(np.abs((((((data["mean_abs_chgbatch_msignal"]) - (((complex(0,1)*np.conjugate(complex(-2.0))) / 2.0)))) + (np.cos((data["abs_maxbatch_slices2_msignal"]))))/2.0)) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) ))) - (np.cos((((data["abs_maxbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.cos((complex(0,1)*np.conjugate(((((complex(3.0)) + (((complex(2.0)) * (((data["medianbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(((np.sin((data["medianbatch_slices2_msignal"]))) - (np.where(np.abs(np.conjugate(np.where(np.abs(complex(0.0)) <= np.abs(np.conjugate(np.tanh((data["maxbatch_msignal"])))),complex(1.), complex(0.) ))) <= np.abs(np.where(np.abs(complex(0.0)) <= np.abs(complex(0,1)*np.conjugate(complex(6.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))))))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(data["medianbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_minbatch_msignal"]) + (((((((data["minbatch_msignal"]) - (data["maxbatch_slices2"]))) + (np.cos((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(((((data["signal_shift_+1"]) * (complex(3.0)))) - (((((np.cos((data["abs_minbatch_slices2_msignal"]))) + (data["maxbatch_slices2_msignal"]))) * 2.0)))),complex(1.), complex(0.) )))))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((np.cos((data["rangebatch_slices2"]))) + (np.where(np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(np.where(np.abs(complex(3.0)) > np.abs(complex(-1.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) <= np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(data["rangebatch_slices2"])) - (((complex(0.0)) / 2.0))))) +

                            0.100000*np.tanh(np.real(((((data["abs_minbatch_msignal"]) * (np.cos((((((((data["maxbatch_slices2_msignal"]) * 2.0)) / 2.0)) / 2.0)))))) - (np.cos((((data["maxbatch_slices2_msignal"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )) * 2.0))) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(data["medianbatch_msignal"]) > np.abs(complex(0,1)*np.conjugate((-((data["medianbatch_msignal"]))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["medianbatch_slices2_msignal"]) + (((np.where(np.abs(((((data["medianbatch_msignal"]) * (((np.where(np.abs(data["minbatch_msignal"]) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) * 2.0)))) * 2.0)) <= np.abs(np.tanh((((data["medianbatch_msignal"]) * 2.0)))),complex(1.), complex(0.) )) - (((np.sin((data["minbatch_msignal"]))) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1_msignal"]) / 2.0)) * (np.sin((((((complex(0,1)*np.conjugate(data["abs_maxbatch_msignal"])) + (np.where(np.abs(((data["abs_maxbatch_msignal"]) + (data["abs_maxbatch_msignal"]))) > np.abs(np.tanh((np.where(np.abs(np.sin((np.cos((data["abs_maxbatch_msignal"]))))) > np.abs(np.tanh((np.cos((data["medianbatch_msignal"]))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))) + (data["abs_maxbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((np.cos((data["abs_avgbatch_slices2_msignal"]))))) / 2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((data["maxbatch_slices2_msignal"]) + (np.cos((np.cos((((data["maxbatch_slices2_msignal"]) + (np.cos((np.where(np.abs(((complex(1.0)) / 2.0)) <= np.abs(np.tanh((((data["maxbatch_slices2_msignal"]) + (np.cos((((data["meanbatch_msignal"]) * (data["meanbatch_slices2_msignal"]))))))))),complex(1.), complex(0.) ))))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs((((complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs((((complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) ))) + (data["abs_minbatch_slices2"]))/2.0)),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) ))) + (data["abs_minbatch_slices2"]))/2.0)) > np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(np.cos((complex(1.96440744400024414))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((-((data["medianbatch_slices2"])))) * (np.cos(((((data["maxbatch_slices2_msignal"]) + ((-((np.where(np.abs(np.conjugate(((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(complex(4.0)),complex(1.), complex(0.) )) + ((-((data["rangebatch_msignal"]))))))) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(complex(3.0)) > np.abs(np.where(np.abs(np.tanh((np.conjugate(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))) <= np.abs(np.where(np.abs(data["signal_shift_+1"]) > np.abs(np.conjugate(data["maxtominbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate(complex(5.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal"]) <= np.abs(np.sin((np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(np.where(np.abs(np.sin((np.conjugate(np.cos((((data["signal_shift_+1_msignal"]) - (np.conjugate(data["signal_shift_+1_msignal"]))))))))) > np.abs(complex(0,1)*np.conjugate(complex(0.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((((data["abs_avgbatch_msignal"]) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((data["minbatch_msignal"]))) <= np.abs((-((complex(0,1)*np.conjugate(((np.where(np.abs(complex(0.0)) <= np.abs(((((data["signal_shift_+1"]) - (data["medianbatch_slices2_msignal"]))) + (((np.sin((data["medianbatch_slices2_msignal"]))) * (data["abs_minbatch_slices2"]))))),complex(1.), complex(0.) )) / 2.0)))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(complex(0,1)*np.conjugate(complex(0.0))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["maxbatch_slices2"]) + ((((-((((complex(-3.0)) + (((data["minbatch_msignal"]) + (np.cos((np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(np.conjugate(data["maxtominbatch_slices2_msignal"]))) <= np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(np.cos((data["abs_minbatch_slices2"])))),complex(1.), complex(0.) ))))))))))) * 2.0)))))) * (((data["maxbatch_slices2"]) / 2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(np.conjugate(np.sin((((complex(0,1)*np.conjugate((-((data["abs_maxbatch_slices2"]))))) / 2.0))))),complex(1.), complex(0.) )) * (np.where(np.abs((((np.sin((data["rangebatch_slices2"]))) + (complex(-2.0)))/2.0)) > np.abs(np.where(np.abs(np.sin((complex(0,1)*np.conjugate((-((complex(0,1)*np.conjugate(((complex(1.0)) * 2.0))))))))) <= np.abs(np.cos((complex(-3.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((complex(0,1)*np.conjugate((-((complex(3.0)))))))) <= np.abs(((np.tanh((data["signal_shift_+1_msignal"]))) - (((np.where(np.abs(complex(0,1)*np.conjugate(complex(1.51755845546722412))) <= np.abs(((np.sin((data["abs_minbatch_slices2_msignal"]))) / 2.0)),complex(1.), complex(0.) )) * (np.sin((complex(0.0)))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-((np.where(np.abs(np.where(np.abs(np.where(np.abs(np.tanh((complex(1.0)))) <= np.abs(((data["medianbatch_msignal"]) / (data["mean_abs_chgbatch_msignal"]))),complex(1.), complex(0.) )) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs((-((data["medianbatch_msignal"])))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(complex(2.43549633026123047)),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real((((complex(0,1)*np.conjugate(((((np.where(np.abs(complex(5.0)) <= np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) )) / 2.0)) / 2.0))) + (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((-((np.sin((((data["signal_shift_+1"]) / 2.0))))))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((np.where(np.abs(np.sin(((-((((data["medianbatch_msignal"]) / 2.0))))))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) - (data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((((((-((np.cos((data["rangebatch_msignal"])))))) + (((complex(0,1)*np.conjugate(np.cos((np.cos((complex(0,1)*np.conjugate(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs((-((data["maxbatch_msignal"])))),complex(1.), complex(0.) )))))))) / 2.0)))) + (np.where(np.abs(complex(-3.0)) <= np.abs(np.cos((np.tanh((np.sin((np.where(np.abs(data["maxtominbatch_slices2"]) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(3.0)))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(((((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2_msignal"])) - ((((((-((complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"]))))) + (((np.where(np.abs(complex(0.0)) > np.abs((-((data["signal_shift_+1_msignal"])))),complex(1.), complex(0.) )) - (((((data["minbatch_msignal"]) / 2.0)) / 2.0)))))) / 2.0)))) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((((((((np.conjugate(np.where(np.abs(complex(-1.0)) > np.abs(((complex(-1.0)) * 2.0)),complex(1.), complex(0.) ))) + (complex(0,1)*np.conjugate((-((np.where(np.abs(np.cos((complex(0,1)*np.conjugate(np.tanh((((complex(2.0)) * (np.conjugate(np.cos((np.tanh((data["abs_maxbatch_msignal"]))))))))))))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))))/2.0)) / 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0)))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["meanbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0)) / ((((((-((np.conjugate(complex(3.0)))))) + (data["meanbatch_slices2_msignal"]))) - (((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])) / 2.0)))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.tanh((np.cos((np.cos((data["medianbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin(((((np.sin((data["abs_avgbatch_slices2"]))) + (((data["signal_shift_-1_msignal"]) * (np.conjugate((((data["medianbatch_msignal"]) + (np.tanh((((data["medianbatch_msignal"]) * 2.0)))))/2.0))))))/2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(np.where(np.abs(((data["abs_minbatch_slices2"]) + (data["abs_maxbatch_slices2"]))) <= np.abs(np.where(np.abs(((data["maxbatch_slices2"]) * (np.conjugate(data["maxbatch_slices2"])))) > np.abs(np.cos((((data["maxbatch_slices2_msignal"]) - (data["meanbatch_slices2_msignal"]))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(np.sin((data["maxtominbatch_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate((-((((np.sin((np.cos((data["maxbatch_msignal"]))))) * 2.0))))))))) + (complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate(np.cos((data["stdbatch_msignal"])))) * 2.0))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(((np.cos((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(((np.where(np.abs(complex(3.0)) <= np.abs(np.cos(((((data["maxbatch_slices2_msignal"]) + (np.sin((data["mean_abs_chgbatch_slices2"]))))/2.0)))),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) )))) * 2.0))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal"]) <= np.abs(np.where(np.abs((-((data["meanbatch_msignal"])))) <= np.abs(np.sin((np.cos((complex(0.0)))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.tanh((((((complex(0.0)) * (np.cos((complex(0.0)))))) * (complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])))))) * (data["abs_minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((-((np.cos(((-((((data["medianbatch_msignal"]) + (complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])))))))))))) > np.abs(((np.tanh((complex(10.0)))) + (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((((((np.conjugate(np.conjugate(np.sin((data["meanbatch_msignal"]))))) + (data["meanbatch_msignal"]))/2.0)) + (data["meanbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(1.0)))) +

                            0.100000*np.tanh(np.real(np.sin((data["signal_shift_-1"])))) +

                            0.100000*np.tanh(np.real(((complex(8.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((-((data["signal_shift_+1"])))) <= np.abs(np.tanh(((-((np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.sin((complex(0,1)*np.conjugate(complex(0.0))))),complex(1.), complex(0.) ))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(np.conjugate(data["abs_avgbatch_slices2"]))) <= np.abs(np.where(np.abs(((np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(np.sin((data["meanbatch_slices2"]))),complex(1.), complex(0.) )) / 2.0)) <= np.abs(complex(0.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))   

        

    def GP_class_8(self,data):

        return self.Output( -3.014039 +

                            0.100000*np.tanh(np.real(data["signal"])) +

                            0.100000*np.tanh(np.real(((((((np.where(np.abs(data["rangebatch_msignal"]) > np.abs(np.sin((data["meanbatch_slices2"]))),complex(1.), complex(0.) )) - (((data["abs_avgbatch_msignal"]) * 2.0)))) - (complex(0,1)*np.conjugate(data["meanbatch_slices2"])))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_slices2"]) - (np.sin((data["maxtominbatch_slices2"]))))) + (complex(-2.0))))) +

                            0.100000*np.tanh(np.real(data["meanbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((((((((complex(-2.0)) * 2.0)) + (complex(-1.0)))) - (data["abs_avgbatch_slices2_msignal"]))) * (data["signal"]))) > np.abs(np.conjugate(((((data["rangebatch_slices2"]) * 2.0)) - (np.sin((((data["signal_shift_+1"]) - (data["signal_shift_+1"])))))))),complex(1.), complex(0.) )) - (((data["abs_avgbatch_slices2_msignal"]) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.tanh((data["maxbatch_slices2"]))) - (data["mean_abs_chgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(data["signal"])) +

                            0.100000*np.tanh(np.real(((((((np.cos((((complex(1.0)) / ((((((data["maxtominbatch"]) + (data["stdbatch_slices2_msignal"]))/2.0)) - (np.sin((np.where(np.abs(np.sin((data["abs_avgbatch_slices2"]))) <= np.abs(complex(3.0)),complex(1.), complex(0.) )))))))))) / 2.0)) + (data["minbatch_msignal"]))) * (np.sin((data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((-(((((((data["mean_abs_chgbatch_msignal"]) + (data["abs_avgbatch_msignal"]))/2.0)) * 2.0)))))) +

                            0.100000*np.tanh(np.real(((complex(-2.0)) + (((((((data["abs_maxbatch"]) + (((((((-((data["rangebatch_msignal"])))) + ((-((np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(np.tanh((data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))/2.0)) * (np.sin((np.conjugate(data["abs_avgbatch_slices2"])))))))) + (complex(-2.0)))) + (complex(-2.0))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"])) + (((np.conjugate(data["maxtominbatch_msignal"])) * (((((complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"])) - (((np.cos(((((data["rangebatch_slices2"]) + (np.tanh((((data["stdbatch_msignal"]) / 2.0)))))/2.0)))) - (complex(3.0)))))) * ((((complex(1.0)) + (data["minbatch_slices2_msignal"]))/2.0))))))))) +

                            0.100000*np.tanh(np.real(((((data["stdbatch_msignal"]) / (np.cos((((data["minbatch_msignal"]) - (np.where(np.abs(((((((data["abs_avgbatch_slices2_msignal"]) + (np.cos((data["minbatch_msignal"]))))/2.0)) + (data["maxbatch_msignal"]))/2.0)) > np.abs(np.where(np.abs(data["minbatch_msignal"]) <= np.abs((((((complex(3.0)) * 2.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))) - (data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real((((((np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(np.cos((np.where(np.abs(complex(-2.0)) <= np.abs((((data["maxbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(((np.sin((data["maxbatch_msignal"]))) - (data["abs_minbatch_msignal"])))))/2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) - (np.cos((data["minbatch_msignal"]))))) + ((-((data["medianbatch_msignal"])))))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin(((-(((-(((((data["minbatch_slices2_msignal"]) + (((complex(0,1)*np.conjugate(((np.conjugate(np.where(np.abs(data["minbatch"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) ))) / 2.0))) * (np.conjugate(((np.sin(((-((data["minbatch_slices2_msignal"])))))) * 2.0))))))/2.0))))))))))) +

                            0.100000*np.tanh(np.real((((-((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) - (((((data["mean_abs_chgbatch_slices2_msignal"]) - (complex(-3.0)))) / 2.0))))))) + (((complex(-3.0)) - ((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) + (np.where(np.abs((((data["abs_minbatch_slices2_msignal"]) + (((((data["maxtominbatch_slices2"]) / 2.0)) * 2.0)))/2.0)) > np.abs(np.sin((complex(-1.0)))),complex(1.), complex(0.) )))/2.0))))))) +

                            0.100000*np.tanh(np.real(data["rangebatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((((-((np.conjugate((-((complex(-2.0))))))))) + (((((data["meanbatch_slices2"]) - (complex(7.39584636688232422)))) - (((complex(-2.0)) - ((-((((complex(-3.0)) / (complex(2.0))))))))))))/2.0)) + (complex(-2.0))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((((data["medianbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_msignal"]))) + (complex(-2.0))))) + ((-((np.cos((((complex(-1.0)) + ((-((complex(0.0))))))))))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((np.sin((data["rangebatch_slices2"]))) * (((complex(0,1)*np.conjugate(np.where(np.abs(data["stdbatch_slices2_msignal"]) <= np.abs(((np.sin((((np.sin((data["stdbatch_slices2"]))) / 2.0)))) / 2.0)),complex(1.), complex(0.) ))) / 2.0)))) <= np.abs(np.conjugate(data["abs_minbatch_msignal"])),complex(1.), complex(0.) )) - (((np.sin((np.sin((data["maxbatch_msignal"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real((((-((np.cos((((data["meanbatch_msignal"]) * (((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(complex(1.0)),complex(1.), complex(0.) )) - (((data["maxtominbatch_msignal"]) + (complex(2.0)))))) / (((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["meanbatch_slices2"]))) - (data["rangebatch_msignal"])))))))))))) - (((data["meanbatch_msignal"]) + (data["meanbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.where(np.abs(np.where(np.abs(np.where(np.abs((-((np.where(np.abs(complex(-2.0)) <= np.abs(np.tanh((np.tanh((data["stdbatch_slices2_msignal"]))))),complex(1.), complex(0.) ))))) > np.abs(data["minbatch"]),complex(1.), complex(0.) )) > np.abs(((data["mean_abs_chgbatch_msignal"]) * ((((data["maxbatch_slices2_msignal"]) + (((complex(1.0)) / 2.0)))/2.0)))),complex(1.), complex(0.) )) <= np.abs((((complex(7.90372467041015625)) + (data["stdbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )) <= np.abs(complex(1.0)),complex(1.), complex(0.) )) + (complex(-3.0))))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["stdbatch_slices2"]) * 2.0)))) - ((((data["signal"]) + (complex(8.58431529998779297)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_msignal"]) * (complex(-3.0)))) - (np.tanh(((((((data["mean_abs_chgbatch_slices2"]) * (((((data["maxbatch_msignal"]) - (data["meanbatch_msignal"]))) - (data["meanbatch_msignal"]))))) + (np.cos((data["medianbatch_slices2_msignal"]))))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["rangebatch_slices2"]) / 2.0)) + (np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(complex(-1.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(complex(0,1)*np.conjugate(((data["abs_minbatch_slices2_msignal"]) - (((((data["abs_maxbatch_msignal"]) / 2.0)) + (((data["medianbatch_msignal"]) * (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((data["abs_maxbatch_slices2"]) / 2.0))))))))))))) - (data["stdbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(data["abs_minbatch_slices2"])) +

                            0.100000*np.tanh(np.real(np.cos((data["abs_maxbatch"])))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_msignal"]) - (data["maxtominbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.sin((complex(1.0)))) + (((complex(0,1)*np.conjugate(data["signal_shift_+1"])) + (complex(-2.0))))))) +

                            0.100000*np.tanh(np.real(((((complex(3.0)) - (((data["abs_maxbatch_slices2"]) - (complex(3.0)))))) - (np.conjugate(complex(3.0)))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["minbatch_slices2_msignal"]))) - (data["mean_abs_chgbatch_slices2"]))) + (((((data["abs_maxbatch_slices2"]) - (((data["abs_avgbatch_slices2_msignal"]) / 2.0)))) * (((data["meanbatch_slices2"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.sin((((data["stdbatch_slices2"]) * 2.0)))) * ((((data["minbatch_slices2_msignal"]) + (((np.sin((np.sin((((data["maxbatch_slices2_msignal"]) * (data["stdbatch_slices2"]))))))) * (np.sin((((((np.sin((((data["minbatch_slices2"]) - (np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )))))) * (data["mean_abs_chgbatch_msignal"]))) * 2.0)))))))/2.0))))) +

                            0.100000*np.tanh(np.real((((((((((-((np.cos((data["maxbatch_msignal"])))))) + (((np.tanh((np.cos((data["maxbatch_msignal"]))))) * (data["stdbatch_msignal"]))))) / 2.0)) / 2.0)) - (((np.cos((data["maxbatch_msignal"]))) * (complex(2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin(((-((((data["meanbatch_slices2_msignal"]) + (((np.where(np.abs(data["rangebatch_slices2_msignal"]) <= np.abs(((((complex(-1.0)) * (((complex(0,1)*np.conjugate(complex(11.20641517639160156))) * 2.0)))) / 2.0)),complex(1.), complex(0.) )) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real((((((data["maxbatch_msignal"]) + (((complex(-2.0)) - (np.where(np.abs(data["maxbatch_slices2"]) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))))/2.0)) * (((data["medianbatch_slices2_msignal"]) * (data["stdbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["medianbatch_slices2_msignal"]) * (((((data["signal_shift_+1"]) * 2.0)) - (((((data["abs_maxbatch_slices2_msignal"]) * ((((data["abs_maxbatch_slices2_msignal"]) + (data["signal_shift_+1"]))/2.0)))) + (((data["maxtominbatch_slices2"]) + (((data["maxbatch_slices2_msignal"]) * (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])),complex(1.), complex(0.) )))))))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos(((((data["rangebatch_slices2"]) + (complex(2.0)))/2.0)))) - (np.where(np.abs(np.sin((((data["abs_avgbatch_slices2"]) - (((((complex(0.0)) * 2.0)) * 2.0)))))) > np.abs(complex(3.0)),complex(1.), complex(0.) )))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_msignal"]) * (((data["signal_shift_-1"]) - ((((data["rangebatch_slices2_msignal"]) + (data["medianbatch_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(((complex(-2.0)) / (np.conjugate(np.tanh((np.cos((((data["meanbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((((complex(0,1)*np.conjugate(((((data["maxbatch_msignal"]) - (((((np.sin((data["minbatch_msignal"]))) - (np.sin(((((np.cos((data["medianbatch_slices2"]))) + (complex(0,1)*np.conjugate(np.sin((data["maxbatch_msignal"])))))/2.0)))))) - (data["maxbatch_msignal"]))))) - ((-((data["maxbatch_msignal"]))))))) - (data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.where(np.abs(np.sin((((np.conjugate(complex(0.0))) / 2.0)))) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(((np.conjugate(data["maxtominbatch_slices2"])) + ((-((np.sin((np.where(np.abs(((data["maxtominbatch"]) + (data["rangebatch_slices2"]))) <= np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) ))))))))) <= np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )) + (data["stdbatch_slices2_msignal"]))) * (((((data["abs_maxbatch_slices2_msignal"]) - (data["mean_abs_chgbatch_slices2"]))) / 2.0))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_msignal"]) / (np.cos(((((data["abs_maxbatch_slices2_msignal"]) + (np.cos((np.cos(((((((np.cos((data["abs_maxbatch_slices2_msignal"]))) + (((np.conjugate(data["minbatch_msignal"])) * 2.0)))/2.0)) * 2.0)))))))/2.0))))))) +

                            0.100000*np.tanh(np.real((((((data["maxtominbatch"]) + (((np.where(np.abs(((data["rangebatch_slices2"]) - (data["stdbatch_slices2_msignal"]))) <= np.abs(np.sin((np.sin((data["rangebatch_slices2"]))))),complex(1.), complex(0.) )) - ((((-(((-(((((-(((((complex(3.0)) + (data["meanbatch_slices2"]))/2.0))))) * 2.0)))))))) / 2.0)))))) + (data["meanbatch_slices2"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["minbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate((((-((data["minbatch_slices2_msignal"])))) - ((((((-((np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(complex(0,1)*np.conjugate(complex(1.0))),complex(1.), complex(0.) ))))) + (complex(2.0)))) + (((data["minbatch_slices2_msignal"]) + (np.sin((data["minbatch_slices2_msignal"])))))))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((complex(-3.0)) - (((((np.tanh((complex(10.0)))) + (complex(0,1)*np.conjugate(((data["abs_minbatch_msignal"]) + (((complex(-3.0)) + (complex(-3.0))))))))) * (np.where(np.abs(complex(-2.0)) > np.abs(complex(0,1)*np.conjugate(data["medianbatch_msignal"])),complex(1.), complex(0.) )))))) / 2.0))) +

                            0.100000*np.tanh(np.real(data["maxbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.cos((data["minbatch_msignal"]))) * (np.conjugate(np.cos(((((data["abs_maxbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.tanh((((np.conjugate(data["abs_maxbatch"])) * (np.tanh(((-((np.tanh(((((((data["maxbatch_slices2"]) / 2.0)) + (np.tanh((np.conjugate(data["abs_minbatch_slices2_msignal"])))))/2.0)))))))))))))))) +

                            0.100000*np.tanh(np.real(((np.cos((complex(2.0)))) - (((data["meanbatch_msignal"]) / (((data["mean_abs_chgbatch_slices2"]) - (np.where(np.abs(np.where(np.abs(np.conjugate(((np.sin((data["rangebatch_slices2_msignal"]))) / 2.0))) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["minbatch_msignal"]))) * (((data["medianbatch_msignal"]) * (((((complex(9.0)) + (data["medianbatch_msignal"]))) + (((complex(-1.0)) * (data["abs_avgbatch_msignal"]))))))))) + (np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(complex(9.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((-((((np.cos((data["minbatch_msignal"]))) * (np.where(np.abs(((complex(5.0)) / 2.0)) <= np.abs(((((data["minbatch_msignal"]) / 2.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) / 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((((data["signal_shift_-1"]) + (((data["maxbatch_slices2"]) * (np.tanh((data["medianbatch_slices2_msignal"]))))))/2.0)) + (np.cos(((((((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"])) + (((data["minbatch_msignal"]) * 2.0)))) + (np.cos((np.where(np.abs(complex(0,1)*np.conjugate(complex(-3.0))) <= np.abs(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)),complex(1.), complex(0.) )))))/2.0)))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["meanbatch_msignal"]) + (np.where(np.abs(np.where(np.abs(((np.where(np.abs(((np.cos((np.where(np.abs(((np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )) / 2.0)) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )))) / 2.0)) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )) / 2.0)) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(complex(3.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(data["minbatch_msignal"])) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((((np.sin((((((complex(0,1)*np.conjugate((-((data["abs_maxbatch"]))))) + ((-((data["medianbatch_slices2_msignal"])))))) * 2.0)))) - ((((data["medianbatch_slices2_msignal"]) + ((-((data["medianbatch_msignal"])))))/2.0)))) + (data["maxtominbatch_slices2_msignal"]))) - ((((data["medianbatch_slices2_msignal"]) + (np.cos((data["stdbatch_msignal"]))))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) <= np.abs(((((data["mean_abs_chgbatch_slices2"]) * (complex(0,1)*np.conjugate(np.where(np.abs(((((np.sin((data["abs_minbatch_slices2_msignal"]))) + (data["medianbatch_slices2"]))) + (data["abs_avgbatch_slices2_msignal"]))) > np.abs(complex(0,1)*np.conjugate(data["rangebatch_msignal"])),complex(1.), complex(0.) ))))) / 2.0)),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(data["maxtominbatch"])) +

                            0.100000*np.tanh(np.real(((((np.sin((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(data["abs_maxbatch_msignal"])),complex(1.), complex(0.) )))) - (((((((((np.sin((data["abs_maxbatch_msignal"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(np.tanh((data["abs_minbatch_msignal"])))) - (((((data["abs_avgbatch_slices2_msignal"]) * (data["abs_maxbatch_msignal"]))) * (data["rangebatch_slices2_msignal"]))))) + (complex(0,1)*np.conjugate(((data["abs_maxbatch_msignal"]) - ((-((data["meanbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(data["meanbatch_msignal"])) - (np.sin((((((data["maxbatch_slices2_msignal"]) + (((complex(-1.0)) + (complex(0,1)*np.conjugate((((data["maxbatch_slices2_msignal"]) + (np.where(np.abs(data["maxbatch_slices2_msignal"]) <= np.abs(np.tanh((((complex(-1.0)) + (complex(0,1)*np.conjugate(((data["maxbatch_slices2_msignal"]) + (complex(-1.0))))))))),complex(1.), complex(0.) )))/2.0))))))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((complex(-3.0)) + (((np.where(np.abs(np.cos(((((complex(0,1)*np.conjugate(np.tanh((np.where(np.abs(np.cos((data["abs_minbatch_msignal"]))) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))) + (np.conjugate(complex(-3.0))))/2.0)))) > np.abs(np.cos((complex(0,1)*np.conjugate(complex(-2.0))))),complex(1.), complex(0.) )) + (data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((((-((data["minbatch_slices2"])))) + (data["mean_abs_chgbatch_msignal"]))/2.0)) * (np.where(np.abs((-((np.cos((np.sin((np.tanh((data["medianbatch_slices2_msignal"])))))))))) <= np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos((((data["maxbatch_slices2_msignal"]) / ((((complex(-3.0)) + (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs((-((np.where(np.abs(complex(-3.0)) <= np.abs(((np.cos((((data["maxbatch_slices2_msignal"]) / ((((complex(-3.0)) + (np.where(np.abs(np.sin((np.conjugate(data["maxbatch_slices2_msignal"])))) <= np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0)))))) - (complex(-3.0)))),complex(1.), complex(0.) ))))),complex(1.), complex(0.) )))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(np.where(np.abs(complex(-3.0)) > np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) - (data["minbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.where(np.abs(complex(0.0)) <= np.abs(complex(0.0)),complex(1.), complex(0.) )) <= np.abs(complex(3.0)),complex(1.), complex(0.) )) > np.abs(((((data["maxbatch_msignal"]) * (data["rangebatch_msignal"]))) + (complex(0.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((np.conjugate(((data["minbatch"]) * 2.0))) - (np.tanh((np.conjugate((((data["abs_avgbatch_msignal"]) + (np.sin((((np.tanh(((-((complex(0,1)*np.conjugate(data["abs_maxbatch"]))))))) / 2.0)))))/2.0))))))))) +

                            0.100000*np.tanh(np.real(((np.cos(((((-((((np.where(np.abs(data["meanbatch_msignal"]) <= np.abs((((((complex(3.0)) / 2.0)) + ((((np.sin((((np.cos((data["maxtominbatch_msignal"]))) / 2.0)))) + (complex(-2.0)))/2.0)))/2.0)),complex(1.), complex(0.) )) * (data["abs_avgbatch_msignal"])))))) * (complex(-3.0)))))) - (np.cos((((data["signal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(data["minbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((data["minbatch_msignal"]) * (np.tanh(((((((np.sin((data["minbatch_msignal"]))) + (np.sin((((data["abs_minbatch_slices2_msignal"]) * (np.tanh((np.cos((data["minbatch_msignal"]))))))))))/2.0)) + (np.cos((data["minbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real((((((((((((-((((((data["mean_abs_chgbatch_slices2_msignal"]) - (data["stdbatch_slices2_msignal"]))) * 2.0))))) + (data["rangebatch_slices2"]))/2.0)) + (data["maxbatch_slices2"]))/2.0)) * (np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (complex(1.46078503131866455)))))))) + (((complex(3.0)) + (data["mean_abs_chgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((np.tanh((complex(0,1)*np.conjugate((-((((data["maxbatch_msignal"]) / 2.0)))))))) + (np.tanh((complex(0,1)*np.conjugate((-((complex(0,1)*np.conjugate((-((((data["maxbatch_msignal"]) / 2.0)))))))))))))/2.0)))) +

                            0.100000*np.tanh(np.real(((np.sin((((np.conjugate(((data["maxbatch_msignal"]) * 2.0))) - (complex(0,1)*np.conjugate(data["minbatch_slices2_msignal"])))))) - (((np.conjugate(data["maxbatch_slices2"])) / 2.0))))) +

                            0.100000*np.tanh(np.real((-((((np.cos((complex(0,1)*np.conjugate(np.cos((data["minbatch_msignal"])))))) + (np.cos((data["minbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])) + (data["abs_maxbatch"]))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real((-(((((((complex(12.07170677185058594)) + ((((((complex(3.0)) + (((((data["maxtominbatch_msignal"]) + (((np.sin((np.cos((np.cos((data["minbatch"]))))))) / (np.sin(((((np.conjugate(np.tanh((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0))))) + (((data["minbatch"]) / 2.0)))/2.0)))))))) * 2.0)))/2.0)) / 2.0)))/2.0)) * 2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(6.26173257827758789)) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1_msignal"])) +

                            0.100000*np.tanh(np.real((((((np.conjugate(np.conjugate(((np.conjugate(np.cos(((((((data["meanbatch_msignal"]) + (complex(0,1)*np.conjugate(complex(6.0))))/2.0)) * 2.0))))) - (((((data["signal_shift_-1_msignal"]) * 2.0)) / 2.0)))))) / 2.0)) + (((np.sin((data["signal_shift_-1_msignal"]))) / 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real((((((data["maxbatch_slices2_msignal"]) + ((((((-((((np.cos((data["meanbatch_slices2"]))) + (data["minbatch_slices2_msignal"])))))) * 2.0)) + (np.cos((data["stdbatch_msignal"]))))))/2.0)) + (np.conjugate(data["maxtominbatch"]))))) +

                            0.100000*np.tanh(np.real(((complex(2.0)) / (((((complex(2.0)) * (np.cos((data["minbatch_msignal"]))))) / (np.tanh((((data["stdbatch_msignal"]) - (((np.tanh((data["abs_maxbatch_msignal"]))) * (np.cos((data["abs_minbatch_slices2"])))))))))))))) +

                            0.100000*np.tanh(np.real((((((((-((data["minbatch"])))) - (np.where(np.abs(data["minbatch"]) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))) / 2.0)) + (data["minbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((data["medianbatch_slices2"]) + (((((data["maxtominbatch"]) / 2.0)) + (data["maxtominbatch"]))))) + ((((((((((data["medianbatch_slices2"]) + (data["maxtominbatch"]))/2.0)) + (data["signal_shift_-1"]))/2.0)) + (((data["medianbatch_slices2"]) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((((data["minbatch_msignal"]) / 2.0)) * ((((np.tanh((np.cos((data["abs_minbatch_slices2"]))))) + (data["signal_shift_+1_msignal"]))/2.0)))))) +

                            0.100000*np.tanh(np.real(((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0)) * (np.sin((np.tanh((np.where(np.abs(np.where(np.abs(np.where(np.abs(np.conjugate(complex(-3.0))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) > np.abs(((complex(9.18558788299560547)) * (data["maxbatch_slices2_msignal"]))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2"]) - (((((np.sin((((data["rangebatch_msignal"]) - (((np.cos((data["signal_shift_+1"]))) - (data["stdbatch_msignal"]))))))) - (((data["rangebatch_msignal"]) - (complex(8.12590789794921875)))))) - (((data["rangebatch_msignal"]) - (complex(8.12590789794921875))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((((data["minbatch_msignal"]) + (data["meanbatch_msignal"]))) + (np.cos((data["abs_avgbatch_slices2"]))))/2.0))))) +

                            0.100000*np.tanh(np.real((-((np.cos((((((-((((complex(-1.0)) - (np.where(np.abs(complex(0,1)*np.conjugate(complex(1.0))) <= np.abs((-((np.tanh((np.where(np.abs(data["minbatch_slices2"]) > np.abs(((data["maxtominbatch_msignal"]) * (data["abs_maxbatch_slices2"]))),complex(1.), complex(0.) ))))))),complex(1.), complex(0.) ))))))) + ((((np.cos((np.tanh((data["stdbatch_msignal"]))))) + (np.cos(((-((data["medianbatch_msignal"])))))))/2.0)))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((complex(3.0)) + (((data["medianbatch_msignal"]) * 2.0))))) +

                            0.100000*np.tanh(np.real((((((complex(0,1)*np.conjugate(((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) * (np.conjugate(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * (data["maxbatch_slices2"]))))))) - (((np.cos((data["abs_maxbatch_slices2_msignal"]))) * (data["maxbatch_slices2"]))))) + (data["stdbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos((((((((np.tanh((data["maxbatch_slices2_msignal"]))) + (((np.sin((((np.tanh(((-((complex(0,1)*np.conjugate(((np.conjugate(np.conjugate(complex(-2.0)))) / 2.0)))))))) / 2.0)))) / 2.0)))/2.0)) + (data["rangebatch_slices2"]))/2.0))))) +

                            0.100000*np.tanh(np.real(((complex(-2.0)) / (np.cos((((((data["stdbatch_slices2"]) - (np.where(np.abs(data["medianbatch_msignal"]) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )))) - (np.where(np.abs(np.where(np.abs(data["medianbatch_msignal"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) )) > np.abs(((complex(-2.0)) / (np.cos((((data["stdbatch_slices2"]) - (data["signal_shift_-1"]))))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.where(np.abs(np.sin((np.sin((((data["minbatch_slices2"]) * 2.0)))))) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))) > np.abs(((data["maxbatch_slices2_msignal"]) + (complex(-3.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["medianbatch_slices2"])) +

                            0.100000*np.tanh(np.real(data["signal_shift_-1_msignal"])) +

                            0.100000*np.tanh(np.real(((np.cos((np.sin((data["maxtominbatch_msignal"]))))) * ((((data["medianbatch_slices2"]) + (data["meanbatch_slices2"]))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((np.cos((((data["maxbatch_slices2_msignal"]) * 2.0)))) + (np.cos((((data["medianbatch_slices2_msignal"]) / 2.0)))))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.tanh((np.cos((data["medianbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real((-((((np.where(np.abs(np.conjugate(data["maxtominbatch_slices2_msignal"])) > np.abs(complex(0,1)*np.conjugate(np.where(np.abs((((((np.where(np.abs(data["abs_maxbatch"]) > np.abs(complex(-1.0)),complex(1.), complex(0.) )) + (data["maxtominbatch_slices2"]))/2.0)) + (data["abs_minbatch_slices2"]))) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) ))),complex(1.), complex(0.) )) / 2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal"]) > np.abs(np.tanh((np.sin((data["signal_shift_-1_msignal"]))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["abs_maxbatch_msignal"]) * 2.0)) + (np.sin((((np.where(np.abs(((np.where(np.abs(complex(0,1)*np.conjugate(((complex(-2.0)) / 2.0))) <= np.abs(data["abs_minbatch_msignal"]),complex(1.), complex(0.) )) + (data["abs_maxbatch_slices2"]))) > np.abs(np.tanh((data["maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )) + (data["abs_maxbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["abs_minbatch_msignal"]) + (np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(((data["medianbatch_msignal"]) + (complex(0,1)*np.conjugate(((((data["signal_shift_-1_msignal"]) * 2.0)) * (((data["stdbatch_slices2"]) + (data["abs_minbatch_msignal"])))))))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_maxbatch"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_maxbatch_slices2_msignal"]) - (((np.where(np.abs(((np.conjugate(((data["abs_maxbatch_slices2_msignal"]) - ((((-((data["signal"])))) / 2.0))))) * 2.0)) <= np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )) - ((((data["minbatch"]) + (np.conjugate(data["meanbatch_slices2_msignal"])))/2.0))))))))) +

                            0.100000*np.tanh(np.real((((-((np.where(np.abs(((data["maxbatch_msignal"]) / 2.0)) <= np.abs(((data["maxbatch_msignal"]) / 2.0)),complex(1.), complex(0.) ))))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.where(np.abs(data["medianbatch_msignal"]) > np.abs(np.where(np.abs(data["signal_shift_+1"]) <= np.abs(np.where(np.abs(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(np.where(np.abs(np.where(np.abs(np.conjugate(np.conjugate(data["abs_maxbatch_msignal"]))) > np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(((((((complex(2.0)) / 2.0)) * ((-((np.conjugate(data["abs_avgbatch_msignal"]))))))) * 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(complex(5.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["mean_abs_chgbatch_slices2"]) * (np.cos((np.where(np.abs(np.tanh((complex(3.0)))) <= np.abs(np.cos((((data["abs_avgbatch_slices2"]) * (data["minbatch_slices2"]))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.where(np.abs(np.where(np.abs(data["signal"]) <= np.abs(np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs((-((np.tanh((((np.where(np.abs(data["signal"]) <= np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) )) + (data["signal_shift_-1_msignal"])))))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) <= np.abs(complex(-1.0)),complex(1.), complex(0.) )) <= np.abs(np.sin((data["signal"]))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin(((((np.cos((complex(0,1)*np.conjugate(((data["signal_shift_+1"]) * 2.0))))) + (np.sin((complex(-3.0)))))/2.0)))))) +

                            0.100000*np.tanh(np.real(((((((np.tanh((complex(-2.0)))) * 2.0)) * (np.conjugate(data["maxtominbatch_msignal"])))) - (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) / ((((np.where(np.abs(complex(0,1)*np.conjugate(complex(0.0))) <= np.abs(np.where(np.abs(complex(0,1)*np.conjugate(data["meanbatch_msignal"])) <= np.abs((((((data["meanbatch_msignal"]) + ((((np.where(np.abs(complex(0.0)) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) + (data["meanbatch_msignal"]))/2.0)))/2.0)) - (complex(0.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (((data["meanbatch_msignal"]) + (complex(0.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real((((((data["maxtominbatch_slices2_msignal"]) - (data["maxbatch_slices2_msignal"]))) + (complex(-2.0)))/2.0))) +

                            0.100000*np.tanh(np.real((((((data["mean_abs_chgbatch_msignal"]) + (data["rangebatch_slices2"]))/2.0)) * (np.sin(((-(((((data["minbatch_msignal"]) + (((np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(complex(0.0)) > np.abs(complex(0,1)*np.conjugate(complex(2.0))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (data["mean_abs_chgbatch_msignal"]))))/2.0)))))))))) +

                            0.100000*np.tanh(np.real((((((np.conjugate(data["abs_maxbatch_slices2_msignal"])) - (np.conjugate(np.cos((data["abs_minbatch_msignal"])))))) + (data["signal_shift_+1"]))/2.0))) +

                            0.100000*np.tanh(np.real(data["maxtominbatch_slices2"])) +

                            0.100000*np.tanh(np.real(np.cos(((((data["rangebatch_msignal"]) + ((((data["maxbatch_slices2"]) + ((((((data["rangebatch_msignal"]) + (((((((((data["minbatch_slices2_msignal"]) + (((((np.where(np.abs(complex(6.0)) <= np.abs(complex(-1.0)),complex(1.), complex(0.) )) / 2.0)) / 2.0)))/2.0)) / 2.0)) + ((((data["abs_maxbatch"]) + (np.tanh((np.sin((((data["mean_abs_chgbatch_slices2"]) + (data["minbatch_slices2_msignal"]))))))))/2.0)))/2.0)))/2.0)) / 2.0)))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxtominbatch_msignal"]) / (np.cos((np.where(np.abs(data["stdbatch_msignal"]) > np.abs(np.cos((np.cos((np.sin((np.cos(((-((complex(0,1)*np.conjugate(np.sin((np.tanh((complex(-2.0)))))))))))))))))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch_msignal"]) + (np.where(np.abs(np.cos((((np.tanh((data["medianbatch_slices2_msignal"]))) + (data["maxbatch_slices2"]))))) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )))) + ((-(((-((data["maxbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.cos((np.cos((np.sin((np.sin((((data["signal_shift_+1_msignal"]) - (np.cos((np.cos((np.cos((np.tanh((((complex(3.0)) * 2.0))))))))))))))))))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1"]) - (((data["abs_avgbatch_slices2"]) + (np.where(np.abs(np.where(np.abs(np.sin((data["maxtominbatch_msignal"]))) > np.abs(np.where(np.abs(((data["signal_shift_+1"]) - (complex(2.0)))) > np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) > np.abs(np.cos((((data["maxtominbatch_slices2"]) / 2.0)))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((-((np.where(np.abs((-((np.where(np.abs(np.cos((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(complex(0.05043984577059746)))))) <= np.abs(np.cos((complex(0.0)))),complex(1.), complex(0.) ))))) <= np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) ))))) * (data["minbatch"])))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["minbatch_msignal"]) - (complex(3.82731294631958008)))))) * (((data["maxbatch_msignal"]) + (np.cos((complex(0,1)*np.conjugate(((data["abs_avgbatch_msignal"]) - (((data["meanbatch_msignal"]) - (np.sin((data["meanbatch_msignal"]))))))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["mean_abs_chgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(0,1)*np.conjugate(data["minbatch"])) <= np.abs(np.tanh((((data["stdbatch_msignal"]) - (complex(-3.0)))))),complex(1.), complex(0.) )) - (np.where(np.abs(((complex(0,1)*np.conjugate(np.tanh((np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(((np.tanh((data["signal_shift_+1"]))) + (data["abs_maxbatch"]))),complex(1.), complex(0.) ))))) / 2.0)) <= np.abs(np.tanh(((-((complex(0,1)*np.conjugate(data["stdbatch_slices2"]))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((complex(4.44991111755371094)) * (((data["maxtominbatch_msignal"]) + (data["rangebatch_msignal"]))))) - (np.sin((((np.conjugate(np.sin((data["rangebatch_msignal"])))) * (complex(4.44991111755371094))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) - ((((np.cos((np.where(np.abs(np.conjugate(((data["abs_maxbatch_msignal"]) * 2.0))) > np.abs((((((np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )) + ((-((np.tanh((np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["maxtominbatch"]),complex(1.), complex(0.) ))))))))/2.0)) * 2.0)),complex(1.), complex(0.) )))) + (data["signal_shift_-1"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["meanbatch_slices2_msignal"]) * 2.0)) <= np.abs(np.where(np.abs(((((data["signal_shift_-1"]) + (np.where(np.abs(np.where(np.abs(((data["rangebatch_msignal"]) / 2.0)) <= np.abs(np.sin((complex(1.0)))),complex(1.), complex(0.) )) <= np.abs(np.conjugate(np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs((-((data["meanbatch_slices2_msignal"])))),complex(1.), complex(0.) ))),complex(1.), complex(0.) )))) / 2.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((np.conjugate(((((np.cos((data["signal_shift_-1_msignal"]))) * (((np.where(np.abs(np.conjugate(data["maxbatch_msignal"])) <= np.abs(np.where(np.abs(data["maxtominbatch_msignal"]) <= np.abs((-((data["signal_shift_-1"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["maxtominbatch"]) / 2.0)) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["minbatch_slices2_msignal"]))) / (((np.conjugate(complex(0.0))) - (np.sin((complex(1.0)))))))) - (np.cos((data["minbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(complex(0.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((np.sin((data["maxtominbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(((((complex(-2.0)) + (np.where(np.abs(np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) > np.abs(np.where(np.abs(np.tanh((data["meanbatch_msignal"]))) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))) + (np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(((complex(1.42254388332366943)) + (complex(-1.0)))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs((((((((((data["abs_minbatch_msignal"]) / 2.0)) / 2.0)) + (data["rangebatch_slices2"]))/2.0)) * 2.0)),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_msignal"]) + (data["medianbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])) > np.abs(np.where(np.abs(((data["rangebatch_msignal"]) * (data["maxtominbatch_slices2_msignal"]))) <= np.abs(complex(1.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxtominbatch_msignal"]) + (np.where(np.abs(data["signal_shift_-1_msignal"]) > np.abs(np.where(np.abs(((np.cos((data["maxtominbatch_msignal"]))) + (np.sin((np.where(np.abs(np.cos((complex(-1.0)))) <= np.abs(((np.sin((((data["maxtominbatch_msignal"]) + (np.where(np.abs(((complex(0.0)) + ((-((data["mean_abs_chgbatch_slices2"])))))) > np.abs(data["maxtominbatch_msignal"]),complex(1.), complex(0.) )))))) * 2.0)),complex(1.), complex(0.) )))))) > np.abs(complex(-3.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["abs_minbatch_slices2_msignal"]) - (data["abs_maxbatch"]))) - (((complex(0,1)*np.conjugate(((data["rangebatch_slices2_msignal"]) + (data["signal_shift_+1_msignal"])))) + (np.sin(((((data["abs_minbatch_slices2"]) + (data["rangebatch_slices2_msignal"]))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate((-((data["signal_shift_+1_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((complex(0.0)) * ((-((((((data["medianbatch_msignal"]) + (np.tanh((((data["maxbatch_msignal"]) * 2.0)))))) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.sin((complex(0,1)*np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(np.sin((np.where(np.abs(data["minbatch"]) > np.abs(np.sin((complex(0,1)*np.conjugate(np.sin((data["maxbatch_slices2"])))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(data["maxbatch_msignal"])) <= np.abs(np.cos((data["signal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((((data["mean_abs_chgbatch_slices2_msignal"]) * (complex(0,1)*np.conjugate(((complex(0.0)) + (((complex(0,1)*np.conjugate((-((data["meanbatch_slices2"]))))) - (data["stdbatch_slices2_msignal"])))))))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((((np.cos((complex(10.0)))) + ((((((((data["rangebatch_msignal"]) + (data["meanbatch_msignal"]))) + (np.sin((data["medianbatch_slices2"]))))/2.0)) - (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(((np.cos((data["rangebatch_slices2_msignal"]))) * (np.cos((data["abs_maxbatch"]))))),complex(1.), complex(0.) )))))) - (np.tanh((data["rangebatch_msignal"])))))) + (complex(2.0))))) +

                            0.100000*np.tanh(np.real((((((np.conjugate(np.where(np.abs((-(((((data["minbatch_slices2"]) + (((((np.cos((np.cos((complex(0,1)*np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs((((np.conjugate(data["medianbatch_slices2_msignal"])) + (complex(0,1)*np.conjugate(complex(3.0))))/2.0)),complex(1.), complex(0.) ))))))) / (complex(-1.0)))) / 2.0)))/2.0))))) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) ))) + (data["abs_minbatch_slices2"]))/2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(np.where(np.abs(((np.conjugate(data["maxtominbatch_slices2"])) + (np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))) > np.abs(np.where(np.abs(np.cos((data["signal_shift_+1_msignal"]))) > np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )) - (((data["signal_shift_+1_msignal"]) / (np.cos((data["abs_minbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["minbatch"]) - (data["rangebatch_msignal"]))) <= np.abs(np.cos((np.where(np.abs(((complex(0,1)*np.conjugate((((data["meanbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0))) - (data["abs_maxbatch_msignal"]))) <= np.abs(((data["maxtominbatch_slices2"]) / 2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch"]) + ((((-((np.cos((((data["signal_shift_-1_msignal"]) * 2.0))))))) * 2.0)))) + (((data["signal_shift_+1"]) + (((data["maxtominbatch"]) + (data["signal_shift_+1"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((((data["medianbatch_slices2_msignal"]) * (np.conjugate(((data["abs_maxbatch_slices2_msignal"]) - (np.sin((((np.sin((((np.sin((((data["medianbatch_slices2_msignal"]) * (np.conjugate(((data["abs_maxbatch_slices2_msignal"]) - (np.tanh((data["abs_maxbatch_slices2_msignal"])))))))))) * 2.0)))) * (data["medianbatch_slices2_msignal"]))))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["maxbatch_msignal"]))) <= np.abs(complex(0.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(((data["maxbatch_slices2_msignal"]) - (np.conjugate(complex(1.0))))) <= np.abs(((complex(1.0)) / 2.0)),complex(1.), complex(0.) )) <= np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.conjugate((-((np.tanh((np.cos((((data["medianbatch_msignal"]) / 2.0)))))))))))) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.sin((data["maxtominbatch_slices2_msignal"]))))) > np.abs(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(data["medianbatch_msignal"]) > np.abs(((np.cos((data["maxtominbatch"]))) / 2.0)),complex(1.), complex(0.) )) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.conjugate(((data["maxbatch_msignal"]) * 2.0))) <= np.abs(np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0)))))),complex(1.), complex(0.) )) + ((-((((((((np.sin((((((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(-1.0)) <= np.abs(complex(0,1)*np.conjugate(((complex(8.71560096740722656)) * (np.where(np.abs(((np.tanh((complex(0,1)*np.conjugate(data["abs_minbatch_slices2_msignal"])))) * (complex(0,1)*np.conjugate(complex(1.0))))) > np.abs(complex(7.0)),complex(1.), complex(0.) ))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.cos(((((complex(0,1)*np.conjugate(((np.conjugate(np.tanh(((((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(complex(-3.0)),complex(1.), complex(0.) )) + (np.where(np.abs(data["medianbatch_msignal"]) > np.abs(np.sin((complex(-1.0)))),complex(1.), complex(0.) )))/2.0))))) * 2.0))) + (data["minbatch"]))/2.0)))),complex(1.), complex(0.) )) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((data["maxbatch_msignal"]))) * (((data["medianbatch_slices2"]) - (np.sin((((np.sin((data["signal"]))) - (np.sin((data["medianbatch_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(complex(2.0)) > np.abs(complex(0,1)*np.conjugate(np.cos((complex(10.89035797119140625))))),complex(1.), complex(0.) )) <= np.abs(np.tanh((complex(0.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((np.cos((np.where(np.abs(data["maxbatch_msignal"]) > np.abs((((((((data["abs_maxbatch_slices2"]) - (np.conjugate(data["meanbatch_slices2"])))) + (np.conjugate((((complex(0,1)*np.conjugate(np.tanh((np.where(np.abs(np.conjugate(complex(2.0))) <= np.abs(((np.cos((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )))) / 2.0)),complex(1.), complex(0.) ))))) + (data["medianbatch_slices2"]))/2.0))))/2.0)) - (data["meanbatch_slices2"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((((data["abs_maxbatch"]) * (np.cos((((((data["rangebatch_msignal"]) * 2.0)) - (((data["abs_maxbatch_slices2"]) * 2.0)))))))) + (np.cos((((data["medianbatch_msignal"]) - (((data["rangebatch_msignal"]) * 2.0)))))))/2.0))) +

                            0.100000*np.tanh(np.real((((((np.cos((data["abs_maxbatch_slices2"]))) + (np.where(np.abs(data["meanbatch_slices2_msignal"]) <= np.abs(((np.sin((complex(0,1)*np.conjugate(complex(1.0))))) / 2.0)),complex(1.), complex(0.) )))/2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.cos((np.where(np.abs(data["minbatch_slices2_msignal"]) > np.abs(np.tanh((np.tanh((data["abs_maxbatch_slices2_msignal"]))))),complex(1.), complex(0.) )))) * 2.0)) <= np.abs(np.cos((np.cos((((data["medianbatch_slices2"]) - (data["maxbatch_slices2"]))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.conjugate(complex(0,1)*np.conjugate(data["abs_maxbatch"]))) > np.abs(data["maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(complex(-3.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((((data["signal_shift_+1_msignal"]) / (((np.conjugate(np.cos((((np.sin((data["meanbatch_msignal"]))) * 2.0))))) / 2.0)))) + (complex(0,1)*np.conjugate(data["signal_shift_+1_msignal"])))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])),complex(1.), complex(0.) )) > np.abs(np.where(np.abs((((data["maxbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))/2.0)) <= np.abs(((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(np.cos((((np.tanh((complex(2.0)))) * 2.0)))),complex(1.), complex(0.) )) * 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((((data["maxtominbatch_slices2_msignal"]) - ((((((data["minbatch"]) + (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(((data["maxtominbatch_slices2_msignal"]) / 2.0))) <= np.abs(((data["maxtominbatch_slices2_msignal"]) + (np.cos((np.sin((complex(3.0)))))))),complex(1.), complex(0.) )))))) + (((np.cos((data["maxtominbatch_slices2_msignal"]))) / 2.0)))/2.0))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["meanbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.real(np.sin((((((data["meanbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate((((-((complex(0,1)*np.conjugate(np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(np.sin(((((-((((data["meanbatch_msignal"]) * 2.0))))) * 2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) / 2.0))) > np.abs(complex(0,1)*np.conjugate(np.sin((data["minbatch_msignal"])))),complex(1.), complex(0.) ))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["signal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.sin((((np.where(np.abs(np.sin((((((complex(3.0)) / 2.0)) * 2.0)))) > np.abs(complex(3.0)),complex(1.), complex(0.) )) * (((data["medianbatch_slices2_msignal"]) - (data["meanbatch_slices2_msignal"]))))))) > np.abs(complex(1.0)),complex(1.), complex(0.) )) > np.abs(((((((np.sin((data["medianbatch_slices2_msignal"]))) / 2.0)) + (np.sin((data["meanbatch_slices2_msignal"]))))) + (data["stdbatch_slices2_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(1.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(np.tanh((((((data["signal_shift_-1_msignal"]) - ((-((data["signal_shift_-1_msignal"])))))) - (((data["signal_shift_+1_msignal"]) * 2.0))))))) > np.abs(data["maxtominbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(np.sin((np.cos((np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(complex(0,1)*np.conjugate((-((((data["signal_shift_-1"]) + ((((((complex(-1.0)) + (complex(0.26934033632278442)))) + (complex(-2.0)))/2.0)))))))),complex(1.), complex(0.) )))))) > np.abs(complex(-2.0)),complex(1.), complex(0.) )) <= np.abs(complex(1.0)),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate((-((np.where(np.abs(((data["abs_maxbatch_slices2_msignal"]) - (((complex(0.0)) / 2.0)))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) + (data["meanbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.tanh((((data["abs_minbatch_slices2_msignal"]) / (complex(2.0))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.tanh((np.sin(((((np.where(np.abs(complex(0,1)*np.conjugate(data["abs_minbatch_msignal"])) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (np.conjugate(data["minbatch"])))/2.0)))))))) +

                            0.100000*np.tanh(np.real((((complex(0.0)) + (np.tanh((np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(np.where(np.abs(np.conjugate(data["abs_avgbatch_slices2"])) > np.abs(np.tanh((np.where(np.abs(complex(1.0)) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))/2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["abs_maxbatch_msignal"]) + ((((data["abs_avgbatch_msignal"]) + (((data["meanbatch_msignal"]) - (complex(0,1)*np.conjugate(complex(-1.0))))))/2.0)))/2.0)))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.where(np.abs(complex(-3.0)) <= np.abs(((((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(complex(7.0)) > np.abs(complex(-1.0)),complex(1.), complex(0.) )))) * (complex(6.0)))) / 2.0)),complex(1.), complex(0.) )) <= np.abs(np.sin((data["signal_shift_+1_msignal"]))),complex(1.), complex(0.) )) > np.abs(data["abs_minbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.cos((data["abs_maxbatch_slices2"]))) * (np.conjugate(np.where(np.abs(((np.where(np.abs(np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(complex(0.0)),complex(1.), complex(0.) )) <= np.abs(np.cos((np.sin((((data["minbatch_msignal"]) * ((((np.sin((data["meanbatch_slices2_msignal"]))) + (np.sin((complex(0.0)))))/2.0)))))))),complex(1.), complex(0.) )) + (complex(-2.0)))) > np.abs(((data["abs_avgbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(np.cos((((np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["maxbatch_msignal"])),complex(1.), complex(0.) )) * (np.cos((data["maxbatch_msignal"]))))))) <= np.abs((-((data["medianbatch_msignal"])))),complex(1.), complex(0.) )) * ((-((np.where(np.abs(data["rangebatch_msignal"]) <= np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) ))))))) - ((((-((data["medianbatch_msignal"])))) - (data["maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["signal_shift_+1"]) > np.abs(((data["rangebatch_slices2"]) + (complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(np.cos(((((np.tanh((np.cos((np.where(np.abs(np.tanh((data["maxbatch_msignal"]))) <= np.abs(np.where(np.abs(data["signal_shift_-1_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["abs_avgbatch_slices2_msignal"])),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) + (complex(0,1)*np.conjugate(data["maxbatch_msignal"])))/2.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["stdbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["medianbatch_msignal"]) <= np.abs((((data["abs_minbatch_slices2"]) + (np.conjugate(np.sin(((((np.where(np.abs(data["maxbatch_slices2"]) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) + (np.conjugate(complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.cos((data["mean_abs_chgbatch_slices2"])))) > np.abs(complex(-1.0)),complex(1.), complex(0.) )))))/2.0))))))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(complex(0.0))) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs((-((data["abs_maxbatch_slices2"])))) > np.abs(complex(1.0)),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))))) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(np.tanh((np.where(np.abs(complex(0,1)*np.conjugate(np.tanh(((((((data["medianbatch_msignal"]) + (data["abs_maxbatch_msignal"]))/2.0)) - (complex(0,1)*np.conjugate(data["signal"]))))))) > np.abs(np.where(np.abs(complex(2.0)) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2"]) * (np.cos(((-((((data["medianbatch_msignal"]) + (np.cos((np.sin(((((np.sin(((((np.where(np.abs(np.cos((np.cos((complex(0.0)))))) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) + (data["medianbatch_msignal"]))/2.0)))) + (data["medianbatch_msignal"]))/2.0)))))))))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate(np.sin((data["signal_shift_+1"])))) / 2.0)))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs((-((((np.where(np.abs((-(((-((np.cos((np.conjugate(((data["abs_maxbatch_slices2"]) / 2.0))))))))))) <= np.abs(complex(-1.0)),complex(1.), complex(0.) )) * (complex(0.0))))))),complex(1.), complex(0.) )) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((data["abs_avgbatch_slices2"]) + (np.conjugate(data["mean_abs_chgbatch_slices2_msignal"])))/2.0)) <= np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) ))))  

    

    def GP_class_9(self,data):

        return self.Output( -3.603840 +

                            0.100000*np.tanh(np.real(((((data["meanbatch_slices2"]) * 2.0)) - (((((np.where(np.abs(complex(-1.0)) > np.abs(complex(0.0)),complex(1.), complex(0.) )) + ((((-((np.where(np.abs(data["maxtominbatch"]) <= np.abs(np.conjugate(data["signal"])),complex(1.), complex(0.) ))))) * 2.0)))) - (data["maxbatch_slices2"])))))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_-1"]) * (data["rangebatch_slices2_msignal"]))) - (np.conjugate(np.where(np.abs(((data["abs_avgbatch_slices2"]) + (data["meanbatch_slices2"]))) > np.abs(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((-((data["minbatch_msignal"])))) <= np.abs(np.conjugate(data["medianbatch_slices2"])),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-(((-(((-((data["abs_avgbatch_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.conjugate(((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2"]) <= np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )) <= np.abs(np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )) > np.abs(np.sin(((-((data["minbatch"])))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0))))) +

                            0.100000*np.tanh(np.real((((complex(1.0)) + (((((data["signal_shift_+1"]) + (np.where(np.abs(np.where(np.abs((-(((((np.cos((complex(0.0)))) + (np.conjugate(((data["signal_shift_+1"]) + (data["signal_shift_-1"])))))/2.0))))) > np.abs(complex(3.0)),complex(1.), complex(0.) )) <= np.abs((((data["signal_shift_+1"]) + (complex(3.0)))/2.0)),complex(1.), complex(0.) )))) / 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate((((data["medianbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0))) / 2.0)) - (data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(data["signal"])) +

                            0.100000*np.tanh(np.real(((data["abs_avgbatch_slices2_msignal"]) / (complex(-3.0))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.sin(((-((((((((((data["medianbatch_msignal"]) / (data["abs_maxbatch"]))) + (data["rangebatch_slices2"]))) * 2.0)) * (np.tanh((np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) ))))))))))) + (complex(1.0)))))) +

                            0.100000*np.tanh(np.real(((((np.conjugate(data["signal_shift_+1"])) + (((((-((((data["minbatch_msignal"]) - (data["signal_shift_-1_msignal"])))))) + (((data["abs_minbatch_slices2_msignal"]) * (complex(-2.0)))))/2.0)))) - (((((complex(3.0)) - (((np.conjugate(data["abs_avgbatch_slices2"])) * (data["signal_shift_+1"]))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(((((data["abs_minbatch_slices2"]) - ((-((((data["mean_abs_chgbatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"])))))))) - ((-(((-((data["medianbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(data["minbatch_msignal"])) / ((((data["meanbatch_slices2_msignal"]) + (np.where(np.abs(np.conjugate(data["minbatch_msignal"])) > np.abs(np.where(np.abs(((np.conjugate(complex(0,1)*np.conjugate(complex(-2.0)))) + (((np.conjugate((-((data["rangebatch_msignal"]))))) + ((-((data["maxtominbatch"])))))))) <= np.abs((-((data["minbatch_msignal"])))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["minbatch_slices2_msignal"]) + (np.cos((((((data["signal_shift_+1"]) * 2.0)) - (data["minbatch_slices2_msignal"]))))))/2.0))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(complex(0.0)) <= np.abs((-((np.sin((data["maxtominbatch_slices2"])))))),complex(1.), complex(0.) )) - (data["abs_minbatch_msignal"]))) * (((((data["meanbatch_slices2"]) + (complex(-2.0)))) * ((-((data["abs_avgbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(np.real((-((((((data["meanbatch_msignal"]) * 2.0)) - ((-((((data["medianbatch_slices2_msignal"]) + (complex(0,1)*np.conjugate(data["abs_avgbatch_msignal"]))))))))))))) +

                            0.100000*np.tanh(np.real((-((((data["abs_avgbatch_msignal"]) + (np.cos((np.where(np.abs(data["abs_avgbatch_msignal"]) <= np.abs(np.where(np.abs(np.cos((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(complex(-2.0)),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate(data["abs_avgbatch_msignal"])),complex(1.), complex(0.) )))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(data["signal"])) +

                            0.100000*np.tanh(np.real(((((np.sin((((data["minbatch_slices2_msignal"]) / 2.0)))) * 2.0)) + (np.conjugate(((((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0)) * (np.sin((((data["abs_maxbatch"]) - (complex(0.88148498535156250)))))))))))) +

                            0.100000*np.tanh(np.real(((((complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"])) - (data["medianbatch_slices2"]))) - (data["stdbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.where(np.abs(np.cos((((((data["abs_avgbatch_slices2"]) / (np.cos((data["stdbatch_slices2"]))))) * 2.0)))) <= np.abs(((data["medianbatch_slices2_msignal"]) / 2.0)),complex(1.), complex(0.) )))) - (((data["abs_avgbatch_slices2"]) / (np.cos((data["medianbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.conjugate(complex(0,1)*np.conjugate(complex(8.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real((-((((np.where(np.abs((((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0)) > np.abs((((((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_msignal"]))) + (complex(0,1)*np.conjugate(data["stdbatch_msignal"])))/2.0)),complex(1.), complex(0.) )) + (((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real((((((np.conjugate(data["mean_abs_chgbatch_slices2_msignal"])) / (np.cos((data["meanbatch_msignal"]))))) + (np.where(np.abs(((data["signal_shift_-1"]) * (complex(-2.0)))) <= np.abs(((np.conjugate(((np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(np.conjugate(data["maxtominbatch"])),complex(1.), complex(0.) )) / 2.0))) * (data["mean_abs_chgbatch_slices2_msignal"]))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((np.tanh((complex(1.0)))) - (((((data["maxbatch_slices2_msignal"]) * (((np.cos((((data["abs_maxbatch_slices2"]) * (data["abs_avgbatch_msignal"]))))) + (data["abs_maxbatch"]))))) - (data["signal_shift_+1"])))))) +

                            0.100000*np.tanh(np.real((((((-((complex(8.11824417114257812))))) / (np.cos((data["medianbatch_msignal"]))))) + (np.where(np.abs(((((((np.cos((data["medianbatch_msignal"]))) / 2.0)) / 2.0)) / (complex(-3.0)))) > np.abs(np.tanh((data["minbatch_msignal"]))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos(((-(((((data["rangebatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2"]) - (np.cos((data["medianbatch_slices2"]))))))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) )) + (data["rangebatch_msignal"])))) + (((((((data["signal_shift_-1"]) + (np.where(np.abs(np.where(np.abs(data["stdbatch_msignal"]) > np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) )))/2.0)) + (data["rangebatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(((np.tanh((data["abs_avgbatch_msignal"]))) - (complex(-2.0))))) +

                            0.100000*np.tanh(np.real((((-((data["abs_maxbatch_slices2"])))) - (((data["medianbatch_slices2"]) - (np.tanh(((-((data["abs_avgbatch_slices2_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real(((data["signal"]) - (((np.where(np.abs(np.tanh((((np.cos((data["minbatch"]))) + (data["minbatch_slices2_msignal"]))))) <= np.abs(data["signal"]),complex(1.), complex(0.) )) + (np.cos((np.cos((np.where(np.abs(data["maxbatch_msignal"]) <= np.abs(((((data["mean_abs_chgbatch_slices2"]) / 2.0)) * 2.0)),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(data["mean_abs_chgbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((((((((((data["mean_abs_chgbatch_slices2_msignal"]) - (data["meanbatch_slices2"]))) * 2.0)) + (data["stdbatch_slices2_msignal"]))) - (((data["stdbatch_slices2"]) - (data["mean_abs_chgbatch_slices2_msignal"]))))) * (np.cos((np.conjugate(data["medianbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.real((-((((((data["medianbatch_slices2_msignal"]) + (((data["medianbatch_slices2_msignal"]) + (complex(1.0)))))) * 2.0)))))) +

                            0.100000*np.tanh(np.real((-((data["medianbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(((np.sin(((((data["minbatch_msignal"]) + (np.cos((np.where(np.abs(np.where(np.abs(np.cos((data["rangebatch_msignal"]))) > np.abs(((data["abs_avgbatch_msignal"]) * 2.0)),complex(1.), complex(0.) )) > np.abs(np.cos((((data["minbatch_msignal"]) / 2.0)))),complex(1.), complex(0.) )))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["minbatch_slices2_msignal"]) / 2.0))))) +

                            0.100000*np.tanh(np.real((-((((np.conjugate((((data["meanbatch_msignal"]) + (np.where(np.abs(((((data["meanbatch_msignal"]) * (data["medianbatch_slices2"]))) / 2.0)) <= np.abs((-((data["meanbatch_slices2"])))),complex(1.), complex(0.) )))/2.0))) * 2.0)))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.sin((data["minbatch_slices2"]))) / 2.0)) <= np.abs(data["abs_minbatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["stdbatch_msignal"])) +

                            0.100000*np.tanh(np.real(((np.sin((((complex(-2.0)) / (((data["medianbatch_slices2"]) + (((((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(((data["rangebatch_slices2"]) / 2.0)),complex(1.), complex(0.) )) * (np.cos((np.where(np.abs(data["minbatch_slices2_msignal"]) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) - (data["mean_abs_chgbatch_msignal"]))))))))) - (data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-((((np.sin(((-((np.sin((np.cos((((data["stdbatch_slices2_msignal"]) * 2.0))))))))))) + (((((((complex(3.0)) / (np.sin((data["maxtominbatch_msignal"]))))) * ((((data["abs_maxbatch_slices2"]) + (complex(0,1)*np.conjugate(data["stdbatch_slices2"])))/2.0)))) + (np.where(np.abs(complex(7.30021286010742188)) <= np.abs((-((complex(3.0))))),complex(1.), complex(0.) ))))))))))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_msignal"]) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((((data["abs_maxbatch"]) / 2.0)) / 2.0)) * (np.sin(((((((((complex(7.0)) / 2.0)) * (np.sin((np.where(np.abs(data["stdbatch_slices2"]) > np.abs(complex(-1.0)),complex(1.), complex(0.) )))))) + (data["minbatch_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(((np.conjugate(((data["abs_avgbatch_msignal"]) * 2.0))) * 2.0))) - (((data["abs_avgbatch_slices2"]) - (((((data["medianbatch_slices2"]) * 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((np.cos((data["minbatch_msignal"]))) * 2.0)))) / 2.0)) * (data["abs_avgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((((np.tanh((((np.tanh((np.tanh((((complex(0.0)) * 2.0)))))) * (np.cos((((data["meanbatch_slices2_msignal"]) - (np.conjugate(np.conjugate(np.tanh((data["abs_maxbatch_slices2"]))))))))))))) * 2.0)))) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) * ((((((np.cos((data["minbatch_slices2_msignal"]))) + (np.cos(((((((data["mean_abs_chgbatch_slices2_msignal"]) + (complex(-2.0)))/2.0)) + (np.where(np.abs(((data["medianbatch_slices2_msignal"]) * (complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"])))) <= np.abs((-((((data["minbatch_msignal"]) * 2.0))))),complex(1.), complex(0.) )))))))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(((np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"]))))) - (data["medianbatch_msignal"]))) <= np.abs(complex(-1.0)),complex(1.), complex(0.) )) + (((data["minbatch_slices2_msignal"]) * (np.sin((np.cos((data["medianbatch_msignal"]))))))))/2.0))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) / 2.0)) - (complex(3.0))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((data["abs_avgbatch_slices2"]) / 2.0)))) * 2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((data["abs_maxbatch_msignal"]))) * (np.tanh((np.where(np.abs(data["maxbatch_slices2"]) > np.abs(((np.sin(((((np.conjugate(complex(7.0))) + (np.sin((data["maxbatch_slices2"]))))/2.0)))) * (np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.sin((np.where(np.abs(np.where(np.abs(data["signal_shift_-1"]) <= np.abs(data["maxtominbatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(np.conjugate((-(((-((np.cos((np.where(np.abs(complex(2.0)) <= np.abs(((complex(1.0)) * (np.conjugate(np.where(np.abs(data["stdbatch_slices2"]) <= np.abs(np.cos((complex(2.0)))),complex(1.), complex(0.) ))))),complex(1.), complex(0.) ))))))))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_msignal"]) - (np.conjugate(((((data["abs_minbatch_slices2"]) / 2.0)) - (((np.where(np.abs(np.conjugate(data["maxbatch_msignal"])) <= np.abs(data["maxtominbatch"]),complex(1.), complex(0.) )) + (data["meanbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real((((-(((((data["meanbatch_msignal"]) + ((((data["meanbatch_slices2_msignal"]) + ((((((-((np.cos(((-((complex(-1.0)))))))))) / 2.0)) * 2.0)))/2.0)))/2.0))))) - (complex(11.62009716033935547))))) +

                            0.100000*np.tanh(np.real(((((((((complex(-3.0)) * (data["meanbatch_msignal"]))) * (data["medianbatch_slices2"]))) + (np.cos((np.conjugate(np.conjugate(complex(-2.0)))))))) - (((((((data["abs_minbatch_slices2"]) * 2.0)) * (np.conjugate(data["maxbatch_slices2"])))) - (((((complex(-1.0)) * (complex(1.0)))) - (((data["medianbatch_slices2"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) * (np.sin(((((-((np.conjugate(np.where(np.abs(data["signal_shift_+1"]) > np.abs(((complex(0,1)*np.conjugate(np.conjugate(data["abs_maxbatch_slices2_msignal"]))) / 2.0)),complex(1.), complex(0.) )))))) + (data["abs_maxbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((((data["abs_maxbatch_slices2_msignal"]) + (np.sin((((complex(-1.0)) + (np.where(np.abs(((((data["signal_shift_+1"]) + (data["abs_maxbatch_slices2_msignal"]))) - (data["abs_maxbatch_slices2_msignal"]))) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))))) * (((((data["signal_shift_+1"]) / 2.0)) - (data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(((((np.conjugate(np.where(np.abs(((data["meanbatch_msignal"]) / 2.0)) <= np.abs((-((data["meanbatch_msignal"])))),complex(1.), complex(0.) ))) + (data["meanbatch_msignal"]))) / 2.0)) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) - (data["meanbatch_msignal"]))) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((((-((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))) - (np.where(np.abs(((np.tanh(((-((((complex(0.0)) + (((data["medianbatch_slices2"]) + (data["abs_minbatch_msignal"])))))))))) / 2.0)) > np.abs(((np.tanh((((np.sin((np.sin((data["abs_maxbatch"]))))) * 2.0)))) * 2.0)),complex(1.), complex(0.) )))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((((np.conjugate(np.cos((data["minbatch_msignal"])))) + ((((((((((np.where(np.abs(data["meanbatch_msignal"]) > np.abs(((np.sin((np.sin((data["minbatch_msignal"]))))) / 2.0)),complex(1.), complex(0.) )) * 2.0)) / 2.0)) / 2.0)) + (data["abs_minbatch_slices2_msignal"]))/2.0)))/2.0)) / (np.cos((data["minbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real((((((((((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)) / 2.0)) + (complex(4.64181756973266602)))/2.0)) / (np.sin((np.sin(((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["minbatch_slices2"]))/2.0)))))))) + ((((np.sin((np.sin((data["minbatch_slices2"]))))) + ((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["minbatch_slices2"]))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.tanh((data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.conjugate(complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate(np.tanh((data["maxbatch_slices2_msignal"])))) + (((complex(0,1)*np.conjugate(((data["signal"]) * (np.tanh((complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"]))))))) + (complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"]))))))))))) +

                            0.100000*np.tanh(np.real((((data["maxbatch_msignal"]) + (complex(-1.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["abs_maxbatch_slices2_msignal"]) - (np.where(np.abs(data["signal_shift_+1"]) <= np.abs(complex(1.0)),complex(1.), complex(0.) )))))) * (((((data["rangebatch_slices2"]) + (data["signal_shift_+1"]))) * (data["abs_avgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin((np.conjugate(data["abs_minbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["rangebatch_slices2"]) + (np.where(np.abs(np.cos((np.cos((complex(0.0)))))) <= np.abs(np.cos((np.where(np.abs(((np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(data["rangebatch_slices2"]) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0)) <= np.abs(complex(0,1)*np.conjugate(data["maxbatch_slices2_msignal"])),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real((((-((data["minbatch"])))) + (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch"]) - (((data["rangebatch_msignal"]) * (np.cos((data["minbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.cos((data["abs_maxbatch_msignal"]))))) * (data["stdbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(data["maxtominbatch_slices2"])) +

                            0.100000*np.tanh(np.real(np.cos(((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(np.where(np.abs(((data["abs_avgbatch_slices2"]) - (complex(2.0)))) > np.abs(np.cos((np.sin((np.tanh((data["rangebatch_slices2"]))))))),complex(1.), complex(0.) ))))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((complex(-1.0)) / 2.0)))) +

                            0.100000*np.tanh(np.real(data["maxbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((((np.cos(((-((data["abs_minbatch_slices2_msignal"])))))) - (np.cos(((((data["maxtominbatch_slices2"]) + (((complex(0,1)*np.conjugate(np.tanh((complex(-3.0))))) * 2.0)))/2.0)))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((complex(-3.0)))) + (np.where(np.abs(np.where(np.abs(complex(4.0)) <= np.abs(complex(10.0)),complex(1.), complex(0.) )) <= np.abs((((((-((data["abs_minbatch_msignal"])))) - (complex(-2.0)))) / 2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((((complex(-3.0)) + (((np.cos((((((np.where(np.abs(((np.where(np.abs(data["signal_shift_-1"]) > np.abs(data["minbatch"]),complex(1.), complex(0.) )) / 2.0)) <= np.abs(((data["minbatch_msignal"]) * (data["signal_shift_-1"]))),complex(1.), complex(0.) )) / 2.0)) + (data["minbatch_msignal"]))))) * (((np.cos((data["signal_shift_+1_msignal"]))) / 2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(data["minbatch_msignal"])) +

                            0.100000*np.tanh(np.real(np.cos((((((data["maxbatch_slices2_msignal"]) + (((np.conjugate(np.cos((complex(0,1)*np.conjugate(complex(0.0)))))) - (complex(0,1)*np.conjugate(np.sin((((np.sin((complex(2.0)))) * (data["abs_maxbatch_msignal"])))))))))) * 2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(data["stdbatch_msignal"])) > np.abs(np.cos((complex(9.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((((complex(0,1)*np.conjugate(data["medianbatch_msignal"])) + (complex(0,1)*np.conjugate(np.where(np.abs(complex(0,1)*np.conjugate(np.sin((data["maxbatch_slices2"])))) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) ))))/2.0)))) +

                            0.100000*np.tanh(np.real(complex(2.0))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) / (np.cos(((((data["minbatch_msignal"]) + ((((complex(0,1)*np.conjugate(np.cos((data["signal_shift_+1_msignal"])))) + (((((((np.cos((data["medianbatch_slices2_msignal"]))) + (data["minbatch_msignal"]))/2.0)) + (np.conjugate(complex(0,1)*np.conjugate(((np.cos((data["rangebatch_msignal"]))) * 2.0)))))/2.0)))/2.0)))/2.0))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.sin((data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(((((data["minbatch"]) * (data["minbatch"]))) - (complex(14.62082481384277344))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.cos((np.sin((np.conjugate(((data["rangebatch_msignal"]) + (data["meanbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(np.real(((((data["stdbatch_slices2_msignal"]) * (np.cos(((-((data["minbatch_msignal"])))))))) * ((((complex(11.17887783050537109)) + (np.where(np.abs(np.cos((((data["stdbatch_slices2_msignal"]) * (complex(11.17887783050537109)))))) <= np.abs((((((data["mean_abs_chgbatch_msignal"]) + (complex(0,1)*np.conjugate(data["minbatch_msignal"])))/2.0)) - (data["minbatch_msignal"]))),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real((((((np.cos((((np.cos((np.conjugate(np.conjugate(data["rangebatch_slices2_msignal"]))))) * (data["mean_abs_chgbatch_msignal"]))))) * (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))) + (data["signal_shift_-1_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(data["signal_shift_-1_msignal"])) +

                            0.100000*np.tanh(np.real(np.conjugate(np.conjugate(np.tanh((np.sin((np.conjugate(data["abs_maxbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["medianbatch_slices2"]) * (data["rangebatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.conjugate(complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate(data["medianbatch_msignal"]))))))) * (np.cos((complex(0,1)*np.conjugate(data["medianbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.real(np.cos(((-((((data["abs_maxbatch"]) + (np.sin((np.cos(((-((data["abs_maxbatch_slices2_msignal"])))))))))))))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.cos((complex(4.0)))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(np.tanh((np.conjugate(complex(-2.0))))),complex(1.), complex(0.) )) + (((((complex(0,1)*np.conjugate(((data["signal_shift_-1_msignal"]) * 2.0))) / 2.0)) / 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["abs_maxbatch_msignal"]) * 2.0)))) - (data["stdbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.tanh((((data["signal_shift_+1_msignal"]) * (((np.conjugate(((complex(1.0)) / 2.0))) / 2.0))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(np.cos((complex(-1.0)))) > np.abs(np.where(np.abs(np.sin((complex(0.0)))) <= np.abs(((((data["rangebatch_slices2_msignal"]) + (complex(1.0)))) + (data["rangebatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((-((np.tanh((np.cos((np.where(np.abs(data["minbatch"]) > np.abs(complex(0,1)*np.conjugate(data["minbatch_msignal"])),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(np.where(np.abs(((((data["medianbatch_slices2_msignal"]) - (((((data["medianbatch_slices2"]) / 2.0)) + (np.where(np.abs(data["minbatch_msignal"]) <= np.abs((((data["rangebatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))))) * (((((complex(1.0)) / 2.0)) * (data["abs_minbatch_slices2_msignal"]))))) > np.abs(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(np.sin((data["signal_shift_+1"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) - (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2_msignal"]) + (np.conjugate(data["meanbatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(np.sin((((complex(0,1)*np.conjugate(np.where(np.abs(data["rangebatch_msignal"]) > np.abs((((np.sin((complex(-1.0)))) + (((np.sin((complex(-1.0)))) * 2.0)))/2.0)),complex(1.), complex(0.) ))) - (np.cos((data["minbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real((((-((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"]))))) - ((((data["mean_abs_chgbatch_slices2"]) + ((((data["rangebatch_slices2"]) + ((((((-((np.conjugate(complex(-3.0)))))) + ((-((((data["stdbatch_slices2_msignal"]) * ((-((((np.tanh((np.sin((data["abs_avgbatch_slices2"]))))) / 2.0)))))))))))) / 2.0)))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) * (((np.where(np.abs(((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(((np.where(np.abs(np.conjugate(np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) ))) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )) + (data["abs_maxbatch_slices2_msignal"]))),complex(1.), complex(0.) )) - (data["abs_maxbatch_slices2_msignal"]))) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) - (data["abs_maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin((data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_-1"]) + (complex(2.0)))) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin((((((data["medianbatch_msignal"]) * 2.0)) - (data["signal_shift_+1_msignal"]))))) * (((((data["signal_shift_-1"]) * 2.0)) + (np.cos((((np.where(np.abs(np.sin((data["abs_minbatch_slices2"]))) > np.abs(np.sin((np.sin((complex(10.0)))))),complex(1.), complex(0.) )) * (np.cos((((((data["medianbatch_msignal"]) * 2.0)) - (data["signal_shift_+1_msignal"])))))))))))))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_-1"]) + (((np.where(np.abs(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(data["minbatch_msignal"]),complex(1.), complex(0.) )) > np.abs(complex(-1.0)),complex(1.), complex(0.) )) - (((np.where(np.abs((((data["abs_maxbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)) <= np.abs((((data["maxtominbatch"]) + (data["abs_maxbatch_msignal"]))/2.0)),complex(1.), complex(0.) )) * 2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(data["signal"]) > np.abs(data["rangebatch_slices2_msignal"]),complex(1.), complex(0.) )) > np.abs(np.where(np.abs(data["minbatch_msignal"]) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(data["signal_shift_-1"]) <= np.abs(((data["rangebatch_slices2_msignal"]) * ((((data["abs_maxbatch"]) + (data["medianbatch_msignal"]))/2.0)))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(complex(0,1)*np.conjugate(((complex(-3.0)) / 2.0))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((data["minbatch_slices2"]) * (np.sin((data["stdbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(np.sin((((np.conjugate(data["signal_shift_+1"])) * (np.where(np.abs(data["abs_maxbatch"]) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.tanh((complex(0.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((((((data["abs_minbatch_msignal"]) - (((data["abs_minbatch_slices2"]) / 2.0)))) * (np.cos((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["meanbatch_slices2"]))))))) + (np.cos((data["abs_minbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real((((((data["signal_shift_-1_msignal"]) + (complex(1.0)))/2.0)) / (complex(1.0))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_slices2_msignal"]) / (np.cos((((data["abs_maxbatch_slices2_msignal"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(data["abs_avgbatch_slices2"])) - (np.cos((((data["abs_avgbatch_slices2"]) + (((data["stdbatch_msignal"]) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.cos((np.where(np.abs(np.cos((np.cos((data["signal_shift_+1"]))))) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((data["abs_minbatch_slices2_msignal"]))) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((((data["abs_minbatch_slices2_msignal"]) + ((((((data["abs_avgbatch_msignal"]) * (data["signal_shift_-1"]))) + (((data["mean_abs_chgbatch_slices2"]) - (np.cos((np.where(np.abs(((((data["abs_avgbatch_msignal"]) / 2.0)) / 2.0)) > np.abs(((data["stdbatch_slices2_msignal"]) - (((np.where(np.abs(np.cos((data["mean_abs_chgbatch_slices2"]))) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )) - (data["minbatch_msignal"]))))),complex(1.), complex(0.) )))))))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1_msignal"]) / 2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) - (((data["medianbatch_slices2_msignal"]) + (np.where(np.abs(np.where(np.abs(((complex(12.21069145202636719)) + (data["medianbatch_slices2_msignal"]))) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((((((data["maxbatch_msignal"]) + (((data["meanbatch_slices2"]) - (data["signal_shift_+1_msignal"]))))) + (data["maxtominbatch_slices2"]))/2.0))) +

                            0.100000*np.tanh(np.real(np.sin((((np.sin((complex(0,1)*np.conjugate(np.tanh(((((complex(0,1)*np.conjugate(((data["maxbatch_slices2"]) / 2.0))) + (np.sin((((np.sin((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate((((complex(0,1)*np.conjugate(complex(2.0))) + (np.cos((data["abs_minbatch_slices2"]))))/2.0)))))) / 2.0)))))/2.0))))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.cos((data["signal_shift_+1"])))) +

                            0.100000*np.tanh(np.real(np.cos((np.cos((np.cos((data["minbatch"])))))))) +

                            0.100000*np.tanh(np.real(((((((data["medianbatch_slices2"]) + (data["maxtominbatch"]))/2.0)) + (data["signal_shift_-1_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1_msignal"])) +

                            0.100000*np.tanh(np.real(((((np.sin((np.sin((np.sin((np.cos((np.sin((np.sin((complex(-1.0)))))))))))))) / 2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(3.0)))) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((((np.cos((((data["maxbatch_msignal"]) * 2.0)))) - (complex(0,1)*np.conjugate((-((complex(0.0)))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((-((complex(0,1)*np.conjugate(complex(3.0)))))) <= np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(((data["rangebatch_slices2"]) * 2.0)) > np.abs(complex(0,1)*np.conjugate((((-((((((np.conjugate(data["abs_maxbatch_msignal"])) + (np.cos((np.cos((data["minbatch"]))))))) * 2.0))))) * (complex(0,1)*np.conjugate(((np.sin(((-((data["maxtominbatch_msignal"])))))) / 2.0)))))),complex(1.), complex(0.) )) > np.abs(data["minbatch_slices2_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((-((((complex(2.0)) * 2.0))))) / 2.0))) +

                            0.100000*np.tanh(np.real(complex(2.0))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch"]) - (data["rangebatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(((data["signal"]) + (np.conjugate(np.where(np.abs(np.cos((data["maxbatch_msignal"]))) > np.abs(((np.sin((data["medianbatch_msignal"]))) + (((complex(-2.0)) + (np.conjugate(np.sin((np.where(np.abs(((data["medianbatch_slices2"]) / 2.0)) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) ))))))))),complex(1.), complex(0.) )))))))/2.0))))) +

                            0.100000*np.tanh(np.real((((((((data["medianbatch_msignal"]) + (((np.sin(((((-((complex(4.0))))) + (((((data["mean_abs_chgbatch_slices2"]) - (((data["signal_shift_+1_msignal"]) - (np.sin((data["rangebatch_slices2"]))))))) / 2.0)))))) - (data["mean_abs_chgbatch_msignal"]))))/2.0)) * 2.0)) + (((np.sin((data["rangebatch_msignal"]))) + (data["maxtominbatch"])))))) +

                            0.100000*np.tanh(np.real(((((((((((((complex(0,1)*np.conjugate(data["meanbatch_slices2_msignal"])) + (((data["signal_shift_-1_msignal"]) * (np.cos((data["signal_shift_-1_msignal"]))))))/2.0)) * (complex(0,1)*np.conjugate(np.sin((data["signal_shift_+1_msignal"])))))) * (complex(2.52313327789306641)))) + (data["signal_shift_-1_msignal"]))/2.0)) * (np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["maxbatch_slices2_msignal"]) + ((((-((np.cos((complex(3.0))))))) / (np.cos((((complex(-3.0)) * 2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(data["abs_minbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((complex(-2.0)) + (((data["signal_shift_-1_msignal"]) * (data["maxbatch_slices2_msignal"]))))/2.0)) + (np.where(np.abs(((complex(-2.0)) * (complex(0,1)*np.conjugate(np.cos((data["signal_shift_-1_msignal"])))))) <= np.abs(np.cos((((np.conjugate(data["signal_shift_-1_msignal"])) / 2.0)))),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((np.cos((((data["signal_shift_-1"]) + (complex(0.0)))))) + (((np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(np.sin((((np.sin((np.sin((np.sin((complex(0.0)))))))) / 2.0)))),complex(1.), complex(0.) )) + (data["maxbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.real(((np.sin((data["rangebatch_msignal"]))) * (np.where(np.abs(np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(((data["abs_avgbatch_msignal"]) / 2.0)),complex(1.), complex(0.) )) <= np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(complex(2.0)) <= np.abs(np.sin((np.conjugate(complex(3.28318190574645996))))),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.sin(((-((np.sin((np.cos((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) <= np.abs(complex(-3.0)),complex(1.), complex(0.) ))))))))))) + (((data["maxbatch_slices2_msignal"]) * (data["maxbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(((complex(0,1)*np.conjugate(data["abs_maxbatch"])) / (complex(1.0)))) <= np.abs(np.conjugate(((complex(2.0)) / 2.0))),complex(1.), complex(0.) )) > np.abs(((complex(0.0)) * (((data["signal_shift_-1_msignal"]) * 2.0)))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((((((data["signal_shift_+1"]) * 2.0)) * (((np.tanh((data["abs_maxbatch_slices2"]))) - (np.cos((data["abs_maxbatch_slices2"]))))))) - (np.cos((data["abs_maxbatch_slices2"]))))) - (np.cos((data["rangebatch_msignal"]))))) > np.abs(data["rangebatch_msignal"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((-((complex(0,1)*np.conjugate(np.where(np.abs(data["maxbatch_msignal"]) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.where(np.abs(data["signal_shift_-1_msignal"]) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-((np.conjugate(data["abs_avgbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["rangebatch_slices2"]) * ((-((((np.sin((((data["minbatch_slices2"]) / 2.0)))) * 2.0))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-((np.where(np.abs(((np.where(np.abs(np.tanh((data["minbatch_msignal"]))) > np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )) + ((((np.where(np.abs(complex(6.0)) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )) + (((((data["minbatch_msignal"]) / (((((data["abs_maxbatch_msignal"]) * (data["abs_minbatch_slices2"]))) + (data["maxbatch_slices2"]))))) + (data["maxbatch_slices2"]))))/2.0)))) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((np.cos((np.where(np.abs(((data["signal_shift_-1"]) / 2.0)) > np.abs(((((((np.sin((data["abs_maxbatch"]))) + (np.conjugate(data["abs_maxbatch"])))/2.0)) + (complex(-3.0)))/2.0)),complex(1.), complex(0.) )))))) > np.abs(complex(0,1)*np.conjugate(np.sin((data["stdbatch_slices2_msignal"])))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) * (((complex(3.0)) / ((-((((((-((np.tanh((np.cos((((data["maxbatch_msignal"]) * 2.0))))))))) + ((((-((np.tanh((np.cos((((data["maxbatch_msignal"]) * 2.0))))))))) * 2.0)))/2.0)))))))))) +

                            0.100000*np.tanh(np.real((((((data["abs_maxbatch_slices2_msignal"]) - ((((((data["abs_maxbatch_msignal"]) + (np.tanh((data["abs_maxbatch_msignal"]))))/2.0)) * 2.0)))) + (((data["minbatch_slices2_msignal"]) * (((complex(1.0)) - (data["abs_maxbatch_slices2_msignal"]))))))/2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.tanh(((-((np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(data["medianbatch_slices2"]) > np.abs(np.sin((((data["maxtominbatch"]) / (((data["maxbatch_slices2"]) / (np.cos((np.cos((((np.conjugate((((((data["meanbatch_slices2"]) + (data["meanbatch_slices2"]))/2.0)) / (data["maxtominbatch_slices2"])))) * 2.0)))))))))))),complex(1.), complex(0.) ))) <= np.abs(data["stdbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_slices2_msignal"]) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((np.conjugate(np.sin((data["abs_maxbatch"])))) + (np.conjugate(data["abs_avgbatch_slices2"])))/2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.cos((data["maxtominbatch_msignal"]))) * 2.0)))) +

                            0.100000*np.tanh(np.real((-((np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs((-((complex(0,1)*np.conjugate((-((((data["mean_abs_chgbatch_msignal"]) / (data["maxtominbatch_msignal"])))))))))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((np.where(np.abs(((complex(1.0)) + (data["mean_abs_chgbatch_msignal"]))) > np.abs(((data["rangebatch_msignal"]) * 2.0)),complex(1.), complex(0.) )) + (np.where(np.abs(complex(1.0)) > np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )))) <= np.abs(((data["abs_maxbatch_slices2"]) * 2.0)),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((np.tanh((np.conjugate(complex(0,1)*np.conjugate((((((data["abs_avgbatch_slices2_msignal"]) + (np.where(np.abs(np.tanh((data["abs_avgbatch_slices2_msignal"]))) > np.abs(complex(0,1)*np.conjugate(complex(2.0))),complex(1.), complex(0.) )))/2.0)) - (data["meanbatch_slices2"]))))))) - (np.where(np.abs(complex(0,1)*np.conjugate(complex(1.0))) > np.abs(np.cos((complex(2.0)))),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.tanh((np.where(np.abs(data["minbatch"]) <= np.abs(np.where(np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) > np.abs(((((data["medianbatch_msignal"]) * 2.0)) + (np.cos((complex(3.0)))))),complex(1.), complex(0.) )) > np.abs(complex(0,1)*np.conjugate((((-((np.tanh((np.cos((np.where(np.abs(data["minbatch"]) <= np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) ))))))))) * (complex(0,1)*np.conjugate(np.cos((complex(3.0)))))))),complex(1.), complex(0.) )),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((((np.where(np.abs(complex(0,1)*np.conjugate(((np.where(np.abs(data["signal_shift_+1"]) <= np.abs(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_msignal"])),complex(1.), complex(0.) )) * (data["signal_shift_+1"])))) <= np.abs(data["signal_shift_+1"]),complex(1.), complex(0.) )) + (np.sin((np.sin((np.where(np.abs(complex(-2.0)) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))))))/2.0)) > np.abs(complex(2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_msignal"]) > np.abs(((data["maxtominbatch_slices2"]) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((np.conjugate(np.cos((((data["abs_avgbatch_msignal"]) / 2.0))))) - (data["rangebatch_slices2"]))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.sin((data["minbatch_slices2_msignal"]))) > np.abs(((((((((complex(0,1)*np.conjugate(data["meanbatch_msignal"])) + (data["stdbatch_slices2_msignal"]))/2.0)) + (complex(0,1)*np.conjugate(np.tanh((data["abs_avgbatch_msignal"])))))) + (np.sin((((np.conjugate(np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) ))) / 2.0)))))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["abs_minbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["minbatch_msignal"]) > np.abs((((data["stdbatch_slices2"]) + (((complex(-3.0)) - (((complex(3.0)) / 2.0)))))/2.0)),complex(1.), complex(0.) )) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.where(np.abs(data["signal_shift_+1"]) <= np.abs(complex(7.83741664886474609)),complex(1.), complex(0.) )) * (data["meanbatch_slices2_msignal"]))) <= np.abs(np.where(np.abs(np.sin((complex(0.0)))) > np.abs((((((((((data["meanbatch_slices2_msignal"]) * 2.0)) + (np.conjugate(np.cos((complex(7.83741664886474609))))))/2.0)) / 2.0)) / 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.sin((np.where(np.abs(complex(2.0)) > np.abs(((((data["minbatch_slices2"]) * 2.0)) * 2.0)),complex(1.), complex(0.) )))))) / 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(((data["signal_shift_-1"]) + (data["maxbatch_msignal"]))) > np.abs(np.cos((np.where(np.abs(((((((complex(-2.0)) / 2.0)) / 2.0)) / 2.0)) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )))),complex(1.), complex(0.) )) > np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real((((data["maxtominbatch"]) + (((((data["signal_shift_-1"]) - (np.conjugate(np.sin((((data["signal_shift_-1"]) / 2.0))))))) + ((-((np.tanh(((-(((((data["abs_maxbatch_slices2_msignal"]) + (((data["signal_shift_-1"]) * 2.0)))/2.0)))))))))))))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos(((((data["rangebatch_slices2"]) + (np.where(np.abs(np.cos(((((data["rangebatch_slices2"]) + (np.sin((np.cos(((((data["minbatch_msignal"]) + (complex(12.21543693542480469)))/2.0)))))))/2.0)))) > np.abs((((data["minbatch_msignal"]) + (np.cos((np.cos(((((data["rangebatch_slices2"]) + (np.sin((((data["rangebatch_slices2"]) / 2.0)))))/2.0)))))))/2.0)),complex(1.), complex(0.) )))/2.0))))) +

                            0.100000*np.tanh(np.real((-(((-((complex(0,1)*np.conjugate(data["minbatch_slices2"]))))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(complex(-2.0)))) +

                            0.100000*np.tanh(np.real(np.sin((complex(0,1)*np.conjugate(((complex(2.0)) * (np.where(np.abs(((data["medianbatch_slices2"]) * 2.0)) <= np.abs(np.where(np.abs(data["medianbatch_slices2"]) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((((((((np.where(np.abs(data["maxtominbatch_msignal"]) <= np.abs(((np.cos((np.cos((data["maxtominbatch_msignal"]))))) / 2.0)),complex(1.), complex(0.) )) + (data["maxtominbatch_msignal"]))) / 2.0)) / 2.0)) + (data["maxtominbatch_msignal"]))) * 2.0))))) +

                            0.100000*np.tanh(np.real((-((np.cos((((data["abs_avgbatch_slices2_msignal"]) - (((np.tanh(((((np.where(np.abs(np.where(np.abs(((data["abs_avgbatch_msignal"]) - (complex(3.0)))) <= np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) > np.abs(data["mean_abs_chgbatch_msignal"]),complex(1.), complex(0.) )) + (data["abs_minbatch_msignal"]))/2.0)))) / 2.0)))))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((complex(0,1)*np.conjugate(((((np.sin((data["medianbatch_slices2_msignal"]))) / 2.0)) / 2.0))) + (data["minbatch_msignal"]))/2.0))))) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_slices2_msignal"]) + (((complex(-1.0)) - (np.where(np.abs(complex(0,1)*np.conjugate(data["meanbatch_msignal"])) <= np.abs(((data["abs_maxbatch_slices2_msignal"]) + (np.sin((complex(0.0)))))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real((-((complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"])))))) +

                            0.100000*np.tanh(np.real(((complex(0.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(data["maxbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin((np.conjugate(np.where(np.abs(data["rangebatch_slices2"]) <= np.abs(np.where(np.abs(np.sin((complex(0.0)))) <= np.abs(np.where(np.abs(np.where(np.abs(np.sin((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(np.cos((((data["mean_abs_chgbatch_slices2"]) - (data["meanbatch_msignal"]))))))))) <= np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )) > np.abs(np.cos((data["rangebatch_slices2"]))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((np.tanh((np.where(np.abs(((data["signal_shift_-1"]) + (complex(0,1)*np.conjugate((-((data["mean_abs_chgbatch_msignal"]))))))) <= np.abs(((complex(0,1)*np.conjugate(data["minbatch_msignal"])) * 2.0)),complex(1.), complex(0.) )))) / 2.0)) <= np.abs(np.tanh((complex(0.0)))),complex(1.), complex(0.) )) / (np.cos((complex(1.0))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(((data["maxbatch_slices2"]) / (((complex(3.0)) / 2.0)))),complex(1.), complex(0.) )) + (data["signal_shift_-1_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) - (np.conjugate(np.conjugate(np.conjugate(data["meanbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.real(np.sin(((((-((data["meanbatch_msignal"])))) - (np.where(np.abs(np.cos(((((-(((-((complex(0.0)))))))) / 2.0)))) > np.abs((((-((data["meanbatch_msignal"])))) - (np.where(np.abs(complex(0.0)) > np.abs(np.sin((np.sin((np.cos((data["abs_maxbatch_msignal"]))))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.cos((np.where(np.abs(np.cos((np.sin((complex(3.0)))))) > np.abs(np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(np.where(np.abs(data["medianbatch_msignal"]) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.sin((((data["abs_maxbatch"]) + (((np.tanh((np.conjugate(data["abs_maxbatch"])))) + (complex(0,1)*np.conjugate(complex(2.0)))))))))) +

                            0.100000*np.tanh(np.real(data["minbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(((((data["stdbatch_slices2"]) * 2.0)) * ((((complex(-3.0)) + (data["abs_maxbatch"]))/2.0))))),complex(1.), complex(0.) )) + (complex(-2.0)))/2.0)) + (data["signal_shift_+1"]))/2.0))))    

    

    def GP_class_10(self,data):

        return self.Output( -4.940363 +

                            0.100000*np.tanh(np.real((-((data["minbatch_slices2"]))))) +

                            0.100000*np.tanh(np.real((((-((data["abs_maxbatch"])))) * 2.0))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2"])) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(complex(0,1)*np.conjugate(np.tanh((((data["abs_minbatch_slices2_msignal"]) * 2.0))))) > np.abs(np.tanh((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0)))),complex(1.), complex(0.) )) > np.abs(data["medianbatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.conjugate(data["signal_shift_+1"])) - (np.cos((complex(1.0)))))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((complex(2.0)) / 2.0)))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((((-((data["maxbatch_slices2"])))) + (data["mean_abs_chgbatch_slices2"]))) > np.abs(np.sin((data["maxtominbatch_msignal"]))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.sin((data["rangebatch_slices2_msignal"]))))) * ((((data["abs_minbatch_msignal"]) + (((data["signal_shift_-1"]) / 2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_maxbatch"]) <= np.abs(complex(3.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.tanh((np.conjugate(np.sin((data["abs_maxbatch"]))))))) +

                            0.100000*np.tanh(np.real(((complex(-2.0)) / 2.0))) +

                            0.100000*np.tanh(np.real(data["minbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch_slices2_msignal"]) * 2.0)) - (np.cos(((-((complex(0,1)*np.conjugate(((np.cos((complex(2.0)))) / 2.0))))))))))) +

                            0.100000*np.tanh(np.real((((((np.conjugate(data["signal_shift_-1"])) - (complex(2.0)))) + (((((data["meanbatch_slices2"]) * (data["mean_abs_chgbatch_slices2"]))) / 2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(-2.0)) <= np.abs(((np.where(np.abs(np.tanh((data["medianbatch_msignal"]))) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) * (complex(8.0)))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs((-((data["mean_abs_chgbatch_slices2"])))) <= np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin(((((np.sin((complex(11.58077621459960938)))) + ((((complex(-1.0)) + (complex(12.52587985992431641)))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(-2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate(np.where(np.abs(np.where(np.abs(np.conjugate(((np.where(np.abs(data["abs_minbatch_slices2"]) <= np.abs(data["meanbatch_msignal"]),complex(1.), complex(0.) )) / 2.0))) > np.abs(complex(0,1)*np.conjugate(((np.cos((np.cos((((np.where(np.abs((-((np.sin((data["abs_minbatch_msignal"])))))) <= np.abs((-((data["meanbatch_msignal"])))),complex(1.), complex(0.) )) / 2.0)))))) * (data["signal_shift_-1_msignal"])))),complex(1.), complex(0.) )) > np.abs(data["abs_maxbatch"]),complex(1.), complex(0.) ))) > np.abs(complex(1.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(np.sin((data["meanbatch_slices2"]))),complex(1.), complex(0.) ))) - (complex(3.0))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.conjugate(((data["minbatch_slices2"]) * (((data["rangebatch_slices2"]) * 2.0))))) <= np.abs(np.sin((data["mean_abs_chgbatch_msignal"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(np.real(((np.sin((np.where(np.abs(data["maxtominbatch_msignal"]) > np.abs(np.conjugate(complex(2.12897348403930664))),complex(1.), complex(0.) )))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.conjugate(data["abs_minbatch_slices2"])) * (data["abs_minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_maxbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((np.tanh((((data["rangebatch_msignal"]) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real((((np.sin((complex(0.0)))) + (((data["maxtominbatch_msignal"]) * (complex(1.0)))))/2.0))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_+1"]) + (data["signal_shift_+1"]))/2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) * (complex(-1.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.where(np.abs(data["abs_avgbatch_slices2_msignal"]) > np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) )) - ((((((((-((np.sin((data["signal"])))))) - (data["abs_avgbatch_slices2"]))) * 2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(complex(11.63314342498779297))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_avgbatch_msignal"]) > np.abs(np.sin((complex(0,1)*np.conjugate(((data["signal_shift_+1"]) + (data["signal_shift_+1_msignal"])))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.conjugate(((np.tanh((data["abs_maxbatch_slices2"]))) * 2.0)))) +

                            0.100000*np.tanh(np.real(np.tanh((data["abs_maxbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((data["signal"]))) <= np.abs(np.tanh((np.tanh((((((np.cos((((((data["signal"]) - (complex(14.21352672576904297)))) - (np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(data["signal_shift_-1_msignal"]),complex(1.), complex(0.) )))))) - (np.cos((data["abs_maxbatch"]))))) / 2.0)))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(11.92458915710449219))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((complex(3.0)) / 2.0)))) - (np.conjugate(((complex(-1.0)) + (np.sin((complex(11.15662956237792969))))))))) * (data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.conjugate(((((np.sin((data["abs_minbatch_slices2"]))) + (data["signal_shift_+1"]))) - (complex(0,1)*np.conjugate(np.where(np.abs(data["rangebatch_slices2"]) > np.abs((-((((data["maxtominbatch"]) + (data["abs_minbatch_slices2_msignal"])))))),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.conjugate(((((data["meanbatch_slices2"]) * (complex(0.0)))) * 2.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["medianbatch_msignal"]))) +

                            0.100000*np.tanh(np.real(np.sin(((((np.sin((data["signal"]))) + (data["rangebatch_slices2"]))/2.0))))) +

                            0.100000*np.tanh(np.real(((complex(-3.0)) * (data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_-1"]) + (complex(3.0)))/2.0))) +

                            0.100000*np.tanh(np.real(np.cos(((((-((data["mean_abs_chgbatch_slices2"])))) + (data["abs_avgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_slices2"]) > np.abs(((data["abs_minbatch_msignal"]) + (np.where(np.abs(data["signal_shift_-1"]) > np.abs(((np.cos(((((((((data["meanbatch_msignal"]) / (data["mean_abs_chgbatch_slices2"]))) * (data["maxbatch_slices2"]))) + (data["signal_shift_-1"]))/2.0)))) - (((complex(0.0)) / (complex(-3.0)))))),complex(1.), complex(0.) )))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_slices2_msignal"]) * (np.where(np.abs(((np.sin((data["meanbatch_slices2_msignal"]))) * 2.0)) <= np.abs(((((complex(0.0)) - (np.cos(((-((np.cos((np.conjugate(data["maxtominbatch_slices2"]))))))))))) / 2.0)),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((complex(0,1)*np.conjugate(data["abs_minbatch_slices2"])) * (data["abs_minbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(-2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(((((complex(10.0)) - (data["rangebatch_slices2_msignal"]))) - (((data["maxtominbatch_slices2"]) + (complex(4.04233646392822266)))))) <= np.abs(data["minbatch"]),complex(1.), complex(0.) )) <= np.abs(((((data["minbatch_msignal"]) * 2.0)) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((((((data["abs_minbatch_slices2"]) / 2.0)) - (data["abs_avgbatch_msignal"]))) - ((((data["meanbatch_msignal"]) + ((-((data["mean_abs_chgbatch_msignal"])))))/2.0)))) - ((((data["abs_avgbatch_msignal"]) + (data["abs_minbatch_slices2"]))/2.0)))) - (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((((np.sin(((-((np.conjugate((-((data["abs_avgbatch_msignal"])))))))))) - (data["abs_avgbatch_msignal"]))) + ((((-((np.where(np.abs(np.conjugate(((data["abs_minbatch_slices2_msignal"]) / 2.0))) > np.abs((-((data["rangebatch_slices2"])))),complex(1.), complex(0.) ))))) * (data["abs_minbatch_slices2_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((data["signal_shift_+1"]) + (complex(0,1)*np.conjugate(complex(2.24728035926818848))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(((complex(0,1)*np.conjugate(np.where(np.abs(((data["signal_shift_+1"]) * 2.0)) <= np.abs(np.tanh((complex(1.0)))),complex(1.), complex(0.) ))) / 2.0)) > np.abs(np.cos((np.sin((data["abs_maxbatch_msignal"]))))),complex(1.), complex(0.) )) - (data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["rangebatch_slices2"]) * (((((np.where(np.abs(data["meanbatch_slices2"]) > np.abs(np.where(np.abs(((np.where(np.abs(complex(4.79061222076416016)) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)) - (np.conjugate(((data["meanbatch_slices2_msignal"]) + (np.tanh((((((np.where(np.abs(data["stdbatch_slices2_msignal"]) > np.abs(complex(0,1)*np.conjugate(data["medianbatch_slices2_msignal"])),complex(1.), complex(0.) )) * 2.0)) * 2.0)))))))))))) +

                            0.100000*np.tanh(np.real((((data["signal"]) + (np.where(np.abs(((data["abs_avgbatch_slices2"]) - ((((data["abs_avgbatch_slices2"]) + (np.tanh(((-((((((data["minbatch_msignal"]) * ((((np.conjugate(data["mean_abs_chgbatch_msignal"])) + (np.tanh((data["minbatch_msignal"]))))/2.0)))) / 2.0))))))))/2.0)))) > np.abs(complex(1.0)),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["maxbatch_msignal"]) > np.abs(((data["abs_maxbatch"]) + (data["signal_shift_-1"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["medianbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate((-((np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(data["medianbatch_msignal"]),complex(1.), complex(0.) )))))),complex(1.), complex(0.) )) - (np.cos((((data["medianbatch_msignal"]) - (complex(0,1)*np.conjugate(complex(3.0)))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((data["maxtominbatch_msignal"]) / 2.0)) > np.abs(((np.where(np.abs(((data["abs_minbatch_slices2"]) - (np.where(np.abs(data["signal_shift_+1_msignal"]) <= np.abs(complex(3.0)),complex(1.), complex(0.) )))) <= np.abs(((((((data["medianbatch_slices2_msignal"]) - (complex(3.0)))) / 2.0)) + ((-((((data["medianbatch_slices2_msignal"]) / 2.0))))))),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((-(((((((data["medianbatch_slices2_msignal"]) * 2.0)) + (((data["abs_avgbatch_slices2"]) - (complex(0,1)*np.conjugate(((((-(((((((((((((-((complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"]))))) / 2.0)) + (data["abs_maxbatch_msignal"]))) + (((np.cos((data["abs_maxbatch"]))) / 2.0)))/2.0)) + (complex(-3.0)))/2.0))))) + (complex(-2.0)))/2.0))))))/2.0))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) - (((data["rangebatch_slices2_msignal"]) - (np.where(np.abs(((data["signal_shift_+1"]) - (np.where(np.abs(((data["meanbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2_msignal"]))) > np.abs((((np.conjugate(data["abs_avgbatch_slices2_msignal"])) + (data["rangebatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))) > np.abs((((((data["meanbatch_slices2_msignal"]) + (np.cos((data["stdbatch_slices2"]))))/2.0)) * (data["signal_shift_+1"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos((complex(0.0))))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_-1"]) + (((((data["signal_shift_-1"]) + (complex(0.0)))) + (data["signal_shift_-1"]))))/2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["signal_shift_+1"]) - (data["rangebatch_msignal"]))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(((data["mean_abs_chgbatch_slices2"]) * (data["medianbatch_slices2"]))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_msignal"]) / ((((-((complex(2.0))))) + (data["stdbatch_msignal"]))))) - (((((data["abs_avgbatch_msignal"]) + (complex(1.0)))) + (((((((data["abs_avgbatch_msignal"]) + (complex(1.0)))) / (complex(1.0)))) * 2.0))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(((complex(8.94131755828857422)) - (((data["meanbatch_slices2"]) - (data["signal_shift_+1"]))))) > np.abs(((data["stdbatch_slices2"]) / 2.0)),complex(1.), complex(0.) )) - (data["meanbatch_slices2_msignal"]))) - (data["meanbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((np.cos(((((data["minbatch_msignal"]) + ((-((complex(0,1)*np.conjugate(np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.where(np.abs(np.tanh((((complex(-2.0)) - (data["medianbatch_slices2_msignal"]))))) > np.abs(((data["rangebatch_slices2"]) * ((((((((complex(-2.0)) - (data["medianbatch_slices2_msignal"]))) + (np.cos((data["maxtominbatch_slices2_msignal"]))))/2.0)) * 2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.conjugate((((((((data["rangebatch_slices2_msignal"]) * (((np.where(np.abs(complex(3.0)) <= np.abs(complex(6.08659887313842773)),complex(1.), complex(0.) )) / 2.0)))) + (((data["stdbatch_slices2_msignal"]) * 2.0)))/2.0)) - ((((data["meanbatch_msignal"]) + (((data["abs_avgbatch_msignal"]) * (np.conjugate(data["maxbatch_slices2"])))))/2.0))))) / 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((((data["minbatch_msignal"]) - (((np.tanh((data["mean_abs_chgbatch_slices2"]))) * 2.0)))) / 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((data["maxbatch_slices2"]) - ((((data["medianbatch_slices2_msignal"]) + (complex(3.0)))/2.0))))) +

                            0.100000*np.tanh(np.real((-(((((-((data["abs_maxbatch_msignal"])))) * ((-(((((np.sin((np.tanh((((np.cos((np.cos((((data["medianbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))))))) / 2.0)))))) + ((((data["medianbatch_slices2_msignal"]) + ((((data["maxbatch_msignal"]) + (complex(3.0)))/2.0)))/2.0)))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((((((complex(-1.0)) - (data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0)))) - (np.tanh((complex(0,1)*np.conjugate(data["meanbatch_slices2_msignal"])))))) - (((complex(3.0)) - (((data["mean_abs_chgbatch_slices2"]) + ((((-((data["mean_abs_chgbatch_msignal"])))) / 2.0))))))))) +

                            0.100000*np.tanh(np.real(np.cos((((((data["minbatch_msignal"]) / 2.0)) - (((data["mean_abs_chgbatch_slices2"]) / 2.0))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs((((-((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"])))))) * (complex(-2.0)))) > np.abs(data["abs_maxbatch_slices2"]),complex(1.), complex(0.) )) - (((((np.tanh((data["maxbatch_msignal"]))) + (data["maxbatch_msignal"]))) + (data["medianbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real((((-((np.conjugate(np.conjugate(np.conjugate(complex(2.0)))))))) + (((data["abs_maxbatch_msignal"]) + (complex(5.06950902938842773))))))) +

                            0.100000*np.tanh(np.real(((((((-((((data["medianbatch_msignal"]) + (((data["maxbatch_msignal"]) + (((((data["medianbatch_slices2_msignal"]) + (complex(1.0)))) + (np.where(np.abs(data["medianbatch_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["medianbatch_msignal"])),complex(1.), complex(0.) ))))))))))) - (data["medianbatch_slices2_msignal"]))) + (((data["abs_avgbatch_slices2_msignal"]) - (complex(0,1)*np.conjugate(((data["medianbatch_msignal"]) / 2.0))))))/2.0))) +

                            0.100000*np.tanh(np.real((((((-((np.tanh((np.cos((data["medianbatch_msignal"])))))))) * 2.0)) - (((complex(0.0)) + (np.where(np.abs(data["stdbatch_msignal"]) > np.abs(((data["medianbatch_msignal"]) * 2.0)),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((((((np.cos((data["signal_shift_-1"]))) - (data["maxbatch_slices2_msignal"]))) - (data["abs_avgbatch_slices2_msignal"]))) + (((complex(0,1)*np.conjugate((-((np.conjugate(data["abs_minbatch_slices2"])))))) - (np.where(np.abs(np.tanh((complex(0.0)))) <= np.abs(data["abs_avgbatch_slices2_msignal"]),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(complex(2.0)),complex(1.), complex(0.) )) - ((((np.tanh((((complex(2.0)) * ((((data["abs_maxbatch"]) + (((data["signal"]) * (data["abs_maxbatch_msignal"]))))/2.0)))))) + ((((complex(2.0)) + (((data["signal"]) * (((data["abs_maxbatch_msignal"]) + (data["meanbatch_msignal"]))))))/2.0)))/2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["signal_shift_+1"]) + ((-((np.where(np.abs(data["abs_maxbatch_msignal"]) > np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) ))))))) * (np.cos((((np.sin((data["meanbatch_slices2_msignal"]))) - (complex(1.0)))))))) - ((-((((np.where(np.abs(data["signal_shift_-1"]) > np.abs(np.tanh((complex(1.0)))),complex(1.), complex(0.) )) - (complex(1.0)))))))))) +

                            0.100000*np.tanh(np.real(((data["stdbatch_slices2_msignal"]) + (((data["medianbatch_msignal"]) * ((((data["stdbatch_msignal"]) + (np.where(np.abs(np.where(np.abs((((data["maxbatch_slices2"]) + (((complex(-1.0)) + (complex(2.0)))))/2.0)) <= np.abs(data["signal_shift_-1"]),complex(1.), complex(0.) )) <= np.abs(np.sin((np.where(np.abs(data["stdbatch_msignal"]) > np.abs((((data["stdbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0)),complex(1.), complex(0.) )))),complex(1.), complex(0.) )))/2.0))))))) +

                            0.100000*np.tanh(np.real((((np.where(np.abs(np.where(np.abs(((complex(-3.0)) / (complex(3.0)))) <= np.abs(np.tanh((np.sin((((data["signal"]) * (data["stdbatch_slices2_msignal"]))))))),complex(1.), complex(0.) )) > np.abs(data["mean_abs_chgbatch_slices2"]),complex(1.), complex(0.) )) + (((data["rangebatch_msignal"]) + ((((data["rangebatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)))))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.tanh((((((((-((complex(2.0))))) - (data["medianbatch_slices2_msignal"]))) + ((((-((np.sin((np.where(np.abs((((-((data["mean_abs_chgbatch_slices2_msignal"])))) / 2.0)) <= np.abs(((data["maxbatch_msignal"]) / (complex(0,1)*np.conjugate(data["abs_maxbatch"])))),complex(1.), complex(0.) ))))))) / 2.0)))/2.0)))) / 2.0)) - (((data["medianbatch_slices2_msignal"]) + (complex(2.0))))))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2_msignal"]) / (np.cos((((data["medianbatch_slices2_msignal"]) * (np.cos((np.sin((((np.where(np.abs(data["abs_maxbatch_slices2"]) > np.abs(np.where(np.abs(((np.conjugate(np.sin((((data["medianbatch_slices2_msignal"]) / (np.cos((((complex(0.0)) * (np.where(np.abs(data["abs_maxbatch_msignal"]) <= np.abs(data["signal"]),complex(1.), complex(0.) ))))))))))) * (complex(-1.0)))) > np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0))))))))))))) +

                            0.100000*np.tanh(np.real(data["abs_maxbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((data["signal"]) - (((complex(3.0)) - (data["signal"])))))) +

                            0.100000*np.tanh(np.real(((((data["abs_avgbatch_slices2"]) / (((complex(-1.0)) - ((((data["meanbatch_msignal"]) + (np.cos((np.sin((data["abs_avgbatch_slices2"]))))))/2.0)))))) - (np.tanh((np.tanh((((data["rangebatch_slices2"]) + (np.sin((data["meanbatch_msignal"])))))))))))) +

                            0.100000*np.tanh(np.real(complex(6.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(((data["meanbatch_slices2"]) / (np.cos(((((data["minbatch_msignal"]) + (np.cos((((np.where(np.abs(np.conjugate(data["meanbatch_slices2"])) <= np.abs(np.sin((data["medianbatch_slices2_msignal"]))),complex(1.), complex(0.) )) / 2.0)))))/2.0)))))))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(complex(2.0)),complex(1.), complex(0.) )) * 2.0)) - ((-((((np.conjugate(data["rangebatch_slices2"])) / (np.sin((((data["medianbatch_slices2_msignal"]) - (np.cos((np.cos((np.sin((data["mean_abs_chgbatch_msignal"]))))))))))))))))))) +

                            0.100000*np.tanh(np.real((((((((data["maxbatch_slices2_msignal"]) + ((((((-((((data["medianbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"])))))) * 2.0)) * 2.0)))) + (np.cos((((((data["rangebatch_slices2"]) + (data["maxbatch_slices2_msignal"]))) * (((data["rangebatch_slices2"]) + (np.tanh((data["medianbatch_msignal"]))))))))))/2.0)) - (((data["maxbatch_msignal"]) + ((((data["meanbatch_slices2"]) + (complex(1.0)))/2.0))))))) +

                            0.100000*np.tanh(np.real(((((data["abs_minbatch_slices2_msignal"]) * (np.cos((data["meanbatch_slices2_msignal"]))))) + ((-((np.where(np.abs(data["maxtominbatch_slices2_msignal"]) > np.abs(data["stdbatch_slices2"]),complex(1.), complex(0.) )))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(complex(2.0))) + (np.cos((data["signal_shift_-1"])))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(np.sin((data["abs_avgbatch_slices2_msignal"]))),complex(1.), complex(0.) )) + (((data["meanbatch_slices2"]) - (((np.where(np.abs(data["medianbatch_slices2_msignal"]) <= np.abs(data["maxbatch_slices2"]),complex(1.), complex(0.) )) - (((np.where(np.abs((((data["signal_shift_+1"]) + (data["meanbatch_msignal"]))/2.0)) <= np.abs(data["signal"]),complex(1.), complex(0.) )) - (data["minbatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.real((((-((((((((-((data["abs_maxbatch_msignal"])))) / 2.0)) + (((data["abs_maxbatch_msignal"]) + ((((((data["abs_avgbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"]))) + (data["abs_avgbatch_slices2_msignal"]))/2.0)))))/2.0))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(((data["abs_maxbatch_slices2"]) + (np.sin((((np.conjugate((-((np.conjugate(((data["abs_maxbatch"]) - (((complex(0,1)*np.conjugate(data["meanbatch_slices2"])) / 2.0))))))))) * 2.0)))))) > np.abs(data["abs_avgbatch_msignal"]),complex(1.), complex(0.) )) * (np.where(np.abs(data["medianbatch_slices2"]) <= np.abs(complex(13.59522247314453125)),complex(1.), complex(0.) )))) - (data["maxbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["minbatch_slices2_msignal"]) / (((data["meanbatch_msignal"]) - (complex(-3.0)))))) + (((complex(-2.0)) / (((data["meanbatch_msignal"]) - (((complex(-2.0)) - (np.tanh((((data["maxbatch_slices2"]) / 2.0))))))))))))) +

                            0.100000*np.tanh(np.real((((((data["maxbatch_slices2"]) + (((np.cos((data["abs_maxbatch_slices2_msignal"]))) + ((((np.where(np.abs(np.cos((np.cos((data["abs_maxbatch_slices2_msignal"]))))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))))/2.0)) / (np.cos((((data["abs_maxbatch_slices2_msignal"]) * (np.cos((complex(3.0))))))))))) +

                            0.100000*np.tanh(np.real((((complex(0.0)) + (data["abs_avgbatch_slices2"]))/2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1"]) + (((np.cos((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0)))) + (np.sin((np.conjugate((((((((data["maxbatch_msignal"]) + ((((data["minbatch_msignal"]) + (np.conjugate(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["meanbatch_slices2_msignal"]),complex(1.), complex(0.) ))))/2.0)))/2.0)) * 2.0)) * 2.0)))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.tanh((data["medianbatch_slices2"])))) + (((np.sin((data["minbatch_slices2"]))) + (np.sin((data["medianbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) - (data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(((((np.where(np.abs(data["meanbatch_slices2"]) <= np.abs(data["mean_abs_chgbatch_slices2_msignal"]),complex(1.), complex(0.) )) / 2.0)) * (data["abs_avgbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((((((data["maxbatch_slices2"]) - (((((((data["maxbatch_slices2_msignal"]) - (np.conjugate(data["abs_maxbatch_msignal"])))) + (data["abs_maxbatch_msignal"]))) * (((data["abs_avgbatch_msignal"]) / (np.cos((((data["abs_maxbatch_msignal"]) * 2.0)))))))))) * 2.0)) * (((data["abs_avgbatch_msignal"]) / (np.cos((((data["abs_maxbatch_msignal"]) * 2.0)))))))) - (data["abs_maxbatch_msignal"])))) +

                            0.100000*np.tanh(np.real((-((((data["abs_avgbatch_slices2_msignal"]) - ((-(((((complex(3.0)) + (np.where(np.abs(((complex(3.0)) / (np.cos((data["abs_avgbatch_slices2"]))))) <= np.abs(((np.where(np.abs(np.cos((((((data["signal"]) - (complex(10.0)))) * (complex(3.0)))))) <= np.abs((-((data["abs_avgbatch_msignal"])))),complex(1.), complex(0.) )) - (np.tanh((np.cos((data["abs_avgbatch_slices2"]))))))),complex(1.), complex(0.) )))/2.0))))))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate(np.sin((((data["medianbatch_msignal"]) + (((((complex(0,1)*np.conjugate(np.sin((((np.conjugate(np.where(np.abs(np.sin((np.sin((data["abs_minbatch_slices2_msignal"]))))) <= np.abs(np.sin((complex(0.0)))),complex(1.), complex(0.) ))) + (((np.cos((data["maxtominbatch_slices2_msignal"]))) * 2.0))))))) * 2.0)) * 2.0))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((np.sin((((((data["medianbatch_slices2"]) / 2.0)) * 2.0)))))) + (np.sin((((np.where(np.abs(np.where(np.abs(data["abs_minbatch_slices2_msignal"]) <= np.abs(complex(0,1)*np.conjugate(data["medianbatch_slices2"])),complex(1.), complex(0.) )) > np.abs(((data["signal_shift_-1_msignal"]) / 2.0)),complex(1.), complex(0.) )) * ((-((data["maxtominbatch_slices2_msignal"])))))))))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((np.where(np.abs(np.where(np.abs(complex(-1.0)) > np.abs(((data["medianbatch_slices2"]) * 2.0)),complex(1.), complex(0.) )) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )) / 2.0)) <= np.abs(np.where(np.abs(np.where(np.abs(data["medianbatch_slices2"]) > np.abs(np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))),complex(1.), complex(0.) )) > np.abs(((np.where(np.abs(((data["signal_shift_+1"]) / 2.0)) > np.abs(np.conjugate(np.tanh((complex(3.0))))),complex(1.), complex(0.) )) / 2.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(((((data["signal_shift_+1"]) + (((data["minbatch_slices2_msignal"]) / 2.0)))) + (complex(0,1)*np.conjugate(data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(data["signal_shift_-1"])) +

                            0.100000*np.tanh(np.real(((np.sin((np.sin((data["meanbatch_msignal"]))))) * 2.0))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate((-(((-((np.where(np.abs(data["abs_minbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["maxbatch_slices2_msignal"]) > np.abs(np.where(np.abs(data["mean_abs_chgbatch_slices2"]) <= np.abs(np.cos((complex(-2.0)))),complex(1.), complex(0.) )),complex(1.), complex(0.) )),complex(1.), complex(0.) )))))))))) +

                            0.100000*np.tanh(np.real(((((data["maxtominbatch"]) - ((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"]))/2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_slices2"]) - (np.cos((((data["abs_maxbatch_slices2"]) - (data["stdbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.real(data["maxtominbatch_slices2"])) +

                            0.100000*np.tanh(np.real(((data["abs_maxbatch_slices2_msignal"]) / (np.cos(((((np.where(np.abs(((np.where(np.abs(((data["mean_abs_chgbatch_msignal"]) * 2.0)) > np.abs(data["abs_avgbatch_slices2"]),complex(1.), complex(0.) )) / 2.0)) <= np.abs(((data["abs_maxbatch_slices2_msignal"]) / (np.cos((((data["maxbatch_slices2_msignal"]) / (np.sin((np.cos((data["abs_minbatch_slices2"]))))))))))),complex(1.), complex(0.) )) + (data["abs_maxbatch_slices2_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) * (((data["rangebatch_msignal"]) - (complex(0.0))))))) +

                            0.100000*np.tanh(np.real(np.cos(((((((((data["minbatch_msignal"]) + (np.where(np.abs(((np.conjugate(np.where(np.abs(complex(1.0)) > np.abs(((data["stdbatch_slices2"]) * 2.0)),complex(1.), complex(0.) ))) / 2.0)) > np.abs((((data["maxtominbatch_msignal"]) + (data["minbatch_msignal"]))/2.0)),complex(1.), complex(0.) )))/2.0)) / 2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.conjugate(np.cos((np.sin((complex(3.0))))))) > np.abs(np.cos((complex(0,1)*np.conjugate(complex(0.0))))),complex(1.), complex(0.) )) + (np.sin((data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.cos((data["stdbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(1.0))) +

                            0.100000*np.tanh(np.real(((data["signal"]) * (np.sin((np.sin((data["meanbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["signal_shift_+1_msignal"])))) +

                            0.100000*np.tanh(np.real(((((complex(3.0)) / 2.0)) - (((np.tanh((data["minbatch_msignal"]))) / 2.0))))) +

                            0.100000*np.tanh(np.real(data["signal_shift_-1_msignal"])) +

                            0.100000*np.tanh(np.real(((((((data["signal_shift_-1_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))) + (np.where(np.abs(np.tanh((complex(-3.0)))) <= np.abs((-((data["rangebatch_msignal"])))),complex(1.), complex(0.) )))) * 2.0))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_+1_msignal"]) + (np.where(np.abs((-(((((data["signal_shift_+1_msignal"]) + (((data["signal_shift_+1_msignal"]) + (np.sin((complex(2.0)))))))/2.0))))) > np.abs(np.conjugate(data["signal_shift_+1_msignal"])),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((((np.sin((data["abs_maxbatch_slices2"]))) / 2.0)) * 2.0))) +

                            0.100000*np.tanh(np.real(((((data["signal_shift_+1_msignal"]) - (((((np.sin((np.where(np.abs(((data["signal_shift_+1_msignal"]) - (complex(10.07381153106689453)))) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))) / 2.0)) + (((((((data["meanbatch_msignal"]) * (complex(0,1)*np.conjugate(((data["abs_avgbatch_msignal"]) / 2.0))))) * (((((data["abs_avgbatch_msignal"]) / 2.0)) + (data["signal_shift_+1_msignal"]))))) * (data["stdbatch_slices2"]))))))) - (data["meanbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(((((data["meanbatch_msignal"]) + (complex(0,1)*np.conjugate(np.sin((np.tanh((np.conjugate(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["signal_shift_-1_msignal"]))))))))))) * (((((((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) / 2.0)) + ((-((data["signal_shift_-1_msignal"])))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(np.conjugate(((((((data["medianbatch_msignal"]) * (data["medianbatch_msignal"]))) + (((np.conjugate(((data["signal_shift_+1_msignal"]) + (data["signal_shift_+1_msignal"])))) + ((-((complex(9.57333183288574219))))))))) - (np.where(np.abs(data["meanbatch_msignal"]) <= np.abs(data["abs_maxbatch_msignal"]),complex(1.), complex(0.) )))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate((-((data["minbatch_slices2_msignal"]))))) > np.abs(((((data["abs_avgbatch_slices2_msignal"]) - (((np.sin(((((((((data["medianbatch_slices2"]) + (data["signal_shift_-1_msignal"]))/2.0)) * (data["minbatch_slices2_msignal"]))) / (data["medianbatch_slices2"]))))) / (np.cos(((((data["minbatch_slices2"]) + (data["signal_shift_+1_msignal"]))/2.0)))))))) / 2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(data["mean_abs_chgbatch_slices2_msignal"])) +

                            0.100000*np.tanh(np.real(((((np.sin((np.sin((np.tanh((((data["minbatch_msignal"]) + (complex(0,1)*np.conjugate(((np.where(np.abs(data["meanbatch_slices2_msignal"]) > np.abs(np.conjugate(np.tanh((data["abs_avgbatch_msignal"])))),complex(1.), complex(0.) )) / 2.0))))))))))) + (data["signal_shift_-1_msignal"]))) + (np.sin((np.sin((data["minbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(((np.conjugate((((((data["signal_shift_+1"]) + (np.sin(((((((complex(-2.0)) * 2.0)) + (((data["abs_avgbatch_slices2"]) / 2.0)))/2.0)))))/2.0)) + (data["maxtominbatch"])))) * 2.0))) +

                            0.100000*np.tanh(np.real(((data["medianbatch_msignal"]) + (complex(10.0))))) +

                            0.100000*np.tanh(np.real(((data["maxtominbatch_slices2_msignal"]) * ((((((data["maxbatch_msignal"]) - (((((data["maxbatch_msignal"]) - (data["abs_minbatch_slices2_msignal"]))) * (complex(-3.0)))))) + (((np.conjugate(data["minbatch"])) * (((np.sin((((data["abs_avgbatch_slices2_msignal"]) - (data["maxtominbatch_slices2_msignal"]))))) * 2.0)))))/2.0))))) +

                            0.100000*np.tanh(np.real((((-((complex(1.0))))) + (((data["signal_shift_+1"]) * (((((((data["abs_minbatch_slices2"]) * (((np.sin((((((((data["abs_minbatch_msignal"]) * (((data["abs_minbatch_slices2"]) / 2.0)))) / 2.0)) / 2.0)))) - (np.sin((((((data["signal_shift_-1_msignal"]) / 2.0)) / 2.0)))))))) / 2.0)) - (np.sin((((data["signal_shift_+1"]) / 2.0))))))))))) +

                            0.100000*np.tanh(np.real(((((-((((complex(5.0)) + (np.where(np.abs(data["abs_maxbatch_slices2_msignal"]) > np.abs(np.conjugate(data["abs_maxbatch_slices2"])),complex(1.), complex(0.) ))))))) + (((data["signal_shift_-1"]) + (np.where(np.abs(((np.tanh((data["abs_avgbatch_slices2_msignal"]))) - (np.cos(((-((data["signal"])))))))) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))/2.0))) +

                            0.100000*np.tanh(np.real((((data["signal_shift_-1_msignal"]) + ((((np.conjugate(data["signal_shift_-1"])) + ((((np.tanh((data["abs_maxbatch_msignal"]))) + (np.where(np.abs(complex(1.0)) > np.abs(((data["signal_shift_+1"]) - (data["maxbatch_slices2"]))),complex(1.), complex(0.) )))/2.0)))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((np.tanh((np.cos((np.conjugate((((-(((-((((data["abs_maxbatch_slices2_msignal"]) * (data["meanbatch_msignal"]))))))))) / 2.0))))))) - (((data["maxbatch_slices2_msignal"]) * (np.conjugate(complex(-2.0)))))))) +

                            0.100000*np.tanh(np.real((((((((data["stdbatch_slices2"]) - (np.where(np.abs(data["medianbatch_slices2"]) > np.abs(((data["signal_shift_+1_msignal"]) + (data["maxtominbatch"]))),complex(1.), complex(0.) )))) + (data["maxtominbatch"]))) + ((((np.conjugate(((complex(0,1)*np.conjugate(data["signal"])) + (np.sin((complex(2.0))))))) + (data["abs_avgbatch_slices2"]))/2.0)))/2.0))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) + (np.sin((np.where(np.abs(np.tanh((np.cos((((data["signal_shift_+1_msignal"]) - (((np.conjugate((-((np.conjugate(np.where(np.abs(((data["abs_maxbatch_msignal"]) / 2.0)) <= np.abs(data["signal"]),complex(1.), complex(0.) ))))))) / 2.0)))))))) > np.abs(((data["signal_shift_+1_msignal"]) + (data["signal_shift_+1_msignal"]))),complex(1.), complex(0.) ))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["abs_minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.cos((((np.cos((((np.tanh((data["signal_shift_+1"]))) - (((((((((((data["abs_maxbatch"]) * 2.0)) * 2.0)) * (data["signal_shift_+1_msignal"]))) * (np.cos((np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(data["abs_maxbatch_slices2_msignal"]),complex(1.), complex(0.) )))))) + (data["signal_shift_+1_msignal"]))))))) + (data["abs_maxbatch"])))))) +

                            0.100000*np.tanh(np.real(((np.cos((((data["abs_minbatch_slices2"]) - ((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))))) - (np.where(np.abs(np.sin((data["maxtominbatch_slices2_msignal"]))) > np.abs(data["signal"]),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.where(np.abs(complex(6.0)) <= np.abs(complex(0,1)*np.conjugate(np.cos((data["abs_avgbatch_slices2"])))),complex(1.), complex(0.) )) > np.abs(np.where(np.abs(data["signal_shift_+1"]) > np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )),complex(1.), complex(0.) )) + (((np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (np.sin((data["meanbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.tanh((((data["maxbatch_msignal"]) * (((np.sin((((((np.where(np.abs((((data["maxbatch_slices2"]) + (data["maxbatch_slices2"]))/2.0)) > np.abs(np.where(np.abs(((np.sin((complex(1.0)))) / 2.0)) > np.abs(complex(1.0)),complex(1.), complex(0.) )),complex(1.), complex(0.) )) / 2.0)) / 2.0)))) / 2.0)))))) > np.abs(complex(-3.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((-((data["abs_avgbatch_msignal"])))) + (data["signal_shift_-1_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1_msignal"])) +

                            0.100000*np.tanh(np.real(np.sin((np.where(np.abs(np.cos((np.tanh((data["rangebatch_msignal"]))))) <= np.abs(np.conjugate(data["maxbatch_slices2"])),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(np.cos(((-(((((data["minbatch_msignal"]) + (((((complex(0,1)*np.conjugate(((data["minbatch_msignal"]) + (((((data["abs_maxbatch_msignal"]) + (complex(0.0)))) - ((-((data["abs_maxbatch_msignal"]))))))))) / 2.0)) * 2.0)))/2.0)))))))) +

                            0.100000*np.tanh(np.real((((((((np.tanh((((data["maxbatch_msignal"]) - (np.tanh((np.where(np.abs(data["signal_shift_-1"]) <= np.abs(data["signal_shift_+1_msignal"]),complex(1.), complex(0.) )))))))) * (data["signal_shift_+1_msignal"]))) * (data["signal_shift_+1_msignal"]))) + (((np.where(np.abs(data["signal_shift_+1_msignal"]) > np.abs(np.sin((data["maxbatch_msignal"]))),complex(1.), complex(0.) )) + (((data["signal_shift_+1_msignal"]) * (complex(12.86174583435058594)))))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["meanbatch_msignal"]) - (np.where(np.abs(data["abs_avgbatch_slices2"]) > np.abs(np.sin((((data["maxbatch_slices2_msignal"]) + (np.tanh((np.cos((np.sin((np.sin((((np.sin((((np.sin((data["meanbatch_msignal"]))) + (np.cos((data["maxbatch_msignal"]))))))) * 2.0)))))))))))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real(((data["signal"]) + (np.where(np.abs(data["stdbatch_msignal"]) > np.abs(np.sin((((((data["maxbatch_slices2"]) * 2.0)) - (data["signal"]))))),complex(1.), complex(0.) ))))) +

                            0.100000*np.tanh(np.real((-((((data["medianbatch_msignal"]) + (((data["maxbatch_msignal"]) - (((data["signal_shift_+1_msignal"]) + ((((np.sin((complex(0,1)*np.conjugate(complex(0.0))))) + (np.tanh((np.tanh((data["medianbatch_msignal"]))))))/2.0)))))))))))) +

                            0.100000*np.tanh(np.real(((complex(0.0)) - (complex(2.0))))) +

                            0.100000*np.tanh(np.real((((((data["medianbatch_slices2"]) + (data["abs_maxbatch"]))) + (complex(0,1)*np.conjugate((-((data["minbatch"]))))))/2.0))) +

                            0.100000*np.tanh(np.real(((data["signal"]) + (complex(0,1)*np.conjugate(((data["stdbatch_slices2"]) + (((np.where(np.abs(data["signal"]) > np.abs(np.conjugate(complex(2.0))),complex(1.), complex(0.) )) * (complex(0.63249725103378296)))))))))) +

                            0.100000*np.tanh(np.real(np.cos((np.sin((np.sin((np.conjugate(data["minbatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.real(np.cos((data["minbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(np.cos((np.where(np.abs(((data["medianbatch_slices2"]) / 2.0)) > np.abs(((np.conjugate(((data["maxbatch_slices2"]) / 2.0))) / 2.0)),complex(1.), complex(0.) )))) <= np.abs(complex(0,1)*np.conjugate(np.tanh((complex(0,1)*np.conjugate((-((data["maxbatch_msignal"])))))))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["meanbatch_slices2"]) > np.abs(complex(0,1)*np.conjugate(((complex(0,1)*np.conjugate(np.sin((data["minbatch"])))) / 2.0))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real((((np.tanh((data["abs_avgbatch_slices2"]))) + (np.where(np.abs(((((np.tanh((((data["abs_minbatch_slices2_msignal"]) / 2.0)))) * (data["abs_minbatch_slices2_msignal"]))) / 2.0)) <= np.abs(data["medianbatch_slices2_msignal"]),complex(1.), complex(0.) )))/2.0))) +

                            0.100000*np.tanh(np.real(((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0))) +

                            0.100000*np.tanh(np.real(((((((np.tanh((data["abs_maxbatch_slices2_msignal"]))) / 2.0)) * (((data["signal_shift_+1"]) / 2.0)))) * (np.sin((data["mean_abs_chgbatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_maxbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(np.sin((np.conjugate(np.conjugate(np.sin((data["minbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate((((data["minbatch"]) + (np.conjugate(data["signal_shift_-1_msignal"])))/2.0))) <= np.abs(np.conjugate(((np.conjugate(data["abs_maxbatch_slices2"])) / 2.0))),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((((((((((np.where(np.abs(complex(1.0)) <= np.abs(complex(-2.0)),complex(1.), complex(0.) )) + (((data["abs_minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"]))))/2.0)) * 2.0)) - (np.where(np.abs(data["abs_maxbatch_slices2"]) <= np.abs(((data["signal_shift_+1"]) * 2.0)),complex(1.), complex(0.) )))) + (data["abs_minbatch_slices2_msignal"]))/2.0))) +

                            0.100000*np.tanh(np.real(data["signal_shift_+1_msignal"])) +

                            0.100000*np.tanh(np.real(((((((((((np.where(np.abs(data["signal_shift_+1"]) > np.abs(data["rangebatch_slices2"]),complex(1.), complex(0.) )) / 2.0)) - (data["abs_minbatch_msignal"]))) + (((data["maxtominbatch"]) - (complex(0,1)*np.conjugate(data["rangebatch_slices2_msignal"])))))) * 2.0)) + (((data["minbatch"]) - (complex(3.0))))))) +

                            0.100000*np.tanh(np.real(np.tanh((data["mean_abs_chgbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((complex(-1.0)) * (np.conjugate(data["signal"]))))) +

                            0.100000*np.tanh(np.real(np.cos((complex(6.45073556900024414))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(complex(0,1)*np.conjugate((((((data["maxbatch_msignal"]) * 2.0)) + (((complex(3.0)) * 2.0)))/2.0))) <= np.abs((((((complex(1.0)) / 2.0)) + (data["rangebatch_slices2"]))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1"]) + (np.conjugate(((complex(3.0)) + (((complex(0,1)*np.conjugate(((((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(complex(3.0))))) - (((complex(3.0)) * 2.0)))) - (data["abs_maxbatch_slices2_msignal"])))) - (((((complex(3.0)) * 2.0)) * 2.0)))))))))) +

                            0.100000*np.tanh(np.real(((data["minbatch_slices2"]) / 2.0))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(((((-(((((-((np.conjugate(((np.cos((complex(0,1)*np.conjugate(complex(4.0))))) / 2.0)))))) * 2.0))))) + (data["maxtominbatch_msignal"]))/2.0)) > np.abs(((((((complex(-1.0)) + (np.tanh((((complex(4.0)) + (complex(3.0)))))))/2.0)) + (np.tanh((((data["rangebatch_slices2_msignal"]) + (complex(3.0)))))))/2.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(np.sin(((-((data["abs_minbatch_msignal"])))))) <= np.abs(np.where(np.abs((-((np.where(np.abs(complex(0.0)) > np.abs(np.conjugate(complex(-1.0))),complex(1.), complex(0.) ))))) > np.abs(np.conjugate(((complex(13.21041870117187500)) / (complex(2.0))))),complex(1.), complex(0.) )),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.where(np.abs(data["signal_shift_-1_msignal"]) <= np.abs(np.tanh((data["signal_shift_-1_msignal"]))),complex(1.), complex(0.) )) * 2.0))) +

                            0.100000*np.tanh(np.real(((np.cos((((np.where(np.abs(np.cos((data["maxbatch_slices2"]))) > np.abs(data["meanbatch_slices2"]),complex(1.), complex(0.) )) * (data["abs_maxbatch"]))))) - (data["stdbatch_msignal"])))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_+1_msignal"]) * (complex(4.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(np.where(np.abs(data["maxtominbatch_slices2"]) <= np.abs((-((data["minbatch"])))),complex(1.), complex(0.) )))) +

                            0.100000*np.tanh(np.real(np.cos(((-((((complex(0,1)*np.conjugate(((((-((np.conjugate(data["abs_avgbatch_slices2_msignal"]))))) + (data["medianbatch_slices2"]))/2.0))) - (((complex(0,1)*np.conjugate(np.conjugate(np.sin((((complex(3.0)) * (data["medianbatch_slices2"]))))))) - ((-(((((data["minbatch_msignal"]) + (np.sin((data["minbatch_msignal"]))))/2.0))))))))))))))) +

                            0.100000*np.tanh(np.real(((data["signal_shift_-1_msignal"]) - ((((((((data["meanbatch_msignal"]) + (data["maxbatch_msignal"]))) * 2.0)) + (np.conjugate(((complex(0,1)*np.conjugate(np.where(np.abs(((data["meanbatch_msignal"]) * ((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0)))) <= np.abs(((((complex(0,1)*np.conjugate(data["maxbatch_msignal"])) * 2.0)) * 2.0)),complex(1.), complex(0.) ))) / 2.0))))/2.0))))) +

                            0.100000*np.tanh(np.real(complex(0,1)*np.conjugate(data["abs_maxbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.real(complex(0.0))) +

                            0.100000*np.tanh(np.real(np.tanh((((data["signal_shift_+1"]) + (((complex(5.0)) + (((data["mean_abs_chgbatch_slices2"]) * (((((data["abs_maxbatch_slices2"]) + (data["mean_abs_chgbatch_slices2"]))) / 2.0))))))))))) +

                            0.100000*np.tanh(np.real(np.sin((np.sin((np.tanh(((((np.where(np.abs(data["mean_abs_chgbatch_slices2"]) > np.abs(((((np.sin((data["medianbatch_slices2"]))) - (np.cos((data["maxbatch_slices2"]))))) / 2.0)),complex(1.), complex(0.) )) + (complex(0.0)))/2.0))))))))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["abs_minbatch_slices2"]) > np.abs(complex(-3.0)),complex(1.), complex(0.) ))) +

                            0.100000*np.tanh(np.real(np.sin((data["abs_avgbatch_slices2"])))) +

                            0.100000*np.tanh(np.real(((((((np.cos((((complex(0.0)) / 2.0)))) - (data["abs_avgbatch_msignal"]))) - ((((((data["abs_maxbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))/2.0)) * 2.0)))) * 2.0))) +

                            0.100000*np.tanh(np.real(np.conjugate(np.sin(((((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.abs(data["abs_minbatch_msignal"]) > np.abs(data["maxbatch_msignal"]),complex(1.), complex(0.) )))/2.0)))))) +

                            0.100000*np.tanh(np.real(((((data["maxbatch_msignal"]) + (complex(0,1)*np.conjugate(np.conjugate(np.tanh((data["abs_maxbatch_slices2_msignal"]))))))) - ((((-((complex(-3.0))))) / 2.0))))) +

                            0.100000*np.tanh(np.real(((((((data["mean_abs_chgbatch_slices2"]) * 2.0)) * 2.0)) * (data["signal_shift_-1_msignal"])))) +

                            0.100000*np.tanh(np.real(np.tanh((data["signal_shift_-1_msignal"])))) +

                            0.100000*np.tanh(np.real(np.where(np.abs(data["mean_abs_chgbatch_msignal"]) > np.abs(np.sin((complex(0,1)*np.conjugate(np.conjugate(((((-((complex(-3.0))))) + (np.where(np.abs(np.tanh((data["signal"]))) > np.abs(np.cos((np.sin(((((data["meanbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0)))))),complex(1.), complex(0.) )))/2.0)))))),complex(1.), complex(0.) ))) ) 
def maddest(d, axis=None):

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def high_pass_filter(x, low_cutoff=1000, sample_rate=10000):



    nyquist = 0.5 * sample_rate

    norm_low_cutoff = low_cutoff / nyquist

    print(norm_low_cutoff)

    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')

    filtered_sig = signal.sosfilt(sos, x)



    return filtered_sig



def denoise_signal( x, wavelet='db4', level=1):

    

    coeff = pywt.wavedec( x, wavelet, mode="per" )

    sigma = (1/0.6745) * maddest( coeff[-level] )

    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )

    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    return pywt.waverec( coeff, wavelet, mode='per' )
def add_rooling_data(df : pd.DataFrame) -> pd.DataFrame:

    window_sizes = [10, 50, 100, 1000]

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()

    return df
base = '../input/liverpool-ion-switching'

train = pd.read_csv(os.path.join(base + '/train.csv'))

test  = pd.read_csv(os.path.join(base + '/test.csv'))
def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch',

                                                    'batch_index', 'batch_slices', 'batch_slices2'

                                                    

                                                   ]]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    return df



train = features(train)

test = features(test)



col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2',

                                             'mean_abs_chgbatch', 'meanbatch', 'rangebatch', 'stdbatch',

                                             'maxbatch', 'medianbatch', 'abs_minbatch', 'abs_avgbatch']]

target = train['open_channels']

train = train[col]
train.replace(np.inf,np.nan,inplace=True)

train.replace(-np.inf,np.nan,inplace=True)

test.replace(np.inf,np.nan,inplace=True)

test.replace(-np.inf,np.nan,inplace=True)

train = train.astype(complex)

train.fillna(complex(0,1),inplace=True)

test[col] = test[col].astype(complex)

test.fillna(complex(0,1),inplace=True)
gp = GP()

train_preds = gp.GrabPredictions(train)

test_preds = gp.GrabPredictions(test)
f1_score(target.values,np.argmax(train_preds.values,axis=1),average='macro')

test['open_channels'] = np.argmax(test_preds.values,axis=1)

test[['time','open_channels']].to_csv('gpsubmission.csv', index=False, float_format='%.4f')
train.replace(complex(0,1),np.nan,inplace=True)

test.replace(complex(0,1),np.nan,inplace=True)

for c in col:

    train[c] = np.real(train[c])

    test[c] = np.real(test[c])
import scipy as sp

from functools import partial

class OptimizedRounder(object):



    def __init__(self):

        self.coef_ = 0



    def loss(self, coef, X, y):

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -metrics.f1_score(y, X_p, average = 'macro')



    def fit(self, X, y):

        loss_partial = partial(self.loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        return (pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).astype(np.int8)



    def coefficients(self):

        return self.coef_['x']





def optimize_predictions(preds, coeffs):

    

    preds[preds <= coeffs[0]] = 0

    preds[np.where(np.logical_and(preds > coeffs[0], preds <= coeffs[1]))] = 1

    preds[np.where(np.logical_and(preds > coeffs[1], preds <= coeffs[2]))] = 2

    preds[np.where(np.logical_and(preds > coeffs[2], preds <= coeffs[3]))] = 3

    preds[np.where(np.logical_and(preds > coeffs[3], preds <= coeffs[4]))] = 4

    preds[np.where(np.logical_and(preds > coeffs[4], preds <= coeffs[5]))] = 5

    preds[np.where(np.logical_and(preds > coeffs[5], preds <= coeffs[6]))] = 6

    preds[np.where(np.logical_and(preds > coeffs[6], preds <= coeffs[7]))] = 7

    preds[np.where(np.logical_and(preds > coeffs[7], preds <= coeffs[8]))] = 8

    preds[np.where(np.logical_and(preds > coeffs[8], preds <= coeffs[9]))] = 9

    preds[preds > coeffs[9]] = 10

    preds = preds.astype(np.int8)

    return preds



def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = metrics.f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)
import lightgbm as lgb

idx = np.zeros(train.shape[0]).astype(bool)

idx[2::5] = True # Chose a different training slice for kicks

train.drop(train.index[idx],inplace=True,axis=0)

gc.collect()

idx = np.logical_not(idx)

train_preds = train_preds[idx]

target = target[idx]

for c in range(11):

    train['gp_'+str(int(c))] = train_preds.values[:,c]

    test['gp_'+str(int(c))] = test_preds.values[:,c]

x1, x2, y1, y2 = model_selection.train_test_split(train, target, test_size=0.2, random_state=7)

del train

gc.collect()

col = x1.columns

params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 

model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=0, early_stopping_rounds=250, feval=MacroF1Metric)

preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)

oof_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)

optR = OptimizedRounder()

optR.fit(oof_lgb,y2)

coeffs = optR.coefficients()

preds_lgb = optimize_predictions(preds_lgb,coeffs)

oof_lgb = optimize_predictions(oof_lgb,coeffs)

print('f1_score',f1_score(y2,oof_lgb,average='macro'))
test['open_channels'] = preds_lgb

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')