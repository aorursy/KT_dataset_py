# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd

import pickle

from sklearn.metrics import mean_squared_error
bilstm_df3= pd.read_csv("../input/model-predictions/bi_lstm_pred_3.csv")

bilstm_df7= pd.read_csv("../input/model-predictions/bi_lstm_pred_7.csv")

bilstm_df15= pd.read_csv("../input/model-predictions/bi_lstm_pred_15.csv")

bilstm_df30= pd.read_csv("../input/model-predictions/bi_lstm_pred_30.csv")



aligned_attentiondf3=pd.read_csv("../input/model-predictions/aligned_attention_3.csv")

aligned_attentiondf7=pd.read_csv("../input/model-predictions/aligned_attention_7.csv")

aligned_attentiondf15=pd.read_csv("../input/model-predictions/aligned_attention_15.csv")

aligned_attentiondf30=pd.read_csv("../input/model-predictions/aligned_attention_30.csv")
with open('../input/model-predictions/mdrm_audio_pred_3.pkl', 'rb') as f:

    mdrm_audio_pred_3=pickle.load(f)

    

with open('../input/model-predictions/mdrm_audio_pred_7.pkl', 'rb') as f:

    mdrm_audio_pred_7=pickle.load(f)

    

with open('../input/model-predictions/mdrm_audio_pred_15.pkl', 'rb') as f:

    mdrm_audio_pred_15=pickle.load(f)

    

with open('../input/model-predictions/mdrm_audio_pred_30.pkl', 'rb') as f:

    mdrm_audio_pred_30=pickle.load(f)

    

mdrm_audio_pred_3=mdrm_audio_pred_3[:,0]

mdrm_audio_pred_7=mdrm_audio_pred_7[:,0]

mdrm_audio_pred_15=mdrm_audio_pred_15[:,0]

mdrm_audio_pred_30=mdrm_audio_pred_30[:,0]
with open('../input/model-predictions/PastRegression_SVR_3.pkl', 'rb') as f:

    PastRegression_SVR_3=pickle.load(f)

    

with open('../input/model-predictions/PastRegression_SVR_7.pkl', 'rb') as f:

    PastRegression_SVR_7=pickle.load(f)

    

with open('../input/model-predictions/PastRegression_SVR_15.pkl', 'rb') as f:

    PastRegression_SVR_15=pickle.load(f)

    

with open('../input/model-predictions/PastRegression_SVR_30.pkl', 'rb') as f:

    PastRegression_SVR_30=pickle.load(f)

    

with open('../input/model-predictions/y_test3.pkl', 'rb') as f:

    y_test3=pickle.load(f)

with open('../input/model-predictions/y_test7.pkl', 'rb') as f:

    y_test7=pickle.load(f)

with open('../input/model-predictions/y_test15.pkl', 'rb') as f:

    y_test15=pickle.load(f)

with open('../input/model-predictions/y_test30.pkl', 'rb') as f:

    y_test30=pickle.load(f)
bilstm_pred3=np.array(bilstm_df3.iloc[:,1])

bilstm_pred7=np.array(bilstm_df7.iloc[:,1])

bilstm_pred15=np.array(bilstm_df15.iloc[:,1])

bilstm_pred30=np.array(bilstm_df30.iloc[:,1])



aligned_attention_pred3=np.array(aligned_attentiondf3.iloc[:,1])

aligned_attention_pred7=np.array(aligned_attentiondf7.iloc[:,1])

aligned_attention_pred15=np.array(aligned_attentiondf15.iloc[:,1])

aligned_attention_pred30=np.array(aligned_attentiondf30.iloc[:,1])
ratio_range=np.linspace(0,1,51)



def combined(duration,y_pred1,y_pred2,y_pred3,y_test):

    



    min_mse=10

    min_alpha=2

    min_beta=2

    mse_list=[]

    alpha_list=[]

    beta_list=[]

    

    y_pred_final=[]

    

    for alpha in ratio_range:

        for beta in ratio_range:

            for gamma in ratio_range:

                if alpha+beta+gamma==1:

                    y_pred_combined=(alpha)*y_pred1+(beta)*y_pred2+(gamma)*y_pred3

                    mse = mean_squared_error(y_test, y_pred_combined)

                    mse_list.append(mse)

                    alpha_list.append(alpha)

                    beta_list.append(beta)

                    if mse<min_mse:

                        min_mse=mse

                        min_alpha=alpha

                        min_beta=beta

                        y_pred_final=y_pred_combined



    print("Min MSE for"+str(duration)+"days=" +str(min_mse))

    print("Ratio at Min MSE: alpha="+str(min_alpha)+" beta="+str(min_beta))

    

    save_pkl='ensemble_y_pred_{}.pkl'.format(duration)

    

    with open(save_pkl,'wb') as f:

        pickle.dump(y_pred_final,f)

    

#     list_of_tuples = list(zip(alpha_list, beta_list,mse_list)) 

#     df = pd.DataFrame(list_of_tuples, columns = ['Alpha', 'Beta','MSE']) 

#     save_path='Ensembe_exp{}.csv'.format(duration)

#     df.to_csv(save_path)





    return
combined(duration=3,y_pred1=aligned_attention_pred3,y_pred2=bilstm_pred3,y_pred3=PastRegression_SVR_3,y_test=y_test3)
combined(duration=7,y_pred1=aligned_attention_pred7,y_pred2=bilstm_pred7,y_pred3=PastRegression_SVR_7,y_test=y_test7)
combined(duration=15,y_pred1=aligned_attention_pred15,y_pred2=bilstm_pred15,y_pred3=PastRegression_SVR_15,y_test=y_test15)
combined(duration=30,y_pred1=aligned_attention_pred30,y_pred2=bilstm_pred30,y_pred3=PastRegression_SVR_30,y_test=y_test30)