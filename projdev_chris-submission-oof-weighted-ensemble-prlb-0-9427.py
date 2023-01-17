## reference: https://www.kaggle.com/ipythonx/optimizing-metrics-out-of-fold-weights-ensemble
!ls ../input/melanoma-oof-and-sub
import pandas as pd, numpy as np

from scipy.optimize import minimize

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import random



%matplotlib inline
np.random.seed(42)

random.seed(42)
oof_01  = pd.read_csv('../input/melanoma-oof-and-sub/oof_0.csv') 

test_01 = pd.read_csv('../input/melanoma-oof-and-sub/sub_0.csv')

oof_01  = oof_01.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_01 = test_01.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_02  = pd.read_csv('../input/melanoma-oof-and-sub/oof_100.csv') 

test_02 = pd.read_csv('../input/melanoma-oof-and-sub/sub_100.csv')

oof_02  = oof_02.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_02 = test_02.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_03  = pd.read_csv('../input/melanoma-oof-and-sub/oof_105.csv') 

test_03 = pd.read_csv('../input/melanoma-oof-and-sub/sub_105.csv')

oof_03  = oof_03.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_03 = test_03.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_04  = pd.read_csv('../input/melanoma-oof-and-sub/oof_108.csv') 

test_04 = pd.read_csv('../input/melanoma-oof-and-sub/sub_108.csv')

oof_04  = oof_04.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_04 = test_04.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_05  = pd.read_csv('../input/melanoma-oof-and-sub/oof_109.csv') 

test_05 = pd.read_csv('../input/melanoma-oof-and-sub/sub_109.csv')

oof_05  = oof_05.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_05 = test_05.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_06  = pd.read_csv('../input/melanoma-oof-and-sub/oof_11.csv') 

test_06 = pd.read_csv('../input/melanoma-oof-and-sub/sub_11.csv')

oof_06  = oof_06.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_06 = test_06.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_07  = pd.read_csv('../input/melanoma-oof-and-sub/oof_110.csv') 

test_07 = pd.read_csv('../input/melanoma-oof-and-sub/sub_110.csv')

oof_07  = oof_07.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_07 = test_07.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_08  = pd.read_csv('../input/melanoma-oof-and-sub/oof_111.csv') 

test_08 = pd.read_csv('../input/melanoma-oof-and-sub/sub_111.csv')

oof_08  = oof_08.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_08 = test_08.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_09  = pd.read_csv('../input/melanoma-oof-and-sub/oof_113.csv') 

test_09 = pd.read_csv('../input/melanoma-oof-and-sub/sub_113.csv')

oof_09  = oof_09.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_09 = test_09.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_10  = pd.read_csv('../input/melanoma-oof-and-sub/oof_116.csv') 

test_10 = pd.read_csv('../input/melanoma-oof-and-sub/sub_116.csv')

oof_10  = oof_10.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_10 = test_10.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_11  = pd.read_csv('../input/melanoma-oof-and-sub/oof_117.csv') 

test_11 = pd.read_csv('../input/melanoma-oof-and-sub/sub_117.csv')

oof_11  = oof_11.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_11 = test_11.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_12  = pd.read_csv('../input/melanoma-oof-and-sub/oof_12.csv') 

test_12 = pd.read_csv('../input/melanoma-oof-and-sub/sub_12.csv')

oof_12  = oof_12.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_12 = test_12.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_13  = pd.read_csv('../input/melanoma-oof-and-sub/oof_120.csv') 

test_13 = pd.read_csv('../input/melanoma-oof-and-sub/sub_120.csv')

oof_13  = oof_13.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_13 = test_13.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_14  = pd.read_csv('../input/melanoma-oof-and-sub/oof_121.csv') 

test_14 = pd.read_csv('../input/melanoma-oof-and-sub/sub_121.csv')

oof_14  = oof_14.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_14 = test_14.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_15  = pd.read_csv('../input/melanoma-oof-and-sub/oof_13.csv') 

test_15 = pd.read_csv('../input/melanoma-oof-and-sub/sub_13.csv')

oof_15  = oof_15.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_15 = test_15.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_16  = pd.read_csv('../input/melanoma-oof-and-sub/oof_15.csv') 

test_16 = pd.read_csv('../input/melanoma-oof-and-sub/sub_15.csv')

oof_16  = oof_16.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_16 = test_16.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)





oof_17  = pd.read_csv('../input/melanoma-oof-and-sub/oof_16.csv') 

test_17 = pd.read_csv('../input/melanoma-oof-and-sub/sub_16.csv')

oof_17  = oof_17.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_17 = test_17.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_18  = pd.read_csv('../input/melanoma-oof-and-sub/oof_2.csv') 

test_18 = pd.read_csv('../input/melanoma-oof-and-sub/sub_2.csv')

oof_18  = oof_18.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_18 = test_18.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_19  = pd.read_csv('../input/melanoma-oof-and-sub/oof_20.csv') 

test_19 = pd.read_csv('../input/melanoma-oof-and-sub/sub_20.csv')

oof_19  = oof_19.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_19 = test_19.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_20  = pd.read_csv('../input/melanoma-oof-and-sub/oof_24.csv') 

test_20 = pd.read_csv('../input/melanoma-oof-and-sub/sub_24.csv')

oof_20  = oof_20.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_20 = test_20.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_21  = pd.read_csv('../input/melanoma-oof-and-sub/oof_28.csv') 

test_21 = pd.read_csv('../input/melanoma-oof-and-sub/sub_28.csv')

oof_21  = oof_21.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_21 = test_21.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_22  = pd.read_csv('../input/melanoma-oof-and-sub/oof_30.csv') 

test_22 = pd.read_csv('../input/melanoma-oof-and-sub/sub_30.csv')

oof_22  = oof_22.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_22 = test_22.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_23  = pd.read_csv('../input/melanoma-oof-and-sub/oof_32.csv') 

test_23 = pd.read_csv('../input/melanoma-oof-and-sub/sub_32.csv')

oof_23  = oof_23.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_23 = test_23.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_24  = pd.read_csv('../input/melanoma-oof-and-sub/oof_33.csv') 

test_24 = pd.read_csv('../input/melanoma-oof-and-sub/sub_33.csv')

oof_24  = oof_24.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_24 = test_24.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_25  = pd.read_csv('../input/melanoma-oof-and-sub/oof_35.csv') 

test_25 = pd.read_csv('../input/melanoma-oof-and-sub/sub_35.csv')

oof_25  = oof_25.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_25 = test_25.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_26  = pd.read_csv('../input/melanoma-oof-and-sub/oof_384.csv') 

test_26 = pd.read_csv('../input/melanoma-oof-and-sub/sub_384.csv')

oof_26  = oof_26.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_26 = test_26.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_27  = pd.read_csv('../input/melanoma-oof-and-sub/oof_385.csv') 

test_27 = pd.read_csv('../input/melanoma-oof-and-sub/sub_385.csv')

oof_27  = oof_27.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_27 = test_27.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_28  = pd.read_csv('../input/melanoma-oof-and-sub/oof_4.csv') 

test_28 = pd.read_csv('../input/melanoma-oof-and-sub/sub_4.csv')

oof_28  = oof_28.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_28 = test_28.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_29  = pd.read_csv('../input/melanoma-oof-and-sub/oof_44.csv') 

test_29 = pd.read_csv('../input/melanoma-oof-and-sub/sub_44.csv')

oof_29  = oof_29.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_29 = test_29.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_30  = pd.read_csv('../input/melanoma-oof-and-sub/oof_54.csv') 

test_30 = pd.read_csv('../input/melanoma-oof-and-sub/sub_54.csv')

oof_30  = oof_30.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_30 = test_30.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_31  = pd.read_csv('../input/melanoma-oof-and-sub/oof_55.csv') 

test_31 = pd.read_csv('../input/melanoma-oof-and-sub/sub_55.csv')

oof_31  = oof_31.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_31 = test_31.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_32  = pd.read_csv('../input/melanoma-oof-and-sub/oof_56.csv') 

test_32 = pd.read_csv('../input/melanoma-oof-and-sub/sub_56.csv')

oof_32  = oof_32.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_32 = test_32.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_33  = pd.read_csv('../input/melanoma-oof-and-sub/oof_57.csv') 

test_33 = pd.read_csv('../input/melanoma-oof-and-sub/sub_57.csv')

oof_33  = oof_33.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_33 = test_33.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_34  = pd.read_csv('../input/melanoma-oof-and-sub/oof_58.csv') 

test_34 = pd.read_csv('../input/melanoma-oof-and-sub/sub_58.csv')

oof_34  = oof_34.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_34 = test_34.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_35  = pd.read_csv('../input/melanoma-oof-and-sub/oof_59.csv') 

test_35 = pd.read_csv('../input/melanoma-oof-and-sub/sub_59.csv')

oof_35  = oof_35.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_35 = test_35.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_36  = pd.read_csv('../input/melanoma-oof-and-sub/oof_6.csv') 

test_36 = pd.read_csv('../input/melanoma-oof-and-sub/sub_6.csv')

oof_36  = oof_36.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_36 = test_36.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_37  = pd.read_csv('../input/melanoma-oof-and-sub/oof_65.csv') 

test_37 = pd.read_csv('../input/melanoma-oof-and-sub/sub_65.csv')

oof_37  = oof_37.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_37 = test_37.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_38  = pd.read_csv('../input/melanoma-oof-and-sub/oof_67.csv') 

test_38 = pd.read_csv('../input/melanoma-oof-and-sub/sub_67.csv')

oof_38  = oof_38.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_38 = test_38.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)



oof_39  = pd.read_csv('../input/melanoma-oof-and-sub/oof_77.csv') 

test_39 = pd.read_csv('../input/melanoma-oof-and-sub/sub_77.csv')

oof_39  = oof_39.sort_values(by=['image_name'],  

                               ascending=True).reset_index(drop=True)

test_39 = test_39.sort_values(by=['image_name'],  

                                ascending=True).reset_index(drop=True)
blend_train = []

blend_test = []



# out of fold prediction

blend_train.append(oof_01.pred)

blend_train.append(oof_02.pred)

blend_train.append(oof_03.pred)

blend_train.append(oof_04.pred)

blend_train.append(oof_05.pred)

blend_train.append(oof_06.pred)

blend_train.append(oof_07.pred)

blend_train.append(oof_08.pred)

blend_train.append(oof_09.pred)

blend_train.append(oof_10.pred)

blend_train.append(oof_11.pred)

blend_train.append(oof_12.pred)

blend_train.append(oof_13.pred)

blend_train.append(oof_14.pred)

blend_train.append(oof_15.pred)

blend_train.append(oof_16.pred)

blend_train.append(oof_17.pred)

blend_train.append(oof_18.pred)

blend_train.append(oof_19.pred)

blend_train.append(oof_20.pred)

blend_train.append(oof_21.pred)

blend_train.append(oof_22.pred)

blend_train.append(oof_23.pred)

blend_train.append(oof_24.pred)

blend_train.append(oof_25.pred)

blend_train.append(oof_26.pred)

blend_train.append(oof_27.pred)

blend_train.append(oof_28.pred)

blend_train.append(oof_29.pred)

blend_train.append(oof_30.pred)

blend_train.append(oof_31.pred)

blend_train.append(oof_32.pred)

blend_train.append(oof_33.pred)

blend_train.append(oof_34.pred)

blend_train.append(oof_35.pred)

blend_train.append(oof_36.pred)

blend_train.append(oof_37.pred)

blend_train.append(oof_38.pred)

blend_train.append(oof_39.pred)

blend_train = np.array(blend_train)



# submission scores

blend_test.append(test_01.target)

blend_test.append(test_02.target)

blend_test.append(test_03.target)

blend_test.append(test_04.target)

blend_test.append(test_05.target)

blend_test.append(test_06.target)

blend_test.append(test_07.target)

blend_test.append(test_08.target)

blend_test.append(test_09.target)

blend_test.append(test_10.target)

blend_test.append(test_11.target)

blend_test.append(test_12.target)

blend_test.append(test_13.target)

blend_test.append(test_14.target)

blend_test.append(test_15.target)

blend_test.append(test_16.target)

blend_test.append(test_17.target)

blend_test.append(test_18.target)

blend_test.append(test_19.target)

blend_test.append(test_20.target)

blend_test.append(test_21.target)

blend_test.append(test_22.target)

blend_test.append(test_23.target)

blend_test.append(test_24.target)

blend_test.append(test_25.target)

blend_test.append(test_26.target)

blend_test.append(test_27.target)

blend_test.append(test_28.target)

blend_test.append(test_29.target)

blend_test.append(test_30.target)

blend_test.append(test_31.target)

blend_test.append(test_32.target)

blend_test.append(test_33.target)

blend_test.append(test_34.target)

blend_test.append(test_35.target)

blend_test.append(test_36.target)

blend_test.append(test_37.target)

blend_test.append(test_38.target)

blend_test.append(test_39.target)

blend_test = np.array(blend_test)
def roc_min_func(weights):

    final_prediction = 0

    for weight, prediction in zip(weights, blend_train):

        final_prediction += weight * prediction

    return roc_auc_score(np.array(oof_01.target), final_prediction)



print('\n Finding Blending Weights ...')

res_list = []

weights_list = []



for k in range(1000):

    #starting_values = np.random.uniform(size=len(blend_train))

    starting_values = np.random.uniform(size=len(blend_train)) * 1/len(blend_train)

    #bounds = [(0, 1)] * len(blend_train)

    #bounds = [(0, 1/len(blend_train))] * len(blend_train)

    bounds = [(0, 1/len(blend_train)) for _ in range(len(blend_train))]

    

    res = minimize(roc_min_func,

                   starting_values,

                   method='L-BFGS-B',

                   bounds=bounds,

                   options={'disp': False,

                            'maxiter': 100000})

    

    res_list.append(res['fun'])

    weights_list.append(res['x'])

    

    print('{iter}\tScore: {score}\tWeights: {weights}'.format(

        iter=(k + 1),

        score=res['fun'],

        weights='\t'.join([str(item) for item in res['x']])))



    

#bestSC   = np.min(res_list)

bestSC = np.max(res_list)

#bestWght = weights_list[np.argmin(res_list)]

bestWght = weights_list[np.argmax(res_list)]

weights  = bestWght

blend_score = round(bestSC, 6)
print('\n Ensemble Score: {best_score}'.format(best_score=bestSC))

print('\n Best Weights: {weights}'.format(weights=bestWght))



train_prices = np.zeros(len(blend_train[0]))

test_prices  = np.zeros(len(blend_test[0]))



print('\n Your final model:')

for k in range(len(blend_test)):

    print(' %.6f * model-%d' % (weights[k], (k + 1)))

    test_prices += blend_test[k] * weights[k]



for k in range(len(blend_train)):

    train_prices += blend_train[k] * weights[k]
## with image and tabular kernel-based

test_01.target = (test_01.target.values*bestWght[0] + 

                 test_02.target.values*bestWght[1] +

                 test_03.target.values*bestWght[2] +

                 test_04.target.values*bestWght[3] +

                 test_05.target.values*bestWght[4] +

                 test_06.target.values*bestWght[5] +

                 test_07.target.values*bestWght[6] +

                 test_08.target.values*bestWght[7] +

                 test_09.target.values*bestWght[8] +

                 test_10.target.values*bestWght[9] +

                 test_11.target.values*bestWght[10] +

                 test_12.target.values*bestWght[11] +

                 test_13.target.values*bestWght[12] +

                 test_14.target.values*bestWght[13] +

                 test_15.target.values*bestWght[14] +

                 test_16.target.values*bestWght[15] +

                 test_17.target.values*bestWght[16] +

                 test_18.target.values*bestWght[17] +

                 test_19.target.values*bestWght[18] +

                 test_20.target.values*bestWght[19] +

                 test_21.target.values*bestWght[20] +

                 test_22.target.values*bestWght[21] +

                 test_23.target.values*bestWght[22] +

                 test_24.target.values*bestWght[23] +

                 test_25.target.values*bestWght[24] +

                 test_26.target.values*bestWght[25] +

                 test_27.target.values*bestWght[26] +

                 test_28.target.values*bestWght[27] +

                 test_29.target.values*bestWght[28] +

                 test_30.target.values*bestWght[29] +

                 test_31.target.values*bestWght[30] +

                 test_32.target.values*bestWght[31] +

                 test_33.target.values*bestWght[32] +

                 test_34.target.values*bestWght[33] +

                 test_35.target.values*bestWght[34] +

                 test_36.target.values*bestWght[35] +

                 test_37.target.values*bestWght[36] +

                 test_38.target.values*bestWght[37] +

                 test_39.target.values*bestWght[38])/sum(bestWght)



test_01.to_csv('final_weighted_average_ensemble.csv', index=False)

test_01.head()