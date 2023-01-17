import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import matthews_corrcoef

import xgboost as xgb

import gc

from numba import jit

from datetime import datetime
def pandas_read_large_csv(file, sample_rate = None, chunksize = 10 ** 5, **kwargs):

    ''' read by chunks '''

    chunks = []

    for chunk in pd.read_csv(file, chunksize = chunksize, **kwargs):

        if sample_rate:

            chunk = chunk.sample(frac = sample_rate)

        chunks.append(chunk)

    return pd.concat(chunks)



def get_data(train_file, test_file = None, return_DMatrix = False, usecols = None):

	print("---Read in train data %s" % datetime.now())

	if usecols:

		train_data = pandas_read_large_csv(

				train_file, 

				chunksize = 100000, 

				usecols = usecols,

				dtype = np.float32

			)

	else:

		train_data = pandas_read_large_csv(

				train_file, 

				chunksize = 100000, 

				dtype = np.float32

			)



	train_id = train_data["Id"].astype(np.int32)

	if train_file == "data/train_numeric.csv":

		y = train_data["Response"].astype(np.int32)

		train_data.drop(labels = ["Id", "Response"], axis = 1, inplace = True)

	else:

		y = cPickle.load(open("y.pkl","rb"))

		train_data.drop(labels = "Id", axis = 1, inplace = True)

	gc.collect()



	if return_DMatrix:

		print("---Convert to DMatrix %s" % datetime.now())

		train_data = xgb.DMatrix(train_data, y)



	if test_file:

		print("---Read in test data %s" % datetime.now())

		if usecols:

			test_data = pandas_read_large_csv(

					test_file, 

					chunksize = 100000, 

					usecols = usecols,

					dtype = np.float32

				)

		else:

			test_data = pandas_read_large_csv(

					test_file, 

					chunksize = 100000, 

					dtype = np.float32

				)



		test_id = test_data["Id"].astype(np.int32)

		test_data.drop(labels = "Id", axis = 1, inplace = True)



		if return_DMatrix:

			print("---Convert to DMatrix %s" % datetime.now())

			test_data = xgb.DMatrix(test_data)

			gc.collect()

		return train_data, test_data, y



	return train_data, None, y
# the following is adopted from https://www.kaggle.com/cpmpml/bosch-production-line-performance/optimizing-probabilities-for-best-mcc/comments

@jit

def mcc(tp, tn, fp, fn):

    sup = tp * tn - fp * fn

    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if inf==0:

        return 0

    else:

        return sup / np.sqrt(inf)



@jit

def eval_mcc(y_true, y_prob, show=False):

    idx = np.argsort(y_prob)

    y_true_sort = y_true[idx]

    n = y_true.shape[0]

    nump = 1.0 * np.sum(y_true) # number of positive

    numn = n - nump # number of negative

    tp = nump

    tn = 0.0

    fp = numn

    fn = 0.0

    best_mcc = 0.0

    best_id = -1

    prev_proba = -1

    best_proba = -1

    mccs = np.zeros(n)

    for i in range(n):

        # all items with idx < i are predicted negative while others are predicted positive

        # only evaluate mcc when probability changes

        proba = y_prob[idx[i]]

        if proba != prev_proba:

            prev_proba = proba

            new_mcc = mcc(tp, tn, fp, fn)

            if new_mcc >= best_mcc:

                best_mcc = new_mcc

                best_id = i

                best_proba = proba

        mccs[i] = new_mcc

        if y_true_sort[i] == 1:

            tp -= 1.0

            fn += 1.0

        else:

            fp -= 1.0

            tn += 1.0

    if show:

        y_pred = (y_prob >= best_proba).astype(int)

        score = matthews_corrcoef(y_true, y_pred)

        print(score, best_mcc)

        #plt.plot(mccs)

        return best_proba, best_mcc, y_pred

    else:

        print(best_proba)

        return best_mcc



def roundn(yprob, scale):

    return np.around(y_prob * scale) / scale



def mcc_eval(y_prob, dtrain):

    y_true = dtrain.get_label()

    best_mcc = eval_mcc(y_true, y_prob)

    return 'MCC', best_mcc
numeric_feature_list = [

	'L1_S24_F829', 'L3_S41_F4006', 'L3_S41_F4004', 'L1_S25_F2131', 'L3_S41_F4008',

	'L1_S25_F2136', 'L3_S30_F3634', 'L3_S30_F3639','L3_S29_F3348','L3_S29_F3342', 

	'L3_S29_F3345', 'L3_S47_F4143', 'L3_S30_F3609', 'L1_S24_F993', 'L3_S31_F3834', 

	'L3_S30_F3494', 'L3_S37_F3950', 'L3_S36_F3918', 'L3_S35_F3889', 'L1_S24_F1180',

	'L3_S30_F3644', 'L0_S12_F334', 'L1_S24_F1818', 'L0_S12_F336', 'L0_S12_F330', 

	'L0_S12_F332', 'L1_S24_F1810', 'L1_S24_F1816', 'L0_S12_F338', 'L1_S24_F1814', 

	'L1_S24_F687', 'L0_S11_F310', 'L0_S11_F314', 'L2_S28_F3292', 'L2_S27_F3133', 

	'L0_S11_F318', 'L1_S24_F1467', 'L1_S25_F2449', 'L1_S24_F902', 'L3_S30_F3544', 

	'L0_S10_F274', 'L1_S24_F1733', 'L1_S25_F2086', 'L1_S25_F1892', 'L1_S24_F1326', 

	'L1_S24_F1738', 'L1_S25_F2247', 'L1_S25_F3001', 'L1_S25_F3009', 'L1_S25_F2847',

	'L1_S24_F1609', 'L3_S50_F4243', 'L1_S24_F1604', 'L3_S29_F3333', 'L3_S29_F3330', 

	'L3_S29_F3336', 'L1_S24_F691', 'L1_S24_F1134', 'L3_S29_F3339', 'L1_S24_F1130', 

	'L1_S24_F1036', 'L1_S24_F1426', 'L1_S24_F1783', 'L0_S2_F60', 'L3_S29_F3427', 

	'L0_S2_F64', 'L0_S23_F667', 'L3_S44_F4115', 'L0_S23_F663', 'L3_S44_F4112', 

	'L2_S26_F3073', 'L0_S20_F461', 'L1_S25_F2101', 'L0_S10_F224', 'L1_S25_F2066', 

	'L3_S30_F3604', 'L1_S24_F1850', 'L3_S30_F3709', 'L3_S30_F3764', 'L3_S30_F3769', 

	'L2_S28_F3248', 'L3_S30_F3504', 'L0_S21_F527', 'L3_S30_F3509', 'L0_S21_F522', 

	'L1_S24_F1672', 'L0_S11_F322', 'L0_S11_F326', 'L2_S26_F3125', 'L2_S27_F3140', 

	'L0_S19_F455', 'L2_S27_F3144', 'L0_S19_F459', 'L1_S24_F808', 'L3_S48_F4198', 

	'L1_S25_F2770', 'L1_S25_F2808', 'L3_S33_F3859', 'L3_S33_F3857', 'L3_S33_F3855',

	'L3_S29_F3354', 'L2_S26_F3106', 'L0_S22_F576', 'L0_S22_F571', 'L0_S10_F249', 

	'L3_S43_F4095', 'L0_S23_F639', 'L1_S24_F1072', 'L0_S10_F244', 'L3_S34_F3882', 

	'L3_S30_F3819', 'L3_S30_F3804', 'L0_S9_F175', 'L0_S9_F170', 'L3_S30_F3809', 

	'L1_S24_F892', 'L0_S23_F643', 'L1_S24_F1778', 'L1_S25_F3026', 'L3_S31_F3842', 

	'L1_S24_F1773', 'L0_S18_F439', 'L1_S24_F1490', 'L1_S24_F1298', 'L1_S24_F1512',

	'L3_S29_F3482', 'L2_S26_F3036', 'L1_S24_F1102', 'L3_S29_F3488', 'L1_S24_F1451',

	'L0_S14_F390', 'L1_S25_F1877', 'L1_S25_F1973', 'L3_S29_F3376', 'L3_S29_F3479', 

	'L3_S29_F3373', 'L1_S24_F1581', 'L3_S29_F3476', 'L3_S47_F4153', 'L3_S29_F3473', 

	'L0_S2_F56', 'L0_S21_F517', 'L0_S21_F512', 'L0_S14_F370', 'L1_S24_F1571', 

	'L0_S14_F374', 'L1_S24_F988', 'L1_S25_F2945', 'L1_S25_F2158', 'L0_S3_F100', 

	'L1_S25_F2155', 'L3_S30_F3799', 'L3_S30_F3794', 'L1_S24_F1808', 'L0_S13_F354', 

	'L0_S13_F356', 'L0_S12_F348', 'L1_S25_F2420', 'L3_S30_F3579', 'L3_S41_F4014', 

	'L3_S30_F3574', 'L1_S24_F1700', 'L3_S41_F4016', 'L1_S25_F2370', 'L1_S24_F1336', 

	'L2_S27_F3199', 'L0_S15_F418', 'L2_S27_F3210', 'L2_S27_F3214', 'L0_S15_F415',

	'L1_S25_F2789', 'L1_S25_F2837', 'L1_S24_F1632', 'L0_S6_F122', 'L0_S10_F219', 

	'L1_S24_F1798', 'L3_S29_F3430', 'L3_S29_F3433', 'L0_S7_F136', 'L3_S29_F3436', 

	'L0_S7_F138', 'L0_S23_F671', 'L0_S0_F22', 'L0_S0_F20', 'L3_S30_F3589', 'L2_S26_F3040',

	'L1_S24_F1567', 'L3_S30_F3584', 'L1_S24_F1848', 'L1_S25_F2111', 'L3_S32_F3850', 

	'L3_S40_F3982', 'L1_S25_F2051', 'L3_S40_F3986', 'L1_S24_F1844', 'L3_S40_F3984', 

	'L1_S24_F1846', 'L3_S30_F3754', 'L3_S29_F3324', 'L3_S29_F3327', 'L3_S30_F3759', 

	'L3_S29_F3321', 'L1_S24_F978', 'L2_S28_F3259', 'L0_S15_F397', 'L3_S30_F3534', 

	'L0_S11_F294', 'L0_S21_F497', 'L0_S11_F290', 'L3_S36_F3938', 'L3_S29_F3357', 

	'L1_S25_F2237', 'L0_S23_F631', 'L2_S27_F3155', 'L3_S43_F4090', 'L1_S25_F2239', 

	'L1_S24_F814', 'L1_S25_F2466', 'L1_S24_F1176', 'L0_S22_F611', 'L1_S24_F1068', 

	'L1_S25_F2997', 'L0_S10_F259', 'L0_S10_F254', 'L1_S25_F2487', 'L1_S24_F1690', 

	'L1_S24_F1346', 'L1_S24_F1695', 'L0_S9_F165', 'L0_S9_F160', 'L3_S48_F4196', 

	'L1_S25_F1909', 'L1_S24_F867', 'L3_S34_F3880', 'L1_S25_F2867', 'L1_S24_F1743', 

	'L1_S24_F1667', 'L0_S22_F546', 'L1_S24_F1748', 'L3_S29_F3315', 'L3_S41_F4020', 

	'L3_S41_F4023', 'L3_S29_F3318', 'L1_S24_F1520', 'L3_S41_F4026', 'L1_S24_F1110',

	'L0_S14_F386', 'L1_S25_F1968', 'L3_S29_F3360', 'L0_S2_F48', 'L1_S24_F1321', 

	'L3_S43_F4085', 'L3_S47_F4163', 'L3_S43_F4080', 'L0_S2_F44', 'L3_S29_F3407',

	'L0_S21_F507', 'L0_S21_F502', 'L0_S14_F362', 'L0_S3_F84', 'L3_S29_F3382',

	'L1_S24_F1235', 'L0_S12_F352', 'L0_S12_F350', 'L3_S30_F3664', 'L3_S30_F3669', 

	'L3_S30_F3784', 'L3_S33_F3873', 'L0_S0_F2', 'L0_S0_F0', 'L0_S0_F6', 'L1_S24_F925',

	'L0_S0_F8', 'L0_S5_F114', 'L0_S5_F116', 'L0_S3_F80', 'L1_S25_F2346', 'L1_S24_F1657',

	'L2_S27_F3206', 'L0_S9_F200', 'L2_S27_F3166', 'L2_S27_F3162', 'L0_S15_F406', 

	'L0_S15_F403', 'L0_S6_F132', 'L1_S25_F2797', 'L1_S24_F1622', 'L1_S24_F1627',

	'L1_S24_F1498', 'L1_S25_F2167', 'L3_S30_F3829', 'L1_S24_F1010', 'L0_S10_F229', 

	'L1_S24_F1494', 'L1_S24_F1718', 'L0_S23_F647', 'L1_S25_F2950', 'L1_S24_F1565',

	'L1_S24_F1713', 'L0_S0_F16', 'L0_S0_F14', 'L0_S0_F12', 'L0_S0_F10', 'L1_S24_F1098',

	'L0_S0_F18', 'L1_S24_F1575', 'L3_S41_F4011', 'L2_S26_F3051', 'L1_S24_F1573', 

	'L1_S25_F2126', 'L0_S17_F431', 'L1_S24_F1578', 'L0_S17_F433', 'L1_S25_F2121', 

	'L3_S30_F3624', 'L3_S30_F3749', 'L0_S1_F28', 'L3_S30_F3744', 'L0_S1_F24', 'L3_S30_F3629',

	'L0_S18_F449', 'L0_S9_F180', 'L0_S9_F185', 'L3_S29_F3351', 'L0_S3_F72', 'L1_S24_F1516', 

	'L3_S38_F3960', 'L0_S21_F477', 'L0_S21_F472', 'L1_S24_F968', 'L1_S24_F963', 

	'L1_S24_F1518', 'L3_S35_F3894', 'L3_S35_F3896', 'L0_S11_F286', 'L0_S11_F282', 

	'L3_S36_F3920', 'L1_S24_F1829', 'L1_S24_F1824', 'L1_S24_F1822', 'L1_S24_F1820',

	'L2_S27_F3129', 'L0_S11_F306', 'L1_S25_F2021', 'L0_S11_F302', 'L1_S25_F2456',

	'L1_S25_F2454', 'L0_S9_F195', 'L1_S25_F2458', 'L0_S9_F190', 'L1_S24_F1788', 

	'L0_S22_F606', 'L0_S22_F601', 'L3_S30_F3554', 'L1_S25_F2096', 'L0_S10_F264',

	'L1_S24_F1723', 'L1_S24_F1351', 'L3_S29_F3461', 'L0_S9_F155', 'L1_S24_F1685',

	'L1_S25_F2385', 'L1_S25_F3034', 'L3_S29_F3464', 'L1_S24_F1753', 'L1_S24_F1758',

	'L1_S24_F1539', 'L1_S24_F683', 'L0_S16_F426', 'L0_S16_F421', 'L1_S25_F2007', 

	'L1_S25_F1958', 'L3_S30_F3704', 'L0_S8_F144', 'L0_S0_F4', 'L0_S23_F619', 'L0_S22_F556',

	'L3_S29_F3379', 'L2_S26_F3121', 'L0_S22_F551', 'L3_S30_F3684', 'L3_S30_F3689',

	'L3_S31_F3846', 'L3_S35_F3913', 'L0_S3_F92', 'L2_S26_F3069', 'L0_S3_F96', 

	'L1_S24_F1544', 'L2_S26_F3062', 'L1_S25_F2176', 'L3_S30_F3679', 'L0_S12_F344', 

	'L3_S30_F3674', 'L0_S12_F346', 'L0_S12_F340', 'L1_S25_F2071', 'L0_S12_F342', 

	'L3_S30_F3774', 'L3_S44_F4118', 'L3_S33_F3865', 'L3_S38_F3952', 'L3_S38_F3956', 

	'L1_S24_F953', 'L1_S24_F958', 'L3_S30_F3514', 'L3_S30_F3519', 'L0_S21_F532', 

	'L1_S24_F1647', 'L1_S24_F882', 'L1_S24_F1316', 'L1_S24_F733', 'L0_S4_F104', 

	'L1_S24_F1207', 'L1_S24_F839', 'L0_S9_F210', 'L0_S4_F109', 'L1_S25_F2767', 'L0_S23_F623', 

	'L0_S7_F142', 'L0_S23_F627', 'L1_S24_F1008', 'L0_S22_F561', 'L2_S26_F3113', 

	'L1_S24_F1004', 'L1_S24_F1006', 'L1_S24_F1000', 'L0_S10_F234', 'L1_S24_F1361', 

	'L3_S47_F4138', 'L0_S2_F32', 'L0_S22_F591', 'L0_S22_F596', 'L0_S2_F36', 'L1_S24_F1768', 

	'L1_S25_F3017', 'L0_S23_F655', 'L0_S23_F651', 'L0_S23_F659'

	]



date_feature_list = [

    'L3_S43_D4062', 'L3_S51_D4255', 'L1_S24_D1163', 'L1_S24_D1511', 'L3_S50_D4242', 'L3_S32_D3852',

    'L0_S1_D30', 'L0_S2_D34', 'L0_S21_D469', 'L1_S25_D2138', 'L3_S34_D3875', 'L0_S23_D617', 

    'L1_S24_D1368', 'L1_S24_D1765', 'L1_S24_D1568', 'L3_S47_D4140', 'L1_S24_D1566', 'L0_S21_D474', 

    'L0_S8_D145', 'L1_S25_D1883', 'L1_S24_D999', 'L1_S25_D2445', 'L0_S16_D423', 'L1_S24_D1277', 

    'L0_S6_D120', 'L1_S24_D1062', 'L1_S24_D1770', 'L1_S24_D1674', 'L1_S24_D1576', 'L0_S19_D454',

    'L3_S43_D4082', 'L1_S24_D1570', 'L1_S25_D1898', 'L3_S39_D3966', 'L0_S15_D395', 'L1_S24_D697',

    'L3_S33_D3856', 'L1_S25_D2058', 'L3_S41_D3997', 'L0_S14_D360', 'L3_S30_D3506', 'L1_S24_D1155',

    'L0_S7_D137', 'L1_S24_D1018', 'L2_S28_D3223', 'L0_S11_D280', 'L1_S24_D772', 'L0_S12_D331', 

    'L3_S38_D3953', 'L1_S25_D1867', 'L2_S26_D3037', 'L1_S25_D2996', 'L1_S25_D2754', 'L3_S49_D4218', 

    'L0_S4_D106', 'L1_S24_D1809', 'L3_S29_D3474', 'L0_S20_D462', 'L0_S10_D221', 'L1_S24_D1558', 

    'L1_S24_D677', 'L1_S25_D2471', 'L3_S30_D3566', 'L1_S25_D1980', 'L1_S25_D2098', 'L1_S25_D2251', 

    'L0_S9_D152', 'L3_S37_D3942', 'L3_S49_D4208', 'L1_S25_D2788', 'L0_S9_D157', 'L0_S4_D111',

    'L1_S24_D1522', 'L3_S40_D3985', 'L0_S5_D115', 'L1_S25_D2674', 'L3_S40_D3981', 'L3_S48_D4194', 

    'L1_S24_D1186', 'L1_S25_D2248', 'L1_S25_D2957', 'L0_S3_D70', 'L1_S25_D2798', 'L0_S0_D1', 

    'L1_S25_D2790', 'L1_S24_D1536', 'L0_S13_D355', 'L1_S25_D1854', 'L3_S44_D4101', 'L1_S24_D807', 

    'L1_S24_D804', 'L1_S25_D1902', 'L1_S24_D801', 'L1_S25_D2230', 'L1_S24_D1583', 'L1_S25_D2238',

    'L2_S26_D3081', 'L1_S25_D2801', 'L1_S24_D1116', 'L3_S29_D3316', 'L3_S35_D3900', 'L0_S1_D26', 

    'L0_S10_D216', 'L2_S27_D3130', 'L0_S17_D432', 'L1_S24_D1826', 'L3_S30_D3496', 'L1_S24_D818', 

    'L1_S25_D2329', 'L3_S36_D3928', 'L1_S24_D813'

]
train_numeric = pandas_read_large_csv(

        "../input/train_numeric.csv", 

        chunksize = 100000, 

        usecols = numeric_feature_list + ["Response"],

        dtype = np.float32

    )

y = train_numeric["Response"].astype(np.int32)

train_numeric.drop(labels = "Response", axis = 1, inplace = True)



train_date = pandas_read_large_csv(

        "../input/train_date.csv",

        chunksize = 100000, 

        usecols = date_feature_list,

        dtype = np.float32

    )



train = pd.concat([train_numeric, train_date], axis = 1)

del [train_numeric, train_date]

gc.collect()
random_state = 36

X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size = 0.33, random_state = random_state)

train = xgb.DMatrix(X_train, y_train)

valid = xgb.DMatrix(X_valid, y_valid)



del [X_train, X_valid, y_train, y_valid]

gc.collect()
print("---fit xgboost")

params = {

        "objective" : "binary:logistic",

        "base_score" : 0.005,

        "learning_rate" : 0.05, 

        "max_depth" : 8

    }



np.random.seed(1412)

bst = xgb.train(

		params = params,

		dtrain = train,

		num_boost_round = 200,

		evals = [(train, "train"), (valid, "valid")],

		feval = mcc_eval,

		maximize = True,

		early_stopping_rounds = 5,

		verbose_eval = True,

    )
features_nums = np.where(clf.feature_importances_ > 0.0008)[0].tolist() 

features_nums = [n + 1 for n in features_nums]
del train_numeric

del clf

gc.collect()



train_numeric = pandas_read_large_csv("../input/train_numeric.csv", usecols = features_nums, chunksize = 100000, dtype = np.float32)

train_numeric.info()
clf = XGBClassifier(

		learning_rate = 0.2,

		silent = True,

		base_score = 0.005,

		n_estimators = 300,

		seed = 36

	)



clf.fit(train_numeric, y)
print("---Try find a threshold")

predProba = clf.predict_proba(train_numeric)[:,1]

_, threshold = find_proba_threshold(predProba, y, verbose = True)





del [train_numeric, y]

gc.collect()