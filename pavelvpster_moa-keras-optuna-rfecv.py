import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')



train.shape
train_target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



train_target.shape
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



test.shape
# From https://www.kaggle.com/carlmcbrideellis/moa-setting-ctl-vehicle-0-improves-score



train.at[train['cp_type'].str.contains('ctl_vehicle'),train.filter(regex='-.*').columns] = 0.0



test.at[test['cp_type'].str.contains('ctl_vehicle'),test.filter(regex='-.*').columns] = 0.0
train_size = train.shape[0]



traintest = pd.concat([train, test])



traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_type'], prefix='cp_type')], axis=1)

traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_time'], prefix='cp_time')], axis=1)

traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_dose'], prefix='cp_dose')], axis=1)



traintest = traintest.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)



train = traintest[:train_size]

test  = traintest[train_size:]



del traintest
train.shape
from sklearn.preprocessing import StandardScaler





g_columns = [ c for c in train.columns if 'g-' in c ]



scaler = StandardScaler()

train[g_columns] = scaler.fit_transform(train[g_columns])

test[g_columns] = scaler.transform(test[g_columns])
x_train = train.drop('sig_id', axis=1)



y_train = train_target.drop('sig_id', axis=1)



x_test = test.drop('sig_id', axis=1)
options = {

    'default': {

        'features': list(x_train.columns)

    }

}
def make_x(option):

    features = options[option]['features']

    return x_train[features], x_test[features]
from sklearn.feature_selection import RFECV

import lightgbm as lgb
params = {

    'objective': 'binary',

    'learning_rate': 0.05,

    'max_depth': -1,

    'num_leaves': 31,

    'num_threads': 4,

    'random_state': 42

}
# feature_selector = RFECV(lgb.LGBMClassifier(**params),

#                          step=10, min_features_to_select=500, scoring='neg_log_loss',

#                          cv=3, verbose=1, n_jobs=-1)



# feature_selector.fit(x_train, y_train['antiviral'])



# print('Features selected:', feature_selector.n_features_)
# selected_features = [f for f in x_train.columns[feature_selector.ranking_ == 1]]



# print(selected_features)
# from RFECV in Version 18 (500 features)

options['500'] = {

    'features': ['g-0', 'g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8', 'g-9', 'g-10', 'g-11', 'g-12', 'g-13', 'g-14', 'g-15', 'g-16', 'g-17', 'g-18', 'g-19', 'g-20', 'g-21', 'g-22', 'g-23', 'g-24', 'g-25', 'g-26', 'g-27', 'g-28', 'g-29', 'g-30', 'g-31', 'g-32', 'g-33', 'g-34', 'g-35', 'g-36', 'g-37', 'g-38', 'g-39', 'g-40', 'g-41', 'g-42', 'g-43', 'g-44', 'g-45', 'g-46', 'g-47', 'g-48', 'g-49', 'g-50', 'g-51', 'g-52', 'g-53', 'g-54', 'g-55', 'g-56', 'g-57', 'g-58', 'g-59', 'g-60', 'g-61', 'g-62', 'g-63', 'g-64', 'g-65', 'g-66', 'g-67', 'g-68', 'g-69', 'g-70', 'g-71', 'g-72', 'g-73', 'g-74', 'g-75', 'g-76', 'g-77', 'g-78', 'g-79', 'g-80', 'g-81', 'g-82', 'g-83', 'g-84', 'g-85', 'g-86', 'g-87', 'g-88', 'g-89', 'g-90', 'g-91', 'g-92', 'g-93', 'g-94', 'g-95', 'g-96', 'g-97', 'g-98', 'g-99', 'g-100', 'g-101', 'g-102', 'g-103', 'g-104', 'g-105', 'g-106', 'g-107', 'g-108', 'g-109', 'g-110', 'g-111', 'g-112', 'g-113', 'g-114', 'g-115', 'g-116', 'g-117', 'g-118', 'g-121', 'g-122', 'g-123', 'g-125', 'g-126', 'g-127', 'g-128', 'g-129', 'g-130', 'g-131', 'g-132', 'g-133', 'g-139', 'g-144', 'g-149', 'g-152', 'g-153', 'g-154', 'g-155', 'g-158', 'g-161', 'g-162', 'g-165', 'g-166', 'g-167', 'g-168', 'g-169', 'g-170', 'g-171', 'g-172', 'g-173', 'g-174', 'g-175', 'g-176', 'g-177', 'g-178', 'g-179', 'g-180', 'g-182', 'g-185', 'g-188', 'g-189', 'g-191', 'g-192', 'g-193', 'g-194', 'g-195', 'g-196', 'g-197', 'g-198', 'g-199', 'g-200', 'g-201', 'g-202', 'g-203', 'g-204', 'g-209', 'g-211', 'g-213', 'g-214', 'g-217', 'g-220', 'g-221', 'g-222', 'g-223', 'g-225', 'g-226', 'g-229', 'g-233', 'g-239', 'g-240', 'g-243', 'g-244', 'g-246', 'g-247', 'g-250', 'g-254', 'g-255', 'g-256', 'g-257', 'g-259', 'g-261', 'g-262', 'g-265', 'g-269', 'g-270', 'g-271', 'g-273', 'g-279', 'g-282', 'g-284', 'g-286', 'g-289', 'g-292', 'g-294', 'g-295', 'g-298', 'g-299', 'g-300', 'g-303', 'g-306', 'g-310', 'g-311', 'g-312', 'g-314', 'g-316', 'g-317', 'g-318', 'g-320', 'g-321', 'g-323', 'g-326', 'g-327', 'g-328', 'g-332', 'g-333', 'g-335', 'g-339', 'g-340', 'g-341', 'g-346', 'g-347', 'g-351', 'g-353', 'g-355', 'g-357', 'g-360', 'g-364', 'g-365', 'g-370', 'g-375', 'g-383', 'g-385', 'g-388', 'g-389', 'g-390', 'g-391', 'g-392', 'g-399', 'g-401', 'g-402', 'g-403', 'g-404', 'g-409', 'g-417', 'g-423', 'g-424', 'g-425', 'g-426', 'g-428', 'g-429', 'g-430', 'g-431', 'g-433', 'g-434', 'g-438', 'g-439', 'g-440', 'g-441', 'g-443', 'g-444', 'g-445', 'g-446', 'g-447', 'g-448', 'g-449', 'g-450', 'g-451', 'g-452', 'g-453', 'g-454', 'g-455', 'g-456', 'g-460', 'g-461', 'g-463', 'g-465', 'g-467', 'g-470', 'g-473', 'g-474', 'g-475', 'g-476', 'g-481', 'g-483', 'g-485', 'g-486', 'g-489', 'g-490', 'g-495', 'g-500', 'g-501', 'g-502', 'g-504', 'g-509', 'g-513', 'g-516', 'g-517', 'g-518', 'g-527', 'g-531', 'g-535', 'g-540', 'g-542', 'g-543', 'g-544', 'g-547', 'g-548', 'g-549', 'g-552', 'g-556', 'g-558', 'g-562', 'g-566', 'g-571', 'g-572', 'g-573', 'g-574', 'g-578', 'g-584', 'g-585', 'g-590', 'g-592', 'g-595', 'g-596', 'g-600', 'g-602', 'g-603', 'g-605', 'g-606', 'g-611', 'g-612', 'g-615', 'g-617', 'g-618', 'g-619', 'g-620', 'g-623', 'g-624', 'g-625', 'g-626', 'g-627', 'g-628', 'g-629', 'g-630', 'g-631', 'g-632', 'g-633', 'g-634', 'g-635', 'g-636', 'g-637', 'g-638', 'g-639', 'g-640', 'g-643', 'g-645', 'g-646', 'g-648', 'g-651', 'g-654', 'g-657', 'g-658', 'g-659', 'g-660', 'g-661', 'g-662', 'g-663', 'g-664', 'g-665', 'g-666', 'g-668', 'g-670', 'g-672', 'g-673', 'g-674', 'g-675', 'g-677', 'g-678', 'g-679', 'g-685', 'g-689', 'g-695', 'g-697', 'g-698', 'g-699', 'g-701', 'g-702', 'g-703', 'g-704', 'g-707', 'g-709', 'g-710', 'g-712', 'g-713', 'g-714', 'g-715', 'g-721', 'g-722', 'g-724', 'g-725', 'g-726', 'g-734', 'g-736', 'g-737', 'g-740', 'g-742', 'g-743', 'g-745', 'g-747', 'g-750', 'g-751', 'g-752', 'g-753', 'g-756', 'g-757', 'g-758', 'g-759', 'g-760', 'g-761', 'g-762', 'g-763', 'g-764', 'g-766', 'g-767', 'g-768', 'g-769', 'g-770', 'g-771', 'c-0', 'c-1', 'c-2', 'c-3', 'c-4', 'c-5', 'c-11', 'c-12', 'c-14', 'c-19', 'c-20', 'c-21', 'c-22', 'c-23', 'c-25', 'c-27', 'c-32', 'c-35', 'c-36', 'c-37', 'c-40', 'c-47', 'c-54', 'c-55', 'c-57', 'c-58', 'c-61', 'c-62', 'c-63', 'c-65', 'c-66', 'c-71', 'c-74', 'c-75', 'c-77', 'c-78', 'c-79', 'c-80', 'c-81', 'c-82', 'c-83', 'c-84', 'c-85', 'c-86', 'c-87', 'c-88', 'c-89', 'c-90', 'c-91', 'c-92', 'c-93', 'c-97']

}
# from RFECV in Version 17 (600 features)

options['600'] = {

    'features': ['g-0', 'g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8', 'g-9', 'g-10', 'g-11', 'g-12', 'g-13', 'g-14', 'g-15', 'g-16', 'g-17', 'g-18', 'g-19', 'g-20', 'g-21', 'g-22', 'g-23', 'g-24', 'g-25', 'g-26', 'g-27', 'g-28', 'g-29', 'g-30', 'g-31', 'g-32', 'g-33', 'g-34', 'g-35', 'g-36', 'g-37', 'g-38', 'g-39', 'g-40', 'g-41', 'g-42', 'g-43', 'g-44', 'g-45', 'g-46', 'g-47', 'g-48', 'g-49', 'g-50', 'g-51', 'g-52', 'g-53', 'g-54', 'g-55', 'g-56', 'g-57', 'g-58', 'g-59', 'g-60', 'g-61', 'g-62', 'g-63', 'g-64', 'g-65', 'g-66', 'g-67', 'g-68', 'g-69', 'g-70', 'g-71', 'g-72', 'g-73', 'g-74', 'g-75', 'g-76', 'g-77', 'g-78', 'g-79', 'g-80', 'g-81', 'g-82', 'g-83', 'g-84', 'g-85', 'g-86', 'g-87', 'g-88', 'g-89', 'g-90', 'g-91', 'g-92', 'g-93', 'g-94', 'g-95', 'g-96', 'g-97', 'g-98', 'g-99', 'g-100', 'g-101', 'g-102', 'g-103', 'g-104', 'g-105', 'g-106', 'g-107', 'g-108', 'g-109', 'g-110', 'g-111', 'g-112', 'g-113', 'g-114', 'g-115', 'g-116', 'g-117', 'g-118', 'g-119', 'g-120', 'g-121', 'g-122', 'g-123', 'g-124', 'g-125', 'g-126', 'g-127', 'g-128', 'g-129', 'g-130', 'g-131', 'g-132', 'g-133', 'g-134', 'g-135', 'g-136', 'g-137', 'g-138', 'g-139', 'g-140', 'g-141', 'g-142', 'g-143', 'g-144', 'g-145', 'g-146', 'g-147', 'g-148', 'g-149', 'g-150', 'g-151', 'g-152', 'g-153', 'g-154', 'g-155', 'g-156', 'g-157', 'g-158', 'g-159', 'g-160', 'g-161', 'g-162', 'g-163', 'g-164', 'g-165', 'g-166', 'g-167', 'g-168', 'g-169', 'g-170', 'g-171', 'g-172', 'g-173', 'g-174', 'g-175', 'g-176', 'g-177', 'g-178', 'g-179', 'g-180', 'g-181', 'g-182', 'g-183', 'g-184', 'g-185', 'g-186', 'g-187', 'g-188', 'g-189', 'g-190', 'g-191', 'g-192', 'g-193', 'g-194', 'g-195', 'g-196', 'g-197', 'g-198', 'g-199', 'g-200', 'g-201', 'g-202', 'g-203', 'g-204', 'g-205', 'g-206', 'g-207', 'g-208', 'g-209', 'g-210', 'g-211', 'g-212', 'g-213', 'g-214', 'g-215', 'g-217', 'g-220', 'g-221', 'g-222', 'g-223', 'g-225', 'g-226', 'g-229', 'g-233', 'g-239', 'g-240', 'g-241', 'g-242', 'g-243', 'g-244', 'g-245', 'g-246', 'g-247', 'g-250', 'g-254', 'g-255', 'g-256', 'g-257', 'g-259', 'g-261', 'g-262', 'g-265', 'g-269', 'g-270', 'g-271', 'g-272', 'g-273', 'g-279', 'g-282', 'g-284', 'g-286', 'g-289', 'g-292', 'g-294', 'g-295', 'g-298', 'g-299', 'g-300', 'g-303', 'g-306', 'g-310', 'g-311', 'g-312', 'g-314', 'g-316', 'g-317', 'g-318', 'g-320', 'g-321', 'g-323', 'g-326', 'g-327', 'g-328', 'g-332', 'g-333', 'g-335', 'g-339', 'g-340', 'g-341', 'g-346', 'g-347', 'g-351', 'g-353', 'g-355', 'g-357', 'g-360', 'g-364', 'g-365', 'g-370', 'g-375', 'g-383', 'g-385', 'g-388', 'g-389', 'g-390', 'g-391', 'g-392', 'g-399', 'g-401', 'g-402', 'g-403', 'g-404', 'g-409', 'g-417', 'g-423', 'g-424', 'g-425', 'g-426', 'g-428', 'g-429', 'g-430', 'g-431', 'g-433', 'g-434', 'g-438', 'g-439', 'g-440', 'g-441', 'g-443', 'g-444', 'g-445', 'g-446', 'g-447', 'g-448', 'g-449', 'g-450', 'g-451', 'g-452', 'g-453', 'g-454', 'g-455', 'g-456', 'g-460', 'g-461', 'g-463', 'g-465', 'g-467', 'g-470', 'g-473', 'g-474', 'g-475', 'g-476', 'g-481', 'g-483', 'g-485', 'g-486', 'g-489', 'g-490', 'g-495', 'g-500', 'g-501', 'g-502', 'g-504', 'g-509', 'g-513', 'g-516', 'g-517', 'g-518', 'g-527', 'g-531', 'g-535', 'g-540', 'g-542', 'g-543', 'g-544', 'g-547', 'g-548', 'g-549', 'g-552', 'g-556', 'g-558', 'g-562', 'g-566', 'g-571', 'g-572', 'g-573', 'g-574', 'g-578', 'g-584', 'g-585', 'g-590', 'g-592', 'g-595', 'g-596', 'g-599', 'g-600', 'g-601', 'g-602', 'g-603', 'g-604', 'g-605', 'g-606', 'g-607', 'g-608', 'g-609', 'g-610', 'g-611', 'g-612', 'g-613', 'g-614', 'g-615', 'g-616', 'g-617', 'g-618', 'g-619', 'g-620', 'g-621', 'g-622', 'g-623', 'g-624', 'g-625', 'g-626', 'g-627', 'g-628', 'g-629', 'g-630', 'g-631', 'g-632', 'g-633', 'g-634', 'g-635', 'g-636', 'g-637', 'g-638', 'g-639', 'g-640', 'g-643', 'g-645', 'g-646', 'g-647', 'g-648', 'g-651', 'g-654', 'g-657', 'g-658', 'g-659', 'g-660', 'g-661', 'g-662', 'g-663', 'g-664', 'g-665', 'g-666', 'g-668', 'g-670', 'g-672', 'g-673', 'g-674', 'g-675', 'g-676', 'g-677', 'g-678', 'g-679', 'g-680', 'g-685', 'g-688', 'g-689', 'g-690', 'g-691', 'g-695', 'g-697', 'g-698', 'g-699', 'g-700', 'g-701', 'g-702', 'g-703', 'g-704', 'g-707', 'g-708', 'g-709', 'g-710', 'g-711', 'g-712', 'g-713', 'g-714', 'g-715', 'g-716', 'g-717', 'g-718', 'g-719', 'g-720', 'g-721', 'g-722', 'g-723', 'g-724', 'g-725', 'g-726', 'g-728', 'g-729', 'g-730', 'g-731', 'g-732', 'g-733', 'g-734', 'g-736', 'g-737', 'g-739', 'g-740', 'g-742', 'g-743', 'g-744', 'g-745', 'g-747', 'g-748', 'g-749', 'g-750', 'g-751', 'g-752', 'g-753', 'g-756', 'g-757', 'g-758', 'g-759', 'g-760', 'g-761', 'g-762', 'g-763', 'g-764', 'g-766', 'g-767', 'g-768', 'g-769', 'g-770', 'g-771', 'c-0', 'c-1', 'c-2', 'c-3', 'c-4', 'c-5', 'c-10', 'c-11', 'c-12', 'c-13', 'c-14', 'c-15', 'c-16', 'c-17', 'c-18', 'c-19', 'c-20', 'c-21', 'c-22', 'c-23', 'c-24', 'c-25', 'c-26', 'c-27', 'c-32', 'c-35', 'c-36', 'c-37', 'c-40', 'c-43', 'c-47', 'c-48', 'c-52', 'c-53', 'c-54', 'c-55', 'c-56', 'c-57', 'c-58', 'c-61', 'c-62', 'c-63', 'c-64', 'c-65', 'c-66', 'c-67', 'c-68', 'c-69', 'c-70', 'c-71', 'c-72', 'c-73', 'c-74', 'c-75', 'c-76', 'c-77', 'c-78', 'c-79', 'c-80', 'c-81', 'c-82', 'c-83', 'c-84', 'c-85', 'c-86', 'c-87', 'c-88', 'c-89', 'c-90', 'c-91', 'c-92', 'c-93', 'c-94', 'c-97']

}
# from RFECV in Version 13 (700 features)

options['700'] = {

    'features': ['g-0', 'g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8', 'g-9', 'g-10', 'g-11', 'g-12', 'g-13', 'g-14', 'g-15', 'g-16', 'g-17', 'g-18', 'g-19', 'g-20', 'g-21', 'g-22', 'g-23', 'g-24', 'g-25', 'g-26', 'g-27', 'g-28', 'g-29', 'g-30', 'g-31', 'g-32', 'g-33', 'g-34', 'g-35', 'g-36', 'g-37', 'g-38', 'g-39', 'g-40', 'g-41', 'g-42', 'g-43', 'g-44', 'g-45', 'g-46', 'g-47', 'g-48', 'g-49', 'g-50', 'g-51', 'g-52', 'g-53', 'g-54', 'g-55', 'g-56', 'g-57', 'g-58', 'g-59', 'g-60', 'g-61', 'g-62', 'g-63', 'g-64', 'g-65', 'g-66', 'g-67', 'g-68', 'g-69', 'g-70', 'g-71', 'g-72', 'g-73', 'g-74', 'g-75', 'g-76', 'g-77', 'g-78', 'g-79', 'g-80', 'g-81', 'g-82', 'g-83', 'g-84', 'g-85', 'g-86', 'g-87', 'g-88', 'g-89', 'g-90', 'g-91', 'g-92', 'g-93', 'g-94', 'g-95', 'g-96', 'g-97', 'g-98', 'g-99', 'g-100', 'g-101', 'g-102', 'g-103', 'g-104', 'g-105', 'g-106', 'g-107', 'g-108', 'g-109', 'g-110', 'g-111', 'g-112', 'g-113', 'g-114', 'g-115', 'g-116', 'g-117', 'g-118', 'g-119', 'g-120', 'g-121', 'g-122', 'g-123', 'g-124', 'g-125', 'g-126', 'g-127', 'g-128', 'g-129', 'g-130', 'g-131', 'g-132', 'g-133', 'g-134', 'g-135', 'g-136', 'g-137', 'g-138', 'g-139', 'g-140', 'g-141', 'g-142', 'g-143', 'g-144', 'g-145', 'g-146', 'g-147', 'g-148', 'g-149', 'g-150', 'g-151', 'g-152', 'g-153', 'g-154', 'g-155', 'g-156', 'g-157', 'g-158', 'g-159', 'g-160', 'g-161', 'g-162', 'g-163', 'g-164', 'g-165', 'g-166', 'g-167', 'g-168', 'g-169', 'g-170', 'g-171', 'g-172', 'g-173', 'g-174', 'g-175', 'g-176', 'g-177', 'g-178', 'g-179', 'g-180', 'g-181', 'g-182', 'g-183', 'g-184', 'g-185', 'g-186', 'g-187', 'g-188', 'g-189', 'g-190', 'g-191', 'g-192', 'g-193', 'g-194', 'g-195', 'g-196', 'g-197', 'g-198', 'g-199', 'g-200', 'g-201', 'g-202', 'g-203', 'g-204', 'g-205', 'g-206', 'g-207', 'g-208', 'g-209', 'g-210', 'g-211', 'g-212', 'g-213', 'g-214', 'g-215', 'g-216', 'g-217', 'g-218', 'g-219', 'g-220', 'g-221', 'g-222', 'g-223', 'g-224', 'g-225', 'g-226', 'g-227', 'g-228', 'g-229', 'g-230', 'g-231', 'g-232', 'g-233', 'g-234', 'g-235', 'g-236', 'g-237', 'g-238', 'g-239', 'g-240', 'g-241', 'g-242', 'g-243', 'g-244', 'g-245', 'g-246', 'g-247', 'g-248', 'g-249', 'g-250', 'g-251', 'g-252', 'g-253', 'g-254', 'g-255', 'g-256', 'g-257', 'g-258', 'g-259', 'g-260', 'g-261', 'g-262', 'g-263', 'g-264', 'g-265', 'g-266', 'g-267', 'g-268', 'g-269', 'g-270', 'g-271', 'g-272', 'g-273', 'g-274', 'g-275', 'g-279', 'g-282', 'g-284', 'g-285', 'g-286', 'g-287', 'g-288', 'g-289', 'g-290', 'g-291', 'g-292', 'g-293', 'g-294', 'g-295', 'g-296', 'g-297', 'g-298', 'g-299', 'g-300', 'g-301', 'g-302', 'g-303', 'g-304', 'g-305', 'g-306', 'g-307', 'g-310', 'g-311', 'g-312', 'g-314', 'g-316', 'g-317', 'g-318', 'g-319', 'g-320', 'g-321', 'g-322', 'g-323', 'g-324', 'g-325', 'g-326', 'g-327', 'g-328', 'g-332', 'g-333', 'g-335', 'g-339', 'g-340', 'g-341', 'g-342', 'g-343', 'g-344', 'g-345', 'g-346', 'g-347', 'g-348', 'g-349', 'g-350', 'g-351', 'g-352', 'g-353', 'g-354', 'g-355', 'g-357', 'g-360', 'g-364', 'g-365', 'g-370', 'g-375', 'g-383', 'g-385', 'g-388', 'g-389', 'g-390', 'g-391', 'g-392', 'g-399', 'g-401', 'g-402', 'g-403', 'g-404', 'g-405', 'g-406', 'g-407', 'g-409', 'g-417', 'g-423', 'g-424', 'g-425', 'g-426', 'g-428', 'g-429', 'g-430', 'g-431', 'g-433', 'g-434', 'g-435', 'g-436', 'g-438', 'g-439', 'g-440', 'g-441', 'g-443', 'g-444', 'g-445', 'g-446', 'g-447', 'g-448', 'g-449', 'g-450', 'g-451', 'g-452', 'g-453', 'g-454', 'g-455', 'g-456', 'g-460', 'g-461', 'g-463', 'g-465', 'g-467', 'g-470', 'g-473', 'g-474', 'g-475', 'g-476', 'g-481', 'g-483', 'g-485', 'g-486', 'g-489', 'g-490', 'g-495', 'g-500', 'g-501', 'g-502', 'g-504', 'g-509', 'g-513', 'g-516', 'g-517', 'g-518', 'g-527', 'g-531', 'g-535', 'g-540', 'g-542', 'g-543', 'g-544', 'g-547', 'g-548', 'g-549', 'g-552', 'g-556', 'g-558', 'g-562', 'g-566', 'g-571', 'g-572', 'g-573', 'g-574', 'g-578', 'g-584', 'g-585', 'g-590', 'g-592', 'g-595', 'g-596', 'g-599', 'g-600', 'g-601', 'g-602', 'g-603', 'g-604', 'g-605', 'g-606', 'g-607', 'g-608', 'g-609', 'g-610', 'g-611', 'g-612', 'g-613', 'g-614', 'g-615', 'g-616', 'g-617', 'g-618', 'g-619', 'g-620', 'g-621', 'g-622', 'g-623', 'g-624', 'g-625', 'g-626', 'g-627', 'g-628', 'g-629', 'g-630', 'g-631', 'g-632', 'g-633', 'g-634', 'g-635', 'g-636', 'g-637', 'g-638', 'g-639', 'g-640', 'g-641', 'g-642', 'g-643', 'g-644', 'g-645', 'g-646', 'g-647', 'g-648', 'g-649', 'g-650', 'g-651', 'g-652', 'g-654', 'g-655', 'g-656', 'g-657', 'g-658', 'g-659', 'g-660', 'g-661', 'g-662', 'g-663', 'g-664', 'g-665', 'g-666', 'g-667', 'g-668', 'g-670', 'g-672', 'g-673', 'g-674', 'g-675', 'g-676', 'g-677', 'g-678', 'g-679', 'g-680', 'g-685', 'g-687', 'g-688', 'g-689', 'g-690', 'g-691', 'g-692', 'g-695', 'g-697', 'g-698', 'g-699', 'g-700', 'g-701', 'g-702', 'g-703', 'g-704', 'g-705', 'g-706', 'g-707', 'g-708', 'g-709', 'g-710', 'g-711', 'g-712', 'g-713', 'g-714', 'g-715', 'g-716', 'g-717', 'g-718', 'g-719', 'g-720', 'g-721', 'g-722', 'g-723', 'g-724', 'g-725', 'g-726', 'g-727', 'g-728', 'g-729', 'g-730', 'g-731', 'g-732', 'g-733', 'g-734', 'g-735', 'g-736', 'g-737', 'g-738', 'g-739', 'g-740', 'g-741', 'g-742', 'g-743', 'g-744', 'g-745', 'g-746', 'g-747', 'g-748', 'g-749', 'g-750', 'g-751', 'g-752', 'g-753', 'g-754', 'g-755', 'g-756', 'g-757', 'g-758', 'g-759', 'g-760', 'g-761', 'g-762', 'g-763', 'g-764', 'g-765', 'g-766', 'g-767', 'g-768', 'g-769', 'g-770', 'g-771', 'c-0', 'c-1', 'c-2', 'c-3', 'c-4', 'c-5', 'c-6', 'c-7', 'c-10', 'c-11', 'c-12', 'c-13', 'c-14', 'c-15', 'c-16', 'c-17', 'c-18', 'c-19', 'c-20', 'c-21', 'c-22', 'c-23', 'c-24', 'c-25', 'c-26', 'c-27', 'c-28', 'c-29', 'c-30', 'c-31', 'c-32', 'c-33', 'c-34', 'c-35', 'c-36', 'c-37', 'c-40', 'c-41', 'c-42', 'c-43', 'c-44', 'c-45', 'c-46', 'c-47', 'c-48', 'c-49', 'c-50', 'c-51', 'c-52', 'c-53', 'c-54', 'c-55', 'c-56', 'c-57', 'c-58', 'c-59', 'c-60', 'c-61', 'c-62', 'c-63', 'c-64', 'c-65', 'c-66', 'c-67', 'c-68', 'c-69', 'c-70', 'c-71', 'c-72', 'c-73', 'c-74', 'c-75', 'c-76', 'c-77', 'c-78', 'c-79', 'c-80', 'c-81', 'c-82', 'c-83', 'c-84', 'c-85', 'c-86', 'c-87', 'c-88', 'c-89', 'c-90', 'c-91', 'c-92', 'c-93', 'c-94', 'c-95', 'c-96', 'c-97']

}
import tensorflow as tf

import tensorflow_addons as tfa



from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import accuracy_score



import tensorflow.keras.backend as K
def make_layer(x, units, dropout_rate):

    t = tfa.layers.WeightNormalization(tf.keras.layers.Dense(units))(x)

    # t = tf.keras.layers.Dense(units)(x)

    t = tf.keras.layers.BatchNormalization()(t)

    t = tf.keras.layers.Activation('relu')(t)

    t = tf.keras.layers.Dropout(dropout_rate)(t)

    return t





def make_model(data, units, dropout_rates):

    

    inputs = tf.keras.layers.Input(shape=(data.shape[1],))

    x = tf.keras.layers.BatchNormalization()(inputs)



    for i in range(len(units)):

        u = units[i]

        d = dropout_rates[i]

        x = make_layer(x, u, d)

       

    y = tf.keras.layers.Dense(206, activation='sigmoid', name='dense_output')(x)

    

    model = tf.keras.Model(inputs=inputs, outputs=y)

    model.compile(loss='binary_crossentropy',

                  optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),

                  metrics=['accuracy'])

    return model
def fit_predict(n_splits, x_train, y_train, units, dropout_rates, epochs, x_test, verbose, random_state):



    histories = []

    scores = []

    y_preds = []



    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in cv.split(x_train, y_train):



        x_train_train = x_train.iloc[train_idx]

        y_train_train = y_train.iloc[train_idx]

        x_train_valid = x_train.iloc[valid_idx]

        y_train_valid = y_train.iloc[valid_idx]



        K.clear_session()



        estimator = make_model(x_train, units, dropout_rates)



        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,

                                              verbose=verbose, mode='min', restore_best_weights=True)



        rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5,

                                                  mode='min', verbose=verbose)



        history = estimator.fit(x_train_train, y_train_train,

                                batch_size=128, epochs=epochs, callbacks=[es, rl],

                                validation_data=(x_train_valid, y_train_valid),

                                verbose=verbose)

        

        if x_test is not None:

            y_part = estimator.predict(x_test)

            y_preds.append(y_part)



        histories.append(history)

        scores.append(history.history['val_loss'][-1])

    

    if x_test is not None:

        y_pred = np.mean(y_preds, axis=0)

    else:

        y_pred = None



    score = np.mean(scores)

    

    return y_pred, histories, score
import optuna



from logging import CRITICAL

optuna.logging.set_verbosity(CRITICAL)
def objective(trial):

    

    n_layers = trial.suggest_int('n_layers', 1, 5)

    

    units = []

    dropout_rates = []

    for i in range(n_layers):

        u = trial.suggest_categorical('units_{}'.format(i+1), [1024, 512, 256, 128])

        units.append(u)

        r = trial.suggest_loguniform('dropout_rate_{}'.format(i+1), 0.1, 0.5)

        dropout_rates.append(r)

    

    print('Units:', units, "Dropout rates:", dropout_rates)

    

    _, _, score = fit_predict(3, x_train, y_train, units, dropout_rates, 25, None, 0, 42)

    return score





# study = optuna.create_study(direction='minimize')

# study.optimize(objective, n_trials=75)
# params = study.best_trial.params

# params
# from Optuna in Version 8

params = {

    'n_layers': 5,

    'units_1': 128,

    'units_2': 256,

    'units_3': 512,

    'units_4': 256,

    'units_5': 1024,

    'dropout_rate_1': 0.3478936880741539,

    'dropout_rate_2': 0.3478936880741539,

    'dropout_rate_3': 0.3478936880741539,

    'dropout_rate_4': 0.3478936880741539,

    'dropout_rate_5': 0.3478936880741539

}



options['default']['params'] = params
# from Optuna in Version 13

params = {

    'n_layers': 3,

    'units_1': 1024,

    'units_2': 512,

    'units_3': 256,

    'dropout_rate_1': 0.4501813451502177,

    'dropout_rate_2': 0.4501813451502177,

    'dropout_rate_3': 0.4501813451502177

}



options['500']['params'] = params

options['600']['params'] = params

options['700']['params'] = params
def fit_predict_option(option, random_state):

    print('Option:', option)

    

    params = options[option]['params']

    

    n_layers = params['n_layers']

    units = []

    dropout_rates = []

    for i in range(n_layers):

        u = params['units_{}'.format(i+1)]

        units.append(u)

        d = params['dropout_rate_{}'.format(i+1)]

        dropout_rates.append(d)



    x_train_option, x_test_option = make_x(option)



    # 7, 100, 2

    y_pred, histories, score = fit_predict(7, x_train_option, y_train, units, dropout_rates, 100, x_test_option, 2, random_state)

    

    print('Score:', score)

    

    return y_pred, histories, score
y_preds = []



for option in options:

    y_pred, histories, score = fit_predict_option(option, 42)

    y_preds.append(y_pred)

    # break
len(y_preds)
y_pred = np.mean(y_preds, axis=0)



y_pred.shape
import matplotlib.pyplot as plt





fig, axs = plt.subplots(2, 2, figsize=(18,18))



# accuracy

for h in histories:

    axs[0,0].plot(h.history['accuracy'], color='g')

axs[0,0].set_title('Model accuracy - Train')

axs[0,0].set_ylabel('Accuracy')

axs[0,0].set_xlabel('Epoch')



for h in histories:

    axs[0,1].plot(h.history['val_accuracy'], color='b')

axs[0,1].set_title('Model accuracy - Test')

axs[0,1].set_ylabel('Accuracy')

axs[0,1].set_xlabel('Epoch')



# loss

for h in histories:

    axs[1,0].plot(h.history['loss'], color='g')

axs[1,0].set_title('Model loss - Train')

axs[1,0].set_ylabel('Loss')

axs[1,0].set_xlabel('Epoch')



for h in histories:

    axs[1,1].plot(h.history['val_loss'], color='b')

axs[1,1].set_title('Model loss - Test')

axs[1,1].set_ylabel('Loss')

axs[1,1].set_xlabel('Epoch')



fig.show()
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



columns = list(submission.columns)

columns.remove('sig_id')



for i in range(len(columns)):

    submission[columns[i]] = y_pred[:,i]



submission.to_csv('submission.csv', index=False)
submission.head()