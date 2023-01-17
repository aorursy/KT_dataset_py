#Define X_trainset and X_testset

X_trainset = normalized_data_train.values[:,:5]

X_testset = normalized_data_test.values[:,:5]

X_testset2 = normalized_data_test2.values[:,:5]

#Define Y_trainset and Y_testset

Y_trainset = data_train.Occupancy

Y_testset = data_test.Occupancy

Y_testset2 = data_test2.Occupancy

#-------------------------------------------------------------------------------------------------------