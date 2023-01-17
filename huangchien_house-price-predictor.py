## Baseline model : Lasso regression (0.48 RMSE) 

import argparse
import pandas as pd
import csv
import numpy as np

## import machine learning package.
from sklearn import preprocessing, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model, svm

## Not recommand without preprocess categorical data.
def extract_data(tra_data_frame, extract, data_type):
    for col in extract:
        avg = tra_data_frame[col].mean()
        tra_data_frame[col] = tra_data_frame[col].fillna(avg)
    ## Collect result & Replace the variable name.
    if data_type == 'train':
        raw_tra_dat, tra_label = tra_data_frame[extract].to_numpy(), tra_data_frame['SalePrice']
        return raw_tra_dat, tra_label
    else:
        return tra_data_frame[extract].to_numpy()
    
## Preprocess the data frame structure, and transfer into numpy data structure.
def transf_df2numpy(tra_data_frame, val_cols):
    all_cols = list(tra_data_frame.keys()) 
    ## extract the categorical columns, and then replace the variable name.
    for col in val_cols : all_cols.remove(col) ; categ_cols = all_cols
    
    ## preprocess the value type columns.
    for col in val_cols:
        avg = tra_data_frame[col].mean()
        tra_data_frame[col] = tra_data_frame[col].fillna(avg)
    ## FIXME : The val_cols are not whole value type of data.
    print(categ_cols)
    ## Transfer the categorical type columns into multiple dummy columns with binary value(0/1). 
    ## HACKME : If you already know the every value of categorical data in each column,
    ##          you may allow to use map function to transfer the categorical type cols :
    ##            df['dummy_col'].map({'cat_val0':val0, 'cat_val1':val1,...}).astype(valTyp)
    one_hot_df = pd.get_dummies(data=tra_data_frame, columns=categ_cols)
    print('cat\n\n\n', one_hot_df[:])
    traing_data = one_hot_df.values
    label = tra_data_frame['SalePrice'].to_numpy()
    return traing_data, label

def prepro_data(raw_data):
    minmax_scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return minmax_scalar.fit_transform(raw_data)

def validation(LinReg, valSet, valLab):
    vali_pred = LinReg.predict(valSet) 
    print('Validation metric : \n')
    print('r2 score : ', metrics.r2_score(valLab, vali_pred))
    print('mean absoulte error : ', metrics.mean_absolute_error(valLab, vali_pred))
    print('\n maximum error : ', metrics.max_error(valLab, vali_pred))
    
def main(args):
    ## self-define key word list of input feature
    keywd_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', \
                  'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', 'GarageArea']

    init_path = '/kaggle/input/house-prices-advanced-regression-techniques'
    tra_data_frame = pd.read_csv(init_path+'/train.csv')
    tst_data_frame = pd.read_csv(init_path+'/test.csv')   

    ## extract the keyword, return corresponding columns data (no recommand)
    #all_tra_data = extract_data(tra_data_frame, extract=keywd_list)
    raw_tra_data, tra_label = extract_data(tra_data_frame, extract=keywd_cols, data_type='train')
    prepro_tra_data = prepro_data(raw_tra_data)
    
    ## Feature selection via randomized decision tree(s) : 
    fea_selector = ExtraTreesClassifier()
    fea_selector.fit(prepro_tra_data, tra_label)
    
    ## FIXME : fea_selector.feature_importances_ return array of values,
    #            we attempt to select important feature by given values.
    #            (the higher, the more important the feature )
    #            we need to build the fea_filter, the following code is wrong way to do that.
    fea_filt = fea_selector.feature_importances_  # display the relative importance of each attribute.
    print('feature importance : ', fea_filt)
    #tra_data = prepro_tra_data[fea_filt]          # extract the better features.
    
    ## Split the train/valide dataset.
    msk = np.random.rand(len(prepro_tra_data)) < 0.8
    traSet, traLab = prepro_tra_data[msk], tra_label[msk]
    valSet, valLab = prepro_tra_data[~msk], tra_label[~msk]
    
    if args.show_data:
        print('Stage 1 : present all data and label : \n\n')
        print('Training set : \n', traSet)
        print('\n Training label : \n', traLab)
    
    ## prepare the testing data.
    tstDat = prepro_data(extract_data(tst_data_frame, extract=keywd_cols, data_type='test'))
    if args.show_data:
        print('\n\nTesting set : \n', tstDat)
    
    ## Build regression model as baseline of prediction :
    #LinReg = linear_model.BayesianRidge() ## Lasso is better.
    LinReg = linear_model.Lasso(alpha=0.1)
    
    LinReg.fit(X=traSet, y=traLab)
    
    ## TODO : complete validation.
    print('\n\n\nStage 2 validation : ')
    validation(LinReg, valSet, valLab)
    
    ## Prediction phase : 
    pred_price = LinReg.predict(tstDat) ## Baseline prediction .. need to imrpove..
    ## TODO : the prediction result should set position : i.e. 2.56, instead of 2.56788...
    print('\n\n\nprediction result : \n\n', pred_price)
    
    ## Stroe the prediction result into csv file
    tstId = tst_data_frame['Id'].to_numpy()
    pred_result = [tstId, pred_price]
    
    with open("houseprice_pred.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(*pred_result))

if __name__ == "__main__":
    ## TODO : for further add the user define parameters..
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_data', type=bool, default=False, help='none')
    parser.add_argument('--all_in', type=bool, default=False, help='do not split validation set, all data use to training')
    args = parser.parse_args(args=[])  ## parse_args(args=[]) due to notebook limitation.
    main(args)
## Dense model : RMSE ()
import argparse
import pandas as pd
import csv
import numpy as np

## import machine learning package.
from sklearn import preprocessing, metrics
from sklearn.ensemble import ExtraTreesClassifier
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

## Not recommand without preprocess categorical data.
def extract_data(tra_data_frame, extract, data_type):
    for col in extract:
        avg = tra_data_frame[col].mean()
        tra_data_frame[col] = tra_data_frame[col].fillna(avg)
    ## Collect result & Replace the variable name.
    if data_type == 'train':
        raw_tra_dat, tra_label = tra_data_frame[extract].to_numpy(), tra_data_frame['SalePrice']
        return raw_tra_dat, tra_label
    else:
        return tra_data_frame[extract].to_numpy()
    
## Preprocess the data frame structure, and transfer into numpy data structure.
def transf_df2numpy(tra_data_frame, val_cols):
    all_cols = list(tra_data_frame.keys()) 
    ## extract the categorical columns, and then replace the variable name.
    for col in val_cols : all_cols.remove(col) ; categ_cols = all_cols
    
    ## preprocess the value type columns.
    for col in val_cols:
        avg = tra_data_frame[col].mean()
        tra_data_frame[col] = tra_data_frame[col].fillna(avg)
    ## FIXME : The val_cols are not whole value type of data.
    print(categ_cols)
    ## Transfer the categorical type columns into multiple dummy columns with binary value(0/1). 
    ## HACKME : If you already know the every value of categorical data in each column,
    ##          you may allow to use map function to transfer the categorical type cols :
    ##            df['dummy_col'].map({'cat_val0':val0, 'cat_val1':val1,...}).astype(valTyp)
    one_hot_df = pd.get_dummies(data=tra_data_frame, columns=categ_cols)
    print('cat\n\n\n', one_hot_df[:])
    traing_data = one_hot_df.values
    label = tra_data_frame['SalePrice'].to_numpy()
    return traing_data, label

def prepro_data(raw_data):
    minmax_scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return minmax_scalar.fit_transform(raw_data)

def validation(LinReg, valSet, valLab):
    vali_pred = LinReg.predict(valSet) 
    print('Validation metric : \n')
    print('r2 score : ', metrics.r2_score(valLab, vali_pred))
    print('mean absoulte error : ', metrics.mean_absolute_error(valLab, vali_pred))
    print('\n maximum error : ', metrics.max_error(valLab, vali_pred))
    
def main(args):
    ## self-define key word list of input feature
    keywd_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', \
                  'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', 'GarageArea']

    init_path = '/kaggle/input/house-prices-advanced-regression-techniques'
    tra_data_frame = pd.read_csv(init_path+'/train.csv')
    tst_data_frame = pd.read_csv(init_path+'/test.csv')   

    ## extract the keyword, return corresponding columns data (no recommand)
    #all_tra_data = extract_data(tra_data_frame, extract=keywd_list)
    raw_tra_data, tra_label = extract_data(tra_data_frame, extract=keywd_cols, data_type='train')
    prepro_tra_data = prepro_data(raw_tra_data)
    
    ## Feature selection via randomized decision tree(s) : 
    fea_selector = ExtraTreesClassifier()
    fea_selector.fit(prepro_tra_data, tra_label)
    
    ## FIXME : fea_selector.feature_importances_ return array of values,
    #            we attempt to select important feature by given values.
    #            (the higher, the more important the feature )
    #            we need to build the fea_filter, the following code is wrong way to do that.
    fea_filt = fea_selector.feature_importances_  # display the relative importance of each attribute.
    fea_filt = [x+1 for x in fea_filt]
    print('feature importance : ', fea_filt)
    #tra_data = prepro_tra_data[fea_filt]          # extract the better features.
    
    ## Split the train/valide dataset.
    msk = np.random.rand(len(prepro_tra_data)) < 1.1
    traSet, traLab = prepro_tra_data[msk], tra_label[msk]
    valSet, valLab = prepro_tra_data[~msk], tra_label[~msk]
    
    if args.show_data:
        print('Stage 1 : present all data and label : \n\n')
        print('Training set : \n', traSet)
        print('\n Training label : \n', traLab)
    
    ## prepare the testing data.
    tstDat = prepro_data(extract_data(tst_data_frame, extract=keywd_cols, data_type='test'))
    if args.show_data:
        print('\n\nTesting set : \n', tstDat)
    
    epochs = 2000
    ## Build regression model as baseline of prediction :
    early_stopping = EarlyStopping(monitor='loss', patience=(epochs/10), verbose=2)

    model = Sequential()
    model.add(Dense(16, input_shape=(9,), activation='selu', \
                    use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(Dense(32, activation='selu',\
                    use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(Dense(1, activation='selu',\
                    use_bias=True, kernel_initializer='glorot_uniform'))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mse'])
    model.fit(traSet, traLab, batch_size=4, epochs=epochs, verbose=2, callbacks=[early_stopping], class_weight=fea_filt)
    
    ## Prediction phase : 
    pred_price = model.predict(tstDat) ## Baseline prediction .. need to imrpove..
    print(type(pred_price))
    ## TODO : the prediction result should set position : i.e. 2.56, instead of 2.56788...
    for idx in range(len(pred_price)):
        pred_price[idx] = pred_price[idx][0]
    
    print('\n\n\nprediction result : \n\n', pred_price)
    
    ## Stroe the prediction result into csv file
    pred_price = pred_price.reshape(-1)
    tstId = tst_data_frame['Id'].to_numpy()
    pred_result = [tstId, pred_price]
    
    with open("houseprice_pred.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(*pred_result))

if __name__ == "__main__":
    ## TODO : for further add the user define parameters..
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_data', type=bool, default=True, help='none')
    args = parser.parse_args(args=[])  ## parse_args(args=[]) due to notebook limitation.
    main(args)
import numpy as np
a = np.array([[1], [1], [1], [1], [1]])

print(a.reshape(-1))
del traLab
