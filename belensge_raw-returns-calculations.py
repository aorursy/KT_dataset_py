import numpy as np
import pandas as pd
pd.options.display.max_rows = 999
import matplotlib
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env() # You can only call make_env() once, so don't lose it!
(mkt_train_df, news_train_df) = env.get_training_data()
asset_codes_lst = mkt_train_df.assetCode.unique()
mkt_train_df_small = mkt_train_df[['time', 'assetCode', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10']].copy()
returns_fails_df = pd.DataFrame({'asset':[],
                                  'wrong_retCloPrevRaw1':[],
                                  'wrong_retOpPrevRaw1':[],
                                  'wrong_retCloPrevRaw10':[],
                                  'wrong_retOpPrevRaw10':[]})
for asset_code in asset_codes_lst[30:40]:
    asset_mkt = mkt_train_df_small[mkt_train_df_small.assetCode == asset_code].reset_index(drop=True)
    
    ret_close_1 = (asset_mkt.close - asset_mkt.close.shift(1)) / asset_mkt.close.shift(1)
    comp_retCloPrevRaw1 = np.round(asset_mkt.returnsClosePrevRaw1[1:],10) == np.round(ret_close_1[1:],10)
    
    ret_open_1 = (asset_mkt.open - asset_mkt.open.shift(1)) / asset_mkt.open.shift(1)
    comp_retOpPrevRaw1 = np.round(asset_mkt.returnsOpenPrevRaw1[1:],10) == np.round(ret_open_1[1:],10)
    
    ret_close_10 = (asset_mkt.close - asset_mkt.close.shift(10)) / asset_mkt.close.shift(10)
    comp_retCloPrevRaw10 = np.round(asset_mkt.returnsClosePrevRaw10[10:],10) == np.round(ret_close_10[10:],10)
    
    ret_open_10 = (asset_mkt.open - asset_mkt.open.shift(10)) / asset_mkt.open.shift(10)
    comp_retOpPrevRaw10 = np.round(asset_mkt.returnsOpenPrevRaw10[10:],10) == np.round(ret_open_10[10:],10)
    
    results = pd.DataFrame({'asset':asset_code,
                            'wrong_retCloPrevRaw1':[sum(~comp_retCloPrevRaw1)],
                            'wrong_retOpPrevRaw1':[sum(~comp_retOpPrevRaw1)],
                            'wrong_retCloPrevRaw10':[sum(~comp_retCloPrevRaw10)],
                            'wrong_retOpPrevRaw10':[sum(~comp_retOpPrevRaw10)]})
    returns_fails_df = returns_fails_df.append(results,ignore_index=True)

returns_fails_df
def show_diff_returns(asset,ret_type, days):
#     Creates a DataFrame with the returns from the input data and the corresponding returns 
#     calculated with (R1 - R2)/R2.
#     Args:
#         asset (str): assetCode
#         ret_type (str): 'Open'/'Close'
#         days (double): 1/10
#     Returns:
#         results (dataframe): input data returns, calcuated returns and percentage of error 
#         between these two returns, for the input assetCode and showing only the ones that differ

    round_per_num = 6    # precision for percentage calculation
    col_name = 'returns' + ret_type + 'PrevRaw' + str(days)
    
    asset_mkt = mkt_train_df_small[mkt_train_df_small.assetCode == asset].reset_index(drop=True)
    cal_ret = (asset_mkt[ret_type.lower()] - asset_mkt[ret_type.lower()].shift(days)) / asset_mkt[ret_type.lower()].shift(days)
    percent = abs((np.round(cal_ret[days:],round_per_num) - np.round(asset_mkt[col_name][days:],round_per_num))*100/np.round(asset_mkt[col_name][days:],round_per_num))
    comp_df = np.round(asset_mkt[col_name][days:],10) == np.round(cal_ret[days:],10)
    results = pd.DataFrame({'time':asset_mkt.time,'open':asset_mkt.open,'close':asset_mkt.close,col_name:asset_mkt[col_name][days:],'expected_return':cal_ret[days:],'error_perc':percent.round(2),'equal':comp_df[days:]})
    
    return results
res = show_diff_returns('AFG.N','Close',1)

res[res['equal'] == False]
res[37:57]
def plot_var(data,typ):
    
    plt.rcParams["figure.figsize"] = (30,10)
    plt.title(data.columns[3])
    plt.plot()
    #plt.plot(data.time,data[typ], linewidth=0.5,color="g",label='close/open')
    plt.plot(data.time,data.returnsClosePrevRaw1 - data.expected_return, marker='o', linewidth=0,markersize=5,color="r",label='wrong data')
    #plt.plot(data.time,(data.equal[data.equal==False].replace(False, 1))*data[typ], marker='o', linewidth=0,markersize=5,color="r",label='wrong data')
    plt.legend()
    plt.show()
    plt.close()
    
plot_var(res,'close')
def plot_var(data,typ):
    
    plt.rcParams["figure.figsize"] = (30,10)
    plt.title(data.columns[3])
    plt.plot()
    plt.plot(data.time,data[typ], linewidth=0.5,color="g",label='close/open')
    plt.plot(data.time,(data.equal[data.equal==False].replace(False, 1))*data[typ], marker='o', linewidth=0,markersize=5,color="r",label='wrong data')
    plt.legend()
    plt.show()
    plt.close()
    
plot_var(res,'close')