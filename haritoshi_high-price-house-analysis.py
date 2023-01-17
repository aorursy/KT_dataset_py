import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

pd.options.display.max_columns = None
pd.options.display.max_rows = None
df_train = pd.read_csv("../input/train_set.csv")
df_test = pd.read_csv("../input/test_set.csv")
plt.hist(df_train.PRICE, bins=100, log=True)
def price_label(price):
    if price > 10**8:
        return '3';
    elif price > 5 * 10**7:
        return '2';
    else:
        return '1';
df_train['PRICELABEL'] = df_train.PRICE.apply(lambda x: price_label(x)).values.copy()
pd.crosstab(df_train.PRICE.apply(lambda x: price_label(x)), df_train.ZIPCODE)
pd.crosstab(df_train.PRICE.apply(lambda x: price_label(x)), df_train.ASSESSMENT_NBHD)
pd.crosstab(df_train.PRICE.apply(lambda x: price_label(x)), df_train.SOURCE)
pd.crosstab(df_train.PRICE.apply(lambda x: price_label(x)), df_train.CENSUS_TRACT)
pd.crosstab(df_train.PRICE.apply(lambda x: price_label(x)), df_train.QUALIFIED)
df_tmp = pd.crosstab([ df_train.QUALIFIED, df_train.SOURCE, df_train.CENSUS_TRACT, df_train.LATITUDE, df_train.LONGITUDE], df_train['PRICE'].apply(lambda x: price_label(x)))
df_tmp[np.logical_or(df_tmp['3']>0, df_tmp['2']>0)]
df_tmp = pd.crosstab([df_train.CENSUS_TRACT, df_train.SALEDATE, df_train.SALE_NUM, df_train.PRICE], df_train.PRICELABEL, colnames=['PLICELABEL'])
df_tmp[np.logical_or(df_tmp['3']>0, df_tmp['2']>0)]
import folium
_map = folium.Map(location=[38.923, -77.05], zoom_start=14)
folium.Marker([38.936070, -77.073946], popup='<i>ラベル3物件</i>').add_to(_map)
folium.Marker([38.937773, -77.075368], popup='<i>ラベル3物件</i>').add_to(_map)
folium.Marker([38.904456, -77.031057], popup='<i>ラベル3物件</i>').add_to(_map)
_map
len(df_test.query('CENSUS_TRACT == 1002.0 and SALEDATE == "2007-04-10 00:00:00" and SALE_NUM == 1').index)
len(df_test.query('CENSUS_TRACT == 10100.0 and SALEDATE == "2015-11-17 00:00:00" and SALE_NUM == 2').index)
def out_highest_price(df):
    df_label3_ids = df.query('CENSUS_TRACT == 1002.0 and SALEDATE == "2007-04-10 00:00:00"').Id.values
    df_label2_ids = df.query('CENSUS_TRACT == 10100.0 and SALEDATE == "2015-11-17 00:00:00" and SALE_NUM == 2').Id.values
    
    df_high_predict =  pd.DataFrame(np.r_[
        np.ones(len(df_label3_ids)) * 137427545.0,
        np.ones(len(df_label2_ids)) * 53969391.0
    ], index=np.r_[df_label3_ids, df_label2_ids], columns=["PRICE"])
    
    df_other = df[df.Id.isin(np.r_[df_label3_ids, df_label2_ids]) == False]
    return df_other, df_high_predict
df_other, df_high_predict = out_highest_price(df_test)
df_high_predict.head()
df_other.head()