# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as mth
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import copy
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

print("sample")
print(np.isclose(10,9, rtol=0.12))

df_train=pd.read_csv("../input/train.csv", sep=',', lineterminator='\n')#.head(1000)
df_test_tmp=pd.read_csv("../input/test.csv", sep=',', lineterminator='\n')#.head(1000)
df_test_tmp['target']=-1



df=df_train#.copy().dropna()
df_test=df_test_tmp#.copy().dropna()
#df_test=df_test.head(100) 
#print(df.index)

#list of columns in dataframe
dynamic_list=df.columns

def check_fix_Null(d,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for i in range(len(a)):
        if d[d[a].columns[i]].isnull().sum() >0 and d[d[a].columns[i]].dtype not in numerics:
            d[d[a].columns[i]]=d[d[a].columns[i]].fillna(value='0')
        elif  d[d[a].columns[i]].isnull().sum() >0 and d[d[a].columns[i]].dtype  in numerics:
            d[d[a].columns[i]]=d[d[a].columns[i]].fillna(value=0)

check_fix_Null(df,dynamic_list)
check_fix_Null(df_test,dynamic_list)


def ListNon_N(d,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    d['new_column']=0   
    for i in range(len(a)):
            #print(i)
            if d[d[a].columns[i]].dtype not in numerics:
                #print(d[a].columns[i])
                #print(d[d[a].columns[i]].unique())
                #print(d[d[a].columns[i]].apply(hash).unique())
                d['new_column']=d[d[a].columns[i]].apply(hash)+d['new_column']
                val_txt=d[a].columns[i]
                print(val_txt)
                var_mod = [val_txt]
                le = LabelEncoder()
                for i in var_mod:
                    d[i] = le.fit_transform(d[i])
                    d.dtypes 
                print(val_txt)
    
    
ListNon_N(df,dynamic_list)
ListNon_N(df_test,dynamic_list)

def drop_dups(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
        if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
          d.drop_duplicates(subset=d[a].columns[i], keep="last")  
          #d.dropna(axis=1, how='all')
           

drop_dups(df,dynamic_list)
#drop_dups(df_test,dynamic_list)            
    
tmp_val=0.1
def corr_check(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
            
            if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
                k=k+1
                print(k)
                print(d[a].columns[i])
                col_name_final=d[a].columns[i]+"_f"
                if k==1:
                    v="new_column"+","+'"'+col_name_final+'"'  
                    col_list=copy.copy(v)
                elif k>1:    
                    z=col_list+','+'"'+col_name_final+'"'
                    col_list=copy.copy(z)
                dict = {True: "2", False: "1"}
                #d["col_name1"]=np.isclose(d[d[a].columns[i]],d[d[a].columns[i]].mean(), rtol=tmp_val)
                d["col_name2"]=np.isclose(d[d[a].columns[i]],d[d[a].columns[i]].quantile(1), rtol=tmp_val)
                d["col_name3"]=np.isclose(d[d[a].columns[i]],d[d[a].columns[i]].quantile(0.75), rtol=tmp_val)
                d["col_name4"]=np.isclose(d[d[a].columns[i]],d[d[a].columns[i]].quantile(0.5), rtol=tmp_val)
                d["col_name5"]=np.isclose(d[d[a].columns[i]],d[d[a].columns[i]].quantile(0.25), rtol=tmp_val)
                d["col_name_final_tmp"]=d["col_name2"].map(dict)+d["col_name3"].map(dict)+d["col_name4"].map(dict)+d["col_name5"].map(dict)
                
                d[col_name_final]=d["col_name_final_tmp"].astype(int)
                print(d[col_name_final])
                dt[col_name_final]=d[col_name_final].astype(int)
                print(col_list)
    return col_list

col_list=""                
#col_list=corr_check(df,dynamic_list)
#col_list=corr_check(df_test,dynamic_list)


def means_all(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
            
            if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
                k=k+1
                print(k)
                print(d[a].columns[i])
                col_name_final=d[a].columns[i]+"_mean"
                
                d["col_name2"]=d.groupby('new_column') [d[a].columns[i]].mean()
                d[col_name_final]=d["col_name2"].fillna(0).astype(int)
                print(d[col_name_final])
                dt[col_name_final]=d[col_name_final].astype(int)
                print(col_name_final)
    #return col_name_final

means_all(df,dynamic_list)
means_all(df_test,dynamic_list)

#print(df_test)



def q_75_all(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
            
            if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
                k=k+1
                print(k)
                print(d[a].columns[i])
                col_name_final=d[a].columns[i]+"_75"
                
                d["col_name2"]=d.groupby('new_column') [d[a].columns[i]].quantile(0.75)
                d[col_name_final]=d["col_name2"].fillna(0).astype(int)
                print(d[col_name_final])
                dt[col_name_final]=d[col_name_final].astype(int)
                print(col_name_final)
    #return col_name_final

#q_75_all(df,dynamic_list)
#q_75_all(df_test,dynamic_list)



def q_25_all(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
            
            if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
                k=k+1
                print(k)
                print(d[a].columns[i])
                col_name_final=d[a].columns[i]+"_25"
                
                d["col_name2"]=d.groupby('new_column') [d[a].columns[i]].quantile(0.25)
                d[col_name_final]=d["col_name2"].fillna(0).astype(int)
                print(d[col_name_final])
                dt[col_name_final]=d[col_name_final].astype(int)
                print(col_name_final)
    #return col_name_final

#q_25_all(df,dynamic_list)
#q_25_all(df_test,dynamic_list)


def median_all(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
            
            if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
                k=k+1
                print(k)
                print(d[a].columns[i])
                col_name_final=d[a].columns[i]+"_median"
                
                d["col_name2"]=d.groupby('new_column') [d[a].columns[i]].quantile(0.5)
                d[col_name_final]=d["col_name2"].fillna(0).astype(int)
                print(d[col_name_final])
                dt[col_name_final]=d[col_name_final].astype(int)
                print(col_name_final)
    #return col_name_final

median_all(df,dynamic_list)
median_all(df_test,dynamic_list)



def min_all(dt,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #dt["final_value"]=0
    k=0
    d=dt#[(dt.target==1)] # taking dataset which is target=1
    for i in range(len(a)):
            
            if d[d[a].columns[i]].dtype  in numerics and d[a].columns[i]!='ID'  and d[a].columns[i]!='target':
                k=k+1
                print(k)
                print(d[a].columns[i])
                col_name_final=d[a].columns[i]+"_min"
                
                d["col_name2"]=d.groupby('new_column') [d[a].columns[i]].min()
                d[col_name_final]=d["col_name2"].fillna(0).astype(int)
                print(d[col_name_final])
                dt[col_name_final]=d[col_name_final].astype(int)
                print(col_name_final)
    #return col_name_final

min_all(df,dynamic_list)
min_all(df_test,dynamic_list)

print(df_test)
        


        

df_test['target']=-1
predictor_var = df[[ 'new_column',
#    "v1_75","v2_75","v4_75","v5_75","v6_75","v7_75","v8_75","v9_75","v10_75","v11_75","v12_75","v13_75","v14_75","v15_75",
#"v16_75","v17_75","v18_75","v19_75","v20_75","v21_75","v23_75","v25_75","v26_75","v27_75","v28_75","v29_75","v32_75","v33_75","v34_75",
#"v35_75","v36_75","v37_75","v38_75","v39_75","v40_75","v41_75","v42_75","v43_75","v44_75","v45_75","v46_75","v48_75","v49_75","v50_75",
#"v51_75","v53_75","v54_75","v55_75","v57_75","v58_75","v59_75","v60_75","v61_75","v62_75","v63_75","v64_75","v65_75","v67_75","v68_75",
#"v69_75","v70_75","v72_75","v73_75","v76_75","v77_75","v78_75","v80_75","v81_75","v82_75","v83_75","v84_75","v85_75","v86_75","v87_75",
#"v88_75","v89_75","v90_75","v92_75","v93_75","v94_75","v95_75","v96_75","v97_75","v98_75","v99_75","v100_75","v101_75","v102_75","v103_75",
#"v104_75","v105_75","v106_75","v108_75","v109_75","v111_75","v114_75","v115_75","v116_75","v117_75","v118_75",
#"v119_75","v120_75","v121_75","v122_75","v123_75","v124_75","v126_75","v127_75","v128_75","v129_75","v130_75","v131_75",                    
#     "v1_25","v2_25","v4_25","v5_25","v6_25","v7_25","v8_25","v9_25","v10_25","v11_25","v12_25","v13_25","v14_25","v15_25",
#"v16_25","v17_25","v18_25","v19_25","v20_25","v21_25","v23_25","v25_25","v26_25","v27_25","v28_25","v29_25","v32_25","v33_25","v34_25",
#"v35_25","v36_25","v37_25","v38_25","v39_25","v40_25","v41_25","v42_25","v43_25","v44_25","v45_25","v46_25","v48_25","v49_25","v50_25",
#"v51_25","v53_25","v54_25","v55_25","v57_25","v58_25","v59_25","v60_25","v61_25","v62_25","v63_25","v64_25","v65_25","v67_25","v68_25",
#"v69_25","v70_25","v72_25","v73_25","v76_25","v77_25","v78_25","v80_25","v81_25","v82_25","v83_25","v84_25","v85_25","v86_25","v87_25",
#"v88_25","v89_25","v90_25","v92_25","v93_25","v94_25","v95_25","v96_25","v97_25","v98_25","v99_25","v100_25","v101_25","v102_25","v103_25",
#"v104_25","v105_25","v106_25","v108_25","v109_25","v111_25","v114_25","v115_25","v116_25","v117_25","v118_25",
#"v119_25","v120_25","v121_25","v122_25","v123_25","v124_25","v126_25","v127_25","v128_25","v129_25","v130_25","v131_25",                   
    "v1_min","v2_min","v4_min","v5_min","v6_min","v7_min","v8_min","v9_min","v10_min","v11_min","v12_min","v13_min","v14_min","v15_min",
"v16_min","v17_min","v18_min","v19_min","v20_min","v21_min","v23_min","v25_min","v26_min","v27_min","v28_min","v29_min","v32_min","v33_min","v34_min",
"v35_min","v36_min","v37_min","v38_min","v39_min","v40_min","v41_min","v42_min","v43_min","v44_min","v45_min","v46_min","v48_min","v49_min","v50_min",
"v51_min","v53_min","v54_min","v55_min","v57_min","v58_min","v59_min","v60_min","v61_min","v62_min","v63_min","v64_min","v65_min","v67_min","v68_min",
"v69_min","v70_min","v72_min","v73_min","v76_min","v77_min","v78_min","v80_min","v81_min","v82_min","v83_min","v84_min","v85_min","v86_min","v87_min",
"v88_min","v89_min","v90_min","v92_min","v93_min","v94_min","v95_min","v96_min","v97_min","v98_min","v99_min","v100_min","v101_min","v102_min","v103_min",
"v104_min","v105_min","v106_min","v108_min","v109_min","v111_min","v114_min","v115_min","v116_min","v117_min","v118_min",
"v119_min","v120_min","v121_min","v122_min","v123_min","v124_min","v126_min","v127_min","v128_min","v129_min","v130_min","v131_min",                
    "v1_median","v2_median","v4_median","v5_median","v6_median","v7_median","v8_median","v9_median","v10_median","v11_median","v12_median","v13_median","v14_median","v15_median",
                    "v16_median","v17_median","v18_median","v19_median","v20_median","v21_median","v23_median","v25_median","v26_median","v27_median","v28_median","v29_median","v32_median","v33_median","v34_median",
                    "v35_median","v36_median","v37_median","v38_median","v39_median","v40_median","v41_median","v42_median","v43_median","v44_median","v45_median","v46_median","v48_median","v49_median","v50_median",
                    "v51_median","v53_median","v54_median","v55_median","v57_median","v58_median","v59_median","v60_median","v61_median","v62_median","v63_median","v64_median","v65_median","v67_median","v68_median",
                    "v69_median","v70_median","v72_median","v73_median","v76_median","v77_median","v78_median","v80_median","v81_median","v82_median","v83_median","v84_median","v85_median","v86_median","v87_median",
                    "v88_median","v89_median","v90_median","v92_median","v93_median","v94_median","v95_median","v96_median","v97_median","v98_median","v99_median","v100_median","v101_median","v102_median","v103_median",
                    "v104_median","v105_median","v106_median","v108_median","v109_median","v111_median","v114_median","v115_median","v116_median","v117_median","v118_median",
                    "v119_median","v120_median","v121_median","v122_median","v123_median","v124_median","v126_median","v127_median","v128_median","v129_median","v130_median","v131_median",
                    "v1_mean","v2_mean","v4_mean","v5_mean","v6_mean","v7_mean","v8_mean","v9_mean","v10_mean","v11_mean","v12_mean","v13_mean","v14_mean","v15_mean",
                    "v16_mean","v17_mean","v18_mean","v19_mean","v20_mean","v21_mean","v23_mean","v25_mean","v26_mean","v27_mean","v28_mean","v29_mean","v32_mean","v33_mean","v34_mean",
                    "v35_mean","v36_mean","v37_mean","v38_mean","v39_mean","v40_mean","v41_mean","v42_mean","v43_mean","v44_mean","v45_mean","v46_mean","v48_mean","v49_mean","v50_mean",
                    "v51_mean","v53_mean","v54_mean","v55_mean","v57_mean","v58_mean","v59_mean","v60_mean","v61_mean","v62_mean","v63_mean","v64_mean","v65_mean","v67_mean","v68_mean",
                    "v69_mean","v70_mean","v72_mean","v73_mean","v76_mean","v77_mean","v78_mean","v80_mean","v81_mean","v82_mean","v83_mean","v84_mean","v85_mean","v86_mean","v87_mean",
                    "v88_mean","v89_mean","v90_mean","v92_mean","v93_mean","v94_mean","v95_mean","v96_mean","v97_mean","v98_mean","v99_mean","v100_mean","v101_mean","v102_mean","v103_mean",
                    "v104_mean","v105_mean","v106_mean","v108_mean","v109_mean","v111_mean","v114_mean","v115_mean","v116_mean","v117_mean","v118_mean",
                    "v119_mean","v120_mean","v121_mean","v122_mean","v123_mean","v124_mean","v126_mean","v127_mean","v128_mean","v129_mean","v130_mean","v131_mean",
                    "v1","v2","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15",
                    "v16","v17","v18","v19","v20","v21","v23","v25","v26","v27","v28","v29","v32","v33","v34",
                    "v35","v36","v37","v38","v39","v40","v41","v42","v43","v44","v45","v46","v48","v49","v50",
                    "v51","v53","v54","v55","v57","v58","v59","v60","v61","v62","v63","v64","v65","v67","v68",
                    "v69","v70","v72","v73","v76","v77","v78","v80","v81","v82","v83","v84","v85","v86","v87",
                    "v88","v89","v90","v92","v93","v94","v95","v96","v97","v98","v99","v100","v101","v102","v103",
                    "v104","v105","v106","v108","v109","v111","v114","v115","v116","v117","v118",
                    "v119","v120","v121","v122","v123","v124","v126","v127","v128","v129","v130","v131",
"v3",
"v22",
"v24",
"v30",
"v31",
"v47",
"v52",
"v56",
"v66",
"v71",
"v74",
"v75",
"v79",
"v91",
"v107",
"v110",
"v112",
"v113",
"v125"]]

predictor_var2 = df_test[['new_column',
#"v1_75","v2_75","v4_75","v5_75","v6_75","v7_75","v8_75","v9_75","v10_75","v11_75","v12_75","v13_75","v14_75","v15_75",
#"v16_75","v17_75","v18_75","v19_75","v20_75","v21_75","v23_75","v25_75","v26_75","v27_75","v28_75","v29_75","v32_75","v33_75","v34_75",
#"v35_75","v36_75","v37_75","v38_75","v39_75","v40_75","v41_75","v42_75","v43_75","v44_75","v45_75","v46_75","v48_75","v49_75","v50_75",
#"v51_75","v53_75","v54_75","v55_75","v57_75","v58_75","v59_75","v60_75","v61_75","v62_75","v63_75","v64_75","v65_75","v67_75","v68_75",
#"v69_75","v70_75","v72_75","v73_75","v76_75","v77_75","v78_75","v80_75","v81_75","v82_75","v83_75","v84_75","v85_75","v86_75","v87_75",
#"v88_75","v89_75","v90_75","v92_75","v93_75","v94_75","v95_75","v96_75","v97_75","v98_75","v99_75","v100_75","v101_75","v102_75","v103_75",
#"v104_75","v105_75","v106_75","v108_75","v109_75","v111_75","v114_75","v115_75","v116_75","v117_75","v118_75",
#"v119_75","v120_75","v121_75","v122_75","v123_75","v124_75","v126_75","v127_75","v128_75","v129_75","v130_75","v131_75",                    
#     "v1_25","v2_25","v4_25","v5_25","v6_25","v7_25","v8_25","v9_25","v10_25","v11_25","v12_25","v13_25","v14_25","v15_25",
#"v16_25","v17_25","v18_25","v19_25","v20_25","v21_25","v23_25","v25_25","v26_25","v27_25","v28_25","v29_25","v32_25","v33_25","v34_25",
#"v35_25","v36_25","v37_25","v38_25","v39_25","v40_25","v41_25","v42_25","v43_25","v44_25","v45_25","v46_25","v48_25","v49_25","v50_25",
#"v51_25","v53_25","v54_25","v55_25","v57_25","v58_25","v59_25","v60_25","v61_25","v62_25","v63_25","v64_25","v65_25","v67_25","v68_25",
#"v69_25","v70_25","v72_25","v73_25","v76_25","v77_25","v78_25","v80_25","v81_25","v82_25","v83_25","v84_25","v85_25","v86_25","v87_25",
#"v88_25","v89_25","v90_25","v92_25","v93_25","v94_25","v95_25","v96_25","v97_25","v98_25","v99_25","v100_25","v101_25","v102_25","v103_25",
#"v104_25","v105_25","v106_25","v108_25","v109_25","v111_25","v114_25","v115_25","v116_25","v117_25","v118_25",
#"v119_25","v120_25","v121_25","v122_25","v123_25","v124_25","v126_25","v127_25","v128_25","v129_25","v130_25","v131_25",  
    "v1_min","v2_min","v4_min","v5_min","v6_min","v7_min","v8_min","v9_min","v10_min","v11_min","v12_min","v13_min","v14_min","v15_min",
"v16_min","v17_min","v18_min","v19_min","v20_min","v21_min","v23_min","v25_min","v26_min","v27_min","v28_min","v29_min","v32_min","v33_min","v34_min",
"v35_min","v36_min","v37_min","v38_min","v39_min","v40_min","v41_min","v42_min","v43_min","v44_min","v45_min","v46_min","v48_min","v49_min","v50_min",
"v51_min","v53_min","v54_min","v55_min","v57_min","v58_min","v59_min","v60_min","v61_min","v62_min","v63_min","v64_min","v65_min","v67_min","v68_min",
"v69_min","v70_min","v72_min","v73_min","v76_min","v77_min","v78_min","v80_min","v81_min","v82_min","v83_min","v84_min","v85_min","v86_min","v87_min",
"v88_min","v89_min","v90_min","v92_min","v93_min","v94_min","v95_min","v96_min","v97_min","v98_min","v99_min","v100_min","v101_min","v102_min","v103_min",
"v104_min","v105_min","v106_min","v108_min","v109_min","v111_min","v114_min","v115_min","v116_min","v117_min","v118_min",
"v119_min","v120_min","v121_min","v122_min","v123_min","v124_min","v126_min","v127_min","v128_min","v129_min","v130_min","v131_min",                
    "v1_median","v2_median","v4_median","v5_median","v6_median","v7_median","v8_median","v9_median","v10_median","v11_median","v12_median","v13_median","v14_median","v15_median",
                    "v16_median","v17_median","v18_median","v19_median","v20_median","v21_median","v23_median","v25_median","v26_median","v27_median","v28_median","v29_median","v32_median","v33_median","v34_median",
                    "v35_median","v36_median","v37_median","v38_median","v39_median","v40_median","v41_median","v42_median","v43_median","v44_median","v45_median","v46_median","v48_median","v49_median","v50_median",
                    "v51_median","v53_median","v54_median","v55_median","v57_median","v58_median","v59_median","v60_median","v61_median","v62_median","v63_median","v64_median","v65_median","v67_median","v68_median",
                    "v69_median","v70_median","v72_median","v73_median","v76_median","v77_median","v78_median","v80_median","v81_median","v82_median","v83_median","v84_median","v85_median","v86_median","v87_median",
                    "v88_median","v89_median","v90_median","v92_median","v93_median","v94_median","v95_median","v96_median","v97_median","v98_median","v99_median","v100_median","v101_median","v102_median","v103_median",
                    "v104_median","v105_median","v106_median","v108_median","v109_median","v111_median","v114_median","v115_median","v116_median","v117_median","v118_median",
                    "v119_median","v120_median","v121_median","v122_median","v123_median","v124_median","v126_median","v127_median","v128_median","v129_median","v130_median","v131_median",
                    "v1_mean","v2_mean","v4_mean","v5_mean","v6_mean","v7_mean","v8_mean","v9_mean","v10_mean","v11_mean","v12_mean","v13_mean","v14_mean","v15_mean",
                    "v16_mean","v17_mean","v18_mean","v19_mean","v20_mean","v21_mean","v23_mean","v25_mean","v26_mean","v27_mean","v28_mean","v29_mean","v32_mean","v33_mean","v34_mean",
                    "v35_mean","v36_mean","v37_mean","v38_mean","v39_mean","v40_mean","v41_mean","v42_mean","v43_mean","v44_mean","v45_mean","v46_mean","v48_mean","v49_mean","v50_mean",
                    "v51_mean","v53_mean","v54_mean","v55_mean","v57_mean","v58_mean","v59_mean","v60_mean","v61_mean","v62_mean","v63_mean","v64_mean","v65_mean","v67_mean","v68_mean",
                    "v69_mean","v70_mean","v72_mean","v73_mean","v76_mean","v77_mean","v78_mean","v80_mean","v81_mean","v82_mean","v83_mean","v84_mean","v85_mean","v86_mean","v87_mean",
                    "v88_mean","v89_mean","v90_mean","v92_mean","v93_mean","v94_mean","v95_mean","v96_mean","v97_mean","v98_mean","v99_mean","v100_mean","v101_mean","v102_mean","v103_mean",
                    "v104_mean","v105_mean","v106_mean","v108_mean","v109_mean","v111_mean","v114_mean","v115_mean","v116_mean","v117_mean","v118_mean",
                    "v119_mean","v120_mean","v121_mean","v122_mean","v123_mean","v124_mean","v126_mean","v127_mean","v128_mean","v129_mean","v130_mean","v131_mean",
                    "v1","v2","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15",
                    "v16","v17","v18","v19","v20","v21","v23","v25","v26","v27","v28","v29","v32","v33","v34",
                    "v35","v36","v37","v38","v39","v40","v41","v42","v43","v44","v45","v46","v48","v49","v50",
                    "v51","v53","v54","v55","v57","v58","v59","v60","v61","v62","v63","v64","v65","v67","v68",
                    "v69","v70","v72","v73","v76","v77","v78","v80","v81","v82","v83","v84","v85","v86","v87",
                    "v88","v89","v90","v92","v93","v94","v95","v96","v97","v98","v99","v100","v101","v102","v103",
                    "v104","v105","v106","v108","v109","v111","v114","v115","v116","v117","v118",
                    "v119","v120","v121","v122","v123","v124","v126","v127","v128","v129","v130","v131",
"v3",
"v22",
"v24",
"v30",
"v31",
"v47",
"v52",
"v56",
"v66",
"v71",
"v74",
"v75",
"v79",
"v91",
"v107",
"v110",
"v112",
"v113",
"v125"]]

outcome_var = df['target']

#model=ExtraTreesClassifier(n_estimators=1000,max_features= 80,criterion= 'entropy',min_samples_split= 8,max_depth= 10, min_samples_leaf= 2, n_jobs = -1)
#ExtraTreesClassifier( n_estimators=1000,max_depth=32,criterion= 'entropy',max_features=692,min_samples_split=2,min_samples_leaf=2,warm_start="true" )


model=ExtraTreesClassifier( n_estimators=1500,max_depth=32,criterion= 'entropy',max_features=468,min_samples_split=2,min_samples_leaf=2,warm_start="true" )


model.fit(predictor_var,outcome_var)
predictions = model.predict_proba(predictor_var2)


#predictions = model.predict(predictor_var2)
#accuracy = metrics.accuracy_score(predictions,outcome_var)
#print("Accuracy : %s" % "{0:.3%}".format(accuracy) )
#predictors=['new_column','final_value']
#outcome='target'



train_predictions = model.predict_proba(predictor_var)[:,1]
print("predictions")
print(train_predictions)
print(outcome_var)
ww = log_loss(outcome_var, train_predictions)
print("Log Loss: {}".format(ww))


'''
#Perform k-fold cross-validation with 5 foldshttps://www.kaggle.com/ozgurb
kf = KFold(df.shape[0], n_folds=5)
error = []

for train, test in kf:
    # Filter training data
    train_predictors = (df[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = df[outcome].iloc[train]
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    #Record error from each cross-validation run
    error.append(model.score(df[predictors].iloc[test,:], df[outcome].iloc[test]))

print( "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)) )
'''

my_submission = pd.DataFrame( { 'Id': df_test.ID,'PredictedProb': predictions[:,1] } )
# you could use any filename. We choose submission here
my_submission.to_csv('sample_submission.csv', index=False)
print("Writing complete")




