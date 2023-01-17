import numpy as np

import pandas as pd

from numba import jit

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler
def Outputs(p):

    return (1./(1.+np.exp(-p)))



    

def GP1(data):

    v = pd.DataFrame()

    v["i1"] = 0.020000*np.tanh((((data["ps_car_12"] + ((data["ps_car_15"] + data["ps_reg_01"]) + data["loo_ps_ind_16_bin"])) * 2.0) * 2.0))

    v["i2"] = 0.019910*np.tanh((((data["loo_ps_car_04_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_car_06_cat"] + data["loo_ps_ind_17_bin"]))) * 2.0) * 2.0))

    v["i3"] = 0.020000*np.tanh((((data["loo_ps_car_05_cat"] + data["loo_ps_ind_16_bin"]) + (data["loo_ps_car_09_cat"] + (data["loo_ps_car_11_cat"] * 2.0))) * 2.0))

    v["i4"] = 0.020000*np.tanh((data["ps_car_15"] + (((data["ps_reg_02"] + data["loo_ps_ind_06_bin"]) + data["ps_reg_03"]) + data["loo_ps_car_11_cat"])))

    v["i5"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]) + data["loo_ps_car_02_cat"]) + data["loo_ps_car_04_cat"]) * 2.0))

    v["i6"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_05_cat"]) + (data["ps_car_13"] + (data["ps_car_13"] + data["ps_reg_02"]))))

    v["i7"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((data["loo_ps_car_03_cat"] + data["ps_car_13"]) * 2.0)) * 2.0))

    v["i8"] = 0.019945*np.tanh(((data["loo_ps_car_03_cat"] + data["loo_ps_car_04_cat"]) + ((data["loo_ps_car_06_cat"] + data["loo_ps_ind_17_bin"]) + data["loo_ps_car_01_cat"])))

    v["i9"] = 0.020000*np.tanh((((data["ps_car_13"] + data["loo_ps_car_07_cat"]) * 2.0) + (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_16_bin"])))

    v["i10"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_06_bin"] + data["loo_ps_car_07_cat"]))) * 2.0) * 2.0))

    v["i11"] = 0.020000*np.tanh(((data["loo_ps_car_11_cat"] + ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_07_cat"])) * 2.0))

    v["i12"] = 0.020000*np.tanh((((((data["ps_car_13"] * 2.0) + data["loo_ps_car_03_cat"])/2.0) + (data["ps_reg_01"] + data["loo_ps_ind_05_cat"])) * 2.0))

    v["i13"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((data["ps_car_13"] + (data["loo_ps_car_01_cat"] + data["loo_ps_car_07_cat"])) * 2.0)) * 2.0))

    v["i14"] = 0.020000*np.tanh((((((data["loo_ps_car_02_cat"] + data["loo_ps_ind_07_bin"]) + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_07_bin"]) * 2.0) * 2.0))

    v["i15"] = 0.020000*np.tanh((data["ps_car_13"] + (data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_06_bin"] + data["loo_ps_car_01_cat"]) + data["loo_ps_car_09_cat"]))))

    v["i16"] = 0.020000*np.tanh(((((data["loo_ps_ind_06_bin"] + data["ps_reg_02"]) + data["ps_car_13"]) + data["loo_ps_car_01_cat"]) + data["ps_car_13"]))

    v["i17"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]) + (data["ps_car_12"] + data["ps_reg_01"])) - data["ps_ind_15"]))

    v["i18"] = 0.020000*np.tanh((((data["ps_car_15"] + ((data["ps_reg_03"] + data["loo_ps_car_09_cat"]) * 2.0)) + data["ps_reg_02"]) * 2.0))

    v["i19"] = 0.020000*np.tanh((data["ps_car_13"] + ((data["ps_reg_03"] + data["loo_ps_car_07_cat"]) - (data["ps_ind_15"] - data["loo_ps_car_03_cat"]))))

    v["i20"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + (data["loo_ps_car_07_cat"] + (data["loo_ps_car_03_cat"] + data["loo_ps_car_09_cat"]))) * 2.0) * 2.0))

    v["i21"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_06_bin"] - data["ps_ind_15"])) + data["ps_reg_02"]) * 2.0))

    v["i22"] = 0.020000*np.tanh(((((((data["loo_ps_ind_17_bin"] + data["ps_reg_03"]) * 2.0) * 2.0) * 2.0) + data["loo_ps_car_11_cat"]) * 2.0))

    v["i23"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] + data["loo_ps_ind_07_bin"]) + (data["loo_ps_ind_05_cat"] + (data["ps_reg_03"] + data["loo_ps_ind_09_bin"]))))

    v["i24"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) + data["ps_ind_03"]) + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_05_cat"]))

    v["i25"] = 0.020000*np.tanh(((data["ps_reg_02"] - ((data["loo_ps_ind_02_cat"] * 2.0) * data["ps_ind_03"])) + data["ps_car_13"]))

    v["i26"] = 0.020000*np.tanh((((data["ps_car_13"] + data["loo_ps_car_09_cat"]) + (data["loo_ps_car_07_cat"] + data["ps_ind_03"])) - data["ps_ind_15"]))

    v["i27"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + data["loo_ps_ind_06_bin"]) + (data["loo_ps_car_01_cat"] + (data["loo_ps_ind_16_bin"] - data["ps_ind_15"]))))

    v["i28"] = 0.020000*np.tanh(((((data["ps_reg_03"] + (data["ps_reg_03"] * 2.0)) + data["loo_ps_ind_04_cat"]) + data["ps_reg_02"]) * 2.0))

    v["i29"] = 0.020000*np.tanh(((data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_17_bin"] + (data["ps_car_13"] + data["ps_ind_01"]))) + data["loo_ps_car_09_cat"]))

    v["i30"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] + (((data["loo_ps_ind_02_cat"] - data["ps_ind_15"]) + data["loo_ps_ind_09_bin"]) + data["ps_reg_01"])))

    v["i31"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + ((data["loo_ps_car_01_cat"] + data["loo_ps_ind_08_bin"]) + data["loo_ps_ind_02_cat"])) + data["loo_ps_ind_07_bin"]))

    v["i32"] = 0.020000*np.tanh((((((data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i33"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] + (data["loo_ps_car_01_cat"] - data["ps_ind_15"])) - data["ps_ind_15"]) + data["loo_ps_car_07_cat"]))

    v["i34"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + (data["loo_ps_ind_16_bin"] - data["loo_ps_ind_18_bin"])) * 2.0) * 2.0))

    v["i35"] = 0.020000*np.tanh((((((data["loo_ps_ind_05_cat"] * 2.0) + (data["loo_ps_ind_07_bin"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0))

    v["i36"] = 0.020000*np.tanh(((((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"]) + (data["loo_ps_ind_05_cat"] * 2.0)) * 2.0))

    v["i37"] = 0.020000*np.tanh((((((data["loo_ps_ind_17_bin"] * 2.0) + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_11_cat"])) * 2.0) * 2.0) * 2.0))

    v["i38"] = 0.020000*np.tanh(((data["ps_car_13"] + (data["loo_ps_car_08_cat"] + (data["ps_ind_03"] * data["ps_ind_03"]))) + data["loo_ps_car_09_cat"]))

    v["i39"] = 0.019996*np.tanh((((((data["ps_car_15"] + data["loo_ps_car_07_cat"]) + (data["ps_reg_03"] * 2.0)) * 2.0) * 2.0) * 2.0))

    v["i40"] = 0.020000*np.tanh(((((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_02_cat"]) * 2.0) * 2.0) + data["ps_car_15"]))

    v["i41"] = 0.020000*np.tanh((1.0 + ((data["loo_ps_ind_04_cat"] - data["ps_ind_15"]) + (data["loo_ps_car_03_cat"] - data["ps_ind_15"]))))

    v["i42"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_05_cat"] * 2.0)) + (data["ps_ind_03"] * data["ps_ind_03"])) * 2.0))

    v["i43"] = 0.020000*np.tanh(((data["loo_ps_ind_06_bin"] + (((data["loo_ps_ind_06_bin"] + data["loo_ps_ind_09_bin"]) * data["ps_ind_03"]) * 2.0)) * 2.0))

    v["i44"] = 0.020000*np.tanh(((((data["loo_ps_car_01_cat"] + data["loo_ps_ind_09_bin"]) + (data["ps_ind_01"] + data["loo_ps_ind_09_bin"])) * 2.0) * 2.0))

    v["i45"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"]) + ((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"]) * 2.0)) * 2.0))

    v["i46"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_04_cat"] + data["ps_ind_01"])) + data["loo_ps_car_09_cat"]) + data["ps_ind_03"]))

    v["i47"] = 0.020000*np.tanh((data["ps_ind_01"] + ((data["loo_ps_car_05_cat"] * (data["ps_ind_01"] * 2.0)) - data["ps_ind_15"])))

    v["i48"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + ((data["loo_ps_ind_16_bin"] * data["ps_car_12"]) + data["loo_ps_ind_05_cat"])))

    v["i49"] = 0.020000*np.tanh(((8.70196723937988281) * ((8.70196723937988281) * (data["ps_ind_03"] * ((0.0) - data["loo_ps_ind_02_cat"])))))

    v["i50"] = 0.020000*np.tanh((data["ps_car_15"] + ((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_17_bin"]) - (data["loo_ps_car_04_cat"] * data["ps_car_11"]))))

    v["i51"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] - data["ps_reg_01"]) * ((7.16651058197021484) * data["loo_ps_car_01_cat"])))

    v["i52"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] - data["ps_car_11"]) - data["ps_ind_15"]) + (data["loo_ps_ind_05_cat"] - data["ps_ind_15"])))

    v["i53"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_06_bin"] * data["loo_ps_ind_05_cat"])))

    v["i54"] = 0.020000*np.tanh(((data["loo_ps_car_06_cat"] + (data["loo_ps_ind_17_bin"] + (data["ps_ind_01"] + data["missing"]))) * data["loo_ps_car_05_cat"]))

    v["i55"] = 0.020000*np.tanh(((data["ps_reg_03"] + ((data["ps_reg_03"] + data["loo_ps_ind_04_cat"]) * 2.0)) * 2.0))

    v["i56"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] + (data["loo_ps_car_01_cat"] - data["loo_ps_car_02_cat"])) * 2.0))

    v["i57"] = 0.020000*np.tanh((-((data["ps_ind_03"] * ((((data["loo_ps_ind_02_cat"] * 2.0) * 2.0) * 2.0) * 2.0)))))

    v["i58"] = 0.019996*np.tanh((data["loo_ps_car_07_cat"] + (data["ps_ind_03"] * (data["ps_ind_03"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_11_cat"])))))

    v["i59"] = 0.020000*np.tanh((((-2.0 + data["loo_ps_ind_05_cat"]) - data["ps_car_11"]) * 2.0))

    v["i60"] = 0.020000*np.tanh((data["loo_ps_ind_09_bin"] * data["loo_ps_car_03_cat"]))

    v["i61"] = 0.020000*np.tanh((((((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_08_bin"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i62"] = 0.020000*np.tanh((data["ps_car_15"] + (data["loo_ps_ind_17_bin"] * data["loo_ps_car_03_cat"])))

    v["i63"] = 0.020000*np.tanh(((data["loo_ps_ind_06_bin"] * ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) - data["loo_ps_car_11_cat"])) * 2.0))

    v["i64"] = 0.019992*np.tanh((data["loo_ps_car_05_cat"] * (data["ps_car_12"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_08_cat"]))))

    v["i65"] = 0.020000*np.tanh(((((data["missing"] * 2.0) * data["loo_ps_ind_02_cat"]) * 2.0) + data["loo_ps_ind_09_bin"]))

    v["i66"] = 0.020000*np.tanh((((((data["missing"] / 2.0) + (data["loo_ps_car_08_cat"] * 2.0)) * 2.0) * 2.0) * data["loo_ps_ind_02_cat"]))

    v["i67"] = 0.020000*np.tanh(((-(((data["ps_reg_02"] * data["loo_ps_car_06_cat"]) + (data["ps_reg_03"] * data["ps_ind_01"])))) * 2.0))

    v["i68"] = 0.020000*np.tanh(((((data["ps_ind_01"] + data["loo_ps_car_09_cat"]) + ((data["loo_ps_car_11_cat"] + data["loo_ps_car_09_cat"])/2.0))/2.0) * data["loo_ps_ind_09_bin"]))

    v["i69"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * (((data["loo_ps_ind_06_bin"] - data["ps_reg_01"]) - data["ps_car_15"]) + data["loo_ps_ind_05_cat"])))

    v["i70"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] + ((((data["ps_ind_15"] * data["loo_ps_ind_18_bin"]) * 2.0) * 2.0) * 2.0)))

    v["i71"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] - (data["ps_car_11"] * data["ps_ind_01"])) - data["ps_car_11"]))

    v["i72"] = 0.019984*np.tanh((((-2.0 - ((data["ps_ind_03"] * 2.0) * 2.0)) * data["loo_ps_ind_02_cat"]) * 2.0))

    v["i73"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["ps_reg_03"]) + (data["loo_ps_ind_16_bin"] * data["loo_ps_car_04_cat"])) * 2.0))

    v["i74"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_car_13"]) * ((data["ps_reg_02"] * 2.0) * 2.0)))

    v["i75"] = 0.019992*np.tanh((((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_07_bin"]) + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"])) - data["loo_ps_car_04_cat"]))

    v["i76"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + data["loo_ps_car_03_cat"]) * data["loo_ps_ind_09_bin"]) + (data["loo_ps_car_05_cat"] * data["ps_ind_03"])))

    v["i77"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + data["ps_ind_01"]) - (data["ps_ind_01"] * data["ps_ind_15"])))

    v["i78"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * ((data["ps_ind_01"] - data["ps_reg_01"]) + (data["loo_ps_ind_07_bin"] - data["ps_reg_01"]))))

    v["i79"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] * data["ps_ind_15"]) + data["loo_ps_ind_07_bin"]) * (data["ps_ind_15"] * 2.0)))

    v["i80"] = 0.019988*np.tanh((((data["loo_ps_car_11_cat"] + data["ps_car_15"])/2.0) * data["missing"]))

    v["i81"] = 0.020000*np.tanh((((data["ps_ind_03"] * (data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_04_cat"] * 2.0))) + data["loo_ps_ind_04_cat"]) * 2.0))

    v["i82"] = 0.020000*np.tanh((data["ps_car_12"] * (((-(data["ps_car_11"])) * 2.0) * 2.0)))

    v["i83"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] * (((data["loo_ps_car_01_cat"] - data["ps_reg_03"]) - data["ps_reg_03"]) - data["ps_ind_14"])))

    v["i84"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) * data["loo_ps_car_04_cat"]) + (data["ps_ind_15"] * data["loo_ps_ind_07_bin"])))

    v["i85"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] + (data["loo_ps_ind_02_cat"] + (data["ps_ind_03"] * (data["loo_ps_car_11_cat"] - data["ps_reg_01"])))))

    v["i86"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"])/2.0) - (data["ps_reg_01"] * (data["ps_ind_03"] + data["loo_ps_ind_08_bin"]))))

    v["i87"] = 0.020000*np.tanh(((((data["loo_ps_car_01_cat"] * data["loo_ps_ind_16_bin"]) + data["loo_ps_ind_16_bin"])/2.0) + (data["loo_ps_car_04_cat"] * data["ps_ind_01"])))

    v["i88"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + data["ps_ind_03"]) * (data["ps_car_12"] - ((data["loo_ps_ind_02_cat"] * 2.0) * 2.0))))

    v["i89"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * (data["ps_reg_02"] + ((data["missing"] + data["loo_ps_car_06_cat"]) + data["loo_ps_car_09_cat"]))))

    v["i90"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * ((data["loo_ps_car_09_cat"] * ((data["loo_ps_ind_06_bin"] + data["loo_ps_car_09_cat"])/2.0)) - data["ps_car_11"])))

    v["i91"] = 0.019984*np.tanh((((data["loo_ps_car_04_cat"] + data["ps_ind_01"]) + data["loo_ps_car_04_cat"]) * (data["loo_ps_ind_05_cat"] - data["ps_reg_03"])))

    v["i92"] = 0.020000*np.tanh((data["ps_reg_03"] * (data["loo_ps_ind_05_cat"] - data["loo_ps_car_01_cat"])))

    v["i93"] = 0.019988*np.tanh(((((data["ps_car_13"] + data["loo_ps_car_03_cat"])/2.0) + ((data["ps_car_15"] + data["loo_ps_ind_08_bin"])/2.0))/2.0))

    v["i94"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["ps_reg_03"]) - data["ps_reg_01"]) * (data["loo_ps_ind_18_bin"] + data["ps_reg_03"])))

    v["i95"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + (data["ps_ind_15"] * (data["loo_ps_ind_18_bin"] - data["loo_ps_car_06_cat"]))) + data["ps_reg_01"]))

    v["i96"] = 0.020000*np.tanh(((data["ps_ind_03"] * ((data["loo_ps_car_05_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_11_bin"] / 2.0)))/2.0)) / 2.0))

    v["i97"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * (data["loo_ps_car_09_cat"] / 2.0)))

    v["i98"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] * ((data["loo_ps_car_09_cat"] + data["ps_ind_01"]) * ((data["loo_ps_car_04_cat"] + data["ps_ind_01"])/2.0))))

    v["i99"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (((data["ps_reg_03"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_05_cat"]) * 2.0)))

    v["i100"] = 0.019988*np.tanh(((data["ps_car_11"] * (np.tanh(data["loo_ps_ind_04_cat"]) * 2.0)) + data["loo_ps_ind_04_cat"]))

    v["i101"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] * data["ps_ind_01"]) - data["loo_ps_ind_04_cat"]) + data["loo_ps_ind_18_bin"]) * data["ps_ind_01"]))

    v["i102"] = 0.020000*np.tanh(((((-1.0 - data["ps_ind_03"]) * 2.0) * np.tanh((data["loo_ps_ind_02_cat"] * 2.0))) * 2.0))

    v["i103"] = 0.020000*np.tanh((((((data["loo_ps_car_09_cat"] + data["loo_ps_ind_07_bin"])/2.0) * data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_17_bin"] * data["loo_ps_car_09_cat"]))/2.0))

    v["i104"] = 0.020000*np.tanh(((((-2.0 + (data["ps_ind_03"] * data["ps_ind_03"])) - data["ps_ind_03"]) * 2.0) * 2.0))

    v["i105"] = 0.020000*np.tanh((((data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_17_bin"] / 2.0)) * data["loo_ps_car_07_cat"]) - data["loo_ps_car_07_cat"]))

    v["i106"] = 0.020000*np.tanh(((((data["loo_ps_ind_17_bin"] + ((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"]))/2.0))/2.0) + data["loo_ps_ind_12_bin"])/2.0))

    v["i107"] = 0.020000*np.tanh((((data["loo_ps_car_11_cat"] * data["loo_ps_car_03_cat"]) - data["loo_ps_car_03_cat"]) - (data["ps_car_15"] * data["loo_ps_car_03_cat"])))

    v["i108"] = 0.019996*np.tanh((((data["loo_ps_car_02_cat"] * data["ps_ind_03"]) + ((data["loo_ps_car_02_cat"] + (data["loo_ps_car_11_cat"] / 2.0))/2.0))/2.0))

    v["i109"] = 0.020000*np.tanh((((data["ps_reg_03"] + data["loo_ps_car_09_cat"]) + data["ps_ind_01"]) * (data["loo_ps_ind_05_cat"] - data["ps_reg_01"])))

    v["i110"] = 0.020000*np.tanh((np.tanh(((data["loo_ps_car_05_cat"] + data["ps_ind_01"])/2.0)) - (data["ps_car_11"] * data["ps_ind_01"])))

    v["i111"] = 0.020000*np.tanh((np.tanh(data["loo_ps_ind_04_cat"]) * (((-((data["loo_ps_car_03_cat"] / 2.0))) + data["loo_ps_ind_04_cat"])/2.0)))

    v["i112"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] * ((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_08_bin"]) + data["loo_ps_ind_02_cat"])) * 2.0))

    v["i113"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * data["loo_ps_car_07_cat"]))

    v["i114"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_16_bin"])/2.0) + data["loo_ps_ind_05_cat"])/2.0)) * data["loo_ps_ind_09_bin"]))

    v["i115"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] / 2.0) * data["loo_ps_ind_04_cat"]) - data["loo_ps_ind_05_cat"]))

    v["i116"] = 0.019871*np.tanh(np.tanh((data["loo_ps_car_05_cat"] * (data["loo_ps_ind_07_bin"] + (data["loo_ps_ind_02_cat"] * 2.0)))))

    v["i117"] = 0.019996*np.tanh((((data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"]) * (data["loo_ps_ind_07_bin"] + data["loo_ps_car_04_cat"])) - data["loo_ps_ind_17_bin"]))

    v["i118"] = 0.019969*np.tanh((((data["loo_ps_ind_08_bin"] + ((data["ps_car_13"] + (data["loo_ps_ind_12_bin"] / 2.0))/2.0)) + data["loo_ps_ind_08_bin"])/2.0))

    v["i119"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] - data["ps_car_15"]) * (data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_17_bin"] - data["ps_car_15"]))))

    v["i120"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] * (data["loo_ps_car_09_cat"] + data["ps_car_13"])) * 2.0) * 2.0))

    v["i121"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["ps_car_12"] + data["ps_car_12"]) * (data["loo_ps_car_07_cat"] * 2.0))))

    v["i122"] = 0.020000*np.tanh((((np.tanh(data["loo_ps_ind_07_bin"]) / 2.0) + np.tanh((data["loo_ps_ind_07_bin"] / 2.0)))/2.0))

    v["i123"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_07_bin"] * (((data["loo_ps_car_08_cat"] + data["loo_ps_ind_17_bin"])/2.0) + data["loo_ps_ind_17_bin"])))/2.0))

    v["i124"] = 0.019906*np.tanh((((-(data["loo_ps_car_04_cat"])) * data["ps_car_11"]) - data["loo_ps_car_04_cat"]))

    v["i125"] = 0.019996*np.tanh(((((((data["loo_ps_car_06_cat"] * data["loo_ps_car_06_cat"]) + data["ps_car_11"])/2.0) * data["loo_ps_car_06_cat"]) + data["loo_ps_car_02_cat"])/2.0))

    v["i126"] = 0.020000*np.tanh((data["ps_reg_03"] * (data["ps_ind_15"] + ((data["loo_ps_ind_05_cat"] + data["ps_ind_15"]) + data["ps_ind_15"]))))

    v["i127"] = 0.020000*np.tanh(((((data["loo_ps_car_09_cat"] - data["ps_ind_01"]) - data["loo_ps_car_06_cat"]) - data["ps_ind_01"]) * data["loo_ps_ind_04_cat"]))

    v["i128"] = 0.020000*np.tanh(((((data["ps_reg_03"] + data["loo_ps_ind_05_cat"])/2.0) - data["ps_reg_01"]) * (data["ps_reg_03"] + data["ps_ind_03"])))

    v["i129"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] * (-(data["loo_ps_ind_18_bin"]))) + (data["loo_ps_ind_09_bin"] * data["loo_ps_car_09_cat"])))

    v["i130"] = 0.020000*np.tanh(((((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) - (data["loo_ps_ind_04_cat"] * 2.0)) * data["loo_ps_ind_02_cat"]) * 2.0))

    v["i131"] = 0.019941*np.tanh(((((data["ps_reg_03"] * 2.0) * (data["ps_reg_03"] * data["ps_ind_03"])) * 2.0) - data["ps_ind_03"]))

    v["i132"] = 0.020000*np.tanh(((((data["loo_ps_ind_07_bin"] - data["loo_ps_ind_07_bin"]) - data["loo_ps_ind_06_bin"]) - data["loo_ps_ind_07_bin"]) * data["loo_ps_ind_04_cat"]))

    v["i133"] = 0.019996*np.tanh((data["ps_reg_01"] * ((((data["ps_reg_01"] + data["missing"])/2.0) - data["loo_ps_ind_05_cat"]) - data["loo_ps_car_09_cat"])))

    v["i134"] = 0.020000*np.tanh((((data["ps_ind_15"] / 2.0) - data["ps_ind_01"]) * ((data["ps_ind_03"] + data["ps_ind_15"])/2.0)))

    v["i135"] = 0.019996*np.tanh(((-(data["ps_car_13"])) - ((((data["ps_car_13"] * data["loo_ps_car_01_cat"]) * 2.0) * 2.0) * 2.0)))

    v["i136"] = 0.020000*np.tanh((((data["ps_reg_03"] + (data["ps_reg_03"] * data["loo_ps_ind_07_bin"])) + data["ps_reg_03"]) * data["loo_ps_ind_07_bin"]))

    v["i137"] = 0.019988*np.tanh((data["loo_ps_car_07_cat"] * (data["ps_reg_02"] / 2.0)))

    v["i138"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["missing"] + (data["loo_ps_car_09_cat"] * data["loo_ps_ind_02_cat"]))/2.0)))

    v["i139"] = 0.019996*np.tanh(((data["ps_reg_02"] + (data["loo_ps_car_06_cat"] / 2.0))/2.0))

    v["i140"] = 0.020000*np.tanh(((((data["loo_ps_ind_06_bin"] * data["ps_ind_15"]) - data["loo_ps_car_06_cat"]) * 2.0) * 2.0))

    v["i141"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * ((data["loo_ps_ind_05_cat"] * data["ps_car_13"]) * data["loo_ps_car_07_cat"])))

    v["i142"] = 0.019980*np.tanh(((data["ps_ind_15"] * (-(data["ps_ind_03"]))) - np.tanh((data["ps_ind_15"] - data["ps_ind_03"]))))

    v["i143"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_car_15"] * data["loo_ps_ind_02_cat"])))

    v["i144"] = 0.019996*np.tanh((((np.tanh(data["loo_ps_ind_10_bin"]) / 2.0) - data["ps_reg_01"]) / 2.0))

    v["i145"] = 0.020000*np.tanh((((data["missing"] + (data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"])) + data["loo_ps_car_08_cat"]) * data["loo_ps_ind_02_cat"]))

    v["i146"] = 0.019984*np.tanh((((data["ps_reg_02"] - data["loo_ps_car_02_cat"]) * (data["ps_reg_01"] * data["loo_ps_car_09_cat"])) / 2.0))

    v["i147"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * ((data["loo_ps_car_09_cat"] * 2.0) + (data["loo_ps_car_09_cat"] * 2.0))))

    v["i148"] = 0.019977*np.tanh(((data["ps_reg_02"] * ((data["ps_car_13"] - data["ps_reg_03"]) * data["loo_ps_ind_06_bin"])) - data["loo_ps_ind_06_bin"]))

    v["i149"] = 0.019996*np.tanh((data["ps_car_11"] * (((data["loo_ps_car_06_cat"] + data["ps_car_11"])/2.0) - data["loo_ps_ind_16_bin"])))

    v["i150"] = 0.020000*np.tanh((((data["ps_ind_14"] / 2.0) + (data["ps_ind_15"] * data["ps_ind_14"]))/2.0))

    return Outputs(-3.274750+v.sum(axis=1))





def GP2(data):

    v = pd.DataFrame()

    v["i1"] = 0.020000*np.tanh((((data["ps_car_15"] + (data["loo_ps_ind_16_bin"] + data["ps_reg_01"])) + data["ps_car_12"]) * 2.0))

    v["i2"] = 0.019910*np.tanh(((((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_17_bin"] + (data["loo_ps_car_06_cat"] + data["loo_ps_car_04_cat"])))/2.0) * 2.0) * 2.0))

    v["i3"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_car_05_cat"] + data["loo_ps_car_11_cat"])) * 2.0) + data["loo_ps_ind_16_bin"]))

    v["i4"] = 0.020000*np.tanh(((((data["ps_reg_03"] + data["loo_ps_car_11_cat"]) + data["loo_ps_ind_06_bin"]) + data["ps_reg_02"]) * 2.0))

    v["i5"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + (data["loo_ps_car_02_cat"] + (data["loo_ps_car_04_cat"] + data["loo_ps_ind_05_cat"]))) * 2.0))

    v["i6"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["ps_car_13"]) * 2.0) + (data["ps_reg_02"] + data["loo_ps_ind_17_bin"])) * 2.0))

    v["i7"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] + ((data["ps_car_13"] + data["loo_ps_ind_16_bin"]) + data["ps_car_13"])) * 2.0) * 2.0))

    v["i8"] = 0.019945*np.tanh(((data["loo_ps_car_03_cat"] + (data["loo_ps_car_08_cat"] + data["loo_ps_car_04_cat"])) + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_01_cat"])))

    v["i9"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + data["loo_ps_car_09_cat"]) + (data["loo_ps_ind_09_bin"] + data["loo_ps_ind_06_bin"])) * 2.0))

    v["i10"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + ((data["ps_car_12"] + data["ps_car_15"]) + (data["ps_reg_01"] + data["loo_ps_car_07_cat"]))))

    v["i11"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + data["loo_ps_car_11_cat"]) + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_06_bin"])) * 2.0))

    v["i12"] = 0.020000*np.tanh(((data["ps_reg_01"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_car_03_cat"] + data["ps_car_13"]))) * 2.0))

    v["i13"] = 0.020000*np.tanh(((((data["ps_car_13"] + (data["loo_ps_car_01_cat"] + data["loo_ps_car_07_cat"])) * 2.0) + data["loo_ps_ind_16_bin"]) * 2.0))

    v["i14"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] + ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_07_bin"]) + data["loo_ps_car_02_cat"])) * 2.0) * 2.0))

    v["i15"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["ps_car_13"]) + data["loo_ps_car_09_cat"]) + data["loo_ps_car_01_cat"]) * 2.0))

    v["i16"] = 0.020000*np.tanh(((data["loo_ps_ind_06_bin"] + (data["ps_reg_02"] + (data["ps_car_13"] + data["loo_ps_car_01_cat"]))) * 2.0))

    v["i17"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + ((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) + (data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"]))))

    v["i18"] = 0.020000*np.tanh(((data["ps_car_15"] + (((data["ps_reg_03"] + data["ps_reg_03"]) + data["loo_ps_car_09_cat"]) * 2.0)) * 2.0))

    v["i19"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + data["ps_reg_03"]) + (data["ps_reg_03"] - (data["ps_ind_15"] - data["ps_car_13"]))))

    v["i20"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_06_cat"])) + data["loo_ps_car_07_cat"]) + data["loo_ps_car_09_cat"]))

    v["i21"] = 0.020000*np.tanh((((data["ps_reg_02"] + data["loo_ps_ind_09_bin"]) + (data["loo_ps_ind_06_bin"] + data["loo_ps_car_11_cat"])) * 2.0))

    v["i22"] = 0.020000*np.tanh((((((data["ps_reg_03"] + data["loo_ps_ind_17_bin"]) * 2.0) * 2.0) + data["loo_ps_car_11_cat"]) * 2.0))

    v["i23"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] + ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_08_bin"]) + data["loo_ps_ind_07_bin"])) * 2.0) * 2.0))

    v["i24"] = 0.020000*np.tanh(((((data["loo_ps_car_03_cat"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_car_01_cat"]) + data["ps_ind_03"]) - data["ps_ind_15"]))

    v["i25"] = 0.020000*np.tanh(((data["ps_reg_02"] + data["ps_reg_02"]) + (data["ps_car_13"] + data["loo_ps_ind_04_cat"])))

    v["i26"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] + (data["loo_ps_car_07_cat"] + data["loo_ps_car_07_cat"])) - data["ps_ind_15"]) * 2.0))

    v["i27"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] + ((data["loo_ps_car_01_cat"] - data["ps_ind_15"]) + (data["ps_reg_03"] + data["loo_ps_car_07_cat"]))))

    v["i28"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] + ((data["loo_ps_car_09_cat"] + data["ps_reg_03"]) + (data["loo_ps_car_11_cat"] + data["loo_ps_car_09_cat"]))))

    v["i29"] = 0.020000*np.tanh(((((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_17_bin"]) + data["ps_car_13"]) - data["ps_ind_15"]) + data["ps_ind_01"]))

    v["i30"] = 0.020000*np.tanh(((data["ps_reg_01"] + ((data["ps_car_13"] + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_07_bin"])) - data["ps_ind_15"]))

    v["i31"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] + data["loo_ps_ind_02_cat"]) + data["loo_ps_ind_02_cat"]) + (data["loo_ps_car_01_cat"] + data["loo_ps_car_09_cat"])))

    v["i32"] = 0.020000*np.tanh(((((((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"]) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i33"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] + data["loo_ps_car_09_cat"]) - (data["ps_ind_15"] - data["loo_ps_car_07_cat"])))

    v["i34"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + data["loo_ps_ind_09_bin"]) + (data["loo_ps_ind_04_cat"] + (data["ps_ind_01"] + data["ps_car_15"]))))

    v["i35"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] * 2.0) + data["ps_ind_03"]) * 2.0) * 2.0))

    v["i36"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) + (data["ps_ind_03"] * data["loo_ps_ind_16_bin"])) * 2.0))

    v["i37"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + ((data["loo_ps_ind_17_bin"] * 2.0) + data["loo_ps_ind_05_cat"])) * 2.0))

    v["i38"] = 0.020000*np.tanh(((10.58713626861572266) * ((data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])) * 2.0)))

    v["i39"] = 0.019996*np.tanh(((((data["ps_car_15"] + (data["ps_reg_03"] * 2.0)) * 2.0) * 2.0) * 2.0))

    v["i40"] = 0.020000*np.tanh(((((data["ps_car_15"] + (data["loo_ps_ind_02_cat"] * 2.0))/2.0) + (data["loo_ps_ind_02_cat"] * 2.0)) + data["loo_ps_ind_17_bin"]))

    v["i41"] = 0.020000*np.tanh((((data["ps_ind_03"] * data["ps_ind_03"]) - data["ps_ind_15"]) + (data["ps_ind_03"] - data["ps_ind_15"])))

    v["i42"] = 0.020000*np.tanh((((((data["ps_ind_03"] * data["loo_ps_ind_07_bin"]) + (data["loo_ps_ind_05_cat"] * 2.0)) * 2.0) * 2.0) * 2.0))

    v["i43"] = 0.020000*np.tanh(((((data["loo_ps_ind_09_bin"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_01_cat"]) + data["loo_ps_ind_09_bin"]) * 2.0))

    v["i44"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] + ((data["ps_car_15"] + data["ps_reg_01"]) + (data["ps_ind_01"] + 1.0))))

    v["i45"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"]) * 2.0) * 2.0))

    v["i46"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_ind_03"] - ((7.0) * (data["ps_ind_03"] + data["ps_ind_03"])))))

    v["i47"] = 0.020000*np.tanh((((data["ps_ind_01"] * data["loo_ps_car_05_cat"]) * 2.0) + (data["loo_ps_car_09_cat"] - data["ps_ind_15"])))

    v["i48"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]) + ((data["loo_ps_ind_12_bin"] + data["loo_ps_ind_17_bin"]) * 2.0)))

    v["i49"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + (data["ps_ind_03"] * ((data["loo_ps_ind_07_bin"] + data["loo_ps_ind_08_bin"])/2.0))) * 2.0))

    v["i50"] = 0.020000*np.tanh((data["ps_car_15"] + (((1.0) + data["loo_ps_ind_17_bin"]) + (data["ps_car_13"] * data["ps_ind_03"]))))

    v["i51"] = 0.020000*np.tanh((((data["loo_ps_car_05_cat"] - data["ps_car_11"]) * 2.0) + data["loo_ps_car_01_cat"]))

    v["i52"] = 0.020000*np.tanh(((((data["ps_ind_15"] * data["loo_ps_ind_18_bin"]) * 2.0) * 2.0) + (data["loo_ps_ind_05_cat"] - data["ps_ind_15"])))

    v["i53"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] + ((data["ps_ind_01"] + data["loo_ps_ind_06_bin"]) * (data["loo_ps_ind_05_cat"] - data["ps_reg_01"]))))

    v["i54"] = 0.020000*np.tanh((data["loo_ps_car_05_cat"] * (((data["loo_ps_car_05_cat"] + data["missing"]) + data["ps_ind_01"]) + data["loo_ps_ind_17_bin"])))

    v["i55"] = 0.020000*np.tanh((((12.35798072814941406) * (data["loo_ps_ind_02_cat"] + data["loo_ps_ind_04_cat"])) + (data["loo_ps_ind_02_cat"] + data["loo_ps_ind_04_cat"])))

    v["i56"] = 0.020000*np.tanh((((data["loo_ps_ind_09_bin"] * data["loo_ps_car_01_cat"]) + data["loo_ps_ind_08_bin"]) + data["loo_ps_ind_07_bin"]))

    v["i57"] = 0.020000*np.tanh(((data["ps_ind_03"] - (((data["ps_ind_03"] * data["loo_ps_ind_02_cat"]) * 2.0) * 2.0)) * 2.0))

    v["i58"] = 0.019996*np.tanh(((data["ps_car_12"] * data["ps_ind_03"]) + (data["loo_ps_car_09_cat"] + (data["ps_car_12"] * data["ps_ind_03"]))))

    v["i59"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + (data["loo_ps_car_01_cat"] + data["loo_ps_car_04_cat"])) * (data["loo_ps_ind_05_cat"] - data["ps_reg_01"])))

    v["i60"] = 0.020000*np.tanh(((data["ps_reg_03"] * (data["loo_ps_ind_17_bin"] - data["ps_reg_01"])) + (data["loo_ps_car_03_cat"] * data["loo_ps_ind_09_bin"])))

    v["i61"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + (data["ps_ind_03"] + ((data["ps_ind_03"] * 2.0) * data["loo_ps_car_05_cat"]))) * 2.0))

    v["i62"] = 0.020000*np.tanh((data["ps_reg_03"] + (data["loo_ps_ind_16_bin"] * ((data["loo_ps_ind_09_bin"] + data["ps_ind_15"]) + data["loo_ps_car_04_cat"]))))

    v["i63"] = 0.020000*np.tanh(((data["loo_ps_ind_06_bin"] * (data["loo_ps_ind_05_cat"] - data["ps_car_11"])) - data["ps_ind_15"]))

    v["i64"] = 0.019992*np.tanh(((((data["loo_ps_car_05_cat"] * data["loo_ps_ind_17_bin"]) - (data["loo_ps_car_01_cat"] * data["ps_reg_02"])) * 2.0) * 2.0))

    v["i65"] = 0.020000*np.tanh(((data["loo_ps_car_11_cat"] * ((data["loo_ps_car_11_cat"] * data["loo_ps_car_09_cat"]) + (-(data["ps_reg_03"])))) * 2.0))

    v["i66"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * (((data["loo_ps_car_08_cat"] + data["loo_ps_ind_02_cat"]) + data["loo_ps_car_08_cat"]) * 2.0)) * 2.0))

    v["i67"] = 0.020000*np.tanh((((data["loo_ps_car_08_cat"] + data["loo_ps_car_08_cat"])/2.0) - (data["ps_ind_01"] * data["ps_reg_03"])))

    v["i68"] = 0.020000*np.tanh((((data["ps_ind_01"] + (data["ps_ind_01"] + data["loo_ps_ind_09_bin"])) + data["loo_ps_car_09_cat"]) * data["loo_ps_ind_09_bin"]))

    v["i69"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_car_15"]) * (data["loo_ps_ind_06_bin"] + (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_06_bin"]))))

    v["i70"] = 0.020000*np.tanh(((data["loo_ps_car_06_cat"] + data["loo_ps_ind_17_bin"]) * (data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"])))

    v["i71"] = 0.020000*np.tanh((((((-(data["ps_car_12"])) - data["ps_car_12"]) - data["ps_ind_01"]) * data["ps_car_11"]) * 2.0))

    v["i72"] = 0.019984*np.tanh((((-(((data["loo_ps_ind_02_cat"] * data["ps_ind_03"]) * 2.0))) * 2.0) - data["ps_car_11"]))

    v["i73"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_05_cat"] + data["ps_reg_03"])) * (data["ps_reg_03"] * data["ps_reg_03"])))

    v["i74"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] * data["ps_reg_02"]) - (data["loo_ps_car_11_cat"] * 2.0)) * data["ps_reg_02"]) * 2.0))

    v["i75"] = 0.019992*np.tanh((((data["loo_ps_car_04_cat"] + data["loo_ps_car_04_cat"]) + data["loo_ps_ind_07_bin"]) * (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_16_bin"])))

    v["i76"] = 0.020000*np.tanh(((-2.0 + (data["ps_ind_03"] * data["ps_ind_03"])) + (data["loo_ps_ind_09_bin"] * data["ps_ind_03"])))

    v["i77"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] + ((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_16_bin"] * data["ps_ind_01"])) + data["loo_ps_ind_04_cat"])))

    v["i78"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] * data["loo_ps_ind_07_bin"]) - data["ps_ind_15"]) * 2.0))

    v["i79"] = 0.020000*np.tanh(((((data["loo_ps_car_09_cat"] + data["ps_reg_02"]) + data["loo_ps_car_01_cat"])/2.0) * (data["loo_ps_ind_17_bin"] - data["ps_reg_03"])))

    v["i80"] = 0.019988*np.tanh((data["loo_ps_car_01_cat"] * ((((data["loo_ps_ind_08_bin"] + data["loo_ps_car_01_cat"])/2.0) + data["loo_ps_car_05_cat"]) - data["ps_car_15"])))

    v["i81"] = 0.020000*np.tanh(((((data["ps_ind_03"] * (data["loo_ps_ind_04_cat"] * 2.0)) * 2.0) + (data["loo_ps_ind_04_cat"] * 2.0)) * 2.0))

    v["i82"] = 0.020000*np.tanh((((-(data["loo_ps_car_11_cat"])) / 2.0) * ((0.0 / 2.0) + data["ps_car_11"])))

    v["i83"] = 0.020000*np.tanh(((((data["loo_ps_car_11_cat"] + data["missing"])/2.0) + (data["loo_ps_car_07_cat"] + data["loo_ps_car_01_cat"])) * data["loo_ps_ind_16_bin"]))

    v["i84"] = 0.020000*np.tanh(((data["ps_car_13"] + (data["ps_ind_15"] * (data["loo_ps_ind_07_bin"] - data["loo_ps_car_04_cat"]))) - data["loo_ps_car_04_cat"]))

    v["i85"] = 0.020000*np.tanh((data["ps_reg_02"] + (data["loo_ps_ind_18_bin"] * (-((data["ps_reg_01"] * 2.0))))))

    v["i86"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - (data["ps_reg_01"] * data["ps_ind_03"])) - (data["ps_reg_01"] * data["ps_ind_03"])))

    v["i87"] = 0.020000*np.tanh((data["loo_ps_car_04_cat"] * data["ps_ind_01"]))

    v["i88"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_07_bin"] + (data["ps_ind_01"] + (data["ps_ind_01"] + data["loo_ps_ind_07_bin"])))))

    v["i89"] = 0.020000*np.tanh((((((data["ps_ind_03"] + 1.0) * (-(data["loo_ps_ind_02_cat"]))) * 2.0) * 2.0) * 2.0))

    v["i90"] = 0.020000*np.tanh(((-3.0 + data["loo_ps_car_09_cat"]) + ((data["ps_car_11"] * data["ps_car_11"]) + -2.0)))

    v["i91"] = 0.019984*np.tanh(((data["ps_ind_15"] * (-((data["ps_ind_01"] * 2.0)))) + data["ps_ind_01"]))

    v["i92"] = 0.020000*np.tanh(((data["ps_reg_03"] * ((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_05_cat"]) - data["loo_ps_car_01_cat"])) - data["loo_ps_ind_05_cat"]))

    v["i93"] = 0.019988*np.tanh((data["ps_ind_03"] * (data["ps_ind_03"] + ((12.93563556671142578) * ((data["loo_ps_ind_08_bin"] + data["ps_reg_03"])/2.0)))))

    v["i94"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((-1.0 - (data["ps_ind_03"] * 2.0)) - (data["ps_ind_03"] * 2.0))))

    v["i95"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + data["loo_ps_car_09_cat"]) - data["loo_ps_car_04_cat"]))

    v["i96"] = 0.020000*np.tanh(((data["ps_ind_03"] * (-(data["ps_reg_01"]))) - (data["loo_ps_ind_02_cat"] + data["ps_ind_03"])))

    v["i97"] = 0.020000*np.tanh((((data["ps_car_13"] - data["ps_reg_03"]) - data["loo_ps_car_09_cat"]) * data["ps_reg_01"]))

    v["i98"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * ((data["loo_ps_car_09_cat"] * (data["loo_ps_car_09_cat"] * data["loo_ps_ind_07_bin"])) * data["loo_ps_car_09_cat"])))

    v["i99"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (((data["loo_ps_ind_05_cat"] * 2.0) + data["loo_ps_ind_05_cat"]) - data["ps_car_15"])))

    v["i100"] = 0.019988*np.tanh(((data["loo_ps_ind_06_bin"] * data["ps_ind_15"]) - (data["ps_ind_15"] * (-(data["ps_reg_01"])))))

    v["i101"] = 0.020000*np.tanh(((data["ps_ind_01"] * ((data["loo_ps_ind_18_bin"] + data["loo_ps_ind_16_bin"])/2.0)) + ((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_05_cat"])/2.0)))

    v["i102"] = 0.020000*np.tanh((data["ps_reg_01"] * (-(((data["loo_ps_car_05_cat"] + data["loo_ps_ind_06_bin"])/2.0)))))

    v["i103"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + data["loo_ps_ind_17_bin"]) * ((data["loo_ps_ind_05_cat"] * 2.0) + data["loo_ps_ind_05_cat"])) * 2.0))

    v["i104"] = 0.020000*np.tanh((((((data["loo_ps_ind_02_cat"] + data["ps_reg_02"])/2.0) * data["ps_car_12"]) + data["loo_ps_ind_02_cat"])/2.0))

    v["i105"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] - data["loo_ps_ind_17_bin"]) * data["ps_ind_15"]) * 2.0))

    v["i106"] = 0.020000*np.tanh((data["ps_car_13"] + (-2.0 + (data["ps_ind_03"] * (data["ps_ind_03"] + data["loo_ps_ind_05_cat"])))))

    v["i107"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] * data["loo_ps_ind_09_bin"]) + ((data["loo_ps_ind_16_bin"] + data["ps_reg_02"]) * data["loo_ps_ind_16_bin"]))/2.0))

    v["i108"] = 0.019996*np.tanh(((data["loo_ps_car_11_cat"] * (((-3.0 - data["ps_ind_03"]) / 2.0) / 2.0)) * data["loo_ps_car_02_cat"]))

    v["i109"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_17_bin"]) * data["loo_ps_car_09_cat"]))

    v["i110"] = 0.020000*np.tanh((((((data["ps_ind_03"] - data["ps_ind_01"]) * data["loo_ps_ind_04_cat"]) * 2.0) * 2.0) * 2.0))

    v["i111"] = 0.020000*np.tanh(((-(((data["loo_ps_ind_04_cat"] + data["loo_ps_car_02_cat"]) * (data["ps_reg_02"] + data["loo_ps_car_06_cat"])))) * 2.0))

    v["i112"] = 0.020000*np.tanh((data["ps_reg_03"] * ((((data["loo_ps_ind_09_bin"] - data["loo_ps_ind_18_bin"]) + data["loo_ps_ind_09_bin"])/2.0) - data["loo_ps_ind_18_bin"])))

    v["i113"] = 0.020000*np.tanh((((data["ps_car_13"] - 2.0) * data["ps_car_13"]) + (data["loo_ps_car_03_cat"] * data["loo_ps_car_06_cat"])))

    v["i114"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["ps_reg_02"] + ((data["loo_ps_car_04_cat"] + data["ps_reg_02"]) + data["loo_ps_car_04_cat"]))))

    v["i115"] = 0.020000*np.tanh((-(((data["loo_ps_car_06_cat"] * 2.0) * (data["loo_ps_ind_04_cat"] + data["ps_reg_02"])))))

    v["i116"] = 0.019871*np.tanh(((((-2.0 - data["ps_ind_03"]) * 2.0) - data["ps_ind_03"]) * (data["loo_ps_ind_02_cat"] * 2.0)))

    v["i117"] = 0.019996*np.tanh((((data["loo_ps_car_11_cat"] + np.tanh(-1.0))/2.0) + (data["loo_ps_car_09_cat"] * data["loo_ps_ind_17_bin"])))

    v["i118"] = 0.019969*np.tanh(((data["loo_ps_ind_02_cat"] * (data["loo_ps_car_08_cat"] * ((data["loo_ps_car_08_cat"] + data["loo_ps_car_08_cat"])/2.0))) * 2.0))

    v["i119"] = 0.020000*np.tanh(((data["ps_car_11"] * data["loo_ps_ind_04_cat"]) + (data["ps_car_11"] * np.tanh(data["loo_ps_ind_04_cat"]))))

    v["i120"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * data["loo_ps_ind_09_bin"]))

    v["i121"] = 0.020000*np.tanh((data["ps_ind_01"] + (data["ps_ind_01"] * (data["ps_ind_01"] * (data["loo_ps_car_03_cat"] * data["ps_ind_01"])))))

    v["i122"] = 0.020000*np.tanh((data["loo_ps_car_06_cat"] * (-2.0 + (data["loo_ps_car_06_cat"] * data["loo_ps_ind_16_bin"]))))

    v["i123"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_07_bin"]) - (data["loo_ps_ind_04_cat"] * (data["loo_ps_ind_07_bin"] * 2.0))))

    v["i124"] = 0.019906*np.tanh((-(((data["loo_ps_ind_17_bin"] + data["loo_ps_car_01_cat"]) * data["ps_car_11"]))))

    v["i125"] = 0.019996*np.tanh((data["loo_ps_car_06_cat"] - data["ps_reg_03"]))

    v["i126"] = 0.020000*np.tanh(((data["loo_ps_ind_08_bin"] + data["loo_ps_car_09_cat"]) * np.tanh((-(data["ps_reg_01"])))))

    v["i127"] = 0.020000*np.tanh(((data["ps_car_15"] + (data["loo_ps_car_04_cat"] + data["loo_ps_ind_17_bin"])) * (-(data["ps_car_15"]))))

    v["i128"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_car_09_cat"] - data["ps_ind_03"]) * data["ps_ind_01"])))

    v["i129"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * ((data["loo_ps_car_09_cat"] * 2.0) - data["ps_ind_03"])) * 2.0))

    v["i130"] = 0.020000*np.tanh((((data["ps_reg_02"] + data["ps_reg_02"])/2.0) - (data["loo_ps_ind_06_bin"] - data["loo_ps_ind_02_cat"])))

    v["i131"] = 0.019941*np.tanh((np.tanh(data["loo_ps_car_09_cat"]) * ((data["loo_ps_ind_08_bin"] + (data["loo_ps_car_09_cat"] * data["loo_ps_car_09_cat"]))/2.0)))

    v["i132"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * (data["loo_ps_car_09_cat"] * data["loo_ps_car_09_cat"])))

    v["i133"] = 0.019996*np.tanh(((data["loo_ps_ind_04_cat"] + (data["missing"] * data["ps_reg_01"]))/2.0))

    v["i134"] = 0.020000*np.tanh(((data["ps_ind_01"] * (data["ps_car_13"] - (data["loo_ps_ind_04_cat"] + data["ps_ind_01"]))) * data["ps_ind_03"]))

    v["i135"] = 0.019996*np.tanh(((((data["ps_ind_03"] - data["loo_ps_car_01_cat"]) - data["loo_ps_car_01_cat"]) - data["loo_ps_car_01_cat"]) * data["ps_car_13"]))

    v["i136"] = 0.020000*np.tanh(((data["ps_reg_01"] * data["loo_ps_ind_05_cat"]) * (data["ps_ind_01"] + (data["loo_ps_car_04_cat"] * data["ps_reg_01"]))))

    v["i137"] = 0.019988*np.tanh((((data["loo_ps_ind_05_cat"] * data["ps_ind_03"]) + (data["ps_ind_03"] * (data["ps_ind_03"] * data["loo_ps_car_01_cat"])))/2.0))

    v["i138"] = 0.020000*np.tanh((((((data["loo_ps_car_03_cat"] + data["loo_ps_car_09_cat"])/2.0) + (data["ps_car_13"] + data["loo_ps_car_03_cat"]))/2.0) * data["loo_ps_ind_09_bin"]))

    v["i139"] = 0.019996*np.tanh(((data["loo_ps_ind_05_cat"] * (data["ps_ind_01"] - data["ps_reg_01"])) - ((-1.0 / 2.0) / 2.0)))

    v["i140"] = 0.020000*np.tanh((data["ps_ind_03"] - (data["ps_ind_15"] * data["ps_ind_03"])))

    v["i141"] = 0.020000*np.tanh(((data["ps_ind_01"] * (data["ps_ind_01"] * (data["ps_ind_01"] * data["loo_ps_ind_08_bin"]))) / 2.0))

    v["i142"] = 0.019980*np.tanh((data["ps_ind_15"] * ((data["ps_ind_14"] + (-(data["loo_ps_car_06_cat"])))/2.0)))

    v["i143"] = 0.020000*np.tanh((-((data["ps_reg_01"] + (data["ps_car_15"] * (data["ps_reg_01"] + data["loo_ps_car_03_cat"]))))))

    v["i144"] = 0.019996*np.tanh(((data["loo_ps_ind_16_bin"] * (data["loo_ps_car_07_cat"] + data["loo_ps_car_09_cat"])) - data["loo_ps_car_10_cat"]))

    v["i145"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["loo_ps_car_08_cat"] + ((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) - data["ps_car_14"]))))

    v["i146"] = 0.019984*np.tanh(((data["ps_reg_02"] + data["loo_ps_car_01_cat"]) * ((data["ps_reg_02"] * data["loo_ps_car_05_cat"]) - data["ps_ind_01"])))

    v["i147"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_05_cat"] + -2.0)/2.0))) - data["loo_ps_ind_05_cat"]))

    v["i148"] = 0.019977*np.tanh(np.tanh((((data["loo_ps_car_06_cat"] * data["loo_ps_ind_02_cat"]) * data["loo_ps_ind_02_cat"]) * data["loo_ps_ind_02_cat"])))

    v["i149"] = 0.019996*np.tanh((data["loo_ps_car_04_cat"] * (((data["loo_ps_car_09_cat"] - data["ps_car_11"]) - data["ps_car_11"]) - data["loo_ps_car_04_cat"])))

    v["i150"] = 0.020000*np.tanh(((((data["ps_ind_15"] * (data["loo_ps_ind_07_bin"] + data["loo_ps_ind_04_cat"])) * data["ps_ind_15"]) / 2.0) * 2.0))

    return Outputs(-3.274750+v.sum(axis=1))





def GP3(data, p=-1):

    v = pd.DataFrame()

    v["i1"] = 0.020000*np.tanh((((data["ps_car_15"] + (data["ps_reg_01"] + data["loo_ps_car_06_cat"])) + data["ps_car_12"]) * (13.12984561920166016)))

    v["i2"] = 0.019910*np.tanh(((((data["loo_ps_car_02_cat"] + data["loo_ps_car_04_cat"]) + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_17_bin"]) * 2.0))

    v["i3"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + data["loo_ps_ind_16_bin"]) + ((data["loo_ps_car_11_cat"] + data["loo_ps_car_05_cat"]) + data["loo_ps_car_11_cat"])))

    v["i4"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] + (data["loo_ps_ind_09_bin"] + data["ps_reg_02"])) + data["loo_ps_car_11_cat"]) * 2.0))

    v["i5"] = 0.020000*np.tanh(((data["loo_ps_car_04_cat"] + (data["loo_ps_ind_07_bin"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]))) + data["loo_ps_ind_05_cat"]))

    v["i6"] = 0.020000*np.tanh((data["ps_reg_02"] + ((data["ps_car_13"] + data["loo_ps_ind_17_bin"]) + (data["loo_ps_ind_05_cat"] + data["ps_car_13"]))))

    v["i7"] = 0.020000*np.tanh(((((data["ps_car_13"] + data["ps_car_13"]) + (data["loo_ps_car_03_cat"] + data["loo_ps_ind_16_bin"])) * 2.0) * 2.0))

    v["i8"] = 0.019945*np.tanh((data["loo_ps_car_06_cat"] + (data["loo_ps_car_04_cat"] + (data["loo_ps_car_01_cat"] + (data["loo_ps_car_03_cat"] + data["loo_ps_ind_17_bin"])))))

    v["i9"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] + (data["loo_ps_ind_16_bin"] + (data["loo_ps_car_07_cat"] * 2.0))) + data["ps_car_13"]) * 2.0))

    v["i10"] = 0.020000*np.tanh(((data["ps_reg_01"] + (data["loo_ps_car_07_cat"] + data["ps_car_12"])) + (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_17_bin"])))

    v["i11"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + (data["loo_ps_car_11_cat"] + data["loo_ps_car_07_cat"])) * 2.0) + data["loo_ps_ind_06_bin"]) * 2.0))

    v["i12"] = 0.020000*np.tanh(((data["ps_car_13"] + ((data["ps_reg_01"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_car_03_cat"])) * 2.0))

    v["i13"] = 0.020000*np.tanh((((data["ps_car_13"] + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_16_bin"]) + (data["ps_car_13"] + data["loo_ps_car_01_cat"])))

    v["i14"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] + (data["loo_ps_car_02_cat"] + data["ps_car_15"])) + data["loo_ps_ind_05_cat"]) * 2.0))

    v["i15"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_05_cat"] + data["ps_car_13"])) + data["loo_ps_car_01_cat"]) * 2.0))

    v["i16"] = 0.020000*np.tanh((((data["ps_car_13"] + (data["ps_reg_02"] + (data["loo_ps_ind_06_bin"] + data["loo_ps_car_01_cat"]))) * 2.0) * 2.0))

    v["i17"] = 0.020000*np.tanh((((data["loo_ps_ind_17_bin"] - data["ps_ind_15"]) + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_07_bin"])) + data["ps_reg_01"]))

    v["i18"] = 0.020000*np.tanh(((data["ps_reg_02"] + (data["ps_car_15"] + ((data["ps_reg_03"] * 2.0) + data["loo_ps_car_09_cat"]))) * 2.0))

    v["i19"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + (data["loo_ps_car_03_cat"] - data["ps_ind_15"])) + (data["ps_reg_03"] + data["ps_car_13"])))

    v["i20"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_car_07_cat"] + (data["loo_ps_car_03_cat"] + data["loo_ps_ind_05_cat"]))) * 2.0) * 2.0))

    v["i21"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] + (data["loo_ps_car_09_cat"] - data["ps_ind_15"])) + data["ps_reg_02"]) * 2.0))

    v["i22"] = 0.020000*np.tanh(((data["ps_reg_03"] + (((data["loo_ps_ind_17_bin"] + data["ps_reg_03"]) * 2.0) * 2.0)) * 2.0))

    v["i23"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_06_bin"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_09_bin"])))))

    v["i24"] = 0.020000*np.tanh(((data["ps_ind_03"] + data["loo_ps_ind_05_cat"]) - (data["ps_ind_15"] - (data["loo_ps_car_01_cat"] - data["ps_ind_15"]))))

    v["i25"] = 0.020000*np.tanh(((((((data["ps_ind_03"] * -3.0) * data["loo_ps_ind_02_cat"]) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i26"] = 0.020000*np.tanh((data["ps_car_13"] + ((data["loo_ps_car_07_cat"] - data["ps_ind_15"]) + (data["ps_reg_03"] * 2.0))))

    v["i27"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + (data["loo_ps_car_01_cat"] + data["loo_ps_ind_16_bin"])) + (data["loo_ps_ind_06_bin"] - data["ps_ind_15"])))

    v["i28"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + data["loo_ps_car_11_cat"]) + (data["ps_reg_03"] + data["loo_ps_ind_04_cat"])) * 2.0))

    v["i29"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + data["loo_ps_ind_17_bin"]) + ((data["loo_ps_ind_08_bin"] + data["loo_ps_ind_17_bin"]) + data["ps_car_13"])))

    v["i30"] = 0.020000*np.tanh((data["loo_ps_ind_09_bin"] + ((data["loo_ps_ind_07_bin"] + (data["loo_ps_car_07_cat"] + data["ps_reg_01"])) + data["ps_car_13"])))

    v["i31"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_16_bin"] + (data["loo_ps_car_03_cat"] + (data["loo_ps_car_01_cat"] + data["loo_ps_car_01_cat"])))))

    v["i32"] = 0.020000*np.tanh(((((data["ps_ind_03"] * data["ps_ind_03"]) + (data["ps_ind_03"] + data["loo_ps_ind_17_bin"])) * 2.0) * 2.0))

    v["i33"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] + data["loo_ps_car_07_cat"]) - (data["ps_ind_15"] - data["loo_ps_ind_02_cat"])) * 2.0))

    v["i34"] = 0.020000*np.tanh((((data["ps_ind_01"] + data["ps_reg_03"]) + (data["loo_ps_ind_09_bin"] + data["ps_car_15"])) + data["loo_ps_car_07_cat"]))

    v["i35"] = 0.020000*np.tanh((data["loo_ps_car_11_cat"] + ((data["loo_ps_ind_05_cat"] + (data["ps_ind_03"] + (data["loo_ps_ind_05_cat"] * 2.0))) * 2.0)))

    v["i36"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((data["loo_ps_ind_05_cat"] + (data["ps_ind_03"] * data["loo_ps_ind_16_bin"])) * 2.0)) * 2.0))

    v["i37"] = 0.020000*np.tanh((((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_07_cat"])) * 2.0))

    v["i38"] = 0.020000*np.tanh(((((data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])) * 2.0) + data["loo_ps_car_07_cat"]) * 2.0))

    v["i39"] = 0.019996*np.tanh((((((((data["ps_reg_03"] * 2.0) + data["ps_car_15"]) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i40"] = 0.020000*np.tanh(((((data["loo_ps_ind_02_cat"] * 2.0) + data["ps_ind_01"]) + (data["loo_ps_ind_17_bin"] + data["ps_car_15"])) * 2.0))

    v["i41"] = 0.020000*np.tanh((((((data["ps_ind_15"] - data["loo_ps_car_09_cat"]) * data["loo_ps_ind_18_bin"]) - data["ps_ind_15"]) * 2.0) * 2.0))

    v["i42"] = 0.020000*np.tanh(((data["ps_ind_03"] + data["loo_ps_ind_05_cat"]) + (data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"])))

    v["i43"] = 0.020000*np.tanh(((((data["loo_ps_ind_08_bin"] + (data["loo_ps_ind_07_bin"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0))

    v["i44"] = 0.020000*np.tanh(((((data["loo_ps_car_01_cat"] + data["loo_ps_ind_09_bin"]) + (data["loo_ps_ind_09_bin"] + data["ps_ind_01"])) * 2.0) * 2.0))

    v["i45"] = 0.020000*np.tanh((((((data["loo_ps_ind_02_cat"] + data["loo_ps_car_07_cat"]) * 2.0) + data["loo_ps_ind_05_cat"]) * 2.0) + data["ps_car_13"]))

    v["i46"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] - ((data["loo_ps_ind_02_cat"] * data["ps_ind_03"]) * 2.0)) + data["loo_ps_car_07_cat"]) * 2.0))

    v["i47"] = 0.020000*np.tanh((((data["loo_ps_car_05_cat"] * data["ps_ind_01"]) + data["loo_ps_car_09_cat"]) - (data["ps_ind_15"] * 2.0)))

    v["i48"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_16_bin"] * data["ps_car_12"]) + data["loo_ps_ind_17_bin"])) + data["loo_ps_ind_04_cat"]))

    v["i49"] = 0.020000*np.tanh((((((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"]) * 2.0) * 2.0) + data["loo_ps_ind_05_cat"]))

    v["i50"] = 0.020000*np.tanh((((data["ps_ind_03"] * data["ps_car_13"]) + data["loo_ps_car_08_cat"]) + (data["loo_ps_car_03_cat"] * data["loo_ps_ind_17_bin"])))

    v["i51"] = 0.020000*np.tanh((((data["loo_ps_car_05_cat"] * (data["ps_ind_01"] + data["loo_ps_ind_17_bin"])) + data["loo_ps_ind_06_bin"]) * 2.0))

    v["i52"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_12_bin"]) - data["ps_ind_15"]) + (data["loo_ps_ind_05_cat"] - data["ps_car_11"])))

    v["i53"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] * data["loo_ps_ind_05_cat"]) * 2.0) - (data["ps_reg_01"] * data["loo_ps_ind_06_bin"])))

    v["i54"] = 0.020000*np.tanh(((data["ps_ind_01"] + ((data["loo_ps_ind_17_bin"] + (data["loo_ps_car_05_cat"] + data["loo_ps_ind_17_bin"]))/2.0)) * data["loo_ps_car_05_cat"]))

    v["i55"] = 0.020000*np.tanh((((data["ps_ind_01"] * (data["loo_ps_ind_18_bin"] - data["ps_reg_03"])) + (data["loo_ps_ind_04_cat"] * 2.0)) * 2.0))

    v["i56"] = 0.020000*np.tanh((data["loo_ps_ind_07_bin"] + ((data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_07_bin"] + 0.0)) + data["loo_ps_car_01_cat"])))

    v["i57"] = 0.020000*np.tanh(((data["loo_ps_ind_09_bin"] + data["ps_ind_03"]) * ((data["ps_ind_03"] * 2.0) + (data["loo_ps_ind_06_bin"] * 2.0))))

    v["i58"] = 0.019996*np.tanh((((data["ps_car_12"] * data["ps_ind_03"]) + -1.0) + (data["ps_ind_03"] * data["ps_ind_03"])))

    v["i59"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - (data["loo_ps_car_04_cat"] * data["ps_car_11"])) + -1.0))

    v["i60"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_07_bin"] + data["ps_reg_03"])) * (data["loo_ps_ind_17_bin"] - data["ps_reg_01"])))

    v["i61"] = 0.020000*np.tanh((data["ps_car_15"] + (data["ps_car_15"] + (((data["ps_reg_03"] + data["ps_reg_03"]) * 2.0) * 2.0))))

    v["i62"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] - (data["loo_ps_car_04_cat"] - ((data["loo_ps_car_04_cat"] + data["loo_ps_ind_17_bin"]) * data["loo_ps_ind_09_bin"]))))

    v["i63"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) - data["loo_ps_car_11_cat"])))

    v["i64"] = 0.019992*np.tanh(((data["loo_ps_car_05_cat"] + ((data["loo_ps_car_05_cat"] - data["loo_ps_car_05_cat"]) - data["loo_ps_ind_18_bin"])) * 2.0))

    v["i65"] = 0.020000*np.tanh(((data["ps_car_14"] - (data["ps_reg_03"] * data["loo_ps_car_11_cat"])) + data["loo_ps_car_09_cat"]))

    v["i66"] = 0.020000*np.tanh((((10.0) * data["loo_ps_ind_02_cat"]) * ((data["missing"] + data["loo_ps_car_08_cat"]) + data["loo_ps_ind_02_cat"])))

    v["i67"] = 0.020000*np.tanh(((-(data["ps_reg_02"])) - (((data["loo_ps_car_06_cat"] * 2.0) * 2.0) * data["ps_reg_02"])))

    v["i68"] = 0.020000*np.tanh(((data["loo_ps_ind_07_bin"] * data["loo_ps_car_05_cat"]) + (data["loo_ps_ind_09_bin"] * data["loo_ps_car_03_cat"])))

    v["i69"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) - data["ps_car_15"]) * data["loo_ps_ind_06_bin"]) - data["ps_ind_15"]))

    v["i70"] = 0.020000*np.tanh(((data["loo_ps_car_04_cat"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_ind_17_bin"])) * (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"])))

    v["i71"] = 0.020000*np.tanh(((-(data["ps_car_11"])) - (data["ps_ind_01"] * (data["ps_car_11"] * 2.0))))

    v["i72"] = 0.019984*np.tanh((data["loo_ps_ind_17_bin"] + (-((data["ps_ind_03"] * (data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_02_cat"] * 2.0)))))))

    v["i73"] = 0.020000*np.tanh((data["missing"] + (data["loo_ps_ind_05_cat"] * (data["ps_reg_03"] * 2.0))))

    v["i74"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - ((np.tanh(data["ps_reg_02"]) + data["loo_ps_car_11_cat"])/2.0)) * data["ps_reg_02"]))

    v["i75"] = 0.019992*np.tanh((data["loo_ps_car_01_cat"] - (data["loo_ps_car_01_cat"] * (data["ps_reg_01"] * 2.0))))

    v["i76"] = 0.020000*np.tanh((((data["ps_ind_03"] * data["loo_ps_car_05_cat"]) + (data["loo_ps_ind_09_bin"] * data["ps_ind_03"])) + data["loo_ps_car_08_cat"]))

    v["i77"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_16_bin"]) * (data["ps_ind_01"] + data["loo_ps_ind_07_bin"])))

    v["i78"] = 0.020000*np.tanh(((data["loo_ps_ind_07_bin"] * (data["ps_ind_15"] + data["loo_ps_ind_17_bin"])) * 2.0))

    v["i79"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] * (data["loo_ps_car_07_cat"] + data["loo_ps_car_09_cat"])) - (data["loo_ps_car_01_cat"] * data["ps_reg_03"])))

    v["i80"] = 0.019988*np.tanh((data["ps_ind_01"] * (2.0 - (data["ps_ind_01"] * (data["ps_ind_01"] * data["ps_ind_01"])))))

    v["i81"] = 0.020000*np.tanh((((1.0 + ((2.0) + data["ps_car_15"])) + data["loo_ps_ind_17_bin"]) + data["ps_car_15"]))

    v["i82"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] - (data["ps_car_11"] * data["ps_car_12"])) - (data["loo_ps_car_07_cat"] * data["ps_car_12"])))

    v["i83"] = 0.020000*np.tanh((((data["loo_ps_ind_16_bin"] * data["loo_ps_car_04_cat"]) + (data["loo_ps_car_01_cat"] * data["loo_ps_ind_16_bin"])) * 2.0))

    v["i84"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"])/2.0) + ((((data["loo_ps_ind_04_cat"] + data["loo_ps_ind_04_cat"])/2.0) + data["loo_ps_ind_05_cat"])/2.0))/2.0))

    v["i85"] = 0.020000*np.tanh(((-((((data["ps_ind_03"] * 2.0) - (-(data["ps_reg_01"]))) * data["loo_ps_ind_02_cat"]))) * 2.0))

    v["i86"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * (data["ps_reg_01"] / 2.0)) - data["ps_ind_03"]) * data["ps_reg_01"]))

    v["i87"] = 0.020000*np.tanh((data["ps_ind_01"] * ((data["loo_ps_car_04_cat"] - data["loo_ps_car_01_cat"]) - data["ps_ind_15"])))

    v["i88"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_07_bin"] + ((data["loo_ps_ind_07_bin"] + data["loo_ps_car_09_cat"]) * 2.0))))

    v["i89"] = 0.020000*np.tanh((data["ps_reg_02"] + (data["loo_ps_car_09_cat"] - np.tanh((data["ps_reg_01"] * data["ps_reg_02"])))))

    v["i90"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * (((data["loo_ps_car_09_cat"] * data["loo_ps_ind_09_bin"]) * data["loo_ps_car_09_cat"]) - data["ps_car_11"])))

    v["i91"] = 0.019984*np.tanh((data["ps_ind_01"] * (data["loo_ps_car_04_cat"] + data["loo_ps_ind_05_cat"])))

    v["i92"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] - data["ps_ind_14"]) - data["loo_ps_car_01_cat"]) - data["loo_ps_car_01_cat"]) * data["ps_reg_03"]))

    v["i93"] = 0.019988*np.tanh(((data["loo_ps_ind_08_bin"] * (data["ps_ind_03"] + data["loo_ps_car_03_cat"])) + data["loo_ps_car_11_cat"]))

    v["i94"] = 0.020000*np.tanh(((data["loo_ps_ind_18_bin"] * (data["loo_ps_ind_12_bin"] - data["ps_reg_01"])) - (data["ps_ind_03"] * data["loo_ps_ind_02_cat"])))

    v["i95"] = 0.020000*np.tanh((((((data["loo_ps_ind_04_cat"] - data["loo_ps_ind_02_cat"]) * 2.0) + data["loo_ps_ind_07_bin"]) * 2.0) * 2.0))

    v["i96"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] - data["ps_car_11"]) - data["loo_ps_car_04_cat"]) * (data["ps_ind_03"] + data["loo_ps_car_04_cat"])))

    v["i97"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + data["ps_reg_02"]) * 2.0) * data["loo_ps_ind_16_bin"]))

    v["i98"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] * ((data["missing"] + (data["ps_ind_01"] + data["loo_ps_car_06_cat"])) + data["loo_ps_car_06_cat"])))

    v["i99"] = 0.020000*np.tanh((data["ps_reg_01"] * ((data["missing"] - data["loo_ps_ind_18_bin"]) - data["loo_ps_ind_05_cat"])))

    v["i100"] = 0.019988*np.tanh(((data["loo_ps_ind_04_cat"] + (data["loo_ps_car_05_cat"] - data["ps_car_11"])) / 2.0))

    v["i101"] = 0.020000*np.tanh((data["loo_ps_ind_09_bin"] * ((data["loo_ps_ind_05_cat"] + (data["ps_ind_01"] + data["loo_ps_ind_18_bin"])) + data["loo_ps_ind_18_bin"])))

    v["i102"] = 0.020000*np.tanh((((((data["ps_ind_03"] * data["loo_ps_ind_02_cat"]) * data["ps_ind_03"]) - data["loo_ps_ind_02_cat"]) * 2.0) * 2.0))

    v["i103"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"]) - data["loo_ps_ind_05_cat"]) + (data["loo_ps_car_09_cat"] * data["loo_ps_ind_17_bin"])))

    v["i104"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * data["ps_car_12"]) - (data["ps_car_11"] * data["ps_car_12"])))

    v["i105"] = 0.020000*np.tanh((data["ps_ind_03"] * ((data["loo_ps_ind_04_cat"] + data["loo_ps_car_11_cat"]) + ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"])/2.0))))

    v["i106"] = 0.020000*np.tanh(((-((data["loo_ps_car_01_cat"] * data["ps_car_15"]))) - (data["ps_car_15"] * data["loo_ps_ind_17_bin"])))

    v["i107"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] + (data["loo_ps_ind_09_bin"] * (((data["loo_ps_ind_09_bin"] * data["loo_ps_car_03_cat"]) + data["loo_ps_ind_09_bin"])/2.0))))

    v["i108"] = 0.019996*np.tanh((((np.tanh((data["loo_ps_ind_04_cat"] * data["ps_ind_03"])) * 2.0) + data["loo_ps_ind_04_cat"]) * 2.0))

    v["i109"] = 0.020000*np.tanh((((data["missing"] + data["missing"])/2.0) + data["loo_ps_ind_17_bin"]))

    v["i110"] = 0.020000*np.tanh((((((-(data["loo_ps_ind_02_cat"])) - data["ps_ind_03"]) * (data["loo_ps_ind_02_cat"] * 2.0)) * 2.0) * 2.0))

    v["i111"] = 0.020000*np.tanh((((-(data["loo_ps_car_11_cat"])) - data["ps_ind_15"]) * data["ps_ind_15"]))

    v["i112"] = 0.020000*np.tanh(((((data["loo_ps_ind_02_cat"] * 2.0) * 2.0) * ((data["loo_ps_car_08_cat"] * 2.0) + data["loo_ps_ind_02_cat"])) * 2.0))

    v["i113"] = 0.020000*np.tanh(((((data["missing"] * data["loo_ps_ind_16_bin"]) + data["ps_car_13"])/2.0) + (data["loo_ps_car_07_cat"] * data["loo_ps_ind_16_bin"])))

    v["i114"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (data["loo_ps_car_04_cat"] + data["ps_reg_02"])) - data["loo_ps_car_04_cat"]))

    v["i115"] = 0.020000*np.tanh(((data["ps_car_12"] * data["loo_ps_ind_07_bin"]) - (data["loo_ps_ind_16_bin"] * data["ps_reg_01"])))

    v["i116"] = 0.019871*np.tanh((data["loo_ps_ind_16_bin"] - data["loo_ps_ind_07_bin"]))

    v["i117"] = 0.019996*np.tanh(((data["loo_ps_ind_17_bin"] + data["loo_ps_car_09_cat"]) * data["loo_ps_ind_05_cat"]))

    v["i118"] = 0.019969*np.tanh((np.tanh((data["ps_ind_15"] * data["loo_ps_ind_07_bin"])) - (data["ps_reg_01"] * data["loo_ps_ind_08_bin"])))

    v["i119"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_04_cat"] * (data["ps_car_11"] + (data["loo_ps_car_09_cat"] + data["ps_car_11"])))))

    v["i120"] = 0.020000*np.tanh((((((data["ps_ind_03"] * 2.0) * data["ps_reg_03"]) * data["ps_reg_03"]) - data["ps_ind_03"]) * 2.0))

    v["i121"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] + data["loo_ps_car_07_cat"])/2.0) * data["loo_ps_ind_09_bin"]))

    v["i122"] = 0.020000*np.tanh((np.tanh((data["loo_ps_car_03_cat"] * (data["loo_ps_car_03_cat"] / 2.0))) - (data["loo_ps_car_03_cat"] * data["ps_car_15"])))

    v["i123"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] * (-(data["ps_reg_02"]))) * 2.0))

    v["i124"] = 0.019906*np.tanh(((0.0 + (((-1.0 / 2.0) - data["ps_car_11"]) * data["loo_ps_car_04_cat"]))/2.0))

    v["i125"] = 0.019996*np.tanh((-(((data["ps_reg_02"] + data["loo_ps_car_06_cat"]) * data["loo_ps_car_02_cat"]))))

    v["i126"] = 0.020000*np.tanh((((data["ps_reg_03"] - (data["loo_ps_car_01_cat"] + data["ps_reg_01"])) - data["ps_car_15"]) * data["ps_reg_03"]))

    v["i127"] = 0.020000*np.tanh((data["ps_ind_01"] * ((data["loo_ps_car_03_cat"] * (data["ps_ind_01"] * data["ps_ind_01"])) * data["ps_ind_01"])))

    v["i128"] = 0.020000*np.tanh((((-((data["loo_ps_car_09_cat"] + data["ps_reg_03"]))) - data["ps_reg_03"]) * data["ps_reg_01"]))

    v["i129"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] * 2.0) - data["ps_ind_03"]) * data["loo_ps_ind_02_cat"]))

    v["i130"] = 0.020000*np.tanh(((((data["loo_ps_car_06_cat"] * data["loo_ps_car_11_cat"]) * data["loo_ps_car_06_cat"]) - data["loo_ps_car_06_cat"]) - data["loo_ps_car_06_cat"]))

    v["i131"] = 0.019941*np.tanh((((-1.0 + data["ps_car_12"])/2.0) + (data["ps_car_12"] + (-2.0 + data["loo_ps_car_09_cat"]))))

    v["i132"] = 0.020000*np.tanh((((data["loo_ps_car_06_cat"] * (data["loo_ps_car_06_cat"] * (2.91371297836303711))) - data["loo_ps_car_06_cat"]) - data["loo_ps_car_11_cat"]))

    v["i133"] = 0.019996*np.tanh((((((data["loo_ps_car_09_cat"] + data["missing"])/2.0) + data["missing"])/2.0) * (data["ps_reg_01"] * data["loo_ps_ind_07_bin"])))

    v["i134"] = 0.020000*np.tanh((((data["ps_ind_15"] - data["ps_ind_01"]) - data["ps_ind_01"]) * data["ps_reg_02"]))

    v["i135"] = 0.019996*np.tanh((((-((((data["ps_ind_15"] * data["ps_car_13"]) + data["ps_ind_15"])/2.0))) / 2.0) * data["ps_car_13"]))

    v["i136"] = 0.020000*np.tanh(((data["ps_reg_03"] * data["loo_ps_ind_07_bin"]) + ((data["ps_reg_02"] + (data["ps_reg_03"] * data["ps_reg_03"]))/2.0)))

    v["i137"] = 0.019988*np.tanh(((data["ps_ind_03"] * data["ps_ind_03"]) + (data["loo_ps_ind_05_cat"] + (-2.0 - data["ps_ind_03"]))))

    v["i138"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] * data["loo_ps_ind_18_bin"]) - data["ps_reg_03"]) * (data["missing"] * 2.0)))

    v["i139"] = 0.019996*np.tanh((-(((data["ps_reg_01"] * (data["ps_reg_03"] + data["loo_ps_car_09_cat"])) + (data["loo_ps_ind_05_cat"] * 2.0)))))

    v["i140"] = 0.020000*np.tanh((data["ps_ind_03"] * (((data["loo_ps_ind_05_cat"] * data["ps_reg_03"]) + (data["ps_car_11"] - data["ps_ind_15"]))/2.0)))

    v["i141"] = 0.020000*np.tanh(((data["ps_ind_01"] * (data["ps_ind_01"] * (data["loo_ps_ind_08_bin"] * data["ps_ind_01"]))) - data["loo_ps_ind_08_bin"]))

    v["i142"] = 0.019980*np.tanh(((((data["ps_reg_02"] + data["missing"])/2.0) + ((data["ps_reg_02"] + data["missing"])/2.0)) + data["ps_reg_02"]))

    v["i143"] = 0.020000*np.tanh(((data["ps_car_15"] * 2.0) * ((data["loo_ps_ind_02_cat"] - data["ps_car_15"]) - (data["loo_ps_ind_17_bin"] * 2.0))))

    v["i144"] = 0.019996*np.tanh((data["loo_ps_ind_02_cat"] * (((data["loo_ps_ind_02_cat"] * 2.0) * (data["loo_ps_ind_02_cat"] * 2.0)) + -3.0)))

    v["i145"] = 0.020000*np.tanh(((data["loo_ps_car_05_cat"] + (data["missing"] - data["loo_ps_car_07_cat"]))/2.0))

    v["i146"] = 0.019984*np.tanh((((-((data["ps_reg_02"] + data["ps_reg_02"]))) - data["ps_reg_02"]) * data["ps_car_15"]))

    v["i147"] = 0.020000*np.tanh((data["ps_ind_15"] * (((data["loo_ps_ind_06_bin"] * data["ps_car_14"]) + data["ps_reg_03"]) * 2.0)))

    v["i148"] = 0.019977*np.tanh((data["loo_ps_ind_05_cat"] * ((data["ps_car_13"] * data["loo_ps_car_07_cat"]) + data["ps_reg_03"])))

    v["i149"] = 0.019996*np.tanh((data["loo_ps_ind_04_cat"] * ((data["loo_ps_ind_04_cat"] + (data["loo_ps_car_04_cat"] * ((data["ps_ind_03"] + data["loo_ps_ind_04_cat"])/2.0)))/2.0)))

    v["i150"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] * data["loo_ps_car_07_cat"]) - (data["loo_ps_ind_04_cat"] + data["loo_ps_ind_07_bin"])) * data["loo_ps_ind_04_cat"]))

    return Outputs(-3.274750+v.sum(axis=1))
@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini



def ProjectOnMean(data1, data2, columnName):

    grpOutcomes = data1.groupby(list([columnName]))['target'].mean().reset_index()

    grpCount = data1.groupby(list([columnName]))['target'].count().reset_index()

    grpOutcomes['cnt'] = grpCount.target

    grpOutcomes.drop('cnt', inplace=True, axis=1)

    outcomes = data2['target'].values

    x = pd.merge(data2[[columnName, 'target']], grpOutcomes,

                 suffixes=('x_', ''),

                 how='left',

                 on=list([columnName]),

                 left_index=True)['target']



    

    return x.fillna(x.mean()).values



def GetData(strdirectory):

    # Project Categorical inputs to Target

    highcardinality = ['ps_car_02_cat',

                       'ps_car_09_cat',

                       'ps_ind_04_cat',

                       'ps_ind_05_cat',

                       'ps_car_03_cat',

                       'ps_ind_08_bin',

                       'ps_car_05_cat',

                       'ps_car_08_cat',

                       'ps_ind_06_bin',

                       'ps_ind_07_bin',

                       'ps_ind_12_bin',

                       'ps_ind_18_bin',

                       'ps_ind_17_bin',

                       'ps_car_07_cat',

                       'ps_car_11_cat',

                       'ps_ind_09_bin',

                       'ps_car_10_cat',

                       'ps_car_04_cat',

                       'ps_car_01_cat',

                       'ps_ind_02_cat',

                       'ps_ind_10_bin',

                       'ps_ind_11_bin',

                       'ps_car_06_cat',

                       'ps_ind_13_bin',

                       'ps_ind_16_bin']



    train = pd.read_csv(strdirectory+'train.csv')

    test = pd.read_csv(strdirectory+'test.csv')



    train['missing'] = (train==-1).sum(axis=1).astype(float)

    test['missing'] = (test==-1).sum(axis=1).astype(float)



    unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

    train.drop(unwanted,inplace=True,axis=1)

    test.drop(unwanted,inplace=True,axis=1)



    test['target'] = np.nan

    feats = list(set(train.columns).difference(set(['id','target'])))

    feats = list(['id'])+feats +list(['target'])

    train = train[feats]

    test = test[feats]

    

    blindloodata = None

    folds = 5

    kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=2017)

    for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]),train.target)):

        print('Fold:',i)

        blindtrain = train.iloc[test_index].copy() 

        vistrain = train.iloc[train_index].copy()



        for c in highcardinality:

            blindtrain.insert(1,'loo_'+c, ProjectOnMean(vistrain,

                                                       blindtrain,c))

        if(blindloodata is None):

            blindloodata = blindtrain.copy()

        else:

            blindloodata = pd.concat([blindloodata,blindtrain])



    for c in highcardinality:

        test.insert(1,'loo_'+c, ProjectOnMean(train,

                                              test,c))

    test.drop(highcardinality,inplace=True,axis=1)



    train = blindloodata

    train.drop(highcardinality,inplace=True,axis=1)



    print('Scale values')

    ss = StandardScaler()

    features = train.columns[1:-1]

    ss.fit(pd.concat([train[features],test[features]]))

    train[features] = ss.transform(train[features] )

    test[features] = ss.transform(test[features] )

    train[features] = np.round(train[features], 6)

    test[features] = np.round(test[features], 6)

    return train, test
strdirectory = '../input/'

gptrain, gptest = GetData(strdirectory)
print(log_loss(gptrain.target,GP1(gptrain)))

print(log_loss(gptrain.target,GP2(gptrain)))

print(log_loss(gptrain.target,GP2(gptrain)))

print(log_loss(gptrain.target,

               (GP1(gptrain)+

                GP2(gptrain)+

                GP3(gptrain))/3.))
print(eval_gini(gptrain.target.ravel(), 

                GP1(gptrain).ravel()))

print(eval_gini(gptrain.target.ravel(), 

                GP2(gptrain).ravel()))

print(eval_gini(gptrain.target.ravel(), 

                GP3(gptrain).ravel()))

print(eval_gini(gptrain.target,

                (GP1(gptrain)+

                 GP2(gptrain)+

                 GP3(gptrain))/3.))
sub = pd.read_csv(strdirectory+'sample_submission.csv')

sub.target = ((GP1(gptest)+

               GP2(gptest)+

               GP3(gptest))/3.).values

sub.to_csv('mediumgp.csv',index=False)