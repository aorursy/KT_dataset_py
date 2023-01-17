import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.preprocessing import StandardScaler

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

traintargets = train.target.values

trainids = train.ID_code.values

testids = test.ID_code.values

del train['target']

del train['ID_code']

del test['ID_code']
ss = StandardScaler()

alldata = pd.concat([train,test],sort=False)

alldata[alldata.columns] = ss.fit_transform(alldata[alldata.columns])

traindata = alldata[:len(traintargets)].copy().reset_index(drop=True)

testdata = alldata[len(traintargets):].copy().reset_index(drop=True)
class GPI:

    def __init__(self):

        self.classes = 2

        self.class_names = [ 'class_p',

                             'class_n']





    def GrabPredictions(self, data, cols):

        oof_preds = np.zeros((len(data), len(self.class_names)))

        oof_preds[:,0] = self.GP_class_p(data,cols)

        oof_preds[:,1] = self.GP_class_n(data,cols)

        oof_df = pd.DataFrame(np.exp(oof_preds), columns=self.class_names)

        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)

        return oof_df





    def GP_class_p(self,data,cols):

        v = pd.DataFrame()

        v["i0"] = -2.146841 * np.ones(data.shape[0])

        v["i1"] = 0.020000*np.tanh(((((((((((((data["var_53"]) + (((((data["var_53"]) + (-3.0))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 

        v["i2"] = 0.020000*np.tanh((((((((((14.10473251342773438)) - ((14.10473632812500000)))) - ((9.0)))) - (((-3.0) - ((14.10473632812500000)))))) - ((14.10473632812500000)))) 

        v["i3"] = 0.020000*np.tanh(((data["var_53"]) - (((((12.09928321838378906)) + ((((((data["var_86"]) + ((10.03234100341796875)))/2.0)) + (((data["var_81"]) * ((8.65062332153320312)))))))/2.0)))) 

        v["i4"] = 0.020000*np.tanh(((((-3.0) - (((((data["var_12"]) * ((13.80687427520751953)))) + ((13.80687046051025391)))))) - ((9.0)))) 

        v["i5"] = 0.020000*np.tanh(((((((((data["var_53"]) - (((data["var_139"]) + (((data["var_81"]) - (-3.0))))))) + (data["var_6"]))) * 2.0)) * 2.0)) 

        v["i6"] = 0.020000*np.tanh(((((((((((((-3.0) + (data["var_6"]))) + (data["var_22"]))) - (data["var_12"]))) + (data["var_110"]))) * 2.0)) * 2.0)) 

        v["i7"] = 0.020000*np.tanh(((((data["var_26"]) + (((data["var_6"]) - ((((((((data["var_81"]) - (data["var_110"]))) + (3.0))/2.0)) * 2.0)))))) * 2.0)) 

        v["i8"] = 0.020000*np.tanh(((((((((data["var_53"]) + (data["var_6"]))) + (((data["var_110"]) - ((((2.38368678092956543)) + (data["var_81"]))))))) * 2.0)) * 2.0)) 

        v["i9"] = 0.020000*np.tanh(((((((((-2.0) + (((((data["var_110"]) - (data["var_81"]))) - (data["var_139"]))))) * 2.0)) + (data["var_53"]))) * 2.0)) 

        v["i10"] = 0.020000*np.tanh(((data["var_26"]) + (((((data["var_6"]) + (((data["var_133"]) - (((3.0) + (data["var_81"]))))))) + (data["var_53"]))))) 

        v["i11"] = 0.020000*np.tanh(((((((((((-2.0) + (((data["var_99"]) + (data["var_6"]))))) + (data["var_190"]))) - (data["var_81"]))) * 2.0)) * 2.0)) 

        v["i12"] = 0.020000*np.tanh(((((((((data["var_190"]) + (((((data["var_53"]) + (-2.0))) + (data["var_6"]))))) - (data["var_21"]))) * 2.0)) * 2.0)) 

        v["i13"] = 0.020000*np.tanh(((((data["var_6"]) + (data["var_133"]))) + (((data["var_110"]) + (((data["var_22"]) - (((3.0) - (data["var_26"]))))))))) 

        v["i14"] = 0.020000*np.tanh(((((data["var_6"]) + (data["var_53"]))) + (((((-2.0) + (((data["var_110"]) + (data["var_26"]))))) - (data["var_76"]))))) 

        v["i15"] = 0.020000*np.tanh(((data["var_22"]) + (((((((-2.0) - (data["var_80"]))) - (data["var_12"]))) + (((data["var_99"]) - (data["var_21"]))))))) 

        v["i16"] = 0.020000*np.tanh(((((data["var_99"]) + (((((data["var_53"]) - (data["var_139"]))) - ((1.75676155090332031)))))) - (((data["var_146"]) - (data["var_6"]))))) 

        v["i17"] = 0.020000*np.tanh(((((-1.0) - (data["var_21"]))) + (((data["var_2"]) + (((((data["var_26"]) - (data["var_139"]))) + (data["var_22"]))))))) 

        v["i18"] = 0.020000*np.tanh(((data["var_94"]) - (((data["var_13"]) - (((((data["var_26"]) + (-1.0))) + (((data["var_6"]) - (data["var_146"]))))))))) 

        v["i19"] = 0.020000*np.tanh(((((((data["var_53"]) + (data["var_191"]))) - (data["var_81"]))) + (((((data["var_133"]) + (data["var_170"]))) + (data["var_40"]))))) 

        v["i20"] = 0.020000*np.tanh(((((data["var_99"]) + (((data["var_0"]) + (data["var_6"]))))) + (((data["var_110"]) + (((data["var_78"]) + (data["var_26"]))))))) 

        v["i21"] = 0.020000*np.tanh(((((data["var_53"]) + (data["var_78"]))) + (((((data["var_2"]) - (data["var_34"]))) + (((data["var_184"]) + (data["var_40"]))))))) 

        v["i22"] = 0.020000*np.tanh(((data["var_1"]) + (((data["var_110"]) + (((((data["var_94"]) + (data["var_99"]))) + (((data["var_22"]) + (data["var_78"]))))))))) 

        v["i23"] = 0.020000*np.tanh(((data["var_2"]) + (((data["var_191"]) + (((((data["var_190"]) + (data["var_22"]))) - (((data["var_115"]) - (data["var_173"]))))))))) 

        v["i24"] = 0.020000*np.tanh(((data["var_170"]) + (((((data["var_190"]) + (((((data["var_1"]) - (data["var_165"]))) + (data["var_133"]))))) - (data["var_92"]))))) 

        v["i25"] = 0.020000*np.tanh(((((data["var_99"]) + (((((data["var_118"]) - (data["var_80"]))) + (data["var_191"]))))) - (((data["var_33"]) - (data["var_0"]))))) 

        v["i26"] = 0.020000*np.tanh(((((((data["var_170"]) + (data["var_53"]))) + (((data["var_179"]) + (data["var_6"]))))) + (((data["var_26"]) + (data["var_184"]))))) 

        v["i27"] = 0.020000*np.tanh(((((((data["var_94"]) + (((data["var_67"]) + (data["var_40"]))))) + (data["var_22"]))) + (((data["var_190"]) + (data["var_110"]))))) 

        v["i28"] = 0.020000*np.tanh(((data["var_147"]) + (((((((data["var_2"]) + (data["var_99"]))) - (((data["var_12"]) - (data["var_78"]))))) + (data["var_67"]))))) 

        v["i29"] = 0.020000*np.tanh(((((((data["var_190"]) - (data["var_154"]))) + (((data["var_26"]) + (((data["var_173"]) - (data["var_121"]))))))) + (data["var_1"]))) 

        v["i30"] = 0.020000*np.tanh(((data["var_22"]) + (((data["var_0"]) + (((data["var_53"]) + (((data["var_94"]) + (((data["var_184"]) + (data["var_6"]))))))))))) 

        v["i31"] = 0.020000*np.tanh(((((data["var_35"]) + (data["var_99"]))) + (((data["var_191"]) + (((((data["var_179"]) - (data["var_123"]))) + (data["var_110"]))))))) 

        v["i32"] = 0.020000*np.tanh(((((data["var_190"]) + (((((data["var_118"]) + (data["var_26"]))) + (((data["var_0"]) + (data["var_40"]))))))) + (data["var_170"]))) 

        v["i33"] = 0.020000*np.tanh(((((data["var_173"]) + (((data["var_67"]) - (data["var_21"]))))) + (((data["var_133"]) + (((data["var_89"]) + (data["var_78"]))))))) 

        v["i34"] = 0.020000*np.tanh(((((data["var_53"]) + (((((data["var_110"]) + (data["var_133"]))) - (data["var_80"]))))) + (((data["var_106"]) + (data["var_91"]))))) 

        v["i35"] = 0.020000*np.tanh(((((((data["var_18"]) - (data["var_192"]))) + (data["var_40"]))) + (((data["var_184"]) + (((data["var_164"]) + (data["var_1"]))))))) 

        v["i36"] = 0.020000*np.tanh(((data["var_26"]) + (((((((data["var_89"]) + (((data["var_94"]) - (data["var_109"]))))) + (data["var_155"]))) + (data["var_147"]))))) 

        v["i37"] = 0.020000*np.tanh(((((data["var_99"]) + (data["var_22"]))) + (((data["var_35"]) + (((data["var_0"]) - (((data["var_33"]) - (data["var_95"]))))))))) 

        v["i38"] = 0.020000*np.tanh(((data["var_18"]) + (((data["var_78"]) + (((((data["var_155"]) + (data["var_118"]))) + (((data["var_48"]) + (data["var_2"]))))))))) 

        v["i39"] = 0.020000*np.tanh(((data["var_133"]) + (((data["var_147"]) - (((data["var_34"]) - (((data["var_179"]) - (((data["var_122"]) - (data["var_22"]))))))))))) 

        v["i40"] = 0.020000*np.tanh(((((((((((data["var_170"]) + (data["var_191"]))) - (data["var_21"]))) - (data["var_76"]))) + (data["var_22"]))) + (data["var_1"]))) 

        v["i41"] = 0.020000*np.tanh(((((data["var_190"]) + (data["var_53"]))) + (((((data["var_91"]) + (((data["var_95"]) + (data["var_106"]))))) + (data["var_173"]))))) 

        v["i42"] = 0.020000*np.tanh(((data["var_184"]) + (((data["var_163"]) + (((((data["var_164"]) - (data["var_123"]))) + (((data["var_67"]) - (data["var_198"]))))))))) 

        v["i43"] = 0.020000*np.tanh(((((data["var_71"]) + (((data["var_35"]) + (data["var_94"]))))) + (((data["var_99"]) + (((data["var_2"]) + (data["var_179"]))))))) 

        v["i44"] = 0.020000*np.tanh(((data["var_1"]) + (((((data["var_26"]) + (data["var_191"]))) - (((data["var_197"]) - (((data["var_6"]) + (data["var_118"]))))))))) 

        v["i45"] = 0.020000*np.tanh(((((((data["var_0"]) + (data["var_184"]))) + (data["var_48"]))) + (((data["var_162"]) + (((data["var_170"]) + (data["var_157"]))))))) 

        v["i46"] = 0.020000*np.tanh(((((data["var_91"]) - (data["var_149"]))) + (((data["var_89"]) + (((data["var_99"]) + (((data["var_184"]) + (data["var_164"]))))))))) 

        v["i47"] = 0.020000*np.tanh(((data["var_71"]) + (((((((data["var_18"]) - (data["var_115"]))) + (data["var_110"]))) - (((data["var_172"]) - (data["var_106"]))))))) 

        v["i48"] = 0.020000*np.tanh(((((data["var_53"]) - (data["var_81"]))) + (((data["var_155"]) + (((((data["var_191"]) + (data["var_2"]))) + (data["var_179"]))))))) 

        v["i49"] = 0.020000*np.tanh(((((data["var_167"]) + (data["var_78"]))) + (((((((data["var_118"]) - (data["var_115"]))) + (data["var_173"]))) + (data["var_40"]))))) 

        v["i50"] = 0.020000*np.tanh(((data["var_180"]) + (((data["var_6"]) + (((((data["var_53"]) + (((data["var_163"]) + (data["var_94"]))))) + (data["var_190"]))))))) 

        v["i51"] = 0.020000*np.tanh(((((data["var_52"]) + (((data["var_32"]) - (data["var_165"]))))) + (((data["var_133"]) + (((data["var_130"]) + (data["var_119"]))))))) 

        v["i52"] = 0.020000*np.tanh(((((((data["var_67"]) + (data["var_179"]))) + (data["var_137"]))) + (((((data["var_5"]) + (data["var_145"]))) + (data["var_95"]))))) 

        v["i53"] = 0.020000*np.tanh(((((data["var_35"]) - (data["var_166"]))) + (((((data["var_1"]) + (data["var_22"]))) + (((data["var_110"]) - (data["var_33"]))))))) 

        v["i54"] = 0.020000*np.tanh(((((data["var_190"]) + (data["var_170"]))) + (((data["var_89"]) + (((data["var_91"]) - (((data["var_198"]) - (data["var_157"]))))))))) 

        v["i55"] = 0.020000*np.tanh(((((data["var_199"]) + (((data["var_106"]) + (((data["var_0"]) + (data["var_173"]))))))) + (((data["var_5"]) + (data["var_163"]))))) 

        v["i56"] = 0.020000*np.tanh(((data["var_167"]) + (((((data["var_26"]) + (data["var_49"]))) + (((((data["var_162"]) + (data["var_18"]))) + (data["var_2"]))))))) 

        v["i57"] = 0.020000*np.tanh(((data["var_155"]) + (((((data["var_180"]) + (data["var_78"]))) - (((data["var_174"]) - (((data["var_191"]) + (data["var_99"]))))))))) 

        v["i58"] = 0.020000*np.tanh(((((((data["var_0"]) + (data["var_67"]))) + (data["var_145"]))) + (((((data["var_133"]) - (data["var_150"]))) + (data["var_112"]))))) 

        v["i59"] = 0.020000*np.tanh(((((((data["var_35"]) + (data["var_119"]))) + (data["var_22"]))) + (((((data["var_48"]) - (data["var_56"]))) + (data["var_195"]))))) 

        v["i60"] = 0.020000*np.tanh(((((data["var_118"]) + (data["var_52"]))) + (((((((data["var_137"]) + (data["var_90"]))) - (data["var_177"]))) - (data["var_21"]))))) 

        v["i61"] = 0.020000*np.tanh(((data["var_110"]) + (((((data["var_151"]) + (((data["var_32"]) + (data["var_164"]))))) + (((data["var_53"]) + (data["var_6"]))))))) 

        v["i62"] = 0.020000*np.tanh(((((((data["var_18"]) + (((data["var_190"]) + (data["var_89"]))))) + (data["var_199"]))) + (((data["var_1"]) + (data["var_91"]))))) 

        v["i63"] = 0.020000*np.tanh(((data["var_179"]) + (((data["var_94"]) + (((((data["var_135"]) + (data["var_155"]))) + (((data["var_78"]) + (data["var_170"]))))))))) 

        v["i64"] = 0.020000*np.tanh(((data["var_130"]) + (((((data["var_5"]) + (((data["var_99"]) + (data["var_49"]))))) + (((data["var_125"]) + (data["var_40"]))))))) 

        v["i65"] = 0.020000*np.tanh(((((data["var_90"]) + (((data["var_145"]) + (data["var_175"]))))) + (((data["var_70"]) + (((data["var_179"]) * (data["var_179"]))))))) 

        v["i66"] = 0.020000*np.tanh(((data["var_119"]) + (((data["var_180"]) + (((((data["var_173"]) + (data["var_147"]))) + (((data["var_164"]) + (data["var_95"]))))))))) 

        v["i67"] = 0.020000*np.tanh(((((data["var_118"]) + (data["var_26"]))) + (((((data["var_67"]) + (data["var_163"]))) + (((data["var_133"]) + (data["var_32"]))))))) 

        v["i68"] = 0.020000*np.tanh(((data["var_1"]) + (((((data["var_2"]) + (((((data["var_167"]) + (data["var_0"]))) + (data["var_111"]))))) + (data["var_48"]))))) 

        v["i69"] = 0.020000*np.tanh(((((((data["var_162"]) + (data["var_125"]))) + (data["var_6"]))) + (((data["var_137"]) + (((data["var_67"]) + (data["var_18"]))))))) 

        v["i70"] = 0.020000*np.tanh(((((((data["var_173"]) - (data["var_139"]))) + (((data["var_195"]) + (((data["var_128"]) + (data["var_91"]))))))) + (data["var_71"]))) 

        v["i71"] = 0.020000*np.tanh(((((((data["var_66"]) + (((data["var_22"]) + (data["var_106"]))))) + (data["var_179"]))) + (((data["var_175"]) + (data["var_145"]))))) 

        v["i72"] = 0.020000*np.tanh(((((data["var_105"]) + (data["var_151"]))) + (((((((data["var_40"]) - (data["var_21"]))) + (data["var_191"]))) + (data["var_90"]))))) 

        v["i73"] = 0.020000*np.tanh(((data["var_199"]) + (((data["var_11"]) + (((((data["var_78"]) + (((data["var_44"]) * (data["var_44"]))))) + (data["var_70"]))))))) 

        v["i74"] = 0.020000*np.tanh(((((((data["var_99"]) + (data["var_94"]))) + (((data["var_51"]) + (data["var_184"]))))) - (((data["var_194"]) - (data["var_190"]))))) 

        v["i75"] = 0.020000*np.tanh(((((data["var_196"]) + (data["var_71"]))) + (((data["var_157"]) + (((((data["var_89"]) + (data["var_119"]))) + (data["var_144"]))))))) 

        v["i76"] = 0.020000*np.tanh(((((data["var_24"]) - (data["var_81"]))) + (((data["var_135"]) + (((data["var_49"]) + (((data["var_94"]) + (data["var_110"]))))))))) 

        v["i77"] = 0.020000*np.tanh(((((data["var_191"]) + (((data["var_167"]) + (((data["var_111"]) + (data["var_35"]))))))) + (((data["var_105"]) + (data["var_164"]))))) 

        v["i78"] = 0.020000*np.tanh(((((data["var_151"]) + (data["var_133"]))) + (((data["var_48"]) + (((((data["var_184"]) + (data["var_137"]))) + (data["var_32"]))))))) 

        v["i79"] = 0.020000*np.tanh(((data["var_5"]) + (((data["var_180"]) + (((((data["var_170"]) + (data["var_6"]))) + (((data["var_90"]) + (data["var_128"]))))))))) 

        v["i80"] = 0.020000*np.tanh(((data["var_196"]) + (((data["var_66"]) - (((data["var_13"]) - (((data["var_99"]) + (((data["var_145"]) + (data["var_130"]))))))))))) 

        v["i81"] = 0.020000*np.tanh(((data["var_162"]) + (((((data["var_106"]) + (data["var_89"]))) + (((data["var_112"]) + (((data["var_70"]) - (data["var_9"]))))))))) 

        v["i82"] = 0.020000*np.tanh(((((data["var_118"]) + (data["var_24"]))) + (((data["var_8"]) + (((data["var_1"]) + (((data["var_190"]) * (data["var_190"]))))))))) 

        v["i83"] = 0.020000*np.tanh(((((data["var_110"]) * (data["var_110"]))) + (((((data["var_51"]) + (((data["var_155"]) + (data["var_67"]))))) + (data["var_74"]))))) 

        v["i84"] = 0.020000*np.tanh(((((data["var_157"]) + (data["var_22"]))) + (((data["var_82"]) + (((data["var_190"]) + (((data["var_55"]) + (data["var_71"]))))))))) 

        v["i85"] = 0.020000*np.tanh(((((data["var_52"]) + (((data["var_180"]) + (((data["var_135"]) - (data["var_169"]))))))) + (((data["var_40"]) + (data["var_18"]))))) 

        v["i86"] = 0.020000*np.tanh(((((((data["var_195"]) + (data["var_163"]))) + (((data["var_137"]) + (data["var_11"]))))) + (((data["var_179"]) + (data["var_78"]))))) 

        v["i87"] = 0.020000*np.tanh(((((((data["var_40"]) + (data["var_26"]))) + (((data["var_170"]) + (((data["var_119"]) + (data["var_164"]))))))) + (data["var_91"]))) 

        v["i88"] = 0.020000*np.tanh(((((data["var_140"]) - (data["var_36"]))) + (((data["var_147"]) - (((data["var_58"]) - (((data["var_48"]) + (data["var_95"]))))))))) 

        v["i89"] = 0.020000*np.tanh(((((((data["var_32"]) + (((data["var_175"]) + (data["var_5"]))))) + (data["var_49"]))) + (((data["var_128"]) - (data["var_139"]))))) 

        v["i90"] = 0.020000*np.tanh(((((((data["var_2"]) + (data["var_145"]))) + (((((data["var_167"]) + (data["var_157"]))) - (data["var_108"]))))) + (data["var_97"]))) 

        v["i91"] = 0.020000*np.tanh(((((data["var_105"]) + (((data["var_199"]) + (data["var_125"]))))) + (((data["var_144"]) + (((data["var_6"]) + (data["var_52"]))))))) 

        v["i92"] = 0.020000*np.tanh(((((data["var_18"]) + (((data["var_138"]) + (((((data["var_151"]) + (data["var_66"]))) - (data["var_123"]))))))) + (data["var_22"]))) 

        v["i93"] = 0.020000*np.tanh(((((data["var_162"]) + (((data["var_94"]) + (data["var_163"]))))) + (((data["var_70"]) + (((data["var_112"]) + (data["var_91"]))))))) 

        v["i94"] = 0.020000*np.tanh(((((data["var_2"]) - (data["var_154"]))) + (((((data["var_74"]) + (data["var_24"]))) + (((data["var_71"]) + (data["var_0"]))))))) 

        v["i95"] = 0.020000*np.tanh(((((data["var_51"]) + (data["var_106"]))) - (((data["var_80"]) - (((data["var_35"]) + (((data["var_173"]) + (data["var_78"]))))))))) 

        v["i96"] = 0.020000*np.tanh(((((data["var_0"]) * (data["var_0"]))) + (((((data["var_191"]) + (data["var_15"]))) + (((data["var_118"]) + (data["var_82"]))))))) 

        v["i97"] = 0.020000*np.tanh(((((((((data["var_190"]) + (data["var_184"]))) + (data["var_110"]))) + (((data["var_164"]) + (data["var_179"]))))) - (data["var_36"]))) 

        v["i98"] = 0.020000*np.tanh(((((data["var_26"]) + (((data["var_135"]) + (((((data["var_196"]) + (data["var_138"]))) + (data["var_8"]))))))) + (data["var_89"]))) 

        v["i99"] = 0.020000*np.tanh(((((((data["var_159"]) + (data["var_48"]))) + (((data["var_148"]) * (data["var_148"]))))) + (((data["var_55"]) + (data["var_11"]))))) 

        v["i100"] = 0.020000*np.tanh(((data["var_67"]) + (((((data["var_171"]) + (((data["var_95"]) + (data["var_137"]))))) + (((data["var_70"]) + (data["var_99"]))))))) 

        v["i101"] = 0.020000*np.tanh(((data["var_51"]) + (((((data["var_74"]) + (((data["var_5"]) + (data["var_180"]))))) + (((data["var_167"]) + (data["var_40"]))))))) 

        v["i102"] = 0.020000*np.tanh(((((((data["var_195"]) + (data["var_94"]))) + (((data["var_140"]) + (data["var_168"]))))) + (((data["var_133"]) + (data["var_125"]))))) 

        v["i103"] = 0.020000*np.tanh(((data["var_119"]) + (((((data["var_191"]) + (data["var_145"]))) + (((data["var_0"]) - (((data["var_75"]) - (data["var_105"]))))))))) 

        v["i104"] = 0.020000*np.tanh(((((data["var_170"]) + (data["var_157"]))) + (((((data["var_175"]) - (data["var_169"]))) + (((data["var_112"]) + (data["var_22"]))))))) 

        v["i105"] = 0.020000*np.tanh(((((data["var_134"]) + (((data["var_196"]) + (((data["var_187"]) + (data["var_62"]))))))) + (((data["var_6"]) - (data["var_109"]))))) 

        v["i106"] = 0.020000*np.tanh(((((data["var_111"]) + (((((data["var_90"]) + (data["var_164"]))) + (data["var_144"]))))) + (((data["var_71"]) + (data["var_173"]))))) 

        v["i107"] = 0.020000*np.tanh(((data["var_91"]) + (((data["var_151"]) + (((((data["var_21"]) * (data["var_21"]))) + (((data["var_18"]) + (data["var_199"]))))))))) 

        v["i108"] = 0.020000*np.tanh(((((((data["var_97"]) + (data["var_52"]))) + (data["var_8"]))) + (((data["var_135"]) + (((data["var_155"]) + (data["var_163"]))))))) 

        v["i109"] = 0.020000*np.tanh(((((data["var_181"]) + (((data["var_82"]) + (data["var_128"]))))) + (((data["var_99"]) + (((data["var_35"]) + (data["var_118"]))))))) 

        v["i110"] = 0.020000*np.tanh(((data["var_26"]) + (((((data["var_130"]) + (data["var_24"]))) + (((((data["var_1"]) + (data["var_159"]))) + (data["var_110"]))))))) 

        v["i111"] = 0.020000*np.tanh(((data["var_119"]) + (((data["var_89"]) + (((((data["var_22"]) - (data["var_34"]))) + (((data["var_32"]) + (data["var_2"]))))))))) 

        v["i112"] = 0.020000*np.tanh(((((data["var_84"]) + (((((data["var_140"]) + (data["var_55"]))) - (data["var_54"]))))) + (((data["var_139"]) * (data["var_139"]))))) 

        v["i113"] = 0.020000*np.tanh(((((((data["var_174"]) * (data["var_174"]))) + (((data["var_66"]) + (data["var_134"]))))) + (((data["var_4"]) + (data["var_62"]))))) 

        v["i114"] = 0.020000*np.tanh(((data["var_179"]) + (((((((data["var_199"]) + (data["var_184"]))) + (((data["var_137"]) + (data["var_95"]))))) + (data["var_106"]))))) 

        v["i115"] = 0.020000*np.tanh(((((((((data["var_170"]) + (data["var_78"]))) + (data["var_99"]))) - (1.0))) + (((data["var_40"]) + (data["var_1"]))))) 

        v["i116"] = 0.020000*np.tanh(((-1.0) + (((((data["var_139"]) * (data["var_139"]))) + (((data["var_139"]) * (data["var_139"]))))))) 

        v["i117"] = 0.020000*np.tanh(((((((data["var_167"]) + (data["var_6"]))) + (((((data["var_179"]) + (data["var_147"]))) + (data["var_53"]))))) + (data["var_190"]))) 

        v["i118"] = 0.020000*np.tanh(((data["var_130"]) + (((((data["var_52"]) + (data["var_157"]))) + (((data["var_5"]) + (((data["var_125"]) - (data["var_21"]))))))))) 

        v["i119"] = 0.020000*np.tanh(((data["var_195"]) + (((data["var_51"]) + (((data["var_15"]) + (((data["var_168"]) + (((data["var_175"]) + (data["var_112"]))))))))))) 

        v["i120"] = 0.020000*np.tanh(((data["var_196"]) + (((data["var_106"]) + (((data["var_155"]) + (((((data["var_90"]) + (data["var_111"]))) + (data["var_164"]))))))))) 

        v["i121"] = 0.020000*np.tanh(((((data["var_128"]) + (((data["var_145"]) + (data["var_151"]))))) + (((((data["var_148"]) * (data["var_148"]))) + (data["var_26"]))))) 

        v["i122"] = 0.020000*np.tanh(((((data["var_0"]) * (data["var_0"]))) + (((((data["var_65"]) + (data["var_144"]))) + (((data["var_187"]) + (data["var_138"]))))))) 

        v["i123"] = 0.020000*np.tanh(((((data["var_12"]) * (data["var_12"]))) + ((((data["var_137"]) + (((data["var_24"]) + (((data["var_49"]) + (-3.0))))))/2.0)))) 

        v["i124"] = 0.020000*np.tanh(((((data["var_180"]) + (data["var_94"]))) + (((data["var_173"]) + (((data["var_105"]) + (((data["var_110"]) + (data["var_70"]))))))))) 

        v["i125"] = 0.020000*np.tanh(((data["var_74"]) + (((((data["var_71"]) + (data["var_184"]))) + (((((data["var_163"]) + (data["var_0"]))) + (data["var_118"]))))))) 

        v["i126"] = 0.020000*np.tanh(((((data["var_91"]) + (data["var_171"]))) - (((data["var_77"]) - (((data["var_3"]) + (((data["var_97"]) + (data["var_89"]))))))))) 

        v["i127"] = 0.020000*np.tanh(((((data["var_25"]) + (data["var_35"]))) + (((((data["var_159"]) + (data["var_8"]))) + (((data["var_157"]) + (data["var_82"]))))))) 

        v["i128"] = 0.020000*np.tanh(((((((((data["var_81"]) * (data["var_81"]))) + ((((((data["var_0"]) * (data["var_0"]))) + (-2.0))/2.0)))) * 2.0)) * 2.0)) 

        v["i129"] = 0.020000*np.tanh(((((((data["var_135"]) + (((((data["var_147"]) + (data["var_40"]))) - (data["var_113"]))))) + (data["var_49"]))) - (data["var_177"]))) 

        v["i130"] = 0.020000*np.tanh(((((data["var_16"]) + ((((((data["var_181"]) + (data["var_130"]))/2.0)) - (data["var_87"]))))) + (((data["var_67"]) * (data["var_67"]))))) 

        v["i131"] = 0.020000*np.tanh(((((data["var_179"]) * (data["var_179"]))) + ((((-1.0*((2.0)))) + (((data["var_22"]) * (data["var_22"]))))))) 

        v["i132"] = 0.020000*np.tanh(((((data["var_74"]) + (((((data["var_167"]) * (data["var_167"]))) + (((data["var_95"]) * (data["var_95"]))))))) - (1.0))) 

        v["i133"] = 0.020000*np.tanh(((((data["var_18"]) + (((data["var_80"]) * (data["var_80"]))))) + (((-2.0) + (((data["var_80"]) * (data["var_80"]))))))) 

        v["i134"] = 0.020000*np.tanh(((((-1.0*((((data["var_179"]) * ((-1.0*((data["var_179"]))))))))) + (((data["var_19"]) + (((data["var_176"]) + (data["var_190"]))))))/2.0)) 

        v["i135"] = 0.020000*np.tanh(((((((data["var_170"]) * (data["var_170"]))) + (-3.0))) + (((data["var_55"]) + (((data["var_170"]) * (data["var_170"]))))))) 

        v["i136"] = 0.020000*np.tanh((((data["var_94"]) + (((((data["var_187"]) + (-2.0))) + (((((data["var_2"]) * (data["var_2"]))) * 2.0)))))/2.0)) 

        v["i137"] = 0.020000*np.tanh((-1.0*((((((((((data["var_198"]) * ((-1.0*((data["var_198"])))))) + (1.0))) + (1.0))) - (data["var_26"])))))) 

        v["i138"] = 0.020000*np.tanh(((((data["var_6"]) * (data["var_6"]))) + (((data["var_48"]) + (((((data["var_29"]) + (data["var_175"]))) + (data["var_159"]))))))) 

        v["i139"] = 0.020000*np.tanh(((data["var_67"]) + (((((data["var_198"]) - (2.0))) + (((((data["var_198"]) * (data["var_198"]))) * 2.0)))))) 

        v["i140"] = 0.020000*np.tanh(((((((((((data["var_99"]) + (data["var_155"]))) + (data["var_119"]))) + (data["var_32"]))) + (data["var_24"]))) + (data["var_133"]))) 

        v["i141"] = 0.020000*np.tanh(((((((((data["var_139"]) * (data["var_139"]))) + (-3.0))) + (((data["var_53"]) * (data["var_53"]))))) + (data["var_61"]))) 

        v["i142"] = 0.020000*np.tanh(((data["var_184"]) + (((((data["var_62"]) + (((data["var_51"]) + (((data["var_95"]) + (data["var_112"]))))))) + (data["var_2"]))))) 

        v["i143"] = 0.020000*np.tanh(((((((data["var_162"]) + (data["var_70"]))) + (-1.0))) + (((data["var_55"]) + (((data["var_40"]) * (data["var_40"]))))))) 

        v["i144"] = 0.020000*np.tanh(((((-1.0*((((2.0) - (((((data["var_12"]) * (data["var_12"]))) * 2.0))))))) + (((data["var_199"]) + (data["var_12"]))))/2.0)) 

        v["i145"] = 0.020000*np.tanh((((((data["var_146"]) * (((data["var_146"]) * (2.0))))) + (((((data["var_84"]) + (data["var_171"]))) + (data["var_4"]))))/2.0)) 

        v["i146"] = 0.020000*np.tanh(((((((((data["var_1"]) + (-2.0))) + (((data["var_81"]) * (((data["var_81"]) + (data["var_81"]))))))) * 2.0)) * 2.0)) 

        v["i147"] = 0.020000*np.tanh(((((((((data["var_21"]) * (data["var_21"]))) + ((((-2.0) + (((data["var_21"]) + (data["var_191"]))))/2.0)))) * 2.0)) * 2.0)) 

        v["i148"] = 0.020000*np.tanh(((-2.0) + (((data["var_5"]) + (((((data["var_86"]) * (data["var_86"]))) + (((data["var_86"]) * (data["var_86"]))))))))) 

        v["i149"] = 0.020000*np.tanh(((((-1.0) - (((np.tanh(((-1.0*(((-1.0*((data["var_53"]))))))))) - (((data["var_53"]) * (data["var_53"]))))))) * 2.0)) 

        v["i150"] = 0.020000*np.tanh(((((((6.08141326904296875)) - (data["var_20"]))) + (data["var_90"]))/2.0)) 

        v["i151"] = 0.020000*np.tanh(((((((data["var_0"]) * (data["var_0"]))) + (((((-2.0) * 2.0)) / 2.0)))) + (((data["var_91"]) * (data["var_91"]))))) 

        v["i152"] = 0.020000*np.tanh(((data["var_105"]) + ((((((data["var_134"]) + (((-2.0) + (data["var_3"]))))/2.0)) + (((data["var_6"]) * (data["var_6"]))))))) 

        v["i153"] = 0.020000*np.tanh((((((data["var_99"]) * (((((data["var_99"]) * (((data["var_99"]) * (data["var_99"]))))) + (data["var_19"]))))) + (data["var_19"]))/2.0)) 

        v["i154"] = 0.020000*np.tanh(((-1.0) + (((data["var_40"]) + (((((-1.0) + (data["var_0"]))) + (((data["var_53"]) * (data["var_53"]))))))))) 

        v["i155"] = 0.020000*np.tanh(((data["var_110"]) + (((data["var_140"]) + (((((data["var_118"]) + (((data["var_111"]) + (data["var_125"]))))) + (data["var_37"]))))))) 

        v["i156"] = 0.020000*np.tanh(((data["var_138"]) + (((data["var_12"]) + (((((data["var_12"]) * (data["var_12"]))) + (data["var_97"]))))))) 

        v["i157"] = 0.020000*np.tanh((((((((data["var_179"]) * (data["var_179"]))) + (data["var_82"]))/2.0)) + (((((data["var_0"]) * (data["var_0"]))) - (data["var_0"]))))) 

        v["i158"] = 0.020000*np.tanh(((((((((((((((data["var_26"]) * (data["var_26"]))) - (data["var_26"]))) * 2.0)) * 2.0)) - (data["var_26"]))) * 2.0)) * 2.0)) 

        v["i159"] = 0.020000*np.tanh(((((data["var_133"]) * (((((data["var_133"]) * 2.0)) / 2.0)))) - (((2.0) - (((data["var_26"]) * (data["var_26"]))))))) 

        v["i160"] = 0.020000*np.tanh(((-2.0) + (((((data["var_148"]) * (data["var_148"]))) + (((((data["var_148"]) * (data["var_148"]))) + (data["var_26"]))))))) 

        v["i161"] = 0.020000*np.tanh(((((data["var_198"]) * (data["var_198"]))) + ((((data["var_198"]) + ((((((data["var_195"]) + (-3.0))/2.0)) + (data["var_190"]))))/2.0)))) 

        v["i162"] = 0.020000*np.tanh(((((data["var_78"]) * (((((((data["var_78"]) * (data["var_78"]))) * (data["var_78"]))) - (data["var_78"]))))) - (data["var_78"]))) 

        v["i163"] = 0.020000*np.tanh(((((data["var_78"]) * (data["var_78"]))) + (((-3.0) + (((((data["var_2"]) * 2.0)) * (data["var_2"]))))))) 

        v["i164"] = 0.020000*np.tanh((((((((data["var_48"]) + (((data["var_157"]) * (data["var_157"]))))) + (-2.0))/2.0)) + ((((data["var_89"]) + (data["var_78"]))/2.0)))) 

        v["i165"] = 0.020000*np.tanh(((((data["var_65"]) + (((data["var_157"]) * (data["var_157"]))))) + (((-1.0) + (((data["var_198"]) * (data["var_198"]))))))) 

        v["i166"] = 0.020000*np.tanh(((((data["var_26"]) + (((((data["var_99"]) - (1.0))) + (((data["var_141"]) * (data["var_141"]))))))) - (1.0))) 

        v["i167"] = 0.020000*np.tanh(((data["var_44"]) * (((((3.0) * 2.0)) * (((3.0) * (((data["var_44"]) + (((2.0) / 2.0)))))))))) 

        v["i168"] = 0.020000*np.tanh((((((data["var_151"]) + (((data["var_167"]) + (data["var_144"]))))) + (((data["var_106"]) + (((data["var_18"]) + (data["var_90"]))))))/2.0)) 

        v["i169"] = 0.020000*np.tanh(((((((data["var_164"]) * (data["var_164"]))) - (data["var_164"]))) + ((((((data["var_34"]) * (data["var_34"]))) + (-2.0))/2.0)))) 

        v["i170"] = 0.020000*np.tanh(((((((data["var_172"]) * (data["var_172"]))) + ((((((data["var_172"]) + (data["var_22"]))/2.0)) + (-1.0))))) * 2.0)) 

        v["i171"] = 0.020000*np.tanh((((((data["var_25"]) + (data["var_164"]))) + (((((data["var_106"]) * (data["var_106"]))) + (((data["var_189"]) + (data["var_137"]))))))/2.0)) 

        v["i172"] = 0.020000*np.tanh(((((((data["var_145"]) * (data["var_145"]))) + (-2.0))) + (((data["var_89"]) * (data["var_89"]))))) 

        v["i173"] = 0.020000*np.tanh(((((data["var_179"]) * (data["var_179"]))) + (((((data["var_148"]) * (data["var_148"]))) - (3.0))))) 

        v["i174"] = 0.020000*np.tanh(((((np.tanh((np.tanh((data["var_44"]))))) + (((data["var_44"]) * (data["var_44"]))))) + (np.tanh((np.tanh((data["var_44"]))))))) 

        v["i175"] = 0.020000*np.tanh(((data["var_188"]) * ((((((10.14662075042724609)) * ((((data["var_188"]) + (1.0))/2.0)))) + (1.0))))) 

        v["i176"] = 0.020000*np.tanh(((((((data["var_109"]) * (data["var_109"]))) + (np.tanh((((data["var_109"]) * ((6.0)))))))) + (np.tanh((data["var_109"]))))) 

        v["i177"] = 0.020000*np.tanh(((((((((data["var_184"]) * (data["var_184"]))) + (data["var_181"]))/2.0)) + (((data["var_11"]) + (((data["var_67"]) * (data["var_67"]))))))/2.0)) 

        v["i178"] = 0.020000*np.tanh(((((((((data["var_1"]) * (data["var_1"]))) + (((data["var_179"]) + (data["var_67"]))))) + (-2.0))) + (data["var_5"]))) 

        v["i179"] = 0.020000*np.tanh(((((((data["var_169"]) * (data["var_169"]))) - (3.0))) + (((data["var_166"]) * (data["var_166"]))))) 

        v["i180"] = 0.020000*np.tanh(((((data["var_110"]) * (data["var_110"]))) + (((((((data["var_2"]) * (data["var_2"]))) + (-1.0))) + (-1.0))))) 

        v["i181"] = 0.020000*np.tanh(((((((((data["var_170"]) * (((data["var_170"]) * (data["var_170"]))))) * (data["var_170"]))) - (1.0))) - (data["var_170"]))) 

        v["i182"] = 0.020000*np.tanh(((data["var_133"]) + (((data["var_80"]) + (((((((data["var_80"]) * (data["var_80"]))) - (((2.0) / 2.0)))) * 2.0)))))) 

        v["i183"] = 0.020000*np.tanh(((((((-2.0) + (((data["var_99"]) * (((data["var_99"]) * 2.0)))))) - (data["var_99"]))) - (data["var_99"]))) 

        v["i184"] = 0.020000*np.tanh(((((data["var_22"]) * (data["var_22"]))) + (((-2.0) - (((data["var_22"]) - (((data["var_22"]) * (data["var_22"]))))))))) 

        v["i185"] = 0.020000*np.tanh((((((((data["var_155"]) * (data["var_155"]))) + (((data["var_22"]) * (data["var_22"]))))) + (-2.0))/2.0)) 

        v["i186"] = 0.020000*np.tanh((((((data["var_108"]) + ((((3.0) + ((((data["var_108"]) + (data["var_108"]))/2.0)))/2.0)))/2.0)) * (data["var_108"]))) 

        v["i187"] = 0.020000*np.tanh(((((((((data["var_164"]) * (data["var_164"]))) + (data["var_82"]))/2.0)) + (((data["var_6"]) * (data["var_6"]))))/2.0)) 

        v["i188"] = 0.020000*np.tanh(((((((data["var_191"]) * (data["var_191"]))) + (((data["var_164"]) - (2.0))))) + (((data["var_115"]) * (data["var_115"]))))) 

        v["i189"] = 0.020000*np.tanh(((((-2.0) - (((data["var_92"]) * ((-1.0*((data["var_92"])))))))) + (((data["var_32"]) + (data["var_53"]))))) 

        v["i190"] = 0.020000*np.tanh(((((data["var_147"]) * 2.0)) * (((data["var_147"]) + (np.tanh(((-1.0*((((((data["var_147"]) + (data["var_147"]))) * 2.0))))))))))) 

        v["i191"] = 0.020000*np.tanh((((((((data["var_37"]) + (data["var_157"]))) + (((((data["var_135"]) * (data["var_135"]))) + (data["var_79"]))))) + (data["var_119"]))/2.0)) 

        v["i192"] = 0.020000*np.tanh(((((((data["var_34"]) * (data["var_34"]))) + (-2.0))) + (((data["var_95"]) * (data["var_95"]))))) 

        v["i193"] = 0.020000*np.tanh(((data["var_13"]) + (((((((data["var_13"]) * (((data["var_13"]) + (data["var_13"]))))) + (data["var_0"]))) - (3.0))))) 

        v["i194"] = 0.020000*np.tanh(((-1.0) + (((((((-1.0) + (((data["var_35"]) * (data["var_35"]))))) + (data["var_110"]))) + (data["var_111"]))))) 

        v["i195"] = 0.020000*np.tanh(((((data["var_165"]) * (data["var_165"]))) + ((((((data["var_24"]) + (((data["var_165"]) + (data["var_147"]))))/2.0)) + (-1.0))))) 

        v["i196"] = 0.020000*np.tanh(((data["var_44"]) * (((data["var_44"]) + (((3.0) + (data["var_44"]))))))) 

        v["i197"] = 0.020000*np.tanh(((((data["var_191"]) * (((data["var_191"]) + (data["var_191"]))))) - (((((2.0) + (data["var_191"]))) - (data["var_91"]))))) 

        v["i198"] = 0.020000*np.tanh(((((((data["var_93"]) * 2.0)) * (data["var_93"]))) + (((data["var_93"]) - (((2.0) - (data["var_162"]))))))) 

        v["i199"] = 0.020000*np.tanh(((((((((data["var_12"]) * (data["var_12"]))) + (((data["var_40"]) * (data["var_40"]))))) - ((1.27164638042449951)))) - ((1.27164995670318604)))) 

        v["i200"] = 0.020000*np.tanh(((((((((((data["var_176"]) + (data["var_159"]))/2.0)) / 2.0)) + (((data["var_172"]) * ((((data["var_172"]) + (data["var_172"]))/2.0)))))/2.0)) * 2.0)) 

        v["i201"] = 0.020000*np.tanh(((((((((data["var_4"]) + (data["var_112"]))/2.0)) + ((((data["var_146"]) + (data["var_145"]))/2.0)))) + (((data["var_146"]) * (data["var_146"]))))/2.0)) 

        v["i202"] = 0.020000*np.tanh(((((((((((-3.0) - (((data["var_109"]) * 2.0)))) + (((data["var_154"]) * (data["var_154"]))))) * 2.0)) * 2.0)) * 2.0)) 

        v["i203"] = 0.020000*np.tanh(((((data["var_33"]) + (((((data["var_33"]) * 2.0)) * (data["var_33"]))))) + (((np.tanh((((data["var_33"]) * 2.0)))) * 2.0)))) 

        v["i204"] = 0.020000*np.tanh(((data["var_1"]) + (((data["var_190"]) + (((np.tanh((((data["var_89"]) + (((data["var_191"]) - (2.0))))))) * 2.0)))))) 

        v["i205"] = 0.020000*np.tanh(((((((((((data["var_155"]) * (data["var_155"]))) - (data["var_155"]))) * (data["var_155"]))) * (data["var_155"]))) - (data["var_155"]))) 

        v["i206"] = 0.020000*np.tanh(((((((data["var_155"]) - (3.0))) + (((data["var_0"]) * (data["var_0"]))))) + (((data["var_1"]) * (data["var_1"]))))) 

        v["i207"] = 0.020000*np.tanh(((((data["var_56"]) * (((data["var_56"]) + (((data["var_56"]) + (1.0))))))) + (((data["var_56"]) + (data["var_56"]))))) 

        v["i208"] = 0.020000*np.tanh((((((data["var_118"]) + (data["var_94"]))/2.0)) + (((((data["var_75"]) + (-1.0))) + (((data["var_75"]) * (data["var_75"]))))))) 

        v["i209"] = 0.020000*np.tanh((((((-1.0) + (((((data["var_52"]) * (data["var_52"]))) + (((((data["var_67"]) * (data["var_67"]))) / 2.0)))))/2.0)) * 2.0)) 

        v["i210"] = 0.020000*np.tanh((((((data["var_199"]) + (((data["var_62"]) - (data["var_56"]))))/2.0)) + ((((data["var_19"]) + (((data["var_90"]) + (data["var_71"]))))/2.0)))) 

        v["i211"] = 0.020000*np.tanh(((((((data["var_47"]) + (((data["var_144"]) * (data["var_144"]))))/2.0)) + (((data["var_47"]) + (((data["var_128"]) * (data["var_128"]))))))/2.0)) 

        v["i212"] = 0.020000*np.tanh(((((-1.0) + (((-1.0) + (((data["var_118"]) * (data["var_118"]))))))) + (((data["var_154"]) * (data["var_154"]))))) 

        v["i213"] = 0.020000*np.tanh((((data["var_3"]) + ((((data["var_125"]) + (((((((data["var_18"]) * (data["var_18"]))) * (data["var_18"]))) * (data["var_18"]))))/2.0)))/2.0)) 

        v["i214"] = 0.020000*np.tanh(((data["var_163"]) - ((-1.0*((((data["var_173"]) - ((-1.0*((((data["var_18"]) - (((3.0) / 2.0)))))))))))))) 

        v["i215"] = 0.020000*np.tanh(((((((((data["var_174"]) + (((((data["var_174"]) + (((data["var_174"]) * (data["var_174"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) 

        v["i216"] = 0.020000*np.tanh(((((((data["var_91"]) + (((data["var_29"]) + (-3.0))))/2.0)) + (((data["var_175"]) + (((data["var_40"]) * (data["var_40"]))))))/2.0)) 

        v["i217"] = 0.020000*np.tanh(((np.tanh((-3.0))) + (((((data["var_76"]) * (data["var_76"]))) - ((-1.0*((np.tanh((((data["var_76"]) * 2.0))))))))))) 

        v["i218"] = 0.020000*np.tanh(((data["var_13"]) + (((((data["var_13"]) + (1.0))) * (((data["var_15"]) + (((data["var_13"]) + (data["var_13"]))))))))) 

        v["i219"] = 0.020000*np.tanh(((((data["var_119"]) * (data["var_119"]))) - (np.tanh((((data["var_119"]) * (((data["var_119"]) * (data["var_119"]))))))))) 

        v["i220"] = 0.020000*np.tanh((((-1.0*((2.0)))) + ((((((((data["var_172"]) * (data["var_172"]))) + (((data["var_139"]) * (data["var_139"]))))/2.0)) * 2.0)))) 

        v["i221"] = 0.020000*np.tanh(((((((((data["var_180"]) + (data["var_122"]))) + (((data["var_140"]) + (data["var_60"]))))/2.0)) + (((data["var_122"]) * (data["var_122"]))))/2.0)) 

        v["i222"] = 0.020000*np.tanh(((((((data["var_22"]) + (((np.tanh((((((-3.0) - (((data["var_109"]) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i223"] = 0.020000*np.tanh((((((((data["var_91"]) * (data["var_91"]))) + (((data["var_139"]) + (((data["var_151"]) * (data["var_151"]))))))) + (-2.0))/2.0)) 

        v["i224"] = 0.020000*np.tanh(((((data["var_173"]) * (data["var_173"]))) + (((data["var_40"]) + ((((((-1.0) + (data["var_170"]))/2.0)) + (-1.0))))))) 

        v["i225"] = 0.020000*np.tanh((((((-1.0*((2.0)))) + (((data["var_166"]) * (data["var_166"]))))) + (((data["var_43"]) * (data["var_43"]))))) 

        v["i226"] = 0.020000*np.tanh(((((((np.tanh((((((data["var_78"]) + ((((-2.0) + (-1.0))/2.0)))) * 2.0)))) * 2.0)) + (data["var_130"]))) * 2.0)) 

        v["i227"] = 0.020000*np.tanh(((((((data["var_123"]) * (((data["var_123"]) + (data["var_123"]))))) + (((-1.0) + (data["var_123"]))))) + (-1.0))) 

        v["i228"] = 0.020000*np.tanh((((((((data["var_163"]) * (data["var_163"]))) - (data["var_163"]))) + ((((data["var_196"]) + (((data["var_2"]) * (data["var_2"]))))/2.0)))/2.0)) 

        v["i229"] = 0.020000*np.tanh(((((((((data["var_53"]) * (data["var_53"]))) + (np.tanh((((data["var_1"]) - (2.0))))))) - (data["var_53"]))) * 2.0)) 

        v["i230"] = 0.020000*np.tanh(((((np.tanh((data["var_56"]))) + (-1.0))) + (((data["var_56"]) * (data["var_56"]))))) 

        v["i231"] = 0.020000*np.tanh((((((((data["var_164"]) + (((data["var_26"]) * (data["var_26"]))))/2.0)) - (1.0))) + (((data["var_106"]) * (data["var_106"]))))) 

        v["i232"] = 0.020000*np.tanh(((((data["var_195"]) + (((data["var_88"]) * (data["var_88"]))))) - (np.tanh(((((4.0)) - (((data["var_164"]) * 2.0)))))))) 

        v["i233"] = 0.020000*np.tanh((((((data["var_186"]) * (data["var_186"]))) + ((((data["var_187"]) + (data["var_95"]))/2.0)))/2.0)) 

        v["i234"] = 0.020000*np.tanh(((np.tanh((((data["var_177"]) * 2.0)))) + ((((((data["var_155"]) + (data["var_53"]))) + (((data["var_6"]) * (data["var_6"]))))/2.0)))) 

        v["i235"] = 0.020000*np.tanh((((((np.tanh((((data["var_121"]) * 2.0)))) + (((data["var_110"]) + (data["var_66"]))))) + (((data["var_157"]) * (data["var_157"]))))/2.0)) 

        v["i236"] = 0.020000*np.tanh((((((data["var_163"]) * (((data["var_163"]) * 2.0)))) + (((data["var_164"]) + (-3.0))))/2.0)) 

        v["i237"] = 0.020000*np.tanh((((((((data["var_167"]) * (data["var_167"]))) + (((((data["var_130"]) * (data["var_130"]))) * 2.0)))/2.0)) - (data["var_130"]))) 

        v["i238"] = 0.020000*np.tanh((((((((((data["var_173"]) * (((data["var_173"]) / 2.0)))) * (data["var_173"]))) * (data["var_173"]))) + (data["var_84"]))/2.0)) 

        v["i239"] = 0.020000*np.tanh(((((((data["var_170"]) * (data["var_170"]))) + ((((-2.0) + (data["var_16"]))/2.0)))) + (np.tanh(((-1.0*((data["var_170"])))))))) 

        v["i240"] = 0.020000*np.tanh(((((((data["var_112"]) * (((data["var_112"]) * (((data["var_112"]) * (data["var_112"]))))))) - (data["var_112"]))) - (data["var_112"]))) 

        v["i241"] = 0.020000*np.tanh(((data["var_170"]) + (((((-1.0) * 2.0)) + (((data["var_112"]) + (((data["var_169"]) * (data["var_169"]))))))))) 

        v["i242"] = 0.020000*np.tanh((((((((data["var_167"]) + (data["var_11"]))) + (-2.0))) + (((data["var_26"]) * (data["var_26"]))))/2.0)) 

        v["i243"] = 0.020000*np.tanh((((((((data["var_67"]) * (data["var_67"]))) + (((data["var_12"]) * (data["var_12"]))))) + (((-1.0) + (-1.0))))/2.0)) 

        v["i244"] = 0.020000*np.tanh((((((((data["var_49"]) * (data["var_49"]))) + (data["var_35"]))) + (((data["var_82"]) + (data["var_176"]))))/2.0)) 

        v["i245"] = 0.020000*np.tanh(((((data["var_191"]) + (-1.0))) + (((((((data["var_157"]) * (data["var_157"]))) + (data["var_51"]))) + (-1.0))))) 

        v["i246"] = 0.020000*np.tanh(((((((data["var_195"]) * (((data["var_195"]) * (((data["var_195"]) * (data["var_195"]))))))) - (data["var_195"]))) - (data["var_195"]))) 

        v["i247"] = 0.020000*np.tanh((((((data["var_67"]) + ((((data["var_6"]) + (data["var_48"]))/2.0)))) + ((((data["var_5"]) + (data["var_55"]))/2.0)))/2.0)) 

        v["i248"] = 0.020000*np.tanh((((((data["var_121"]) * (data["var_121"]))) + (((((data["var_147"]) * (data["var_147"]))) + (-3.0))))/2.0)) 

        v["i249"] = 0.020000*np.tanh(((((((data["var_69"]) + (-3.0))/2.0)) + (((((((data["var_2"]) * (data["var_2"]))) + ((-1.0*((data["var_2"])))))) * 2.0)))/2.0)) 

        v["i250"] = 0.020000*np.tanh(((((data["var_111"]) * (data["var_111"]))) + (((-2.0) - ((-1.0*((((data["var_11"]) * (data["var_11"])))))))))) 

        v["i251"] = 0.020000*np.tanh(((((((((((data["var_179"]) * 2.0)) * (data["var_179"]))) - (1.0))) - (1.0))) - (((data["var_179"]) * 2.0)))) 

        v["i252"] = 0.020000*np.tanh(((data["var_146"]) + (((data["var_146"]) * (data["var_146"]))))) 

        v["i253"] = 0.020000*np.tanh(((((np.tanh((((((-3.0) + (((data["var_78"]) * 2.0)))) * 2.0)))) * 2.0)) + (((data["var_44"]) * (data["var_44"]))))) 

        v["i254"] = 0.020000*np.tanh(((((((data["var_179"]) * (data["var_179"]))) + (-3.0))) + (((data["var_150"]) * (data["var_150"]))))) 

        v["i255"] = 0.020000*np.tanh(((((((((data["var_1"]) + (data["var_135"]))/2.0)) + (((data["var_70"]) * (data["var_70"]))))) + ((((data["var_69"]) + (data["var_151"]))/2.0)))/2.0)) 

        v["i256"] = 0.020000*np.tanh(((((((data["var_144"]) + (data["var_49"]))/2.0)) + (((data["var_124"]) + (((data["var_128"]) + (((data["var_70"]) - (data["var_80"]))))))))/2.0)) 

        v["i257"] = 0.020000*np.tanh((((((data["var_147"]) * (((data["var_147"]) / 2.0)))) + ((((data["var_71"]) + (((data["var_71"]) * (data["var_71"]))))/2.0)))/2.0)) 

        v["i258"] = 0.020000*np.tanh((((((data["var_106"]) * (data["var_106"]))) + ((((data["var_97"]) + (((((data["var_48"]) * (data["var_48"]))) + (data["var_97"]))))/2.0)))/2.0)) 

        v["i259"] = 0.020000*np.tanh(((((((-1.0) + (((data["var_151"]) * (data["var_151"]))))/2.0)) + (((((data["var_146"]) * (data["var_146"]))) / 2.0)))/2.0)) 

        v["i260"] = 0.020000*np.tanh(((((((((data["var_122"]) * (data["var_122"]))) + (data["var_105"]))/2.0)) + (((data["var_53"]) * (((data["var_53"]) / 2.0)))))/2.0)) 

        v["i261"] = 0.020000*np.tanh((((-1.0) + (((data["var_177"]) * (((((((((data["var_177"]) / 2.0)) * (data["var_177"]))) * (data["var_177"]))) / 2.0)))))/2.0)) 

        v["i262"] = 0.020000*np.tanh((((((((-3.0) + (-3.0))/2.0)) + (((data["var_162"]) * (data["var_162"]))))) + (((data["var_162"]) * (data["var_162"]))))) 

        v["i263"] = 0.020000*np.tanh(((((data["var_22"]) * (((data["var_22"]) - (1.0))))) - (1.0))) 

        v["i264"] = 0.020000*np.tanh(((((((((data["var_190"]) * (data["var_190"]))) - (data["var_190"]))) * 2.0)) + ((((data["var_190"]) + (-3.0))/2.0)))) 

        v["i265"] = 0.020000*np.tanh((((((data["var_138"]) * (data["var_138"]))) + (((data["var_138"]) * (data["var_138"]))))/2.0)) 

        v["i266"] = 0.020000*np.tanh(((data["var_154"]) + (((((((((np.tanh((data["var_154"]))) * 2.0)) + (((data["var_154"]) * (data["var_154"]))))) * 2.0)) * 2.0)))) 

        v["i267"] = 0.020000*np.tanh((((-1.0) + (((((data["var_184"]) * (data["var_184"]))) + ((((((data["var_22"]) * (data["var_22"]))) + (-1.0))/2.0)))))/2.0)) 

        v["i268"] = 0.020000*np.tanh((((((((((data["var_85"]) * (data["var_85"]))) * (((data["var_85"]) * (data["var_85"]))))) + (data["var_85"]))/2.0)) + (data["var_85"]))) 

        v["i269"] = 0.020000*np.tanh(((((((data["var_28"]) + (np.tanh((data["var_109"]))))) + (((data["var_28"]) * (data["var_28"]))))) - (data["var_28"]))) 

        v["i270"] = 0.020000*np.tanh((((-2.0) + (((data["var_95"]) * (((-2.0) + (((data["var_95"]) + (((data["var_95"]) + (data["var_95"]))))))))))/2.0)) 

        v["i271"] = 0.020000*np.tanh(((((((((data["var_149"]) * (data["var_149"]))) + (((data["var_35"]) * (data["var_35"]))))) + (-1.0))) + (-1.0))) 

        v["i272"] = 0.020000*np.tanh(((((((data["var_165"]) * (data["var_165"]))) + (-2.0))) + (((data["var_165"]) + (((data["var_132"]) * (data["var_132"]))))))) 

        v["i273"] = 0.020000*np.tanh((((((((data["var_1"]) * (((data["var_1"]) * (((data["var_1"]) * (((data["var_1"]) / 2.0)))))))) + (data["var_199"]))/2.0)) / 2.0)) 

        v["i274"] = 0.020000*np.tanh((((((data["var_26"]) * (data["var_26"]))) + (((((((data["var_138"]) + (data["var_181"]))/2.0)) + (-2.0))/2.0)))/2.0)) 

        v["i275"] = 0.020000*np.tanh(((((((((((data["var_18"]) * (data["var_18"]))) + (data["var_50"]))/2.0)) + (data["var_50"]))/2.0)) + (((data["var_50"]) * (data["var_50"]))))) 

        v["i276"] = 0.020000*np.tanh((((((data["var_195"]) + (((((data["var_145"]) * (((data["var_145"]) * (data["var_145"]))))) * (data["var_145"]))))) + (-2.0))/2.0)) 

        v["i277"] = 0.020000*np.tanh(((((((data["var_108"]) * ((((((data["var_108"]) * (data["var_108"]))) + (data["var_108"]))/2.0)))) * (data["var_108"]))) + (-1.0))) 

        v["i278"] = 0.020000*np.tanh(((data["var_32"]) * ((((((((((((data["var_89"]) / 2.0)) + (-3.0))/2.0)) + (data["var_32"]))) * 2.0)) * 2.0)))) 

        v["i279"] = 0.020000*np.tanh(((((-1.0) + (data["var_32"]))) + (((((-1.0) + (data["var_9"]))) + (((data["var_9"]) * (data["var_9"]))))))) 

        v["i280"] = 0.020000*np.tanh(((data["var_75"]) + (((((data["var_75"]) + (data["var_75"]))) * (((data["var_75"]) - (np.tanh((((data["var_75"]) * 2.0)))))))))) 

        v["i281"] = 0.020000*np.tanh((((data["var_44"]) + (((data["var_44"]) * (data["var_44"]))))/2.0)) 

        v["i282"] = 0.020000*np.tanh(((((((data["var_79"]) + (data["var_189"]))/2.0)) + ((((((data["var_89"]) * (data["var_89"]))) + (((data["var_110"]) + (data["var_106"]))))/2.0)))/2.0)) 

        v["i283"] = 0.020000*np.tanh((((data["var_43"]) + (((((data["var_43"]) * (data["var_43"]))) + ((((((data["var_43"]) * (data["var_43"]))) + (data["var_89"]))/2.0)))))/2.0)) 

        v["i284"] = 0.020000*np.tanh(((data["var_154"]) * ((((data["var_154"]) + (np.tanh((((((data["var_154"]) * (data["var_154"]))) - (data["var_154"]))))))/2.0)))) 

        v["i285"] = 0.020000*np.tanh((((((((data["var_123"]) * 2.0)) + ((((data["var_123"]) + (data["var_89"]))/2.0)))) + (((data["var_123"]) * (((data["var_123"]) * 2.0)))))/2.0)) 

        v["i286"] = 0.020000*np.tanh(((((((((data["var_167"]) * (data["var_167"]))) * (((((data["var_167"]) * (data["var_167"]))) / 2.0)))) - (data["var_167"]))) * 2.0)) 

        v["i287"] = 0.020000*np.tanh(((((((3.0) / 2.0)) - (data["var_133"]))) * (((((((data["var_162"]) - (((3.0) / 2.0)))) * 2.0)) * 2.0)))) 

        v["i288"] = 0.020000*np.tanh((((((((((((data["var_0"]) * (data["var_0"]))) / 2.0)) + (data["var_0"]))/2.0)) * (data["var_0"]))) + ((-1.0*((data["var_0"])))))) 

        v["i289"] = 0.020000*np.tanh((((((((data["var_80"]) * (data["var_80"]))) * ((((((data["var_80"]) * (data["var_80"]))) + (data["var_80"]))/2.0)))) + (data["var_0"]))/2.0)) 

        v["i290"] = 0.020000*np.tanh(((((((data["var_167"]) + ((((data["var_190"]) + (data["var_195"]))/2.0)))/2.0)) + (((((data["var_195"]) * (data["var_195"]))) - (data["var_195"]))))/2.0)) 

        v["i291"] = 0.020000*np.tanh((((((-1.0) + ((((((data["var_60"]) + (data["var_2"]))/2.0)) + ((((data["var_91"]) + (data["var_91"]))/2.0)))))) + (data["var_180"]))/2.0)) 

        v["i292"] = 0.020000*np.tanh(((np.tanh(((-1.0*((data["var_94"])))))) + ((((-1.0*((data["var_94"])))) - ((((-1.0*((data["var_94"])))) * (data["var_94"]))))))) 

        v["i293"] = 0.020000*np.tanh(((data["var_114"]) * (((data["var_114"]) + ((-1.0*((np.tanh((((data["var_114"]) + (((data["var_134"]) + (data["var_114"])))))))))))))) 

        v["i294"] = 0.020000*np.tanh(((((((data["var_53"]) + (((np.tanh((((((((data["var_130"]) - ((1.55873215198516846)))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i295"] = 0.020000*np.tanh(((((((data["var_76"]) + (((data["var_76"]) * (data["var_76"]))))) - (3.0))) + (((data["var_76"]) * (data["var_76"]))))) 

        v["i296"] = 0.020000*np.tanh(((((data["var_164"]) * 2.0)) * (((data["var_164"]) * ((((((data["var_164"]) * (data["var_164"]))) + (-3.0))/2.0)))))) 

        v["i297"] = 0.020000*np.tanh(((((data["var_99"]) * (data["var_99"]))) + (((-1.0) + (((data["var_94"]) + (-1.0))))))) 

        v["i298"] = 0.020000*np.tanh(((data["var_33"]) + (((np.tanh(((-1.0*((((data["var_33"]) * (data["var_33"])))))))) + (((data["var_33"]) * (data["var_33"]))))))) 

        v["i299"] = 0.020000*np.tanh(((((data["var_78"]) - (np.tanh((np.tanh((((data["var_78"]) * 2.0)))))))) + ((-1.0*((np.tanh((np.tanh((data["var_78"])))))))))) 

        v["i300"] = 0.020000*np.tanh((((((data["var_56"]) + (((data["var_56"]) * (data["var_56"]))))) + ((((data["var_117"]) + (((data["var_134"]) + (data["var_78"]))))/2.0)))/2.0)) 

        v["i301"] = 0.020000*np.tanh(((((((data["var_198"]) + (data["var_91"]))/2.0)) + ((((((data["var_147"]) * (data["var_147"]))) + (((data["var_198"]) * (data["var_198"]))))/2.0)))/2.0)) 

        v["i302"] = 0.020000*np.tanh((((np.tanh((((((data["var_190"]) * (data["var_190"]))) - (2.0))))) + (((((data["var_105"]) * (data["var_105"]))) * 2.0)))/2.0)) 

        v["i303"] = 0.020000*np.tanh((((((data["var_114"]) + (((((data["var_114"]) * (data["var_114"]))) + (data["var_114"]))))/2.0)) * (((data["var_114"]) * (data["var_114"]))))) 

        v["i304"] = 0.020000*np.tanh((((((((((((((-3.0) + (((((data["var_177"]) / 2.0)) / 2.0)))/2.0)) - (data["var_177"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 

        v["i305"] = 0.020000*np.tanh(((((((((data["var_180"]) * (data["var_180"]))) + (-3.0))/2.0)) + (((data["var_148"]) * (data["var_148"]))))/2.0)) 

        v["i306"] = 0.020000*np.tanh(((((data["var_166"]) * 2.0)) * (((data["var_166"]) - (((-2.0) - (((((data["var_166"]) / 2.0)) / 2.0)))))))) 

        v["i307"] = 0.020000*np.tanh(((((((((data["var_6"]) * (data["var_6"]))) + (((data["var_1"]) * (data["var_1"]))))/2.0)) + ((((data["var_157"]) + (data["var_78"]))/2.0)))/2.0)) 

        v["i308"] = 0.020000*np.tanh((((((data["var_115"]) * (data["var_115"]))) + (((((data["var_191"]) * (data["var_191"]))) / 2.0)))/2.0)) 

        v["i309"] = 0.020000*np.tanh((((-1.0*((((1.0) - ((((-1.0*((((1.0) - (((data["var_118"]) * (data["var_118"])))))))) * 2.0))))))) * 2.0)) 

        v["i310"] = 0.020000*np.tanh(((((((data["var_188"]) * 2.0)) * ((((data["var_188"]) + ((((3.0) + ((((data["var_188"]) + (2.0))/2.0)))/2.0)))/2.0)))) * 2.0)) 

        v["i311"] = 0.020000*np.tanh(((data["var_150"]) + (((((data["var_150"]) + (((data["var_150"]) * (data["var_150"]))))) * (((data["var_150"]) * (data["var_150"]))))))) 

        v["i312"] = 0.020000*np.tanh(((data["var_127"]) * (((((data["var_127"]) + (((((data["var_127"]) + (1.0))) + (1.0))))) + (1.0))))) 

        v["i313"] = 0.020000*np.tanh((((((((data["var_119"]) * (data["var_119"]))) + ((((-1.0*(((3.0))))) + (((data["var_13"]) * (data["var_13"]))))))/2.0)) * 2.0)) 

        v["i314"] = 0.020000*np.tanh((((((((data["var_164"]) * (data["var_164"]))) - (data["var_164"]))) + (((data["var_50"]) * (data["var_50"]))))/2.0)) 

        v["i315"] = 0.020000*np.tanh(((((((((data["var_111"]) / 2.0)) * (data["var_111"]))) * (((data["var_111"]) * (data["var_111"]))))) - (data["var_111"]))) 

        v["i316"] = 0.020000*np.tanh(((-2.0) + (((data["var_164"]) - (((data["var_172"]) * (((0.0) - (data["var_172"]))))))))) 

        v["i317"] = 0.020000*np.tanh(((data["var_192"]) * (((data["var_192"]) + (((((data["var_192"]) * (data["var_192"]))) + (-2.0))))))) 

        v["i318"] = 0.020000*np.tanh((((data["var_111"]) + ((((((data["var_111"]) + (((data["var_196"]) * (((data["var_196"]) * (data["var_196"]))))))/2.0)) * (data["var_196"]))))/2.0)) 

        v["i319"] = 0.020000*np.tanh(((((((((data["var_74"]) + (data["var_24"]))) + (data["var_53"]))/2.0)) + ((((data["var_164"]) + (data["var_1"]))/2.0)))/2.0)) 

        v["i320"] = 0.020000*np.tanh(((((((((data["var_133"]) + (data["var_133"]))/2.0)) * ((((data["var_168"]) + (data["var_133"]))/2.0)))) + (((data["var_168"]) * (data["var_168"]))))/2.0)) 

        v["i321"] = 0.020000*np.tanh(((((data["var_122"]) * 2.0)) * ((((np.tanh(((-1.0*((((((data["var_122"]) * 2.0)) * 2.0))))))) + (data["var_122"]))/2.0)))) 

        v["i322"] = 0.020000*np.tanh(((-1.0) + ((((((data["var_9"]) * (data["var_9"]))) + (((-1.0) + (((data["var_173"]) * (data["var_173"]))))))/2.0)))) 

        v["i323"] = 0.020000*np.tanh(((np.tanh((data["var_13"]))) + (((((data["var_13"]) * (data["var_13"]))) + (-2.0))))) 

        v["i324"] = 0.020000*np.tanh((((((-1.0) + (((((((data["var_192"]) + (data["var_193"]))/2.0)) + (data["var_82"]))/2.0)))/2.0)) + (((data["var_193"]) * (data["var_193"]))))) 

        v["i325"] = 0.020000*np.tanh((((np.tanh((((((((((data["var_139"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) + (((data["var_175"]) * (data["var_175"]))))/2.0)) 

        v["i326"] = 0.020000*np.tanh(((((((((((((data["var_141"]) * (data["var_141"]))) + (data["var_141"]))) * 2.0)) + (data["var_141"]))) * 2.0)) + (data["var_173"]))) 

        v["i327"] = 0.020000*np.tanh((((((((data["var_43"]) + (((data["var_43"]) * (data["var_43"]))))/2.0)) * (((data["var_43"]) * (data["var_43"]))))) + (data["var_43"]))) 

        v["i328"] = 0.020000*np.tanh(((data["var_36"]) * (((((data["var_36"]) * (((data["var_36"]) * ((((data["var_36"]) + ((-1.0*((-1.0)))))/2.0)))))) / 2.0)))) 

        v["i329"] = 0.020000*np.tanh((((((data["var_168"]) * (((data["var_168"]) + (data["var_168"]))))) + ((((((data["var_66"]) * (data["var_66"]))) + (data["var_67"]))/2.0)))/2.0)) 

        v["i330"] = 0.020000*np.tanh(((((((((data["var_187"]) * (data["var_187"]))) + ((((data["var_125"]) + (data["var_82"]))/2.0)))/2.0)) + ((((data["var_53"]) + (data["var_134"]))/2.0)))/2.0)) 

        v["i331"] = 0.020000*np.tanh(((-3.0) + (((((data["var_196"]) * (data["var_196"]))) + (((data["var_184"]) * (data["var_184"]))))))) 

        v["i332"] = 0.020000*np.tanh((((((((((data["var_162"]) + (data["var_181"]))/2.0)) + (data["var_190"]))/2.0)) + (((((data["var_194"]) * (data["var_194"]))) - (1.0))))/2.0)) 

        v["i333"] = 0.020000*np.tanh(((data["var_72"]) * ((((data["var_72"]) + (data["var_72"]))/2.0)))) 

        v["i334"] = 0.020000*np.tanh(((data["var_167"]) * ((((data["var_167"]) + ((((-1.0) + ((((-1.0) + (data["var_167"]))/2.0)))/2.0)))/2.0)))) 

        v["i335"] = 0.020000*np.tanh((((((data["var_181"]) * (data["var_181"]))) + (1.0))/2.0)) 

        v["i336"] = 0.020000*np.tanh((((((((((data["var_83"]) * (data["var_83"]))) + (((data["var_81"]) * (data["var_81"]))))/2.0)) - (0.318310))) - (0.318310))) 

        v["i337"] = 0.020000*np.tanh((((((data["var_197"]) * (data["var_197"]))) + ((((((data["var_95"]) * (data["var_95"]))) + (data["var_197"]))/2.0)))/2.0)) 

        v["i338"] = 0.020000*np.tanh((((((data["var_44"]) * (data["var_44"]))) + (((((data["var_119"]) * (data["var_119"]))) + (-2.0))))/2.0)) 

        v["i339"] = 0.020000*np.tanh((((((data["var_38"]) + (((((((data["var_15"]) * (data["var_15"]))) / 2.0)) * (((data["var_15"]) * (data["var_15"]))))))/2.0)) / 2.0)) 

        v["i340"] = 0.020000*np.tanh((((((((data["var_145"]) * (((((((data["var_145"]) * (data["var_145"]))) / 2.0)) * (data["var_145"]))))) + (data["var_0"]))/2.0)) / 2.0)) 

        v["i341"] = 0.020000*np.tanh((((((((((-3.0) + (data["var_48"]))/2.0)) + (data["var_137"]))/2.0)) + ((((data["var_145"]) + (((data["var_8"]) + (data["var_147"]))))/2.0)))/2.0)) 

        v["i342"] = 0.020000*np.tanh(((((((((((data["var_49"]) + (data["var_78"]))) - (data["var_23"]))) + (data["var_144"]))/2.0)) + ((((data["var_5"]) + (data["var_46"]))/2.0)))/2.0)) 

        v["i343"] = 0.020000*np.tanh((((((((((data["var_151"]) * (data["var_151"]))) / 2.0)) - (data["var_190"]))) + (((data["var_190"]) * (data["var_190"]))))/2.0)) 

        v["i344"] = 0.020000*np.tanh((((-1.0) + ((((((data["var_107"]) * (((data["var_107"]) * 2.0)))) + (((data["var_110"]) * (data["var_110"]))))/2.0)))/2.0)) 

        v["i345"] = 0.020000*np.tanh(((2.0) * (((((data["var_71"]) * (((data["var_71"]) - (np.tanh((data["var_71"]))))))) - (np.tanh((data["var_71"]))))))) 

        v["i346"] = 0.020000*np.tanh(((data["var_71"]) + (((np.tanh((((np.tanh((((((-3.0) + (((data["var_128"]) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)))) 

        v["i347"] = 0.020000*np.tanh(((((((data["var_190"]) + (((data["var_90"]) * (data["var_90"]))))/2.0)) + ((((-2.0) + ((((data["var_71"]) + (data["var_71"]))/2.0)))/2.0)))/2.0)) 

        v["i348"] = 0.020000*np.tanh((((((-1.0) + (((data["var_133"]) * (data["var_133"]))))/2.0)) + ((((-2.0) + (((data["var_109"]) * (data["var_109"]))))/2.0)))) 

        v["i349"] = 0.020000*np.tanh((((((((data["var_145"]) + (((data["var_123"]) * (data["var_123"]))))) + (((((data["var_192"]) * (data["var_192"]))) / 2.0)))/2.0)) / 2.0)) 

        v["i350"] = 0.020000*np.tanh(((((((((data["var_125"]) / 2.0)) - (np.tanh(((((((3.0) + (data["var_125"]))/2.0)) - (data["var_40"]))))))) * 2.0)) * 2.0)) 

        v["i351"] = 0.020000*np.tanh(((((((data["var_5"]) + (((data["var_162"]) + (data["var_31"]))))/2.0)) + ((((data["var_69"]) + (((data["var_155"]) + (data["var_164"]))))/2.0)))/2.0)) 

        v["i352"] = 0.020000*np.tanh((((((((data["var_32"]) / 2.0)) * (data["var_32"]))) + (((data["var_149"]) * (data["var_149"]))))/2.0)) 

        v["i353"] = 0.020000*np.tanh(((((data["var_48"]) * 2.0)) * (((data["var_48"]) - (np.tanh((((data["var_48"]) * 2.0)))))))) 

        v["i354"] = 0.020000*np.tanh((((((data["var_52"]) * (data["var_52"]))) + ((((((data["var_56"]) * (data["var_56"]))) + ((-1.0*((data["var_52"])))))/2.0)))/2.0)) 

        v["i355"] = 0.020000*np.tanh((((((-1.0) + (((((data["var_51"]) * (((data["var_51"]) * (data["var_51"]))))) / 2.0)))/2.0)) * (data["var_51"]))) 

        v["i356"] = 0.020000*np.tanh((((((((data["var_180"]) * (data["var_180"]))) + (((((-2.0) + (((data["var_58"]) * (data["var_58"]))))) * 2.0)))/2.0)) * 2.0)) 

        v["i357"] = 0.020000*np.tanh((((((((-1.0) - (data["var_69"]))) + (((-1.0) + (data["var_140"]))))/2.0)) + (((data["var_69"]) * (data["var_69"]))))) 

        v["i358"] = 0.020000*np.tanh(((((data["var_138"]) / 2.0)) * ((((data["var_138"]) + (((((((data["var_138"]) * (data["var_138"]))) / 2.0)) * (data["var_138"]))))/2.0)))) 

        v["i359"] = 0.020000*np.tanh(((((((((((data["var_162"]) * (data["var_162"]))) * (data["var_162"]))) * (data["var_162"]))) - (data["var_162"]))) / 2.0)) 

        v["i360"] = 0.020000*np.tanh(((((data["var_87"]) * (data["var_87"]))) - (np.tanh((((data["var_87"]) * (((data["var_87"]) * 2.0)))))))) 

        v["i361"] = 0.020000*np.tanh((((data["var_137"]) + (((-1.0) + (((data["var_51"]) + (((data["var_135"]) * (data["var_135"]))))))))/2.0)) 

        v["i362"] = 0.020000*np.tanh((((((data["var_145"]) * (data["var_145"]))) + ((((data["var_22"]) + (((data["var_32"]) - (data["var_145"]))))/2.0)))/2.0)) 

        v["i363"] = 0.020000*np.tanh(((((((((data["var_145"]) - (np.tanh((data["var_145"]))))) - (np.tanh((np.tanh((np.tanh((data["var_145"]))))))))) * 2.0)) * 2.0)) 

        v["i364"] = 0.020000*np.tanh((-1.0*((((1.0) - (((((-1.0*((((1.0) - (((data["var_191"]) * (data["var_191"])))))))) + (data["var_145"]))/2.0))))))) 

        v["i365"] = 0.020000*np.tanh((((((((data["var_11"]) * (data["var_11"]))) + (data["var_82"]))) + ((((((data["var_159"]) * (data["var_159"]))) + (data["var_145"]))/2.0)))/2.0)) 

        v["i366"] = 0.020000*np.tanh((((((data["var_195"]) * (data["var_195"]))) + (np.tanh((((data["var_177"]) * 2.0)))))/2.0)) 

        v["i367"] = 0.020000*np.tanh((((np.tanh((((-1.0) + (data["var_35"]))))) + ((((((data["var_87"]) * (data["var_87"]))) + (data["var_192"]))/2.0)))/2.0)) 

        v["i368"] = 0.020000*np.tanh(((np.tanh((((data["var_165"]) * ((-1.0*((data["var_165"])))))))) - (((data["var_165"]) * ((-1.0*((((data["var_165"]) / 2.0))))))))) 

        v["i369"] = 0.020000*np.tanh((((((data["var_43"]) * (data["var_43"]))) + (((((data["var_125"]) * (data["var_125"]))) / 2.0)))/2.0)) 

        v["i370"] = 0.020000*np.tanh(((data["var_104"]) * ((((data["var_104"]) + ((((((data["var_104"]) + (((((((data["var_104"]) / 2.0)) / 2.0)) / 2.0)))/2.0)) / 2.0)))/2.0)))) 

        v["i371"] = 0.020000*np.tanh((((((data["var_26"]) * (((data["var_26"]) / 2.0)))) + (((data["var_128"]) * (data["var_128"]))))/2.0)) 

        v["i372"] = 0.020000*np.tanh((((((data["var_80"]) + (data["var_80"]))/2.0)) * ((((np.tanh((((data["var_80"]) * (data["var_80"]))))) + (data["var_80"]))/2.0)))) 

        v["i373"] = 0.020000*np.tanh((((((data["var_33"]) * (data["var_33"]))) + (data["var_33"]))/2.0)) 

        v["i374"] = 0.020000*np.tanh(((((((((data["var_2"]) * (data["var_2"]))) * (((((data["var_2"]) * (((data["var_2"]) / 2.0)))) / 2.0)))) / 2.0)) / 2.0)) 

        v["i375"] = 0.020000*np.tanh((-1.0*(((((-1.0*((data["var_112"])))) * (((data["var_112"]) - (np.tanh((1.0)))))))))) 

        v["i376"] = 0.020000*np.tanh((((np.tanh((np.tanh((data["var_56"]))))) + ((((((data["var_171"]) * (data["var_171"]))) + (((data["var_28"]) * (data["var_28"]))))/2.0)))/2.0)) 

        v["i377"] = 0.020000*np.tanh(((((((((((((data["var_12"]) * (data["var_12"]))) + (data["var_12"]))/2.0)) * (data["var_12"]))) * (data["var_12"]))) + (data["var_12"]))/2.0)) 

        v["i378"] = 0.020000*np.tanh(((((((((data["var_16"]) / 2.0)) * (data["var_16"]))) * (((data["var_16"]) / 2.0)))) * (data["var_16"]))) 

        v["i379"] = 0.020000*np.tanh((((np.tanh((data["var_87"]))) + ((((((data["var_87"]) * (data["var_87"]))) + (data["var_67"]))/2.0)))/2.0)) 

        v["i380"] = 0.020000*np.tanh(((((((data["var_99"]) + (((np.tanh((((((data["var_94"]) - (2.0))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i381"] = 0.020000*np.tanh(((((((data["var_105"]) + ((((data["var_187"]) + (data["var_135"]))/2.0)))/2.0)) + ((((((data["var_91"]) * (data["var_91"]))) + (data["var_69"]))/2.0)))/2.0)) 

        v["i382"] = 0.020000*np.tanh((((((data["var_167"]) * ((((((((data["var_167"]) * (data["var_167"]))) + (-1.0))/2.0)) * (data["var_167"]))))) + (-1.0))/2.0)) 

        v["i383"] = 0.020000*np.tanh((-1.0*(((((((4.0)) + ((((-1.0*((data["var_1"])))) * 2.0)))) * 2.0))))) 

        v["i384"] = 0.020000*np.tanh((((((((data["var_151"]) * (data["var_151"]))) + (((data["var_89"]) * (((data["var_89"]) * (((data["var_89"]) / 2.0)))))))/2.0)) / 2.0)) 

        v["i385"] = 0.020000*np.tanh(((np.tanh((((-2.0) - (data["var_92"]))))) + ((((data["var_112"]) + (((data["var_128"]) * (data["var_128"]))))/2.0)))) 

        v["i386"] = 0.020000*np.tanh(((((((data["var_60"]) + (data["var_175"]))/2.0)) + (((data["var_74"]) * (((((data["var_74"]) * (data["var_74"]))) * (data["var_74"]))))))/2.0)) 

        v["i387"] = 0.020000*np.tanh(((np.tanh((np.tanh(((-1.0*((data["var_1"])))))))) + (((np.tanh(((-1.0*((data["var_1"])))))) + (data["var_1"]))))) 

        v["i388"] = 0.020000*np.tanh(((data["var_88"]) * ((((data["var_88"]) + ((((((data["var_88"]) + (data["var_88"]))/2.0)) / 2.0)))/2.0)))) 

        v["i389"] = 0.020000*np.tanh((-1.0*((((data["var_146"]) * ((-1.0*((((((((data["var_146"]) + (data["var_146"]))/2.0)) + (((((3.47107243537902832)) + (data["var_146"]))/2.0)))/2.0)))))))))) 

        v["i390"] = 0.020000*np.tanh(((((data["var_36"]) * ((((data["var_36"]) + (data["var_36"]))/2.0)))) / 2.0)) 

        v["i391"] = 0.020000*np.tanh((((-1.0*((data["var_112"])))) + (((data["var_112"]) * (((((((data["var_112"]) * (data["var_112"]))) / 2.0)) * (data["var_112"]))))))) 

        v["i392"] = 0.020000*np.tanh((((((np.tanh((((data["var_92"]) * 2.0)))) + ((((data["var_52"]) + ((((data["var_128"]) + (data["var_159"]))/2.0)))/2.0)))) + (data["var_112"]))/2.0)) 

        v["i393"] = 0.020000*np.tanh(((((((np.tanh(((((-1.0*((((3.0) - ((((-1.0*((data["var_92"])))) * 2.0))))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) 

        v["i394"] = 0.020000*np.tanh(((data["var_157"]) * ((((data["var_157"]) + (((data["var_157"]) - ((((3.83017754554748535)) - (data["var_157"]))))))/2.0)))) 

        v["i395"] = 0.020000*np.tanh((((((data["var_157"]) + ((((((data["var_18"]) * (data["var_18"]))) + (((data["var_157"]) * (data["var_157"]))))/2.0)))/2.0)) / 2.0)) 

        v["i396"] = 0.020000*np.tanh((((((data["var_60"]) * (data["var_60"]))) + ((((data["var_198"]) + (((data["var_198"]) * (data["var_198"]))))/2.0)))/2.0)) 

        v["i397"] = 0.019990*np.tanh(((((((data["var_110"]) * (((data["var_110"]) * (((data["var_110"]) - (1.0))))))) - (1.0))) * (data["var_110"]))) 

        v["i398"] = 0.020000*np.tanh((((-2.0) + (((data["var_155"]) * ((((-2.0) + (((data["var_155"]) * (((data["var_155"]) * (data["var_155"]))))))/2.0)))))/2.0)) 

        v["i399"] = 0.020000*np.tanh((((((data["var_114"]) + (data["var_114"]))/2.0)) + (((-1.0) + (((data["var_114"]) * (data["var_114"]))))))) 

        v["i400"] = 0.020000*np.tanh((((((data["var_155"]) + (-2.0))) + ((((data["var_97"]) + (((((data["var_143"]) * (data["var_143"]))) * 2.0)))/2.0)))/2.0)) 

        v["i401"] = 0.020000*np.tanh(((((1.0) - ((((((data["var_165"]) * (data["var_165"]))) + (np.tanh((data["var_165"]))))/2.0)))) * (data["var_165"]))) 

        v["i402"] = 0.020000*np.tanh(((((((data["var_118"]) * (((data["var_118"]) * (data["var_118"]))))) - (data["var_118"]))) - (data["var_118"]))) 

        v["i403"] = 0.020000*np.tanh(((((((np.tanh((((((((data["var_9"]) + (0.318310))) * 2.0)) + (0.318310))))) * 2.0)) * 2.0)) * 2.0)) 

        v["i404"] = 0.020000*np.tanh((((((((((((((data["var_9"]) * (data["var_9"]))) + (data["var_9"]))) + (np.tanh((data["var_9"]))))/2.0)) * 2.0)) * 2.0)) * 2.0)) 

        v["i405"] = 0.020000*np.tanh(((((((((data["var_67"]) * (data["var_67"]))) + (((((data["var_11"]) * 2.0)) * (data["var_11"]))))/2.0)) + (0.318310))/2.0)) 

        v["i406"] = 0.020000*np.tanh(((((((((((data["var_5"]) * (data["var_5"]))) * (data["var_5"]))) / 2.0)) - (data["var_5"]))) * (data["var_5"]))) 

        v["i407"] = 0.020000*np.tanh((((((data["var_94"]) * (data["var_94"]))) + (((np.tanh(((-1.0*((((data["var_94"]) * (data["var_94"])))))))) * 2.0)))/2.0)) 

        v["i408"] = 0.020000*np.tanh(((((data["var_156"]) * (((((data["var_156"]) * (((data["var_156"]) * (data["var_156"]))))) / 2.0)))) + (np.tanh((data["var_156"]))))) 

        v["i409"] = 0.020000*np.tanh(((((((((data["var_169"]) * (data["var_169"]))) + (data["var_169"]))/2.0)) + (((((data["var_106"]) / 2.0)) * (data["var_106"]))))/2.0)) 

        v["i410"] = 0.020000*np.tanh(((np.tanh(((-1.0*((((data["var_127"]) * (data["var_127"])))))))) + ((((data["var_187"]) + (((data["var_127"]) * (data["var_127"]))))/2.0)))) 

        v["i411"] = 0.020000*np.tanh(((data["var_168"]) * ((((((data["var_168"]) * (((data["var_168"]) * (data["var_168"]))))) + ((-1.0*(((1.79895091056823730))))))/2.0)))) 

        v["i412"] = 0.020000*np.tanh((((((((((((data["var_168"]) + (data["var_168"]))/2.0)) + (data["var_162"]))/2.0)) + (((data["var_125"]) * (data["var_125"]))))/2.0)) / 2.0)) 

        v["i413"] = 0.020000*np.tanh(((np.tanh(((((data["var_77"]) + ((((((data["var_77"]) + (data["var_77"]))/2.0)) * 2.0)))/2.0)))) + (((data["var_77"]) * (data["var_77"]))))) 

        v["i414"] = 0.020000*np.tanh((((((((((data["var_107"]) * (((data["var_107"]) + (data["var_107"]))))) + (((data["var_95"]) - (data["var_77"]))))/2.0)) / 2.0)) / 2.0)) 

        v["i415"] = 0.020000*np.tanh(((((((data["var_90"]) + (((data["var_193"]) * (data["var_193"]))))/2.0)) + (np.tanh(((((data["var_126"]) + (-1.0))/2.0)))))/2.0)) 

        v["i416"] = 0.020000*np.tanh(((((((data["var_83"]) * (data["var_83"]))) + (((np.tanh((data["var_83"]))) / 2.0)))) + (data["var_83"]))) 

        v["i417"] = 0.017802*np.tanh(((((((((3.0) + (((data["var_20"]) * 2.0)))/2.0)) * (((((((data["var_20"]) * 2.0)) * 2.0)) * 2.0)))) + (data["var_20"]))/2.0)) 

        v["i418"] = 0.020000*np.tanh(((data["var_184"]) * (((((data["var_184"]) * (((data["var_184"]) / 2.0)))) + (np.tanh(((-1.0*((data["var_184"])))))))))) 

        v["i419"] = 0.020000*np.tanh((((((((data["var_62"]) * (data["var_62"]))) + ((((((data["var_83"]) * (data["var_83"]))) + (data["var_4"]))/2.0)))/2.0)) / 2.0)) 

        v["i420"] = 0.020000*np.tanh(((((((data["var_63"]) + (data["var_170"]))/2.0)) + (((((data["var_63"]) * (data["var_63"]))) + (data["var_63"]))))/2.0)) 

        v["i421"] = 0.020000*np.tanh((((((3.0) + (data["var_96"]))/2.0)) / 2.0)) 

        v["i422"] = 0.020000*np.tanh(((((data["var_82"]) * (((data["var_82"]) / 2.0)))) * (((((((((data["var_82"]) / 2.0)) / 2.0)) * (data["var_82"]))) / 2.0)))) 

        v["i423"] = 0.020000*np.tanh((((((((data["var_95"]) * (((data["var_95"]) * (data["var_95"]))))) + ((-1.0*((data["var_95"])))))/2.0)) + ((-1.0*((data["var_95"])))))) 

        v["i424"] = 0.019990*np.tanh(((((((data["var_95"]) + ((((data["var_111"]) + (data["var_60"]))/2.0)))/2.0)) + (np.tanh((np.tanh((((((data["var_86"]) * 2.0)) * 2.0)))))))/2.0)) 

        v["i425"] = 0.020000*np.tanh(((((((((data["var_184"]) * (data["var_184"]))) + (data["var_170"]))/2.0)) + (np.tanh((np.tanh((np.tanh((((data["var_81"]) * 2.0)))))))))/2.0)) 

        v["i426"] = 0.020000*np.tanh((((((data["var_51"]) * (((data["var_51"]) * (((((data["var_51"]) * (data["var_51"]))) / 2.0)))))) + ((-1.0*((data["var_51"])))))/2.0)) 

        v["i427"] = 0.020000*np.tanh((((data["var_76"]) + (((data["var_76"]) * (((((((data["var_139"]) + (data["var_76"]))/2.0)) + (data["var_76"]))/2.0)))))/2.0)) 

        v["i428"] = 0.020000*np.tanh((((((((((data["var_71"]) * (data["var_71"]))) + (data["var_133"]))/2.0)) - (1.0))) / 2.0)) 

        v["i429"] = 0.020000*np.tanh(((data["var_119"]) * ((((((((((data["var_119"]) * (data["var_119"]))) * (data["var_119"]))) / 2.0)) + (-2.0))/2.0)))) 

        v["i430"] = 0.020000*np.tanh((((data["var_119"]) + (((((((data["var_22"]) / 2.0)) * (((((data["var_22"]) / 2.0)) * (data["var_22"]))))) * (data["var_22"]))))/2.0)) 

        v["i431"] = 0.020000*np.tanh(((((((((((((data["var_18"]) - ((((2.96681714057922363)) / 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - ((2.96681714057922363)))) 

        v["i432"] = 0.020000*np.tanh(((data["var_56"]) * ((((data["var_56"]) + (((((4.0)) + ((((((data["var_56"]) / 2.0)) + (data["var_56"]))/2.0)))/2.0)))/2.0)))) 

        v["i433"] = 0.020000*np.tanh((((data["var_141"]) + ((((((((data["var_141"]) + (((data["var_141"]) * (data["var_141"]))))/2.0)) * (data["var_141"]))) * (data["var_141"]))))/2.0)) 

        v["i434"] = 0.020000*np.tanh(((((((((((((data["var_190"]) / 2.0)) * (data["var_190"]))) / 2.0)) / 2.0)) * (((data["var_190"]) * (data["var_190"]))))) / 2.0)) 

        v["i435"] = 0.020000*np.tanh(((((((data["var_99"]) / 2.0)) * (((data["var_99"]) / 2.0)))) + (np.tanh((((data["var_91"]) + (-2.0))))))) 

        v["i436"] = 0.020000*np.tanh(((((((data["var_157"]) * (((((data["var_157"]) / 2.0)) * (((data["var_157"]) * (((data["var_157"]) / 2.0)))))))) / 2.0)) / 2.0)) 

        v["i437"] = 0.020000*np.tanh((((((data["var_127"]) * (((data["var_127"]) * ((((data["var_127"]) + (((data["var_127"]) * (data["var_127"]))))/2.0)))))) + (data["var_127"]))/2.0)) 

        v["i438"] = 0.020000*np.tanh(((((4.0)) + (((3.0) * 2.0)))/2.0)) 

        v["i439"] = 0.020000*np.tanh(((((((data["var_21"]) - (np.tanh((data["var_21"]))))) * (data["var_21"]))) + (np.tanh((data["var_21"]))))) 

        v["i440"] = 0.020000*np.tanh(((((((data["var_199"]) + (data["var_199"]))) * (((((data["var_199"]) * (data["var_199"]))) + (-1.0))))) * 2.0)) 

        v["i441"] = 0.020000*np.tanh(((((((((data["var_81"]) * (data["var_109"]))) + (((data["var_19"]) * (data["var_19"]))))/2.0)) + (np.tanh((data["var_109"]))))/2.0)) 

        v["i442"] = 0.017499*np.tanh(((((((((data["var_95"]) / 2.0)) * (data["var_95"]))) / 2.0)) * (((((data["var_95"]) / 2.0)) * (((data["var_95"]) / 2.0)))))) 

        v["i443"] = 0.020000*np.tanh((((((data["var_165"]) + (((data["var_165"]) * (((((data["var_165"]) * (((data["var_165"]) * (data["var_165"]))))) / 2.0)))))/2.0)) / 2.0)) 

        v["i444"] = 0.020000*np.tanh(((((((((((((data["var_40"]) * (data["var_40"]))) * (data["var_40"]))) / 2.0)) * (data["var_40"]))) - (data["var_40"]))) / 2.0)) 

        v["i445"] = 0.020000*np.tanh(((((((data["var_110"]) + ((((data["var_125"]) + (data["var_90"]))/2.0)))/2.0)) + (((((((data["var_8"]) + (data["var_196"]))/2.0)) + (data["var_40"]))/2.0)))/2.0)) 

        v["i446"] = 0.020000*np.tanh(((((((((-1.0*((((data["var_139"]) * (data["var_139"])))))) * (data["var_139"]))) + (data["var_139"]))/2.0)) + (data["var_139"]))) 

        v["i447"] = 0.020000*np.tanh(((((((data["var_122"]) * (data["var_122"]))) * ((((data["var_122"]) + (((data["var_122"]) * (data["var_122"]))))/2.0)))) / 2.0)) 

        v["i448"] = 0.019980*np.tanh(((((((((data["var_0"]) + (((data["var_139"]) * (data["var_139"]))))/2.0)) / 2.0)) + ((((data["var_97"]) + (((data["var_157"]) / 2.0)))/2.0)))/2.0)) 

        v["i449"] = 0.020000*np.tanh(((((data["var_174"]) * (((data["var_174"]) / 2.0)))) + (np.tanh(((-1.0*((((data["var_174"]) * (data["var_174"])))))))))) 

        v["i450"] = 0.020000*np.tanh(((((((((-2.0) + (data["var_128"]))/2.0)) + (((data["var_5"]) * ((((((data["var_147"]) / 2.0)) + (data["var_5"]))/2.0)))))/2.0)) / 2.0)) 

        v["i451"] = 0.020000*np.tanh((((np.tanh((np.tanh(((-1.0*((((2.0) - (data["var_164"])))))))))) + (((data["var_106"]) * (((data["var_106"]) / 2.0)))))/2.0)) 

        v["i452"] = 0.020000*np.tanh((((((data["var_85"]) * ((((data["var_85"]) + ((((data["var_81"]) + (data["var_85"]))/2.0)))/2.0)))) + ((((data["var_100"]) + (data["var_85"]))/2.0)))/2.0)) 

        v["i453"] = 0.020000*np.tanh(np.tanh((np.tanh((((((data["var_198"]) * (data["var_198"]))) + (((data["var_198"]) * 2.0)))))))) 

        v["i454"] = 0.019990*np.tanh((((((((((data["var_6"]) * (data["var_11"]))) / 2.0)) + (((data["var_37"]) * (data["var_37"]))))/2.0)) / 2.0)) 

        v["i455"] = 0.019980*np.tanh((((((data["var_1"]) + (((((data["var_2"]) * (data["var_2"]))) / 2.0)))/2.0)) * ((((((data["var_2"]) + (data["var_1"]))/2.0)) / 2.0)))) 

        v["i456"] = 0.020000*np.tanh((((((3.0) / 2.0)) + (((((((1.0) * (data["var_53"]))) / 2.0)) * (data["var_53"]))))/2.0)) 

        v["i457"] = 0.020000*np.tanh((((((np.tanh(((((((-1.0*((data["var_78"])))) - (data["var_78"]))) - (data["var_78"]))))) + (data["var_78"]))/2.0)) * (data["var_78"]))) 

        v["i458"] = 0.020000*np.tanh(((((((data["var_18"]) + (((np.tanh((((data["var_18"]) * ((-1.0*((data["var_18"])))))))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i459"] = 0.020000*np.tanh((((((((data["var_115"]) / 2.0)) * (data["var_115"]))) + ((((data["var_88"]) + (((data["var_88"]) * (data["var_88"]))))/2.0)))/2.0)) 

        v["i460"] = 0.020000*np.tanh(((((-1.0*((data["var_49"])))) + (((((((data["var_49"]) / 2.0)) * (data["var_49"]))) * (data["var_49"]))))/2.0)) 

        v["i461"] = 0.020000*np.tanh(((data["var_166"]) * (((data["var_166"]) + (((np.tanh((((np.tanh((np.tanh(((-1.0*((data["var_166"])))))))) * 2.0)))) * 2.0)))))) 

        v["i462"] = 0.020000*np.tanh((((((((data["var_50"]) / 2.0)) * (((((((data["var_50"]) / 2.0)) / 2.0)) / 2.0)))) + (((data["var_50"]) * (data["var_50"]))))/2.0)) 

        v["i463"] = 0.020000*np.tanh((((((((data["var_194"]) * (data["var_194"]))) + (((data["var_174"]) * (data["var_174"]))))/2.0)) + (((np.tanh((data["var_174"]))) * 2.0)))) 

        v["i464"] = 0.020000*np.tanh((((((data["var_89"]) + (((((data["var_144"]) * (data["var_144"]))) + (((data["var_144"]) * (data["var_144"]))))))/2.0)) / 2.0)) 

        v["i465"] = 0.020000*np.tanh((((((((data["var_4"]) * (data["var_4"]))) + ((((data["var_15"]) + ((((data["var_118"]) + (data["var_118"]))/2.0)))/2.0)))/2.0)) / 2.0)) 

        v["i466"] = 0.020000*np.tanh((((((((data["var_49"]) * (data["var_49"]))) + (((data["var_130"]) * (data["var_130"]))))/2.0)) / 2.0)) 

        v["i467"] = 0.020000*np.tanh((((((((data["var_190"]) * (data["var_190"]))) + ((((data["var_15"]) + (((data["var_184"]) * (data["var_184"]))))/2.0)))/2.0)) / 2.0)) 

        v["i468"] = 0.020000*np.tanh((((((((data["var_47"]) + (((((data["var_62"]) * (data["var_62"]))) * (((data["var_62"]) * (data["var_62"]))))))/2.0)) / 2.0)) / 2.0)) 

        v["i469"] = 0.020000*np.tanh(((data["var_18"]) * (((((data["var_18"]) * (((data["var_18"]) * (((((data["var_18"]) / 2.0)) / 2.0)))))) - (data["var_18"]))))) 

        v["i470"] = 0.020000*np.tanh((((np.tanh((np.tanh((np.tanh((((data["var_113"]) * 2.0)))))))) + (((((data["var_35"]) * (data["var_35"]))) / 2.0)))/2.0)) 

        v["i471"] = 0.020000*np.tanh(((data["var_147"]) * (((((data["var_147"]) * (((((((data["var_147"]) * (((data["var_147"]) / 2.0)))) / 2.0)) / 2.0)))) / 2.0)))) 

        v["i472"] = 0.020000*np.tanh(((((data["var_135"]) * (((((data["var_135"]) * (data["var_135"]))) / 2.0)))) - (data["var_135"]))) 

        v["i473"] = 0.020000*np.tanh((((((data["var_170"]) + ((((((data["var_76"]) * (data["var_76"]))) + (((data["var_76"]) / 2.0)))/2.0)))/2.0)) / 2.0)) 

        v["i474"] = 0.020000*np.tanh((-1.0*((((data["var_63"]) - (((data["var_63"]) + (((((-2.0) - (data["var_108"]))) * 2.0))))))))) 

        v["i475"] = 0.019990*np.tanh(((((-1.0*((data["var_179"])))) + (((((((data["var_179"]) / 2.0)) * (data["var_179"]))) * (((data["var_179"]) * (data["var_179"]))))))/2.0)) 

        v["i476"] = 0.020000*np.tanh(((((((-3.0) + (((((((((-3.0) + (((data["var_164"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) 

        v["i477"] = 0.020000*np.tanh((((((((data["var_184"]) + (data["var_2"]))/2.0)) * ((((data["var_2"]) + ((((data["var_184"]) + (((data["var_184"]) / 2.0)))/2.0)))/2.0)))) / 2.0)) 

        v["i478"] = 0.020000*np.tanh(((((data["var_75"]) / 2.0)) * ((((((((((data["var_75"]) + (data["var_75"]))/2.0)) / 2.0)) / 2.0)) + (((data["var_75"]) / 2.0)))))) 

        v["i479"] = 0.019990*np.tanh((((1.0) + ((((np.tanh((data["var_80"]))) + ((((data["var_99"]) + (np.tanh((np.tanh((np.tanh((data["var_23"]))))))))/2.0)))/2.0)))/2.0)) 

        v["i480"] = 0.020000*np.tanh((((((((((data["var_22"]) * (data["var_22"]))) / 2.0)) * (data["var_22"]))) + (((data["var_115"]) - (data["var_22"]))))/2.0)) 

        v["i481"] = 0.020000*np.tanh((((((data["var_44"]) * ((((data["var_44"]) + (((data["var_44"]) / 2.0)))/2.0)))) + ((((data["var_44"]) + (np.tanh((data["var_44"]))))/2.0)))/2.0)) 

        v["i482"] = 0.020000*np.tanh(((((((data["var_135"]) * (((((data["var_135"]) * (((data["var_135"]) * (data["var_135"]))))) / 2.0)))) / 2.0)) / 2.0)) 

        v["i483"] = 0.020000*np.tanh((((((((data["var_1"]) * (data["var_22"]))) + (((data["var_26"]) * (((data["var_26"]) * (((data["var_26"]) / 2.0)))))))/2.0)) / 2.0)) 

        v["i484"] = 0.020000*np.tanh(((((((((np.tanh((((((((-2.0) + (data["var_94"]))) * 2.0)) * 2.0)))) * 2.0)) + (data["var_49"]))) * 2.0)) * 2.0)) 

        v["i485"] = 0.020000*np.tanh((((((((data["var_21"]) * (data["var_21"]))) + (data["var_21"]))/2.0)) * (((data["var_21"]) * (((((data["var_21"]) / 2.0)) / 2.0)))))) 

        v["i486"] = 0.020000*np.tanh((((((((((data["var_18"]) * (((data["var_18"]) / 2.0)))) + (((data["var_51"]) * (data["var_51"]))))) + (data["var_161"]))/2.0)) / 2.0)) 

        v["i487"] = 0.019990*np.tanh(((((((((((data["var_176"]) * (data["var_176"]))) + (data["var_18"]))/2.0)) + (((data["var_178"]) * (data["var_178"]))))/2.0)) / 2.0)) 

        v["i488"] = 0.019990*np.tanh(((data["var_57"]) * ((((data["var_12"]) + (((((((data["var_57"]) * (data["var_12"]))) * (data["var_12"]))) / 2.0)))/2.0)))) 

        v["i489"] = 0.020000*np.tanh(((data["var_163"]) * ((((data["var_163"]) + (-1.0))/2.0)))) 

        v["i490"] = 0.019971*np.tanh((((((((data["var_163"]) + (((data["var_61"]) * (data["var_61"]))))) + (((((data["var_46"]) * (data["var_46"]))) / 2.0)))/2.0)) / 2.0)) 

        v["i491"] = 0.020000*np.tanh(np.tanh((np.tanh((((data["var_134"]) - (((np.tanh((((np.tanh((((data["var_122"]) - (-2.0))))) * 2.0)))) * 2.0)))))))) 

        v["i492"] = 0.020000*np.tanh(((((-1.0*((data["var_190"])))) + (((((((data["var_190"]) * (data["var_190"]))) / 2.0)) * (data["var_190"]))))/2.0)) 

        v["i493"] = 0.020000*np.tanh(((((data["var_181"]) * ((((data["var_181"]) + (np.tanh((data["var_190"]))))/2.0)))) / 2.0)) 

        v["i494"] = 0.020000*np.tanh(((((((((((data["var_55"]) * (data["var_55"]))) - ((1.0)))) - ((1.0)))) - ((1.0)))) / 2.0)) 

        v["i495"] = 0.020000*np.tanh(((((((((((((((data["var_137"]) * (data["var_137"]))) / 2.0)) / 2.0)) * (data["var_137"]))) * (data["var_137"]))) / 2.0)) / 2.0)) 

        v["i496"] = 0.020000*np.tanh(((((data["var_199"]) * ((((((((((data["var_199"]) * (data["var_199"]))) + (data["var_199"]))/2.0)) / 2.0)) * (data["var_199"]))))) / 2.0)) 

        v["i497"] = 0.020000*np.tanh(((((data["var_131"]) * (data["var_131"]))) * (((-1.0) + (((((((data["var_131"]) / 2.0)) / 2.0)) * (data["var_131"]))))))) 

        v["i498"] = 0.020000*np.tanh(((((((data["var_177"]) * (data["var_177"]))) / 2.0)) * (((((((data["var_177"]) / 2.0)) / 2.0)) * (((data["var_177"]) / 2.0)))))) 

        v["i499"] = 0.020000*np.tanh(((data["var_53"]) * (((((data["var_53"]) - (np.tanh((data["var_53"]))))) - (np.tanh((data["var_53"]))))))) 

        v["i500"] = 0.020000*np.tanh(((((((((((data["var_33"]) / 2.0)) * (data["var_33"]))) / 2.0)) * (((data["var_33"]) * (((data["var_33"]) / 2.0)))))) / 2.0)) 

        v["i501"] = 0.020000*np.tanh((((-2.0) + ((((((((-2.0) + (((data["var_134"]) * (data["var_134"]))))/2.0)) * (data["var_134"]))) * (data["var_134"]))))/2.0)) 

        v["i502"] = 0.020000*np.tanh(((data["var_63"]) * (((((((data["var_63"]) * ((((1.0) + (data["var_63"]))/2.0)))) / 2.0)) * (data["var_63"]))))) 

        v["i503"] = 0.020000*np.tanh((((((((-1.0) + (data["var_25"]))) + (data["var_25"]))/2.0)) * ((((((data["var_25"]) / 2.0)) + (data["var_25"]))/2.0)))) 

        v["i504"] = 0.020000*np.tanh(((data["var_67"]) * (((data["var_67"]) + (np.tanh(((((-1.0*((((((data["var_67"]) * 2.0)) - (data["var_180"])))))) * 2.0)))))))) 

        v["i505"] = 0.020000*np.tanh(((((((-1.0) + (((data["var_123"]) * (data["var_123"]))))/2.0)) + (np.tanh((data["var_123"]))))/2.0)) 

        v["i506"] = 0.020000*np.tanh(((((((data["var_119"]) + (data["var_134"]))/2.0)) + ((((data["var_133"]) + ((((-2.0) + (((data["var_190"]) + (data["var_160"]))))/2.0)))/2.0)))/2.0)) 

        v["i507"] = 0.020000*np.tanh(((((((data["var_130"]) + (((np.tanh((((((((data["var_49"]) + (-2.0))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i508"] = 0.019922*np.tanh(((data["var_68"]) * (((np.tanh(((2.70949316024780273)))) + (data["var_68"]))))) 

        v["i509"] = 0.020000*np.tanh(((((((((data["var_8"]) * (((data["var_8"]) * (data["var_8"]))))) * (((data["var_8"]) * (data["var_8"]))))) / 2.0)) / 2.0)) 

        v["i510"] = 0.020000*np.tanh(((((((data["var_125"]) * (((data["var_125"]) * (((data["var_125"]) / 2.0)))))) * (((((data["var_125"]) / 2.0)) / 2.0)))) / 2.0)) 

        v["i511"] = 0.020000*np.tanh(((((((data["var_133"]) * (((data["var_133"]) * (((((((data["var_133"]) / 2.0)) * (data["var_133"]))) / 2.0)))))) / 2.0)) / 2.0)) 

        v["i512"] = 0.019873*np.tanh((((((data["var_133"]) + (((((((((data["var_91"]) * (data["var_91"]))) * (data["var_91"]))) * (data["var_91"]))) / 2.0)))/2.0)) / 2.0))

        if(cols!=-1):

            return v[v.columns[:cols]].sum(axis=1)

        return v

    

    def GP_class_n(self,data,cols):

        v = pd.DataFrame()

        v["i0"] = -0.124263 * np.ones(data.shape[0])

        v["i1"] = 0.020000*np.tanh((((13.27900791168212891)) + (((((((data["var_174"]) + (((data["var_174"]) + (((data["var_146"]) * 2.0)))))) + (data["var_169"]))) * 2.0)))) 

        v["i2"] = 0.020000*np.tanh((((9.0)) + (((((((data["var_139"]) + (((data["var_146"]) + (((data["var_81"]) * 2.0)))))) * 2.0)) + (data["var_166"]))))) 

        v["i3"] = 0.020000*np.tanh(((((((data["var_81"]) * 2.0)) * 2.0)) + ((8.0)))) 

        v["i4"] = 0.020000*np.tanh(((((((((data["var_12"]) * 2.0)) + (((((((data["var_81"]) * 2.0)) + ((((8.0)) / 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i5"] = 0.020000*np.tanh(((((data["var_12"]) * 2.0)) + ((8.0)))) 

        v["i6"] = 0.020000*np.tanh((((((14.64460372924804688)) + ((((9.0)) + (data["var_166"]))))) + (((data["var_174"]) * ((14.64460372924804688)))))) 

        v["i7"] = 0.020000*np.tanh((((((8.67539978027343750)) + ((7.56956768035888672)))) + (data["var_166"]))) 

        v["i8"] = 0.020000*np.tanh((((7.99196910858154297)) + ((((((8.04000568389892578)) * (data["var_81"]))) + ((3.77778506278991699)))))) 

        v["i9"] = 0.020000*np.tanh(((((data["var_12"]) * ((8.0)))) + ((12.89955425262451172)))) 

        v["i10"] = 0.020000*np.tanh(((((((((6.54421567916870117)) + ((7.0)))/2.0)) + (((((data["var_139"]) * 2.0)) * 2.0)))) + (data["var_12"]))) 

        v["i11"] = 0.020000*np.tanh(((((((data["var_81"]) * 2.0)) + (data["var_76"]))) + ((((data["var_81"]) + (((((10.0)) + ((6.0)))/2.0)))/2.0)))) 

        v["i12"] = 0.020000*np.tanh(((((((((((3.0) + (data["var_139"]))) + (data["var_81"]))) + (data["var_198"]))) + (data["var_12"]))) + (data["var_174"]))) 

        v["i13"] = 0.020000*np.tanh(((((data["var_12"]) + (((data["var_146"]) + (((data["var_80"]) + (((data["var_81"]) + (2.0))))))))) + (data["var_139"]))) 

        v["i14"] = 0.020000*np.tanh(((((((((((data["var_146"]) + (2.0))) + (data["var_148"]))) + (data["var_76"]))) - (data["var_190"]))) + (data["var_174"]))) 

        v["i15"] = 0.020000*np.tanh(((((((data["var_76"]) - (data["var_110"]))) + (((data["var_80"]) + (((data["var_146"]) + (data["var_165"]))))))) + (2.0))) 

        v["i16"] = 0.020000*np.tanh(((((((1.0) + (((data["var_21"]) + (data["var_166"]))))) + (((data["var_139"]) + (data["var_12"]))))) + (data["var_76"]))) 

        v["i17"] = 0.020000*np.tanh(((1.0) + (((data["var_146"]) + (((((data["var_76"]) + (data["var_139"]))) + (((data["var_165"]) + (data["var_81"]))))))))) 

        v["i18"] = 0.020000*np.tanh(((((((data["var_44"]) - (((data["var_6"]) - (data["var_21"]))))) + (data["var_174"]))) + (((data["var_34"]) + (1.0))))) 

        v["i19"] = 0.020000*np.tanh(((((data["var_13"]) + (((data["var_139"]) + (((data["var_80"]) - (data["var_133"]))))))) + (((data["var_21"]) - (data["var_2"]))))) 

        v["i20"] = 0.020000*np.tanh(((((data["var_148"]) + (data["var_166"]))) + (((data["var_165"]) + (((((data["var_149"]) + (data["var_174"]))) + (data["var_169"]))))))) 

        v["i21"] = 0.020000*np.tanh(((data["var_139"]) + (((data["var_192"]) + (((((((data["var_76"]) + (data["var_33"]))) + (data["var_12"]))) - (data["var_184"]))))))) 

        v["i22"] = 0.020000*np.tanh(((data["var_76"]) + (((((data["var_108"]) + (data["var_174"]))) + (((((data["var_146"]) + (data["var_198"]))) + (data["var_139"]))))))) 

        v["i23"] = 0.020000*np.tanh(((((((data["var_166"]) + (((((data["var_109"]) + (data["var_198"]))) + (data["var_21"]))))) + (data["var_169"]))) + (data["var_149"]))) 

        v["i24"] = 0.020000*np.tanh(((data["var_81"]) + (((((((data["var_115"]) + (data["var_109"]))) + (data["var_12"]))) + (((data["var_122"]) + (data["var_154"]))))))) 

        v["i25"] = 0.020000*np.tanh(((data["var_44"]) + (((((((data["var_148"]) + (((data["var_13"]) + (data["var_80"]))))) - (data["var_179"]))) + (data["var_198"]))))) 

        v["i26"] = 0.020000*np.tanh(((((((data["var_108"]) + (data["var_81"]))) + (((((data["var_198"]) + (data["var_192"]))) + (data["var_165"]))))) + (data["var_9"]))) 

        v["i27"] = 0.020000*np.tanh(((data["var_192"]) + (((data["var_34"]) + (((data["var_13"]) + (((data["var_122"]) + (((data["var_148"]) + (data["var_146"]))))))))))) 

        v["i28"] = 0.020000*np.tanh(((((data["var_169"]) + (((((data["var_44"]) - (data["var_6"]))) + (((data["var_139"]) + (data["var_174"]))))))) + (data["var_127"]))) 

        v["i29"] = 0.020000*np.tanh(((data["var_34"]) + (((data["var_108"]) + (((data["var_172"]) + (((data["var_92"]) + (((data["var_80"]) + (data["var_149"]))))))))))) 

        v["i30"] = 0.020000*np.tanh(((((data["var_148"]) + (data["var_115"]))) + (((data["var_166"]) + (((data["var_107"]) + (((data["var_33"]) + (data["var_174"]))))))))) 

        v["i31"] = 0.020000*np.tanh(((data["var_12"]) + (((((data["var_146"]) + (((data["var_21"]) + (data["var_154"]))))) - (((data["var_2"]) - (data["var_9"]))))))) 

        v["i32"] = 0.020000*np.tanh(((data["var_165"]) + (((data["var_172"]) + (((data["var_76"]) + (((data["var_92"]) + (((data["var_169"]) + (data["var_166"]))))))))))) 

        v["i33"] = 0.020000*np.tanh(((((data["var_139"]) + (data["var_121"]))) + (((((((data["var_149"]) + (data["var_44"]))) + (data["var_198"]))) + (data["var_13"]))))) 

        v["i34"] = 0.020000*np.tanh(((data["var_34"]) + (((((data["var_115"]) + (((data["var_107"]) + (data["var_36"]))))) + (((data["var_127"]) + (data["var_122"]))))))) 

        v["i35"] = 0.020000*np.tanh(((((((data["var_87"]) + (data["var_123"]))) + (((data["var_75"]) + (data["var_76"]))))) + (((data["var_146"]) + (data["var_81"]))))) 

        v["i36"] = 0.020000*np.tanh(((((data["var_177"]) + (data["var_108"]))) - (((((data["var_110"]) - (((data["var_115"]) + (data["var_92"]))))) - (data["var_9"]))))) 

        v["i37"] = 0.020000*np.tanh(((((data["var_174"]) + (data["var_87"]))) + (((((data["var_12"]) + (data["var_148"]))) + (((data["var_33"]) + (data["var_165"]))))))) 

        v["i38"] = 0.020000*np.tanh(((data["var_139"]) + (((data["var_154"]) + (((data["var_172"]) + (((((data["var_86"]) + (data["var_44"]))) + (data["var_149"]))))))))) 

        v["i39"] = 0.020000*np.tanh(((((data["var_13"]) + (((data["var_169"]) + (data["var_107"]))))) + (((data["var_93"]) + (((data["var_109"]) + (data["var_192"]))))))) 

        v["i40"] = 0.020000*np.tanh(((((data["var_36"]) + (data["var_148"]))) - (((data["var_40"]) - (((((data["var_165"]) + (data["var_166"]))) + (data["var_197"]))))))) 

        v["i41"] = 0.020000*np.tanh(((((((data["var_146"]) + (data["var_56"]))) + (((data["var_86"]) + (data["var_36"]))))) + (((data["var_81"]) + (data["var_92"]))))) 

        v["i42"] = 0.020000*np.tanh(((((((((data["var_166"]) + (data["var_127"]))) + (data["var_177"]))) + (((data["var_9"]) + (data["var_174"]))))) + (data["var_12"]))) 

        v["i43"] = 0.020000*np.tanh(((((((data["var_76"]) + (data["var_75"]))) - (data["var_133"]))) + (((((data["var_148"]) + (data["var_121"]))) + (data["var_108"]))))) 

        v["i44"] = 0.020000*np.tanh(((((data["var_44"]) + (data["var_80"]))) + (((data["var_127"]) + (((((data["var_123"]) + (data["var_188"]))) + (data["var_139"]))))))) 

        v["i45"] = 0.020000*np.tanh(((data["var_86"]) + (((data["var_33"]) + (((data["var_107"]) + (((data["var_146"]) + (((data["var_165"]) + (data["var_13"]))))))))))) 

        v["i46"] = 0.020000*np.tanh(((((data["var_21"]) + (((data["var_192"]) + (data["var_169"]))))) + (((data["var_34"]) + (((data["var_36"]) + (data["var_121"]))))))) 

        v["i47"] = 0.020000*np.tanh(((data["var_87"]) + (((((data["var_12"]) + (data["var_81"]))) + (((((data["var_122"]) - (data["var_71"]))) + (data["var_75"]))))))) 

        v["i48"] = 0.020000*np.tanh(((((data["var_109"]) - (((data["var_26"]) - (data["var_154"]))))) + (((data["var_169"]) + (((data["var_80"]) + (data["var_198"]))))))) 

        v["i49"] = 0.020000*np.tanh(((((data["var_131"]) + (data["var_186"]))) + (((((data["var_148"]) + (((data["var_9"]) + (data["var_31"]))))) + (data["var_93"]))))) 

        v["i50"] = 0.020000*np.tanh(((data["var_122"]) + (((((data["var_127"]) + (data["var_177"]))) + (((((data["var_108"]) - (data["var_0"]))) - (data["var_147"]))))))) 

        v["i51"] = 0.020000*np.tanh(((((data["var_192"]) + (data["var_81"]))) + (((data["var_13"]) + (((data["var_23"]) + (((data["var_139"]) + (data["var_149"]))))))))) 

        v["i52"] = 0.020000*np.tanh(((data["var_188"]) + (((data["var_92"]) + (((((((data["var_80"]) + (data["var_172"]))) + (data["var_107"]))) + (data["var_34"]))))))) 

        v["i53"] = 0.020000*np.tanh(((data["var_154"]) + (((((data["var_87"]) + (data["var_197"]))) + (((data["var_56"]) + (((data["var_146"]) + (data["var_141"]))))))))) 

        v["i54"] = 0.020000*np.tanh(((((data["var_86"]) + (((data["var_76"]) + (data["var_148"]))))) + (((((data["var_75"]) - (data["var_24"]))) + (data["var_139"]))))) 

        v["i55"] = 0.020000*np.tanh(((data["var_123"]) - (((data["var_95"]) - (((data["var_58"]) + (((((data["var_121"]) + (data["var_44"]))) - (data["var_147"]))))))))) 

        v["i56"] = 0.020000*np.tanh(((data["var_109"]) + (((((data["var_36"]) + (data["var_104"]))) + (((data["var_192"]) + (((data["var_93"]) + (data["var_122"]))))))))) 

        v["i57"] = 0.020000*np.tanh(((((((data["var_31"]) + (((((data["var_165"]) + (data["var_174"]))) + (data["var_12"]))))) + (data["var_169"]))) + (data["var_76"]))) 

        v["i58"] = 0.020000*np.tanh(((data["var_150"]) + (((((data["var_186"]) + (((data["var_149"]) - (data["var_40"]))))) + (((data["var_92"]) + (data["var_21"]))))))) 

        v["i59"] = 0.020000*np.tanh(((data["var_166"]) + (((data["var_87"]) + (((data["var_13"]) + (((((data["var_23"]) + (data["var_115"]))) + (data["var_44"]))))))))) 

        v["i60"] = 0.020000*np.tanh(((data["var_9"]) + (((data["var_132"]) + (((((data["var_197"]) + (((data["var_34"]) + (data["var_131"]))))) - (data["var_128"]))))))) 

        v["i61"] = 0.020000*np.tanh(((((((data["var_188"]) + (((data["var_109"]) + (((data["var_172"]) + (data["var_107"]))))))) + (data["var_12"]))) + (data["var_80"]))) 

        v["i62"] = 0.020000*np.tanh(((((((data["var_36"]) + (data["var_108"]))) + (data["var_141"]))) + (((((data["var_28"]) + (data["var_43"]))) + (data["var_146"]))))) 

        v["i63"] = 0.020000*np.tanh(((((data["var_123"]) + (((data["var_104"]) + (((((data["var_186"]) + (data["var_154"]))) + (data["var_198"]))))))) + (data["var_122"]))) 

        v["i64"] = 0.020000*np.tanh(((((((data["var_93"]) + (data["var_31"]))) - (((data["var_157"]) - (data["var_139"]))))) + (((data["var_33"]) - (data["var_184"]))))) 

        v["i65"] = 0.020000*np.tanh(((((data["var_166"]) + (((data["var_149"]) + (((((data["var_177"]) + (data["var_121"]))) + (data["var_75"]))))))) + (data["var_108"]))) 

        v["i66"] = 0.020000*np.tanh(((((data["var_58"]) + (((data["var_86"]) + (data["var_192"]))))) + (((data["var_115"]) + (((data["var_169"]) + (data["var_127"]))))))) 

        v["i67"] = 0.020000*np.tanh(((data["var_76"]) + (((data["var_34"]) + (((((data["var_44"]) - (data["var_24"]))) - (((data["var_170"]) - (data["var_131"]))))))))) 

        v["i68"] = 0.020000*np.tanh(((data["var_172"]) + (((data["var_9"]) + (((data["var_188"]) + (((data["var_85"]) + (((data["var_81"]) + (data["var_43"]))))))))))) 

        v["i69"] = 0.020000*np.tanh(((((data["var_165"]) - (((data["var_26"]) + (data["var_35"]))))) - (((((data["var_112"]) - (data["var_23"]))) - (data["var_13"]))))) 

        v["i70"] = 0.020000*np.tanh(((((data["var_92"]) + (((data["var_114"]) + (((data["var_104"]) + (data["var_28"]))))))) + (((data["var_93"]) + (data["var_123"]))))) 

        v["i71"] = 0.020000*np.tanh(((((data["var_80"]) + (((data["var_109"]) + (((data["var_148"]) + (((data["var_33"]) + (data["var_174"]))))))))) + (data["var_56"]))) 

        v["i72"] = 0.020000*np.tanh(((data["var_169"]) + (((((data["var_150"]) + (data["var_198"]))) - (((data["var_130"]) - (((data["var_121"]) + (data["var_107"]))))))))) 

        v["i73"] = 0.020000*np.tanh(((((data["var_146"]) - (((data["var_52"]) + (data["var_53"]))))) + (((data["var_192"]) - (((data["var_147"]) - (data["var_87"]))))))) 

        v["i74"] = 0.020000*np.tanh(((((data["var_154"]) + (((data["var_197"]) + (data["var_116"]))))) + (((((data["var_12"]) - (data["var_195"]))) + (data["var_33"]))))) 

        v["i75"] = 0.020000*np.tanh(((data["var_141"]) + (((data["var_34"]) - (((data["var_0"]) - (((((data["var_31"]) + (data["var_148"]))) + (data["var_23"]))))))))) 

        v["i76"] = 0.020000*np.tanh(((data["var_166"]) + (((((((data["var_122"]) + (data["var_115"]))) + (((data["var_131"]) + (data["var_149"]))))) - (data["var_163"]))))) 

        v["i77"] = 0.020000*np.tanh(((((((data["var_172"]) - (((data["var_2"]) + (data["var_125"]))))) + (((data["var_127"]) + (data["var_76"]))))) + (data["var_36"]))) 

        v["i78"] = 0.020000*np.tanh(((((data["var_56"]) + (data["var_92"]))) + (((data["var_58"]) + (((((data["var_21"]) + (data["var_44"]))) + (data["var_75"]))))))) 

        v["i79"] = 0.020000*np.tanh(((((data["var_150"]) - (((data["var_174"]) * (data["var_174"]))))) + (((data["var_188"]) + (((data["var_28"]) + (data["var_83"]))))))) 

        v["i80"] = 0.020000*np.tanh(((data["var_86"]) + (((((data["var_198"]) + (((data["var_20"]) + (((data["var_43"]) + (data["var_108"]))))))) - (data["var_95"]))))) 

        v["i81"] = 0.020000*np.tanh(((data["var_174"]) + (((data["var_132"]) + (((data["var_123"]) + (((((data["var_80"]) + (data["var_85"]))) + (data["var_12"]))))))))) 

        v["i82"] = 0.020000*np.tanh(((data["var_87"]) + (((((data["var_186"]) + (data["var_154"]))) + (((data["var_165"]) + (((data["var_102"]) + (data["var_45"]))))))))) 

        v["i83"] = 0.020000*np.tanh(((data["var_177"]) + (((data["var_93"]) - (((data["var_1"]) - (((((data["var_114"]) + (data["var_149"]))) + (data["var_75"]))))))))) 

        v["i84"] = 0.020000*np.tanh(((data["var_141"]) + (((((((data["var_109"]) + (data["var_197"]))) + (data["var_178"]))) + (((data["var_146"]) + (data["var_104"]))))))) 

        v["i85"] = 0.020000*np.tanh(((data["var_107"]) + (((((((((data["var_121"]) + (data["var_132"]))) - (data["var_173"]))) + (data["var_13"]))) + (data["var_122"]))))) 

        v["i86"] = 0.020000*np.tanh(((data["var_166"]) + (((((data["var_194"]) + (((data["var_115"]) - (data["var_53"]))))) - (((data["var_110"]) - (data["var_116"]))))))) 

        v["i87"] = 0.020000*np.tanh(((data["var_31"]) + (((((((data["var_165"]) + (((data["var_76"]) + (data["var_186"]))))) + (data["var_131"]))) + (data["var_127"]))))) 

        v["i88"] = 0.020000*np.tanh(((((((data["var_34"]) + (((data["var_28"]) + (((data["var_192"]) + (data["var_56"]))))))) + (data["var_45"]))) + (data["var_148"]))) 

        v["i89"] = 0.020000*np.tanh(((((data["var_141"]) + (((data["var_88"]) + (((data["var_86"]) - (data["var_133"]))))))) + (((data["var_85"]) - (data["var_111"]))))) 

        v["i90"] = 0.020000*np.tanh(((((((data["var_188"]) - (data["var_155"]))) + (data["var_81"]))) + (((data["var_197"]) + (((data["var_114"]) + (data["var_83"]))))))) 

        v["i91"] = 0.020000*np.tanh(((((data["var_150"]) + (((((data["var_33"]) + (data["var_177"]))) + (data["var_169"]))))) + (((data["var_121"]) + (data["var_20"]))))) 

        v["i92"] = 0.020000*np.tanh(((data["var_54"]) + (((((((data["var_146"]) + (data["var_172"]))) + (((data["var_109"]) + (data["var_127"]))))) + (data["var_198"]))))) 

        v["i93"] = 0.020000*np.tanh(((((((data["var_21"]) + (data["var_77"]))) + (((data["var_174"]) + (data["var_23"]))))) + (((data["var_43"]) + (data["var_9"]))))) 

        v["i94"] = 0.020000*np.tanh(((((((((data["var_92"]) - (data["var_1"]))) - (data["var_130"]))) + (data["var_193"]))) + (((data["var_142"]) + (data["var_156"]))))) 

        v["i95"] = 0.020000*np.tanh(((data["var_116"]) + (((((data["var_44"]) + (((data["var_93"]) + (((data["var_166"]) - (data["var_53"]))))))) + (data["var_114"]))))) 

        v["i96"] = 0.020000*np.tanh(((((((data["var_76"]) + (data["var_172"]))) + (((data["var_57"]) - (data["var_49"]))))) + (((data["var_87"]) - (data["var_90"]))))) 

        v["i97"] = 0.020000*np.tanh(((((((data["var_104"]) + (data["var_115"]))) + (data["var_12"]))) + (((data["var_139"]) + (((data["var_178"]) + (data["var_45"]))))))) 

        v["i98"] = 0.020000*np.tanh(((data["var_43"]) + (((((data["var_80"]) + (data["var_107"]))) + (((((data["var_85"]) - (data["var_32"]))) + (data["var_142"]))))))) 

        v["i99"] = 0.020000*np.tanh(((((data["var_56"]) + (data["var_186"]))) + (((data["var_149"]) + (((((data["var_9"]) + (data["var_34"]))) + (data["var_131"]))))))) 

        v["i100"] = 0.020000*np.tanh(((((data["var_102"]) + (((data["var_50"]) + (((data["var_122"]) + (data["var_58"]))))))) + (((data["var_194"]) + (data["var_198"]))))) 

        v["i101"] = 0.020000*np.tanh(((((data["var_44"]) + (data["var_88"]))) + (((((data["var_188"]) + (((data["var_63"]) + (data["var_86"]))))) + (data["var_192"]))))) 

        v["i102"] = 0.020000*np.tanh(((((((data["var_28"]) - (data["var_147"]))) + (data["var_174"]))) + (((data["var_13"]) + (((data["var_132"]) + (data["var_165"]))))))) 

        v["i103"] = 0.020000*np.tanh(((((data["var_116"]) + (data["var_81"]))) + (((((data["var_108"]) - (data["var_162"]))) + (((data["var_177"]) + (data["var_166"]))))))) 

        v["i104"] = 0.020000*np.tanh(((data["var_21"]) + (((((data["var_43"]) + (((((data["var_156"]) + (data["var_143"]))) + (data["var_154"]))))) + (data["var_148"]))))) 

        v["i105"] = 0.020000*np.tanh(((((((data["var_146"]) + (data["var_123"]))) + (data["var_197"]))) + (((data["var_31"]) + (((data["var_9"]) + (data["var_150"]))))))) 

        v["i106"] = 0.020000*np.tanh((((((data["var_33"]) + (data["var_93"]))/2.0)) - (((((data["var_44"]) * (data["var_44"]))) + ((((-2.0) + (data["var_49"]))/2.0)))))) 

        v["i107"] = 0.020000*np.tanh(((data["var_12"]) + (((np.tanh((((((((((1.0) + (data["var_81"]))) * 2.0)) * 2.0)) + (data["var_13"]))))) * 2.0)))) 

        v["i108"] = 0.020000*np.tanh(((data["var_92"]) + (((((((data["var_76"]) + (data["var_115"]))) + (data["var_20"]))) + (((data["var_87"]) + (data["var_83"]))))))) 

        v["i109"] = 0.020000*np.tanh(((((data["var_127"]) + (((((data["var_77"]) - (data["var_66"]))) + (data["var_121"]))))) + (((data["var_178"]) + (data["var_194"]))))) 

        v["i110"] = 0.020000*np.tanh(((((data["var_186"]) + (((data["var_72"]) + (data["var_193"]))))) + (((((data["var_139"]) + (data["var_104"]))) + (data["var_149"]))))) 

        v["i111"] = 0.020000*np.tanh(((((data["var_114"]) + (data["var_141"]))) + (((data["var_12"]) - (((data["var_11"]) - (((data["var_23"]) + (data["var_131"]))))))))) 

        v["i112"] = 0.020000*np.tanh(((((data["var_64"]) + (((data["var_13"]) + (((data["var_54"]) + (((data["var_198"]) + (data["var_57"]))))))))) + (data["var_102"]))) 

        v["i113"] = 0.020000*np.tanh(((((((data["var_169"]) - (data["var_133"]))) + (data["var_36"]))) + (((((data["var_86"]) - (data["var_180"]))) + (data["var_132"]))))) 

        v["i114"] = 0.020000*np.tanh(((((data["var_188"]) + (data["var_142"]))) + (((((data["var_58"]) + (((data["var_45"]) + (data["var_165"]))))) + (data["var_174"]))))) 

        v["i115"] = 0.020000*np.tanh(((((data["var_177"]) - (((data["var_67"]) - (data["var_31"]))))) - (((data["var_48"]) - (((data["var_33"]) + (data["var_101"]))))))) 

        v["i116"] = 0.020000*np.tanh(((((((data["var_166"]) + (((np.tanh((np.tanh((((((data["var_146"]) - (-1.0))) * 2.0)))))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i117"] = 0.020000*np.tanh(((((data["var_192"]) + (((data["var_107"]) - (((data["var_190"]) * (data["var_190"]))))))) + (((1.0) + (data["var_123"]))))) 

        v["i118"] = 0.020000*np.tanh(((((data["var_50"]) + (((data["var_122"]) + (data["var_85"]))))) + (((data["var_28"]) + (((data["var_172"]) + (data["var_150"]))))))) 

        v["i119"] = 0.020000*np.tanh(((((data["var_115"]) + (((data["var_72"]) - (data["var_162"]))))) + (((data["var_108"]) + (((data["var_88"]) + (data["var_75"]))))))) 

        v["i120"] = 0.020000*np.tanh(((((((data["var_152"]) + (data["var_56"]))) + (((data["var_44"]) + (((data["var_154"]) + (data["var_93"]))))))) + (data["var_109"]))) 

        v["i121"] = 0.020000*np.tanh(((((data["var_26"]) + ((((data["var_80"]) + (3.0))/2.0)))) + ((((-1.0*((((data["var_26"]) * (data["var_26"])))))) * 2.0)))) 

        v["i122"] = 0.020000*np.tanh(((((2.0) - (((data["var_12"]) * (data["var_12"]))))) - (((data["var_146"]) * (data["var_146"]))))) 

        v["i123"] = 0.020000*np.tanh(((data["var_23"]) + (((((data["var_104"]) + (data["var_68"]))) + (((((data["var_63"]) + (data["var_12"]))) - (data["var_18"]))))))) 

        v["i124"] = 0.020000*np.tanh(((((data["var_148"]) + (data["var_34"]))) + (((data["var_20"]) - (((data["var_191"]) - (((data["var_149"]) + (data["var_120"]))))))))) 

        v["i125"] = 0.020000*np.tanh(((data["var_57"]) + (((((data["var_197"]) + (((data["var_116"]) + (data["var_83"]))))) + (((data["var_193"]) + (data["var_76"]))))))) 

        v["i126"] = 0.020000*np.tanh(((((data["var_33"]) + (data["var_156"]))) + (((((data["var_121"]) + (data["var_92"]))) + (((data["var_174"]) + (data["var_141"]))))))) 

        v["i127"] = 0.020000*np.tanh(((data["var_86"]) + (((((1.0) + (((data["var_198"]) + (data["var_139"]))))) + (((data["var_80"]) - (data["var_53"]))))))) 

        v["i128"] = 0.020000*np.tanh(((((((data["var_58"]) + (data["var_40"]))) + (data["var_143"]))) + (((data["var_194"]) - (((data["var_40"]) * (data["var_40"]))))))) 

        v["i129"] = 0.020000*np.tanh(((((((((np.tanh((np.tanh((((((1.0) + (data["var_44"]))) * 2.0)))))) * 2.0)) - (data["var_22"]))) * 2.0)) * 2.0)) 

        v["i130"] = 0.020000*np.tanh(((data["var_178"]) + (((data["var_169"]) + (((data["var_36"]) + (((data["var_115"]) + (((data["var_114"]) + (data["var_142"]))))))))))) 

        v["i131"] = 0.020000*np.tanh((((((((data["var_75"]) + (data["var_9"]))/2.0)) - (data["var_11"]))) + (((data["var_122"]) - (((data["var_78"]) - (data["var_13"]))))))) 

        v["i132"] = 0.020000*np.tanh((((2.61301231384277344)) + (((data["var_132"]) + (((((data["var_110"]) * 2.0)) * (((data["var_83"]) - (((data["var_110"]) * 2.0)))))))))) 

        v["i133"] = 0.020000*np.tanh(((data["var_127"]) + (((((((((data["var_45"]) - (data["var_167"]))) + (data["var_188"]))) - (data["var_15"]))) + (data["var_186"]))))) 

        v["i134"] = 0.020000*np.tanh(((data["var_50"]) + (((((data["var_102"]) + (data["var_43"]))) + (((data["var_131"]) - (((data["var_106"]) * (data["var_106"]))))))))) 

        v["i135"] = 0.020000*np.tanh(((2.0) + (((((data["var_76"]) - (((data["var_81"]) * (((((data["var_81"]) * 2.0)) * 2.0)))))) - (data["var_81"]))))) 

        v["i136"] = 0.020000*np.tanh(((data["var_28"]) + (((data["var_88"]) + (((data["var_154"]) + (((((data["var_192"]) - (data["var_66"]))) + (data["var_85"]))))))))) 

        v["i137"] = 0.020000*np.tanh((((((data["var_110"]) + (((((data["var_110"]) * ((-1.0*((data["var_110"])))))) * 2.0)))) + (data["var_64"]))/2.0)) 

        v["i138"] = 0.020000*np.tanh(((((data["var_93"]) + (((((data["var_63"]) - (data["var_145"]))) - (data["var_151"]))))) + (((data["var_148"]) + (data["var_165"]))))) 

        v["i139"] = 0.020000*np.tanh(((((data["var_21"]) - (data["var_196"]))) + (((data["var_109"]) + (((data["var_59"]) + (((data["var_132"]) + (data["var_123"]))))))))) 

        v["i140"] = 0.020000*np.tanh(((((((data["var_156"]) - (data["var_195"]))) - (data["var_91"]))) + (((((data["var_182"]) + (data["var_150"]))) - (data["var_170"]))))) 

        v["i141"] = 0.020000*np.tanh((((((data["var_166"]) + (((data["var_34"]) + (data["var_9"]))))) + (((data["var_108"]) + (data["var_20"]))))/2.0)) 

        v["i142"] = 0.020000*np.tanh((-1.0*((((data["var_76"]) * (data["var_76"])))))) 

        v["i143"] = 0.020000*np.tanh((((((((2.0) + ((((-1.0*((data["var_81"])))) - (data["var_71"]))))/2.0)) - (((data["var_81"]) * (data["var_81"]))))) * 2.0)) 

        v["i144"] = 0.020000*np.tanh(((((data["var_172"]) - (((data["var_110"]) * (data["var_110"]))))) - ((((((data["var_93"]) * (data["var_93"]))) + (-3.0))/2.0)))) 

        v["i145"] = 0.020000*np.tanh(((1.0) + (((1.0) + (((((data["var_133"]) * 2.0)) * (((1.0) - (((data["var_133"]) * 2.0)))))))))) 

        v["i146"] = 0.020000*np.tanh(((data["var_197"]) + (((((((data["var_198"]) - (((data["var_133"]) + (data["var_6"]))))) + (data["var_75"]))) + (data["var_101"]))))) 

        v["i147"] = 0.020000*np.tanh((((data["var_31"]) + (((((data["var_147"]) * ((-1.0*((((data["var_147"]) + (((data["var_147"]) / 2.0))))))))) + (data["var_116"]))))/2.0)) 

        v["i148"] = 0.020000*np.tanh(((np.tanh((((((data["var_149"]) * 2.0)) * 2.0)))) + (((((data["var_190"]) * ((-1.0*((data["var_190"])))))) - (-1.0))))) 

        v["i149"] = 0.020000*np.tanh(((((data["var_12"]) - (data["var_53"]))) + (((data["var_120"]) + (((((data["var_68"]) - (data["var_164"]))) + (data["var_193"]))))))) 

        v["i150"] = 0.020000*np.tanh(((data["var_76"]) + (((data["var_92"]) + ((((2.26304459571838379)) + (((((data["var_123"]) * ((-1.0*((data["var_123"])))))) * 2.0)))))))) 

        v["i151"] = 0.020000*np.tanh(((data["var_13"]) + (((((data["var_56"]) + (((data["var_64"]) + (((data["var_23"]) + (data["var_114"]))))))) - (data["var_52"]))))) 

        v["i152"] = 0.020000*np.tanh(((((data["var_5"]) * ((-1.0*((data["var_5"])))))) + (((data["var_21"]) + (((3.0) / 2.0)))))) 

        v["i153"] = 0.020000*np.tanh(((((data["var_104"]) + (((((data["var_107"]) - (data["var_128"]))) + (data["var_174"]))))) + (((data["var_81"]) - (data["var_168"]))))) 

        v["i154"] = 0.020000*np.tanh((((((data["var_141"]) + (data["var_152"]))) + (((((((data["var_87"]) + (data["var_72"]))) + (data["var_113"]))) + (data["var_121"]))))/2.0)) 

        v["i155"] = 0.020000*np.tanh(((data["var_42"]) + ((((((data["var_115"]) + (data["var_115"]))/2.0)) * ((-1.0*(((((data["var_115"]) + (data["var_115"]))/2.0))))))))) 

        v["i156"] = 0.020000*np.tanh(((((data["var_165"]) + ((((-1.0*((((data["var_198"]) * (data["var_198"])))))) + (((1.0) + (1.0))))))) * 2.0)) 

        v["i157"] = 0.020000*np.tanh(((data["var_12"]) + (((((data["var_36"]) + (data["var_169"]))) + (((1.0) + (((data["var_109"]) - (data["var_179"]))))))))) 

        v["i158"] = 0.020000*np.tanh(((((data["var_194"]) + (((data["var_83"]) + (data["var_54"]))))) + (((data["var_127"]) - (((data["var_163"]) + (data["var_22"]))))))) 

        v["i159"] = 0.020000*np.tanh(((data["var_80"]) * (np.tanh((((data["var_142"]) - (data["var_80"]))))))) 

        v["i160"] = 0.020000*np.tanh(((((((np.tanh(((((1.67784368991851807)) + (data["var_109"]))))) * 2.0)) + (((data["var_155"]) * ((-1.0*((data["var_155"])))))))) * 2.0)) 

        v["i161"] = 0.020000*np.tanh((((((((data["var_108"]) - (data["var_173"]))) + (((((data["var_57"]) - (data["var_35"]))) - (data["var_8"]))))) + (data["var_122"]))/2.0)) 

        v["i162"] = 0.020000*np.tanh((((((data["var_86"]) + (data["var_153"]))) + (((((data["var_45"]) - (data["var_135"]))) - (((data["var_78"]) * (data["var_78"]))))))/2.0)) 

        v["i163"] = 0.020000*np.tanh((((((data["var_178"]) - (((data["var_78"]) - (((data["var_58"]) - (data["var_53"]))))))) + (((data["var_115"]) - (data["var_170"]))))/2.0)) 

        v["i164"] = 0.020000*np.tanh(((((data["var_174"]) + (((np.tanh((((np.tanh((((((data["var_192"]) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)) 

        v["i165"] = 0.020000*np.tanh(((((2.0) + (((data["var_78"]) * ((-1.0*((data["var_78"])))))))) + (((data["var_198"]) * ((-1.0*((data["var_198"])))))))) 

        v["i166"] = 0.020000*np.tanh((-1.0*((((((data["var_146"]) - (-1.0))) * ((((((data["var_146"]) + (-1.0))/2.0)) - ((-1.0*((data["var_146"]))))))))))) 

        v["i167"] = 0.020000*np.tanh((((((data["var_154"]) + (data["var_33"]))) + (((data["var_132"]) + (((((data["var_44"]) + (data["var_129"]))) + (data["var_177"]))))))/2.0)) 

        v["i168"] = 0.020000*np.tanh(((((data["var_146"]) - (data["var_180"]))) - (np.tanh((((-2.0) + (((data["var_40"]) - (data["var_166"]))))))))) 

        v["i169"] = 0.020000*np.tanh(((((((data["var_139"]) * (((-2.0) - (data["var_139"]))))) - (data["var_139"]))) + (((data["var_139"]) + (data["var_139"]))))) 

        v["i170"] = 0.020000*np.tanh(((((((((((data["var_76"]) * ((-1.0*((data["var_76"])))))) * 2.0)) + (((data["var_139"]) + (2.0))))) * 2.0)) * 2.0)) 

        v["i171"] = 0.020000*np.tanh(((((((((data["var_166"]) * ((-1.0*((data["var_166"])))))) - (-1.0))) * 2.0)) + ((-1.0*((data["var_166"])))))) 

        v["i172"] = 0.020000*np.tanh(((((data["var_51"]) * ((-1.0*((data["var_51"])))))) + (np.tanh((((data["var_51"]) + (((data["var_51"]) + (data["var_51"]))))))))) 

        v["i173"] = 0.020000*np.tanh(((((1.0) + (((data["var_148"]) * ((-1.0*((data["var_148"])))))))) + (((data["var_148"]) * ((-1.0*((data["var_148"])))))))) 

        v["i174"] = 0.020000*np.tanh(((((data["var_174"]) * (((((np.tanh((-2.0))) - (np.tanh((data["var_174"]))))) - (data["var_174"]))))) - (-2.0))) 

        v["i175"] = 0.020000*np.tanh((((((data["var_188"]) + (((data["var_131"]) + (data["var_166"]))))) + (((data["var_44"]) * ((-1.0*((data["var_44"])))))))/2.0)) 

        v["i176"] = 0.020000*np.tanh(((((data["var_174"]) - (((data["var_130"]) - (data["var_188"]))))) + (((((data["var_34"]) - (-1.0))) - (data["var_51"]))))) 

        v["i177"] = 0.020000*np.tanh((((((data["var_89"]) * ((-1.0*((data["var_89"])))))) + (((data["var_77"]) + (((data["var_44"]) - (-1.0))))))/2.0)) 

        v["i178"] = 0.020000*np.tanh(((((((1.0) + (((data["var_109"]) + (1.0))))) - (((data["var_180"]) * (data["var_180"]))))) * 2.0)) 

        v["i179"] = 0.020000*np.tanh((((((data["var_31"]) - ((13.32383632659912109)))) + (((data["var_43"]) + (((((((-3.0) + (data["var_23"]))/2.0)) + (-3.0))/2.0)))))/2.0)) 

        v["i180"] = 0.020000*np.tanh((((((data["var_88"]) + (data["var_197"]))) + (((data["var_80"]) + (((((data["var_31"]) - (data["var_78"]))) + (data["var_198"]))))))/2.0)) 

        v["i181"] = 0.020000*np.tanh((((((data["var_184"]) * ((-1.0*((data["var_184"])))))) + (((data["var_182"]) + (((data["var_143"]) + (data["var_92"]))))))/2.0)) 

        v["i182"] = 0.020000*np.tanh((((data["var_107"]) + (((((data["var_102"]) - (data["var_170"]))) + (((data["var_28"]) + (((data["var_43"]) + (data["var_172"]))))))))/2.0)) 

        v["i183"] = 0.020000*np.tanh(((-1.0) + ((((((((((-1.0) + (data["var_142"]))) - (data["var_49"]))) + (data["var_14"]))/2.0)) - (data["var_61"]))))) 

        v["i184"] = 0.020000*np.tanh((((((data["var_6"]) * ((((-1.0*((data["var_6"])))) - (data["var_6"]))))) + (((data["var_6"]) - ((-1.0*((data["var_20"])))))))/2.0)) 

        v["i185"] = 0.020000*np.tanh(((((((data["var_139"]) - (np.tanh((((-3.0) - (((data["var_75"]) + (data["var_75"]))))))))) * 2.0)) * 2.0)) 

        v["i186"] = 0.020000*np.tanh(((((((data["var_50"]) - (data["var_134"]))) - ((((-3.0) + (data["var_2"]))/2.0)))) - (((data["var_99"]) * (data["var_99"]))))) 

        v["i187"] = 0.020000*np.tanh((((((1.0) + (((data["var_86"]) + (data["var_108"]))))) + (((((data["var_186"]) - (data["var_184"]))) - (data["var_99"]))))/2.0)) 

        v["i188"] = 0.020000*np.tanh(((data["var_166"]) * (((-2.0) - ((((((data["var_56"]) + (-2.0))/2.0)) + (data["var_166"]))))))) 

        v["i189"] = 0.020000*np.tanh(((data["var_44"]) + (((((((1.0) - (((data["var_95"]) + (data["var_52"]))))) - (data["var_168"]))) + (data["var_166"]))))) 

        v["i190"] = 0.020000*np.tanh((((-1.0*((((data["var_197"]) * (data["var_197"])))))) - ((((data["var_197"]) + ((((data["var_181"]) + (((data["var_197"]) * 2.0)))/2.0)))/2.0)))) 

        v["i191"] = 0.020000*np.tanh((((((2.0) + (data["var_9"]))/2.0)) + (((data["var_58"]) * ((-1.0*((data["var_58"])))))))) 

        v["i192"] = 0.020000*np.tanh((((-1.0*(((((-3.0) + ((((data["var_125"]) + (data["var_165"]))/2.0)))/2.0))))) - (((data["var_165"]) * (data["var_165"]))))) 

        v["i193"] = 0.020000*np.tanh((((((data["var_186"]) * (((np.tanh((-2.0))) - (data["var_186"]))))) + (((-3.0) + (data["var_17"]))))/2.0)) 

        v["i194"] = 0.020000*np.tanh((-1.0*((((data["var_86"]) * (((data["var_86"]) - (((-2.0) - (data["var_86"])))))))))) 

        v["i195"] = 0.020000*np.tanh(((((((((data["var_93"]) + (data["var_80"]))) + (data["var_165"]))) + ((1.44516742229461670)))) + (data["var_154"]))) 

        v["i196"] = 0.020000*np.tanh((((((data["var_18"]) - (((data["var_18"]) * (data["var_18"]))))) + (((((((data["var_150"]) + (data["var_86"]))/2.0)) + (data["var_13"]))/2.0)))/2.0)) 

        v["i197"] = 0.020000*np.tanh(((((((((((data["var_72"]) + (data["var_136"]))) + (data["var_141"]))/2.0)) - (((data["var_107"]) * (data["var_107"]))))) + (data["var_152"]))/2.0)) 

        v["i198"] = 0.020000*np.tanh(((((((((((0.318310) + (((data["var_139"]) * ((-1.0*((data["var_139"])))))))) * 2.0)) + (0.318310))) * 2.0)) * 2.0)) 

        v["i199"] = 0.020000*np.tanh((((((data["var_193"]) + (((data["var_123"]) - (data["var_155"]))))) + ((((((data["var_197"]) - (data["var_175"]))) + (data["var_116"]))/2.0)))/2.0)) 

        v["i200"] = 0.020000*np.tanh(((((data["var_26"]) * ((-1.0*((data["var_26"])))))) + (((((data["var_99"]) * ((-1.0*((data["var_99"])))))) + (2.0))))) 

        v["i201"] = 0.020000*np.tanh(((((np.tanh(((((-1.0*((data["var_109"])))) * 2.0)))) * 2.0)) + (((data["var_109"]) * ((-1.0*((data["var_109"])))))))) 

        v["i202"] = 0.020000*np.tanh((((((-1.0*((data["var_109"])))) * (data["var_109"]))) + ((-1.0*((data["var_109"])))))) 

        v["i203"] = 0.020000*np.tanh((-1.0*((((data["var_179"]) - ((((-1.0*((((data["var_179"]) - (((np.tanh((((data["var_179"]) * 2.0)))) * 2.0))))))) * 2.0))))))) 

        v["i204"] = 0.020000*np.tanh(((data["var_33"]) + (((((data["var_13"]) + (data["var_109"]))) + (((np.tanh((((data["var_122"]) + (1.0))))) * 2.0)))))) 

        v["i205"] = 0.020000*np.tanh((((-1.0*((((((data["var_95"]) * (data["var_95"]))) * (((data["var_95"]) * (data["var_95"])))))))) + (data["var_95"]))) 

        v["i206"] = 0.020000*np.tanh(((data["var_115"]) * ((((-1.0*((data["var_115"])))) - (data["var_115"]))))) 

        v["i207"] = 0.020000*np.tanh(((((((1.0) + (1.0))) - (((data["var_6"]) * (data["var_6"]))))) - (((data["var_99"]) * (data["var_99"]))))) 

        v["i208"] = 0.020000*np.tanh((((data["var_59"]) + (((((data["var_44"]) + (data["var_56"]))) - ((((-1.0*(((-1.0*((data["var_75"]))))))) * (data["var_75"]))))))/2.0)) 

        v["i209"] = 0.020000*np.tanh(((((np.tanh((((3.0) - (((data["var_133"]) * 2.0)))))) - (((data["var_188"]) * (data["var_188"]))))) * 2.0)) 

        v["i210"] = 0.020000*np.tanh((((((((2.0)) + (((data["var_109"]) + ((-1.0*((((data["var_33"]) * (data["var_33"])))))))))) * 2.0)) * 2.0)) 

        v["i211"] = 0.020000*np.tanh(((((((3.0) + (data["var_57"]))/2.0)) + (((data["var_75"]) + (((data["var_12"]) * (((data["var_57"]) - (data["var_12"]))))))))/2.0)) 

        v["i212"] = 0.020000*np.tanh(((((((-1.0*((((data["var_53"]) * (data["var_53"])))))) + ((-1.0*((((data["var_76"]) * (data["var_76"])))))))/2.0)) + (1.0))) 

        v["i213"] = 0.020000*np.tanh(((((((((0.318310) - (((data["var_91"]) * (data["var_91"]))))) + (data["var_91"]))) * 2.0)) + (np.tanh((data["var_91"]))))) 

        v["i214"] = 0.020000*np.tanh((((-1.0*((((((data["var_149"]) * 2.0)) * (((data["var_149"]) * 2.0))))))) + (((np.tanh((((data["var_149"]) * 2.0)))) * 2.0)))) 

        v["i215"] = 0.020000*np.tanh(((data["var_21"]) + (((data["var_174"]) + (np.tanh((((((3.0) + (((data["var_33"]) * 2.0)))) * 2.0)))))))) 

        v["i216"] = 0.020000*np.tanh((((((((data["var_146"]) + (((data["var_174"]) + (data["var_104"]))))) - (data["var_91"]))) + (((data["var_166"]) + (data["var_188"]))))/2.0)) 

        v["i217"] = 0.020000*np.tanh((((((data["var_87"]) * ((-1.0*((data["var_87"])))))) + ((((((data["var_72"]) + (data["var_23"]))) + (data["var_85"]))/2.0)))/2.0)) 

        v["i218"] = 0.020000*np.tanh((((((data["var_56"]) + (((data["var_113"]) + (data["var_39"]))))) + (((((data["var_160"]) - (data["var_171"]))) - (data["var_135"]))))/2.0)) 

        v["i219"] = 0.020000*np.tanh(((((((data["var_13"]) + (((np.tanh((((((((data["var_121"]) + ((1.18997478485107422)))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i220"] = 0.020000*np.tanh((((data["var_40"]) + (((data["var_40"]) - (((((data["var_40"]) * (data["var_40"]))) * (((data["var_40"]) * (data["var_40"]))))))))/2.0)) 

        v["i221"] = 0.020000*np.tanh(((((data["var_86"]) * ((-1.0*((data["var_86"])))))) + (np.tanh((((data["var_177"]) + ((2.0)))))))) 

        v["i222"] = 0.020000*np.tanh(((((data["var_133"]) * ((-1.0*((data["var_133"])))))) + ((((data["var_133"]) + (((data["var_116"]) - (data["var_195"]))))/2.0)))) 

        v["i223"] = 0.020000*np.tanh((((((data["var_155"]) + (((data["var_120"]) + (((data["var_13"]) / 2.0)))))) + (((data["var_155"]) * ((-1.0*((data["var_155"])))))))/2.0)) 

        v["i224"] = 0.020000*np.tanh((((data["var_26"]) + (((data["var_26"]) * (((((data["var_26"]) * (((data["var_26"]) * ((-1.0*((data["var_26"])))))))) / 2.0)))))/2.0)) 

        v["i225"] = 0.020000*np.tanh((((((data["var_81"]) * ((-1.0*((data["var_81"])))))) + ((((((-1.0*((data["var_157"])))) * (data["var_157"]))) + (data["var_150"]))))/2.0)) 

        v["i226"] = 0.020000*np.tanh((-1.0*((((data["var_125"]) * (((-1.0) + (data["var_125"])))))))) 

        v["i227"] = 0.020000*np.tanh((((((data["var_0"]) + (((data["var_0"]) * ((-1.0*((data["var_0"])))))))/2.0)) * 2.0)) 

        v["i228"] = 0.020000*np.tanh(((((data["var_146"]) * ((-1.0*((data["var_146"])))))) + (((2.0) + (data["var_123"]))))) 

        v["i229"] = 0.020000*np.tanh((((((data["var_92"]) + (data["var_81"]))) + ((((((-1.0*((data["var_0"])))) - (data["var_125"]))) + (data["var_156"]))))/2.0)) 

        v["i230"] = 0.020000*np.tanh(((data["var_164"]) * (((((((((1.0) - (((data["var_164"]) - (((1.0) / 2.0)))))) * 2.0)) * 2.0)) * 2.0)))) 

        v["i231"] = 0.020000*np.tanh(((((((data["var_148"]) * (((np.tanh((((data["var_148"]) * 2.0)))) - (((data["var_148"]) * 2.0)))))) - (data["var_148"]))) * 2.0)) 

        v["i232"] = 0.020000*np.tanh(((2.0) + (((((2.0) - (data["var_164"]))) - (((data["var_164"]) * 2.0)))))) 

        v["i233"] = 0.020000*np.tanh(((((((data["var_110"]) * 2.0)) + (data["var_110"]))) * ((-1.0*((((-3.0) + (((data["var_110"]) * 2.0))))))))) 

        v["i234"] = 0.020000*np.tanh(((((((data["var_158"]) + (data["var_148"]))/2.0)) + ((((data["var_56"]) + (((data["var_63"]) + (((data["var_149"]) + (data["var_83"]))))))/2.0)))/2.0)) 

        v["i235"] = 0.020000*np.tanh(((((2.0) - (((((data["var_164"]) / 2.0)) * (data["var_164"]))))) - (((data["var_110"]) * (data["var_110"]))))) 

        v["i236"] = 0.020000*np.tanh((((((data["var_78"]) * ((-1.0*((data["var_78"])))))) + (np.tanh((np.tanh((((((data["var_78"]) * 2.0)) * 2.0)))))))/2.0)) 

        v["i237"] = 0.020000*np.tanh((-1.0*(((((((-3.0) + (((np.tanh((data["var_174"]))) * 2.0)))/2.0)) + (((data["var_174"]) * (data["var_174"])))))))) 

        v["i238"] = 0.020000*np.tanh(((((data["var_177"]) - (-3.0))) - (((data["var_21"]) + (((((data["var_21"]) * (data["var_21"]))) * 2.0)))))) 

        v["i239"] = 0.020000*np.tanh((((((data["var_51"]) * ((-1.0*((data["var_51"])))))) + ((((np.tanh((data["var_51"]))) + (((data["var_148"]) - (-2.0))))/2.0)))/2.0)) 

        v["i240"] = 0.020000*np.tanh(((((-1.0*((((data["var_163"]) * (data["var_163"])))))) + (np.tanh((((data["var_93"]) - (np.tanh((-3.0))))))))/2.0)) 

        v["i241"] = 0.020000*np.tanh((((data["var_174"]) + ((((((((data["var_183"]) + (data["var_177"]))/2.0)) - (data["var_112"]))) - (data["var_119"]))))/2.0)) 

        v["i242"] = 0.020000*np.tanh(((((((data["var_33"]) + (((data["var_34"]) + (data["var_153"]))))/2.0)) + (((data["var_54"]) - (((data["var_141"]) * (data["var_141"]))))))/2.0)) 

        v["i243"] = 0.020000*np.tanh(((((data["var_165"]) * 2.0)) * ((-1.0*(((((data["var_165"]) + ((-1.0*((np.tanh((data["var_165"])))))))/2.0))))))) 

        v["i244"] = 0.020000*np.tanh((((((((data["var_135"]) * 2.0)) / 2.0)) + (((((1.0) * 2.0)) - (((((data["var_135"]) * (data["var_135"]))) * 2.0)))))/2.0)) 

        v["i245"] = 0.020000*np.tanh((((data["var_133"]) + (((data["var_133"]) * (((((data["var_133"]) / 2.0)) * (((data["var_133"]) * ((-1.0*((data["var_133"])))))))))))/2.0)) 

        v["i246"] = 0.020000*np.tanh(((data["var_81"]) * (((((data["var_81"]) * (((data["var_141"]) - ((9.02286720275878906)))))) - ((9.02286720275878906)))))) 

        v["i247"] = 0.020000*np.tanh(((((2.0) + (data["var_80"]))) * (((((2.0) + (data["var_166"]))) * (((data["var_108"]) - (-2.0))))))) 

        v["i248"] = 0.020000*np.tanh((((((data["var_170"]) - (((data["var_186"]) * (data["var_186"]))))) + (((data["var_170"]) * ((-1.0*((data["var_170"])))))))/2.0)) 

        v["i249"] = 0.020000*np.tanh(((np.tanh((((data["var_142"]) + (((data["var_76"]) + (1.0))))))) + (((data["var_23"]) * ((-1.0*((data["var_23"])))))))) 

        v["i250"] = 0.020000*np.tanh((((data["var_165"]) + (((((data["var_101"]) + (data["var_68"]))) - (data["var_2"]))))/2.0)) 

        v["i251"] = 0.020000*np.tanh((-1.0*((((((data["var_86"]) * (data["var_86"]))) + (data["var_86"])))))) 

        v["i252"] = 0.020000*np.tanh(((((((data["var_13"]) + (data["var_107"]))) + (data["var_86"]))) + (((np.tanh((((data["var_36"]) + (1.0))))) * 2.0)))) 

        v["i253"] = 0.020000*np.tanh(((((data["var_1"]) * ((-1.0*((data["var_1"])))))) + (((np.tanh((data["var_1"]))) + (data["var_1"]))))) 

        v["i254"] = 0.020000*np.tanh((-1.0*(((((((data["var_110"]) * (((data["var_110"]) / 2.0)))) + ((((data["var_179"]) + (((data["var_110"]) * (data["var_110"]))))/2.0)))/2.0))))) 

        v["i255"] = 0.020000*np.tanh((((((((7.0)) / 2.0)) - ((((-1.0*((data["var_80"])))) * 2.0)))) * ((((-1.0*((data["var_80"])))) * 2.0)))) 

        v["i256"] = 0.020000*np.tanh(((((((-1.0*((((data["var_80"]) * (data["var_80"])))))) / 2.0)) + (((data["var_122"]) * ((((-1.0*((data["var_122"])))) / 2.0)))))/2.0)) 

        v["i257"] = 0.020000*np.tanh(((((3.0) + (((data["var_9"]) * 2.0)))) * (((((((3.0) + (((data["var_177"]) * 2.0)))) * 2.0)) * 2.0)))) 

        v["i258"] = 0.020000*np.tanh(((((((3.0) - (((data["var_32"]) * 2.0)))) * 2.0)) * (((((3.0) - (((data["var_94"]) * 2.0)))) * 2.0)))) 

        v["i259"] = 0.020000*np.tanh(((((data["var_36"]) + (((((data["var_36"]) * 2.0)) * 2.0)))) * (((np.tanh((((data["var_36"]) * 2.0)))) - (data["var_36"]))))) 

        v["i260"] = 0.020000*np.tanh(((((((((data["var_177"]) + (((np.tanh(((((-1.0*((data["var_177"])))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) + (data["var_177"]))) 

        v["i261"] = 0.020000*np.tanh(((np.tanh((((3.0) + (((data["var_177"]) * 2.0)))))) - (((((data["var_93"]) * ((1.0)))) * (data["var_93"]))))) 

        v["i262"] = 0.020000*np.tanh((((-1.0*((data["var_92"])))) + ((((-1.0*((np.tanh((data["var_92"])))))) + (((data["var_92"]) * ((-1.0*((data["var_92"])))))))))) 

        v["i263"] = 0.020000*np.tanh(((((((data["var_81"]) + (((data["var_177"]) - (data["var_179"]))))/2.0)) + ((-1.0*((((data["var_92"]) * (data["var_92"])))))))/2.0)) 

        v["i264"] = 0.020000*np.tanh((((((data["var_125"]) * ((-1.0*((data["var_125"])))))) + ((((((data["var_22"]) * ((-1.0*((data["var_22"])))))) + (data["var_92"]))/2.0)))/2.0)) 

        v["i265"] = 0.020000*np.tanh(((((data["var_75"]) + (((np.tanh((((data["var_34"]) + (((((data["var_34"]) + (2.0))) * 2.0)))))) * 2.0)))) * 2.0)) 

        v["i266"] = 0.020000*np.tanh(((((((((data["var_34"]) + (((np.tanh(((-1.0*((((data["var_34"]) * 2.0))))))) * 2.0)))) * 2.0)) * 2.0)) + (data["var_34"]))) 

        v["i267"] = 0.020000*np.tanh(((data["var_154"]) + (((((data["var_180"]) + (np.tanh((1.0))))) + (((data["var_180"]) * ((-1.0*((data["var_180"])))))))))) 

        v["i268"] = 0.020000*np.tanh(((((((((((data["var_127"]) - (((data["var_180"]) + (data["var_22"]))))) - (data["var_22"]))) + (data["var_14"]))/2.0)) + (data["var_85"]))/2.0)) 

        v["i269"] = 0.020000*np.tanh(((((2.0) + (data["var_197"]))) + (((data["var_56"]) * ((-1.0*((data["var_56"])))))))) 

        v["i270"] = 0.020000*np.tanh(((((data["var_52"]) * ((-1.0*((data["var_52"])))))) + (((2.0) - (((data["var_5"]) * (data["var_5"]))))))) 

        v["i271"] = 0.020000*np.tanh(((((data["var_93"]) * ((-1.0*((data["var_93"])))))) - (data["var_93"]))) 

        v["i272"] = 0.020000*np.tanh(((((((data["var_34"]) + (data["var_92"]))/2.0)) + ((((data["var_85"]) + ((((-1.0*((data["var_25"])))) - (data["var_95"]))))/2.0)))/2.0)) 

        v["i273"] = 0.020000*np.tanh(((((((3.0) + (((data["var_121"]) + (data["var_87"]))))/2.0)) + (((data["var_190"]) * ((-1.0*((data["var_190"])))))))/2.0)) 

        v["i274"] = 0.020000*np.tanh(((data["var_62"]) * ((-1.0*((((data["var_62"]) - (np.tanh((np.tanh((np.tanh((data["var_62"])))))))))))))) 

        v["i275"] = 0.020000*np.tanh((-1.0*((((((-3.0) + (((((((((-3.0) + (((data["var_94"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0))))) 

        v["i276"] = 0.020000*np.tanh(((data["var_89"]) * (((((3.0) - (data["var_89"]))) - (data["var_89"]))))) 

        v["i277"] = 0.020000*np.tanh((((((data["var_191"]) * (((data["var_191"]) * (((((data["var_191"]) * ((-1.0*((data["var_191"])))))) / 2.0)))))) + (data["var_191"]))/2.0)) 

        v["i278"] = 0.020000*np.tanh((((((((data["var_93"]) + (data["var_165"]))/2.0)) / 2.0)) - (((data["var_89"]) * (((((data["var_89"]) / 2.0)) / 2.0)))))) 

        v["i279"] = 0.020000*np.tanh((((((((((-1.0*((((data["var_107"]) * (data["var_107"])))))) - (data["var_107"]))) * 2.0)) - (data["var_107"]))) * 2.0)) 

        v["i280"] = 0.020000*np.tanh(((((-1.0*((((data["var_114"]) * (((((data["var_114"]) * (((data["var_114"]) / 2.0)))) * (data["var_114"])))))))) + (data["var_107"]))/2.0)) 

        v["i281"] = 0.020000*np.tanh((((np.tanh(((8.84132480621337891)))) + (((data["var_21"]) * (((np.tanh((np.tanh((data["var_21"]))))) - (data["var_21"]))))))/2.0)) 

        v["i282"] = 0.020000*np.tanh(((((-1.0*((-2.0)))) + (((data["var_75"]) + (((data["var_115"]) * (((data["var_115"]) * (-2.0))))))))/2.0)) 

        v["i283"] = 0.020000*np.tanh(((((((((data["var_33"]) - (-2.0))) * 2.0)) + ((-1.0*((((data["var_108"]) * (data["var_108"])))))))) * 2.0)) 

        v["i284"] = 0.020000*np.tanh(((((data["var_33"]) - (np.tanh((np.tanh((np.tanh((data["var_33"]))))))))) - (np.tanh((((data["var_33"]) * 2.0)))))) 

        v["i285"] = 0.020000*np.tanh(((((data["var_197"]) * ((-1.0*(((((((data["var_197"]) + (data["var_197"]))/2.0)) - (-1.0)))))))) - (-1.0))) 

        v["i286"] = 0.020000*np.tanh(((((data["var_147"]) + (data["var_147"]))) + ((-1.0*((((((data["var_147"]) * (data["var_147"]))) * (data["var_147"])))))))) 

        v["i287"] = 0.020000*np.tanh((((((((((((-3.0) + (-3.0))/2.0)) + (data["var_197"]))/2.0)) + ((((data["var_44"]) + (data["var_177"]))/2.0)))/2.0)) / 2.0)) 

        v["i288"] = 0.020000*np.tanh(((2.0) - (((((data["var_121"]) * 2.0)) * (((2.0) - (((data["var_121"]) * (data["var_121"]))))))))) 

        v["i289"] = 0.020000*np.tanh(((np.tanh((data["var_91"]))) + (((np.tanh((((data["var_91"]) * (data["var_91"]))))) - (((data["var_91"]) * (data["var_91"]))))))) 

        v["i290"] = 0.020000*np.tanh(((((((((-1.0*((data["var_123"])))) * (data["var_123"]))) + ((-1.0*((((data["var_131"]) * (data["var_131"])))))))/2.0)) + (1.0))) 

        v["i291"] = 0.020000*np.tanh((((((((-1.0*((data["var_5"])))) * (((data["var_5"]) / 2.0)))) * (((data["var_5"]) * 2.0)))) + (((data["var_5"]) * 2.0)))) 

        v["i292"] = 0.020000*np.tanh(((((((((data["var_63"]) - (((data["var_78"]) * 2.0)))) + (data["var_43"]))/2.0)) + (((np.tanh((((data["var_78"]) * 2.0)))) * 2.0)))/2.0)) 

        v["i293"] = 0.020000*np.tanh(((((((data["var_178"]) + (((((data["var_80"]) - (data["var_15"]))) - (data["var_95"]))))/2.0)) + ((((data["var_9"]) + (data["var_129"]))/2.0)))/2.0)) 

        v["i294"] = 0.020000*np.tanh(((((((((data["var_108"]) + (-2.0))/2.0)) + (((((-1.0*((data["var_94"])))) + (-3.0))/2.0)))/2.0)) / 2.0)) 

        v["i295"] = 0.020000*np.tanh((((-1.0) + ((((data["var_98"]) + ((((data["var_76"]) + (data["var_146"]))/2.0)))/2.0)))/2.0)) 

        v["i296"] = 0.020000*np.tanh((((((((((((data["var_198"]) + (data["var_123"]))/2.0)) + (data["var_154"]))/2.0)) / 2.0)) + (-2.0))/2.0)) 

        v["i297"] = 0.020000*np.tanh((((((data["var_6"]) * (((np.tanh((data["var_53"]))) - (data["var_6"]))))) + (np.tanh((data["var_6"]))))/2.0)) 

        v["i298"] = 0.020000*np.tanh(((((np.tanh((((((((np.tanh((data["var_180"]))) * 2.0)) - (data["var_180"]))) * 2.0)))) * 2.0)) - (np.tanh((data["var_180"]))))) 

        v["i299"] = 0.020000*np.tanh(((((((data["var_45"]) + (((data["var_33"]) - (data["var_167"]))))/2.0)) + ((((data["var_131"]) + (((data["var_33"]) - (data["var_5"]))))/2.0)))/2.0)) 

        v["i300"] = 0.020000*np.tanh(((1.0) + (((1.0) + ((((((data["var_109"]) + (data["var_134"]))/2.0)) - (((data["var_134"]) * (data["var_134"]))))))))) 

        v["i301"] = 0.020000*np.tanh((((((data["var_18"]) + ((-1.0*((((data["var_18"]) * (((data["var_18"]) * (data["var_18"])))))))))/2.0)) + (data["var_18"]))) 

        v["i302"] = 0.020000*np.tanh(((((((data["var_18"]) * (((1.0) - (data["var_18"]))))) + (1.0))) + (((1.0) - (data["var_18"]))))) 

        v["i303"] = 0.020000*np.tanh((((data["var_114"]) + (((((((((-1.0*((data["var_194"])))) * (data["var_194"]))) + (data["var_56"]))/2.0)) - (data["var_18"]))))/2.0)) 

        v["i304"] = 0.020000*np.tanh(((((((data["var_198"]) + (-3.0))/2.0)) + ((((((((-3.0) + (data["var_194"]))) + (data["var_33"]))/2.0)) / 2.0)))/2.0)) 

        v["i305"] = 0.020000*np.tanh(((((((((-1.0*((((data["var_68"]) * (data["var_68"])))))) + (data["var_188"]))/2.0)) + (1.0))) * 2.0)) 

        v["i306"] = 0.020000*np.tanh((((((data["var_56"]) + (data["var_166"]))/2.0)) - ((((-1.0) + (((data["var_147"]) * (data["var_147"]))))/2.0)))) 

        v["i307"] = 0.020000*np.tanh((((((((((data["var_77"]) + (data["var_198"]))/2.0)) + (data["var_12"]))/2.0)) + ((((data["var_172"]) + (((data["var_123"]) - (data["var_26"]))))/2.0)))/2.0)) 

        v["i308"] = 0.020000*np.tanh((((((((-1.0*((((data["var_49"]) * (data["var_49"])))))) + (((data["var_169"]) + (1.0))))) + (1.0))) * 2.0)) 

        v["i309"] = 0.020000*np.tanh(((((-1.0*((((data["var_109"]) * (data["var_109"])))))) + (((((data["var_2"]) * ((-1.0*((data["var_2"])))))) + (2.0))))/2.0)) 

        v["i310"] = 0.020000*np.tanh(((((((data["var_43"]) + ((((-1.0*((data["var_33"])))) * (data["var_33"]))))/2.0)) + ((((data["var_73"]) + (data["var_141"]))/2.0)))/2.0)) 

        v["i311"] = 0.020000*np.tanh(((data["var_188"]) - ((((((data["var_171"]) * (data["var_171"]))) + (((-2.0) + (((data["var_188"]) * (data["var_171"]))))))/2.0)))) 

        v["i312"] = 0.020000*np.tanh((((((1.0) - (((data["var_132"]) * (data["var_132"]))))) + ((((data["var_150"]) + ((((data["var_150"]) + (data["var_88"]))/2.0)))/2.0)))/2.0)) 

        v["i313"] = 0.020000*np.tanh(((data["var_127"]) - (np.tanh((np.tanh((((((data["var_127"]) + (data["var_127"]))) + (data["var_127"]))))))))) 

        v["i314"] = 0.020000*np.tanh(((((((data["var_114"]) + (((((-1.0*((data["var_6"])))) + (data["var_154"]))/2.0)))/2.0)) + ((((data["var_127"]) + (((data["var_33"]) / 2.0)))/2.0)))/2.0)) 

        v["i315"] = 0.020000*np.tanh((((((1.0)) - (((((data["var_163"]) * (data["var_163"]))) / 2.0)))) - (((((data["var_1"]) * (data["var_1"]))) / 2.0)))) 

        v["i316"] = 0.020000*np.tanh(((((-1.0*((((data["var_28"]) * ((((data["var_28"]) + (data["var_28"]))/2.0))))))) + (data["var_150"]))/2.0)) 

        v["i317"] = 0.020000*np.tanh(((((3.0) - ((((-1.0*((data["var_107"])))) * 2.0)))) * (((3.0) - (((data["var_164"]) * 2.0)))))) 

        v["i318"] = 0.020000*np.tanh(((np.tanh((data["var_164"]))) + ((((-1.0*((((data["var_164"]) * (data["var_164"])))))) + (((np.tanh((data["var_164"]))) * 2.0)))))) 

        v["i319"] = 0.020000*np.tanh(((np.tanh((np.tanh((((((((3.0) - (((data["var_53"]) * 2.0)))) * 2.0)) * 2.0)))))) * (((data["var_53"]) * 2.0)))) 

        v["i320"] = 0.020000*np.tanh((((-1.0*((data["var_121"])))) - (((((((((((-3.0) - (((data["var_121"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) 

        v["i321"] = 0.020000*np.tanh((((data["var_162"]) + ((((-1.0*((((((((data["var_162"]) * (data["var_162"]))) * (data["var_162"]))) * (data["var_162"])))))) / 2.0)))/2.0)) 

        v["i322"] = 0.020000*np.tanh((((np.tanh((((data["var_109"]) * (data["var_109"]))))) + ((-1.0*((((data["var_109"]) * (data["var_109"])))))))/2.0)) 

        v["i323"] = 0.020000*np.tanh((((data["var_82"]) + (((data["var_82"]) - (((((data["var_82"]) * (((data["var_82"]) * (data["var_82"]))))) * (data["var_82"]))))))/2.0)) 

        v["i324"] = 0.020000*np.tanh(((data["var_86"]) * ((((-1.0*((data["var_86"])))) - (np.tanh((((((data["var_82"]) - (data["var_86"]))) - (data["var_86"]))))))))) 

        v["i325"] = 0.020000*np.tanh(((((((((2.31803345680236816)) - (data["var_155"]))) - (data["var_53"]))) + (((((1.0) - (data["var_53"]))) * (data["var_53"]))))/2.0)) 

        v["i326"] = 0.020000*np.tanh(((data["var_67"]) - (((((((((data["var_67"]) * (data["var_67"]))) / 2.0)) * (data["var_67"]))) * (((data["var_67"]) / 2.0)))))) 

        v["i327"] = 0.020000*np.tanh(((data["var_141"]) + (((((1.0) - (((data["var_139"]) * (data["var_139"]))))) + (1.0))))) 

        v["i328"] = 0.020000*np.tanh((((data["var_43"]) + ((((data["var_141"]) + (((data["var_75"]) - (((data["var_118"]) + (data["var_191"]))))))/2.0)))/2.0)) 

        v["i329"] = 0.020000*np.tanh(((((((1.0) + (((((((((data["var_166"]) + (1.0))) * 2.0)) + (1.0))) * 2.0)))) * 2.0)) * 2.0)) 

        v["i330"] = 0.020000*np.tanh(((data["var_4"]) * (((data["var_4"]) * (((((data["var_4"]) * ((((-1.0*((((data["var_4"]) / 2.0))))) / 2.0)))) / 2.0)))))) 

        v["i331"] = 0.020000*np.tanh((((np.tanh((data["var_31"]))) + (((((((np.tanh((data["var_31"]))) + (np.tanh((-1.0))))/2.0)) + (-1.0))/2.0)))/2.0)) 

        v["i332"] = 0.020000*np.tanh((-1.0*((((data["var_34"]) * (((np.tanh(((-1.0*((((data["var_34"]) * 2.0))))))) + (((0.318310) + (data["var_34"])))))))))) 

        v["i333"] = 0.020000*np.tanh((((((((((-1.0*((data["var_22"])))) - ((((-3.0) + ((-1.0*((((0.318310) / 2.0))))))/2.0)))) * 2.0)) * 2.0)) * 2.0)) 

        v["i334"] = 0.020000*np.tanh((((((((-1.0*((data["var_12"])))) * ((((-1.0*((data["var_12"])))) * (data["var_12"]))))) / 2.0)) + ((-1.0*((data["var_12"])))))) 

        v["i335"] = 0.020000*np.tanh((((((((0.74987429380416870)) * 2.0)) * 2.0)) - (((data["var_108"]) * (data["var_108"]))))) 

        v["i336"] = 0.020000*np.tanh((((-1.0*((((data["var_75"]) * (((data["var_75"]) / 2.0))))))) + (np.tanh((((data["var_75"]) * (data["var_75"]))))))) 

        v["i337"] = 0.020000*np.tanh(np.tanh((((((((np.tanh(((6.0)))) / 2.0)) - (((-3.0) + (((data["var_78"]) * 2.0)))))) * 2.0)))) 

        v["i338"] = 0.020000*np.tanh(((((((((np.tanh((data["var_186"]))) + (data["var_34"]))/2.0)) + (np.tanh((((data["var_78"]) * 2.0)))))) + (np.tanh((data["var_22"]))))/2.0)) 

        v["i339"] = 0.020000*np.tanh((((((data["var_0"]) * (((data["var_0"]) * (((((data["var_0"]) * ((-1.0*((data["var_0"])))))) / 2.0)))))) + (data["var_0"]))/2.0)) 

        v["i340"] = 0.020000*np.tanh(((np.tanh((((3.0) - (((data["var_5"]) * 2.0)))))) + (((data["var_170"]) * ((-1.0*((((data["var_170"]) / 2.0))))))))) 

        v["i341"] = 0.020000*np.tanh(((((data["var_5"]) * 2.0)) + ((-1.0*((((data["var_5"]) * ((((((data["var_5"]) * (data["var_5"]))) + (data["var_5"]))/2.0))))))))) 

        v["i342"] = 0.020000*np.tanh((((-1.0*((data["var_109"])))) + (((((((data["var_109"]) * ((-1.0*((data["var_109"])))))) / 2.0)) * ((-1.0*((data["var_109"])))))))) 

        v["i343"] = 0.020000*np.tanh(((np.tanh((((((3.0) - (((data["var_130"]) * 2.0)))) * 2.0)))) * (((3.0) * 2.0)))) 

        v["i344"] = 0.020000*np.tanh(((data["var_133"]) * ((((np.tanh((data["var_133"]))) + ((-1.0*((data["var_133"])))))/2.0)))) 

        v["i345"] = 0.020000*np.tanh((((np.tanh((data["var_130"]))) + ((((data["var_197"]) + (((data["var_21"]) * ((-1.0*(((((data["var_21"]) + (data["var_21"]))/2.0))))))))/2.0)))/2.0)) 

        v["i346"] = 0.020000*np.tanh((-1.0*((((((data["var_23"]) * (((((data["var_23"]) / 2.0)) * ((-1.0*((-3.0)))))))) / 2.0))))) 

        v["i347"] = 0.020000*np.tanh((-1.0*((((data["var_173"]) * ((((data["var_173"]) + (np.tanh((np.tanh(((-1.0*((data["var_173"])))))))))/2.0))))))) 

        v["i348"] = 0.020000*np.tanh((((np.tanh((((np.tanh((((data["var_128"]) * 2.0)))) * 2.0)))) + (np.tanh((data["var_145"]))))/2.0)) 

        v["i349"] = 0.020000*np.tanh((((((-1.0*(((((((data["var_186"]) + (data["var_186"]))/2.0)) * (data["var_186"])))))) + (((data["var_136"]) / 2.0)))) / 2.0)) 

        v["i350"] = 0.020000*np.tanh((((np.tanh((np.tanh((np.tanh((((data["var_173"]) * 2.0)))))))) + (((np.tanh((-1.0))) * 2.0)))/2.0)) 

        v["i351"] = 0.020000*np.tanh((((((data["var_155"]) + (((data["var_155"]) * 2.0)))) + ((-1.0*((((data["var_155"]) * (((data["var_155"]) * (data["var_155"])))))))))/2.0)) 

        v["i352"] = 0.020000*np.tanh(((((2.0) - (((data["var_179"]) + (((data["var_179"]) - (((2.0) / 2.0)))))))) * 2.0)) 

        v["i353"] = 0.020000*np.tanh((((np.tanh((((data["var_81"]) * (data["var_139"]))))) + ((((-1.0*((((data["var_148"]) / 2.0))))) * (data["var_148"]))))/2.0)) 

        v["i354"] = 0.020000*np.tanh(((((((np.tanh(((-1.0*((((-3.0) - (((data["var_121"]) * 2.0))))))))) * 2.0)) * 2.0)) * 2.0)) 

        v["i355"] = 0.020000*np.tanh(((((data["var_121"]) * ((-1.0*((data["var_121"])))))) + (((np.tanh(((((-1.0*((data["var_121"])))) * 2.0)))) * 2.0)))) 

        v["i356"] = 0.020000*np.tanh(((((((((-1.0*((data["var_80"])))) * (data["var_80"]))) / 2.0)) + (((data["var_121"]) / 2.0)))/2.0)) 

        v["i357"] = 0.020000*np.tanh(((((np.tanh((np.tanh((data["var_40"]))))) - (((data["var_40"]) / 2.0)))) * (data["var_40"]))) 

        v["i358"] = 0.020000*np.tanh((((((np.tanh((((data["var_179"]) * 2.0)))) + ((((data["var_76"]) + (data["var_33"]))/2.0)))) + (np.tanh((np.tanh((data["var_40"]))))))/2.0)) 

        v["i359"] = 0.020000*np.tanh(((((((3.0) + (((data["var_177"]) * 2.0)))) * (((3.0) - (((data["var_94"]) * 2.0)))))) * (3.0))) 

        v["i360"] = 0.020000*np.tanh(((3.0) * (((3.0) * (((data["var_137"]) * (((((3.0) / 2.0)) - (data["var_137"]))))))))) 

        v["i361"] = 0.020000*np.tanh(((((((data["var_41"]) + ((((data["var_186"]) + (data["var_194"]))/2.0)))/2.0)) + ((((((data["var_94"]) + (data["var_156"]))/2.0)) / 2.0)))/2.0)) 

        v["i362"] = 0.020000*np.tanh(((data["var_34"]) * (((data["var_34"]) * ((((-1.0*((((data["var_34"]) * (data["var_34"])))))) - (-2.0))))))) 

        v["i363"] = 0.020000*np.tanh(((((((((((data["var_81"]) - (data["var_137"]))) - (data["var_138"]))) + (data["var_132"]))/2.0)) + ((((data["var_42"]) + (data["var_72"]))/2.0)))/2.0)) 

        v["i364"] = 0.020000*np.tanh((-1.0*((((((data["var_152"]) - ((((-1.0*((1.0)))) / 2.0)))) * (data["var_152"])))))) 

        v["i365"] = 0.020000*np.tanh(((((data["var_87"]) / 2.0)) + (np.tanh((((((12.27435970306396484)) + (((((((data["var_177"]) * 2.0)) * 2.0)) * 2.0)))/2.0)))))) 

        v["i366"] = 0.020000*np.tanh(((((data["var_45"]) + (((np.tanh((np.tanh((np.tanh((((-3.0) * (data["var_45"]))))))))) * 2.0)))) * 2.0)) 

        v["i367"] = 0.020000*np.tanh(((((np.tanh((((((data["var_70"]) * 2.0)) * 2.0)))) - (((((data["var_70"]) * 2.0)) / 2.0)))) * 2.0)) 

        v["i368"] = 0.020000*np.tanh(((((((((data["var_146"]) + (data["var_148"]))/2.0)) + (np.tanh((data["var_125"]))))) + (((((-1.0*((data["var_196"])))) + (data["var_152"]))/2.0)))/2.0)) 

        v["i369"] = 0.020000*np.tanh((((((((((((2.0)) - (data["var_170"]))) * 2.0)) * 2.0)) * 2.0)) - ((2.0)))) 

        v["i370"] = 0.020000*np.tanh(((((((np.tanh((((data["var_82"]) * 2.0)))) * 2.0)) - (data["var_82"]))) + (((np.tanh((data["var_82"]))) - (data["var_82"]))))) 

        v["i371"] = 0.020000*np.tanh(((data["var_127"]) + (((((data["var_127"]) + (((((((((data["var_127"]) + ((2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))) 

        v["i372"] = 0.020000*np.tanh(((np.tanh((data["var_26"]))) + ((((-1.0*((((((((data["var_26"]) * (data["var_26"]))) * (data["var_26"]))) / 2.0))))) / 2.0)))) 

        v["i373"] = 0.020000*np.tanh(((((-1.0*((-3.0)))) + (((data["var_169"]) * ((-1.0*((((data["var_169"]) * ((((-1.0*((-3.0)))) / 2.0))))))))))/2.0)) 

        v["i374"] = 0.020000*np.tanh((((((((((-1.0*((-1.0)))) + (data["var_175"]))/2.0)) + ((-1.0*((((data["var_175"]) * (data["var_175"])))))))) + (0.318310))/2.0)) 

        v["i375"] = 0.020000*np.tanh(((1.0) - (((((1.0) - (((((data["var_33"]) / 2.0)) * (data["var_33"]))))) * (data["var_33"]))))) 

        v["i376"] = 0.020000*np.tanh(((((-1.0*((data["var_197"])))) + ((-1.0*(((((-1.0*((((((data["var_197"]) / 2.0)) * (data["var_197"])))))) * (data["var_197"])))))))/2.0)) 

        v["i377"] = 0.020000*np.tanh(((((data["var_111"]) * 2.0)) * (((((np.tanh((np.tanh((((data["var_111"]) * 2.0)))))) * 2.0)) + ((-1.0*((data["var_111"])))))))) 

        v["i378"] = 0.020000*np.tanh(((((0.318310) + (0.318310))) + ((((((data["var_188"]) * ((-1.0*((data["var_188"])))))) + (data["var_12"]))/2.0)))) 

        v["i379"] = 0.020000*np.tanh(((((np.tanh((data["var_105"]))) + (((data["var_105"]) * ((-1.0*((data["var_105"])))))))) + (np.tanh(((-1.0*((-1.0)))))))) 

        v["i380"] = 0.020000*np.tanh((((((((data["var_50"]) / 2.0)) + ((((-3.0) + (-3.0))/2.0)))/2.0)) / 2.0)) 

        v["i381"] = 0.020000*np.tanh((((data["var_128"]) + (((((data["var_128"]) * (((data["var_128"]) * ((-1.0*((data["var_128"])))))))) + (data["var_128"]))))/2.0)) 

        v["i382"] = 0.020000*np.tanh(((((((((-1.0*((data["var_67"])))) + (data["var_80"]))/2.0)) * ((((data["var_67"]) + ((-1.0*((data["var_80"])))))/2.0)))) / 2.0)) 

        v["i383"] = 0.020000*np.tanh((((-1.0*(((((((3.0)) + (((data["var_86"]) / 2.0)))) / 2.0))))) / 2.0)) 

        v["i384"] = 0.020000*np.tanh((((((np.tanh((((data["var_28"]) + (data["var_28"]))))) + ((-1.0*((((data["var_28"]) * (data["var_28"])))))))/2.0)) / 2.0)) 

        v["i385"] = 0.020000*np.tanh((((((((np.tanh(((((data["var_149"]) + (data["var_86"]))/2.0)))) + (data["var_86"]))) + (data["var_33"]))/2.0)) / 2.0)) 

        v["i386"] = 0.020000*np.tanh(((((((2.08380508422851562)) - (((((data["var_93"]) / 2.0)) * (data["var_93"]))))) + ((-1.0*((((data["var_66"]) * (data["var_66"])))))))/2.0)) 

        v["i387"] = 0.020000*np.tanh(((((((data["var_109"]) + (data["var_80"]))/2.0)) + ((((data["var_83"]) + (((np.tanh((data["var_112"]))) - (data["var_71"]))))/2.0)))/2.0)) 

        v["i388"] = 0.020000*np.tanh(((((((((np.tanh(((((((((data["var_92"]) + (2.0))/2.0)) * 2.0)) * 2.0)))) * 2.0)) - (data["var_184"]))) * 2.0)) * 2.0)) 

        v["i389"] = 0.020000*np.tanh((-1.0*(((((((data["var_154"]) + (((((data["var_13"]) / 2.0)) / 2.0)))/2.0)) * (((data["var_154"]) / 2.0))))))) 

        v["i390"] = 0.020000*np.tanh(((data["var_115"]) + (((np.tanh((((((((((data["var_77"]) + (1.0))) * 2.0)) + (1.0))) * 2.0)))) * 2.0)))) 

        v["i391"] = 0.020000*np.tanh(((np.tanh((((data["var_181"]) * (data["var_181"]))))) + ((-1.0*((((((data["var_181"]) / 2.0)) * (data["var_181"])))))))) 

        v["i392"] = 0.020000*np.tanh((((((((((-1.0*((((data["var_146"]) * (data["var_146"])))))) + (data["var_13"]))/2.0)) + ((((data["var_103"]) + (data["var_58"]))/2.0)))/2.0)) / 2.0)) 

        v["i393"] = 0.020000*np.tanh(((((((((data["var_139"]) + (-2.0))/2.0)) + ((((data["var_38"]) + ((((data["var_146"]) + (data["var_139"]))/2.0)))/2.0)))/2.0)) / 2.0)) 

        v["i394"] = 0.020000*np.tanh(((((((data["var_149"]) + (((((((data["var_121"]) / 2.0)) / 2.0)) - (data["var_151"]))))/2.0)) + ((-1.0*((((data["var_179"]) / 2.0))))))/2.0)) 

        v["i395"] = 0.020000*np.tanh(((data["var_165"]) * ((-1.0*(((((data["var_165"]) + ((-1.0*((np.tanh((((data["var_165"]) * 2.0))))))))/2.0))))))) 

        v["i396"] = 0.020000*np.tanh(((((1.0) + (((1.0) + (((((data["var_166"]) * ((-1.0*((data["var_166"])))))) + (1.0))))))) * 2.0)) 

        v["i397"] = 0.019990*np.tanh((((((((((data["var_146"]) + ((-1.0*((data["var_6"])))))/2.0)) + ((-1.0*((((data["var_198"]) * (data["var_198"])))))))/2.0)) + (0.318310))/2.0)) 

        v["i398"] = 0.020000*np.tanh((-1.0*((((((((((((1.0) + (data["var_110"]))/2.0)) * (data["var_110"]))) + ((((1.0) + (data["var_110"]))/2.0)))/2.0)) / 2.0))))) 

        v["i399"] = 0.020000*np.tanh(((np.tanh((((data["var_97"]) * 2.0)))) + ((((-1.0*((((data["var_97"]) / 2.0))))) * (data["var_97"]))))) 

        v["i400"] = 0.020000*np.tanh((((((data["var_174"]) * ((-1.0*(((((data["var_174"]) + (data["var_174"]))/2.0))))))) + (((data["var_114"]) + ((-1.0*((data["var_174"])))))))/2.0)) 

        v["i401"] = 0.020000*np.tanh(((((((data["var_59"]) + ((((data["var_116"]) + (data["var_174"]))/2.0)))/2.0)) + ((((data["var_174"]) + (np.tanh(((-1.0*((data["var_174"])))))))/2.0)))/2.0)) 

        v["i402"] = 0.020000*np.tanh((((data["var_192"]) + ((-1.0*((((((((data["var_192"]) * (data["var_192"]))) * (data["var_192"]))) * (data["var_192"])))))))/2.0)) 

        v["i403"] = 0.020000*np.tanh((((data["var_9"]) + (((1.0) + (((1.0) - (((data["var_32"]) * (data["var_32"]))))))))/2.0)) 

        v["i404"] = 0.020000*np.tanh(((((data["var_9"]) / 2.0)) + (((data["var_9"]) + (((np.tanh((((data["var_9"]) * (data["var_9"]))))) * 2.0)))))) 

        v["i405"] = 0.020000*np.tanh(((((data["var_9"]) + (np.tanh((((data["var_9"]) * (data["var_9"]))))))) + (np.tanh((((data["var_9"]) * (data["var_9"]))))))) 

        v["i406"] = 0.020000*np.tanh((((np.tanh((((data["var_9"]) * (data["var_9"]))))) + ((-1.0*((((((data["var_9"]) * (data["var_9"]))) / 2.0))))))/2.0)) 

        v["i407"] = 0.020000*np.tanh(((data["var_187"]) - (((((data["var_187"]) * (((data["var_187"]) / 2.0)))) * (((data["var_187"]) * (data["var_187"]))))))) 

        v["i408"] = 0.020000*np.tanh(np.tanh(((((3.0) + ((((-1.0*((data["var_9"])))) * (data["var_9"]))))/2.0)))) 

        v["i409"] = 0.020000*np.tanh((((data["var_156"]) + (((((((data["var_156"]) + ((-1.0*((data["var_149"])))))/2.0)) + (((data["var_9"]) + ((1.0)))))/2.0)))/2.0)) 

        v["i410"] = 0.020000*np.tanh((((-1.0*((((data["var_2"]) * ((((data["var_2"]) + (data["var_32"]))/2.0))))))) / 2.0)) 

        v["i411"] = 0.020000*np.tanh(np.tanh((((((((np.tanh((((((data["var_34"]) + ((2.0)))) * 2.0)))) * 2.0)) - (data["var_180"]))) * 2.0)))) 

        v["i412"] = 0.020000*np.tanh(((((((data["var_6"]) * ((-1.0*((data["var_6"])))))) / 2.0)) + (np.tanh((((data["var_6"]) * (data["var_6"]))))))) 

        v["i413"] = 0.020000*np.tanh(((((((1.0) + (data["var_76"]))/2.0)) + (((((1.0) - (((data["var_94"]) * (data["var_94"]))))) + (1.0))))/2.0)) 

        v["i414"] = 0.020000*np.tanh((((((np.tanh((((((data["var_94"]) * (data["var_94"]))) * 2.0)))) * 2.0)) + ((-1.0*((((data["var_102"]) * (data["var_102"])))))))/2.0)) 

        v["i415"] = 0.020000*np.tanh((((((((data["var_152"]) * ((-1.0*((data["var_152"])))))) / 2.0)) + (((data["var_20"]) + (np.tanh(((-1.0*((data["var_20"])))))))))/2.0)) 

        v["i416"] = 0.020000*np.tanh(((((((np.tanh((((((((2.0) + (data["var_13"]))) * 2.0)) * 2.0)))) - (((data["var_144"]) / 2.0)))) * 2.0)) * 2.0)) 

        v["i417"] = 0.017802*np.tanh(((((np.tanh((np.tanh((((((-1.0) + (((data["var_184"]) * 2.0)))) * 2.0)))))) * 2.0)) / 2.0)) 

        v["i418"] = 0.020000*np.tanh(((((data["var_20"]) - (np.tanh((((np.tanh((data["var_20"]))) + ((((0.318310) + (data["var_20"]))/2.0)))))))) * 2.0)) 

        v["i419"] = 0.020000*np.tanh((((np.tanh((data["var_60"]))) + (((1.0) + (((data["var_60"]) * ((-1.0*((data["var_60"])))))))))/2.0)) 

        v["i420"] = 0.020000*np.tanh((((((np.tanh((((((data["var_170"]) * 2.0)) * 2.0)))) + (np.tanh((data["var_94"]))))) + ((((data["var_83"]) + (data["var_20"]))/2.0)))/2.0)) 

        v["i421"] = 0.020000*np.tanh(((((((((data["var_108"]) + (((np.tanh((((((2.0) - (data["var_78"]))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) 

        v["i422"] = 0.020000*np.tanh((((np.tanh((np.tanh((((data["var_8"]) * 2.0)))))) + ((((data["var_63"]) + (np.tanh((((((data["var_90"]) * 2.0)) * 2.0)))))/2.0)))/2.0)) 

        v["i423"] = 0.020000*np.tanh(((((((((-1.0*((((data["var_8"]) + (data["var_168"])))))) / 2.0)) + (np.tanh((((data["var_22"]) * 2.0)))))/2.0)) / 2.0)) 

        v["i424"] = 0.019990*np.tanh((((((data["var_132"]) + (np.tanh(((-1.0*((((((data["var_132"]) * 2.0)) * 2.0))))))))/2.0)) * 2.0)) 

        v["i425"] = 0.020000*np.tanh((((1.0) + (((data["var_86"]) * ((-1.0*(((((1.0) + (((data["var_86"]) * ((-1.0*((data["var_86"])))))))/2.0))))))))/2.0)) 

        v["i426"] = 0.020000*np.tanh((((((((((3.0) + (((data["var_198"]) + (data["var_36"]))))/2.0)) + (data["var_183"]))/2.0)) + ((((data["var_158"]) + (data["var_123"]))/2.0)))/2.0)) 

        v["i427"] = 0.020000*np.tanh((((((data["var_139"]) * (data["var_133"]))) + (np.tanh((((((data["var_133"]) * 2.0)) * 2.0)))))/2.0)) 

        v["i428"] = 0.020000*np.tanh(((((((((-1.0*((data["var_92"])))) * (data["var_92"]))) / 2.0)) + (np.tanh((((((data["var_92"]) * 2.0)) * (data["var_92"]))))))/2.0)) 

        v["i429"] = 0.020000*np.tanh(np.tanh((((((-1.0*((np.tanh((data["var_132"])))))) + ((((data["var_81"]) + ((((((data["var_179"]) / 2.0)) + (data["var_76"]))/2.0)))/2.0)))/2.0)))) 

        v["i430"] = 0.020000*np.tanh((((((-1.0*((((data["var_177"]) / 2.0))))) * ((((data["var_81"]) + (data["var_177"]))/2.0)))) / 2.0)) 

        v["i431"] = 0.020000*np.tanh(((((-1.0*((((data["var_188"]) * (data["var_188"])))))) + ((-1.0*((data["var_188"])))))/2.0)) 

        v["i432"] = 0.020000*np.tanh((-1.0*((((data["var_75"]) * ((((-1.0*((((data["var_75"]) * (((data["var_75"]) / 2.0))))))) - (-1.0)))))))) 

        v["i433"] = 0.020000*np.tanh((((((data["var_132"]) + (data["var_56"]))/2.0)) * ((((((data["var_56"]) * (data["var_56"]))) + ((((data["var_56"]) + (data["var_56"]))/2.0)))/2.0)))) 

        v["i434"] = 0.020000*np.tanh(((((data["var_141"]) - ((((np.tanh((data["var_92"]))) + (np.tanh((np.tanh((data["var_156"]))))))/2.0)))) - (np.tanh((data["var_141"]))))) 

        v["i435"] = 0.020000*np.tanh((((((data["var_131"]) * ((-1.0*((data["var_131"])))))) + ((((-1.0*((((data["var_131"]) + (data["var_131"])))))) / 2.0)))/2.0)) 

        v["i436"] = 0.020000*np.tanh(((((-1.0*((data["var_131"])))) + (((((data["var_131"]) * ((-1.0*((((data["var_131"]) / 2.0))))))) * ((-1.0*((data["var_131"])))))))/2.0)) 

        v["i437"] = 0.020000*np.tanh(((((data["var_131"]) + (2.0))) * (((data["var_109"]) + (((((((data["var_109"]) + (2.0))) * 2.0)) * 2.0)))))) 

        v["i438"] = 0.020000*np.tanh(((3.0) - (((((data["var_195"]) * (data["var_195"]))) + (((((data["var_163"]) - (data["var_56"]))) - (data["var_21"]))))))) 

        v["i439"] = 0.020000*np.tanh(((((((data["var_127"]) + (((data["var_188"]) + (data["var_131"]))))/2.0)) + (np.tanh((data["var_163"]))))/2.0)) 

        v["i440"] = 0.020000*np.tanh(((((((data["var_92"]) + (np.tanh((data["var_15"]))))/2.0)) + ((((data["var_21"]) + (np.tanh((data["var_170"]))))/2.0)))/2.0)) 

        v["i441"] = 0.020000*np.tanh(np.tanh((np.tanh((np.tanh(((((((0.318310) + (((((((data["var_127"]) + (0.318310))/2.0)) + (data["var_199"]))/2.0)))/2.0)) * 2.0)))))))) 

        v["i442"] = 0.017499*np.tanh(((((((0.318310) - (data["var_142"]))) - (((((((data["var_142"]) * 2.0)) / 2.0)) * (data["var_142"]))))) * 2.0)) 

        v["i443"] = 0.020000*np.tanh(np.tanh((np.tanh(((((-1.0*((((-3.0) + (((data["var_196"]) * 2.0))))))) * 2.0)))))) 

        v["i444"] = 0.020000*np.tanh(((((((np.tanh((data["var_196"]))) + (np.tanh((data["var_196"]))))/2.0)) + ((((data["var_142"]) + (data["var_141"]))/2.0)))/2.0)) 

        v["i445"] = 0.020000*np.tanh(((data["var_191"]) * (((((-1.0*((((data["var_191"]) * (((data["var_191"]) * (((data["var_191"]) / 2.0))))))))) + (data["var_191"]))/2.0)))) 

        v["i446"] = 0.020000*np.tanh((((((((-1.0*((data["var_51"])))) + (np.tanh((data["var_31"]))))/2.0)) + ((((((data["var_140"]) / 2.0)) + (data["var_165"]))/2.0)))/2.0)) 

        v["i447"] = 0.020000*np.tanh(((data["var_78"]) + (((((data["var_78"]) - (((data["var_78"]) * (data["var_78"]))))) * (data["var_78"]))))) 

        v["i448"] = 0.019980*np.tanh((((np.tanh((np.tanh((data["var_78"]))))) + ((((((data["var_122"]) + (data["var_139"]))/2.0)) - (((data["var_81"]) * (data["var_6"]))))))/2.0)) 

        v["i449"] = 0.020000*np.tanh((((((data["var_160"]) / 2.0)) + (((((((((((data["var_44"]) + (data["var_114"]))/2.0)) * 2.0)) + (data["var_30"]))/2.0)) / 2.0)))/2.0)) 

        v["i450"] = 0.020000*np.tanh((((((-1.0*((data["var_35"])))) / 2.0)) * ((((data["var_35"]) + ((((data["var_134"]) + ((((data["var_35"]) + (data["var_104"]))/2.0)))/2.0)))/2.0)))) 

        v["i451"] = 0.020000*np.tanh(((data["var_94"]) * ((((np.tanh((data["var_94"]))) + ((((data["var_12"]) + ((-1.0*((data["var_94"])))))/2.0)))/2.0)))) 

        v["i452"] = 0.020000*np.tanh(((((((data["var_198"]) + ((((data["var_88"]) + (data["var_104"]))/2.0)))/2.0)) + ((((data["var_85"]) + (np.tanh(((13.87759876251220703)))))/2.0)))/2.0)) 

        v["i453"] = 0.020000*np.tanh(((((data["var_107"]) * 2.0)) * (((((((np.tanh((np.tanh((data["var_107"]))))) - (((data["var_107"]) / 2.0)))) * 2.0)) * 2.0)))) 

        v["i454"] = 0.019990*np.tanh((((((((data["var_37"]) * (((data["var_37"]) * (((np.tanh((data["var_37"]))) - (data["var_37"]))))))) + (data["var_198"]))/2.0)) / 2.0)) 

        v["i455"] = 0.019980*np.tanh(((((data["var_21"]) * ((((((((data["var_40"]) - (data["var_23"]))) + (data["var_1"]))) + (data["var_62"]))/2.0)))) / 2.0)) 

        v["i456"] = 0.020000*np.tanh(((((((3.0) + (((((((3.0) + (((data["var_92"]) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) 

        v["i457"] = 0.020000*np.tanh((((((((data["var_89"]) + (((data["var_89"]) * ((-1.0*((data["var_89"])))))))/2.0)) + (0.318310))) / 2.0)) 

        v["i458"] = 0.020000*np.tanh(((((((np.tanh((data["var_37"]))) + ((-1.0*((((1.0) / 2.0))))))/2.0)) + ((-1.0*((((1.0) / 2.0))))))/2.0)) 

        v["i459"] = 0.020000*np.tanh((((0.318310) + (np.tanh((np.tanh((((((((data["var_18"]) * 2.0)) * 2.0)) * 2.0)))))))/2.0)) 

        v["i460"] = 0.020000*np.tanh((((((data["var_88"]) + ((((((((((data["var_88"]) + (data["var_156"]))/2.0)) / 2.0)) / 2.0)) / 2.0)))/2.0)) - (np.tanh((data["var_88"]))))) 

        v["i461"] = 0.020000*np.tanh(((((((np.tanh((np.tanh(((-1.0*((data["var_92"])))))))) + ((-1.0*((data["var_99"])))))/2.0)) + (np.tanh((np.tanh((data["var_99"]))))))/2.0)) 

        v["i462"] = 0.020000*np.tanh(((((np.tanh((data["var_174"]))) - (((((np.tanh((data["var_174"]))) - (data["var_174"]))) * (data["var_174"]))))) * (data["var_174"]))) 

        v["i463"] = 0.020000*np.tanh((((((((data["var_76"]) / 2.0)) + (data["var_88"]))/2.0)) / 2.0)) 

        v["i464"] = 0.020000*np.tanh((-1.0*((((-3.0) + (((data["var_130"]) * (data["var_130"])))))))) 

        v["i465"] = 0.020000*np.tanh((((data["var_15"]) + (((data["var_15"]) * (((((data["var_15"]) * (((data["var_15"]) * ((-1.0*((data["var_15"])))))))) / 2.0)))))/2.0)) 

        v["i466"] = 0.020000*np.tanh(((((data["var_174"]) + (((((((((data["var_174"]) + (2.0))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) 

        v["i467"] = 0.020000*np.tanh(((((((np.tanh((data["var_99"]))) + (((data["var_45"]) / 2.0)))/2.0)) + (((data["var_139"]) / 2.0)))/2.0)) 

        v["i468"] = 0.020000*np.tanh((((-1.0*(((((data["var_108"]) + (((data["var_108"]) * (data["var_108"]))))/2.0))))) + (np.tanh((((data["var_108"]) * (data["var_108"]))))))) 

        v["i469"] = 0.020000*np.tanh((((((data["var_108"]) + (data["var_113"]))/2.0)) / 2.0)) 

        v["i470"] = 0.020000*np.tanh((((((((1.0) + (((data["var_5"]) * (((data["var_5"]) * (-1.0))))))/2.0)) / 2.0)) * (((data["var_5"]) / 2.0)))) 

        v["i471"] = 0.020000*np.tanh(((((data["var_53"]) * ((((data["var_53"]) + ((-1.0*((((data["var_53"]) * (data["var_53"])))))))/2.0)))) / 2.0)) 

        v["i472"] = 0.020000*np.tanh(np.tanh(((((data["var_53"]) + ((((-1.0*((((data["var_53"]) * (((data["var_53"]) / 2.0))))))) * (data["var_53"]))))/2.0)))) 

        v["i473"] = 0.020000*np.tanh(((((np.tanh((((data["var_154"]) * (data["var_154"]))))) * 2.0)) + ((-1.0*((((data["var_154"]) * (data["var_154"])))))))) 

        v["i474"] = 0.020000*np.tanh((((0.0) + (((np.tanh(((((data["var_20"]) + ((-1.0*((((2.0) * 2.0))))))/2.0)))) * 2.0)))/2.0)) 

        v["i475"] = 0.019990*np.tanh(((((((((((data["var_53"]) / 2.0)) * (((data["var_6"]) * (data["var_53"]))))) * (data["var_53"]))) + (data["var_53"]))) / 2.0)) 

        v["i476"] = 0.020000*np.tanh((((-1.0) + ((((-1.0) + ((((((-1.0) + ((((np.tanh((data["var_170"]))) + (-1.0))/2.0)))/2.0)) * 2.0)))/2.0)))/2.0)) 

        v["i477"] = 0.020000*np.tanh(((np.tanh((data["var_164"]))) - (((((((((data["var_179"]) + (((((data["var_179"]) / 2.0)) / 2.0)))/2.0)) / 2.0)) + (data["var_164"]))/2.0)))) 

        v["i478"] = 0.020000*np.tanh(((((((((np.tanh((((((((2.0) + (data["var_141"]))) * 2.0)) * 2.0)))) * 2.0)) + (data["var_80"]))) * 2.0)) * 2.0)) 

        v["i479"] = 0.019990*np.tanh(((((data["var_113"]) - (np.tanh((((data["var_113"]) * 2.0)))))) - (((((np.tanh((((data["var_113"]) * 2.0)))) / 2.0)) / 2.0)))) 

        v["i480"] = 0.020000*np.tanh((((np.tanh((np.tanh((np.tanh((data["var_51"]))))))) + (np.tanh((np.tanh((np.tanh((np.tanh((data["var_170"]))))))))))/2.0)) 

        v["i481"] = 0.020000*np.tanh((((((((np.tanh((data["var_121"]))) + (data["var_121"]))) + (((data["var_121"]) * (data["var_121"]))))/2.0)) * (((data["var_121"]) * 2.0)))) 

        v["i482"] = 0.020000*np.tanh(((((((((data["var_43"]) / 2.0)) + (((data["var_44"]) / 2.0)))/2.0)) + ((((data["var_115"]) + (np.tanh(((-1.0*((data["var_121"])))))))/2.0)))/2.0)) 

        v["i483"] = 0.020000*np.tanh((-1.0*((((((((data["var_197"]) / 2.0)) / 2.0)) * ((((((data["var_197"]) / 2.0)) + (data["var_197"]))/2.0))))))) 

        v["i484"] = 0.020000*np.tanh((((((-2.0) + (((((((data["var_113"]) + (-2.0))/2.0)) + ((((((-2.0) / 2.0)) + (-1.0))/2.0)))/2.0)))/2.0)) / 2.0)) 

        v["i485"] = 0.020000*np.tanh(((((((data["var_154"]) + ((-1.0*((np.tanh((data["var_154"])))))))) * 2.0)) + ((-1.0*((np.tanh((data["var_154"])))))))) 

        v["i486"] = 0.020000*np.tanh(((((((((np.tanh((((2.0) - (data["var_1"]))))) * 2.0)) * 2.0)) * (data["var_1"]))) * (data["var_1"]))) 

        v["i487"] = 0.019990*np.tanh(np.tanh(((((np.tanh((data["var_26"]))) + ((((np.tanh((((data["var_190"]) * 2.0)))) + ((-1.0*((data["var_51"])))))/2.0)))/2.0)))) 

        v["i488"] = 0.019990*np.tanh(np.tanh(((((1.0) + ((((((((data["var_172"]) / 2.0)) - (data["var_170"]))) + (data["var_116"]))/2.0)))/2.0)))) 

        v["i489"] = 0.020000*np.tanh(((((((np.tanh((((3.0) - (((data["var_78"]) * 2.0)))))) * 2.0)) * (data["var_78"]))) * (data["var_78"]))) 

        v["i490"] = 0.019971*np.tanh((((np.tanh((np.tanh((data["var_78"]))))) + ((-1.0*((((data["var_150"]) * (((((((data["var_150"]) * 2.0)) / 2.0)) / 2.0))))))))/2.0)) 

        v["i491"] = 0.020000*np.tanh((((((np.tanh((((data["var_18"]) * 2.0)))) + (((data["var_18"]) * ((-1.0*((data["var_18"])))))))/2.0)) / 2.0)) 

        v["i492"] = 0.020000*np.tanh(((((((np.tanh((data["var_26"]))) - (data["var_26"]))) * (data["var_26"]))) / 2.0)) 

        v["i493"] = 0.020000*np.tanh((((((np.tanh((data["var_1"]))) * (np.tanh((data["var_1"]))))) + (((data["var_40"]) * (((data["var_172"]) / 2.0)))))/2.0)) 

        v["i494"] = 0.020000*np.tanh(((data["var_0"]) * ((-1.0*((data["var_0"])))))) 

        v["i495"] = 0.020000*np.tanh(((data["var_93"]) * (((-1.0) - (((data["var_93"]) * (((((data["var_93"]) / 2.0)) - (data["var_93"]))))))))) 

        v["i496"] = 0.020000*np.tanh(((np.tanh((((data["var_43"]) * (((data["var_43"]) + (data["var_43"]))))))) - (((data["var_43"]) * (((data["var_43"]) / 2.0)))))) 

        v["i497"] = 0.020000*np.tanh(((((data["var_175"]) * ((((((((((((data["var_90"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) + ((-1.0*((data["var_175"])))))/2.0)))) / 2.0)) 

        v["i498"] = 0.020000*np.tanh((((np.tanh((((data["var_40"]) * (data["var_40"]))))) + (((data["var_40"]) * (((((-1.0*((data["var_40"])))) + (data["var_44"]))/2.0)))))/2.0)) 

        v["i499"] = 0.020000*np.tanh((((((np.tanh((((((np.tanh((((data["var_21"]) * 2.0)))) * 2.0)) * 2.0)))) + (np.tanh((((data["var_134"]) * 2.0)))))/2.0)) / 2.0)) 

        v["i500"] = 0.020000*np.tanh(((data["var_87"]) * ((-1.0*(((((((data["var_87"]) + ((-1.0*((((np.tanh((data["var_87"]))) / 2.0))))))/2.0)) / 2.0))))))) 

        v["i501"] = 0.020000*np.tanh((((data["var_70"]) + (((data["var_70"]) * ((-1.0*((data["var_70"])))))))/2.0)) 

        v["i502"] = 0.020000*np.tanh((((((((data["var_75"]) * (((np.tanh((((data["var_75"]) / 2.0)))) - (data["var_75"]))))) + ((-1.0*((data["var_70"])))))/2.0)) / 2.0)) 

        v["i503"] = 0.020000*np.tanh(((((((data["var_63"]) / 2.0)) + (np.tanh((((((((((data["var_9"]) - (-2.0))) * 2.0)) * 2.0)) * 2.0)))))) * 2.0)) 

        v["i504"] = 0.020000*np.tanh((((((((data["var_160"]) * ((((-1.0*((data["var_160"])))) / 2.0)))) + (np.tanh((data["var_49"]))))/2.0)) / 2.0)) 

        v["i505"] = 0.020000*np.tanh((((((((data["var_141"]) * ((((-1.0*((data["var_141"])))) / 2.0)))) + ((-1.0*((np.tanh((data["var_141"])))))))/2.0)) / 2.0)) 

        v["i506"] = 0.020000*np.tanh(((((((((data["var_123"]) - (((data["var_83"]) * (data["var_199"]))))) - (((data["var_83"]) * (data["var_83"]))))) / 2.0)) / 2.0)) 

        v["i507"] = 0.020000*np.tanh((((-1.0*((data["var_131"])))) * ((((((data["var_131"]) + ((((((((data["var_131"]) / 2.0)) / 2.0)) + (data["var_131"]))/2.0)))/2.0)) / 2.0)))) 

        v["i508"] = 0.019922*np.tanh(((np.tanh((np.tanh((((((np.tanh(((((-1.0*((0.318310)))) + (((data["var_133"]) * 2.0)))))) * 2.0)) * 2.0)))))) / 2.0)) 

        v["i509"] = 0.020000*np.tanh(((((((data["var_68"]) + (np.tanh((((((data["var_8"]) * 2.0)) * 2.0)))))/2.0)) + (np.tanh((((((data["var_8"]) * 2.0)) * 2.0)))))/2.0)) 

        v["i510"] = 0.020000*np.tanh((((((data["var_81"]) * ((((-1.0*((data["var_81"])))) / 2.0)))) + (np.tanh((((data["var_164"]) * (data["var_164"]))))))/2.0)) 

        v["i511"] = 0.020000*np.tanh((((np.tanh((np.tanh((data["var_134"]))))) + ((((0.318310) + (np.tanh((data["var_1"]))))/2.0)))/2.0)) 

        v["i512"] = 0.019873*np.tanh((((((data["var_157"]) * (data["var_93"]))) + (np.tanh((np.tanh((((data["var_40"]) * (data["var_40"]))))))))/2.0))

        if(cols!=-1):

            return v[v.columns[:cols]].sum(axis=1)

        return v
gp = GPI()

roc_auc_score(traintargets,gp.GrabPredictions(traindata,514).class_p)
pd.DataFrame({'ID_code':testids,

              'target':gp.GrabPredictions(testdata,514).class_p.values}).to_csv('gpsubmission.csv',index=False)