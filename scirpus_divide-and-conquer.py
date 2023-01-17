import numpy as np

import pandas as pd

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import gc

import warnings

warnings.filterwarnings("ignore")
os.listdir('../input/')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train.head()
structures = pd.read_csv('../input/structures.csv')



def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')

test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')
for f in ['type', 'atom_0', 'atom_1']:

    lbl = LabelEncoder()

    lbl.fit(list(train[f].values) + list(train[f].values))

    train[f] = lbl.transform(list(train[f].values))

    test[f] = lbl.transform(list(test[f].values))
def GP0(data):

    return (94.962006 +

            1.0*(((data["atom_index_1"]) - ((((((((1.0) <= (data["dist_to_type_mean"]))*1.)) + ((((1.0) <= (data["dist_to_type_mean"]))*1.)))) * ((4.90689849853515625)))))) +

            1.0*(((((14.0) * ((((data["dist_to_type_mean"]) <= (0.995526))*1.)))) + (np.minimum(((-1.0)), ((((14.0) - (data["atom_index_0"])))))))) +

            1.0*((((((((((((data["dist_to_type_mean"]) <= (0.991079))*1.)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

            0.948217*((((-1.0*((((np.minimum((((((((((6.0)) > (data["x_0"]))*1.)) > (data["dist_to_type_mean"]))*1.))), ((data["dist_to_type_mean"])))) * 2.0))))) * 2.0)) +

            1.0*(((((((((data["dist"]) * ((((((data["dist"]) > (1.100827))*1.)) * 2.0)))) * (data["dist"]))) * 2.0)) * (data["dist"]))) +

            1.0*((-1.0*(((((((data["dist_to_type_mean"]) > (0.997387))*1.)) - ((((12.0) > (data["atom_index_0"]))*1.))))))) +

            1.0*(((((0.318310) * ((11.50162220001220703)))) * ((((0.997117) > (data["dist_to_type_mean"]))*1.)))) +

            1.0*((((((((((data["dist_to_type_mean"]) <= (0.986908))*1.)) * 2.0)) * 2.0)) - (((((((data["dist_to_type_mean"]) > (0.986908))*1.)) > (data["dist_to_type_mean"]))*1.)))) +

            1.0*(((0.984850) * ((((8.0)) * (((0.984850) * ((((8.0)) * ((((0.984850) > (data["dist_to_type_mean"]))*1.)))))))))) +

            1.0*((((((data["dist_to_type_mean"]) > (np.maximum(((1.0)), ((((data["atom_index_1"]) - (2.152171)))))))*1.)) - ((((1.013243) > (data["dist_to_type_mean"]))*1.)))) +

            1.0*((((((((1.013243) <= (data["dist_to_type_mean"]))*1.)) * ((10.21938037872314453)))) * 2.0)) +

            1.0*(((((((-1.0*(((((((data["y_0"]) > (-1.0))*1.)) / 2.0))))) + ((((data["atom_index_0"]) <= (12.0))*1.)))/2.0)) * 2.0)) +

            1.0*(((-0.072825) + (((-0.072825) + (((-0.072825) + ((((0.998071) > (np.maximum(((data["dist_to_type_mean"])), ((data["y_1"])))))*1.)))))))) +

            0.922325*(((((5.0)) > ((((np.minimum((((((data["y_0"]) + (data["z_1"]))/2.0))), ((data["y_0"])))) + (data["atom_index_0"]))/2.0)))*1.)) +

            0.977528*(((((-0.543418) * ((((data["x_0"]) <= ((((data["x_1"]) + (-0.283168))/2.0)))*1.)))) * ((((-2.0) <= (data["x_1"]))*1.)))) +

            1.0*(((((-1.0*((((((1.0)) > (data["dist_to_type_mean"]))*1.))))) + (((((6.0)) > (((((data["y_1"]) * 2.0)) * 2.0)))*1.)))/2.0)))



def GP1(data):

    return (47.511509 +

            1.0*((((((((((-1.0*((data["dist"])))) + ((((((data["dist_to_type_mean"]) <= (0.998991))*1.)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

            1.0*(((3.141593) * (((1.003744) * ((((((((1.003744) > (data["dist_to_type_mean"]))*1.)) * 2.0)) - (1.387789))))))) +

            0.979971*(np.minimum(((((4.740009) - (data["dist"])))), ((((1.570796) * (((data["atom_index_1"]) * ((((0.994081) > (data["dist_to_type_mean"]))*1.))))))))) +

            0.950660*(((0.076053) - ((((((((((data["atom_index_0"]) + (-2.0))/2.0)) / 2.0)) / 2.0)) * ((((data["dist_to_type_mean"]) <= (0.998991))*1.)))))) +

            1.0*(np.minimum((((((((0.994834) > (data["dist_to_type_mean"]))*1.)) * (((0.636620) + (0.994834)))))), ((np.maximum(((data["atom_index_1"])), ((0.636620))))))) +

            1.0*(((((-1.0*(((((np.maximum(((data["z_1"])), (((-1.0*((data["z_1"]))))))) > (0.110836))*1.))))) + ((((data["atom_index_1"]) > (7.0))*1.)))/2.0)) +

            1.0*((((((((((((data["atom_index_1"]) + (((data["dist_to_type_mean"]) * 2.0)))) > (data["atom_index_0"]))*1.)) * 2.0)) * 2.0)) * 2.0)) +

            1.0*((((data["atom_index_0"]) <= (((np.maximum(((3.141593)), ((((data["atom_index_1"]) + ((((3.141593) > (((data["atom_index_1"]) / 2.0)))*1.))))))) * 2.0)))*1.)) +

            1.0*(((np.minimum((((((-1.0*(((((((data["y_0"]) / 2.0)) + (data["z_1"]))/2.0))))) / 2.0))), ((((3.0) + (data["y_0"])))))) / 2.0)) +

            0.916952*(np.minimum((((((1.0) > (data["z_0"]))*1.))), ((((((((0.0) + (data["z_0"]))/2.0)) + ((((data["dist_to_type_mean"]) > (1.0))*1.)))/2.0))))) +

            1.0*((((((((1.021625) > (data["dist"]))*1.)) - (data["dist"]))) * ((((1.021625) + (((1.021625) * 2.0)))/2.0)))) +

            1.0*((((data["dist_to_type_mean"]) <= ((((data["dist"]) > (1.011806))*1.)))*1.)) +

            1.0*((((0.998991) > (((((((0.998991) + (((0.998991) * (data["dist_to_type_mean"]))))/2.0)) > ((((data["dist_to_type_mean"]) > (0.998991))*1.)))*1.)))*1.)) +

            1.0*((((np.minimum(((data["dist_to_type_mean"])), ((((((data["atom_index_0"]) / 2.0)) / 2.0))))) <= ((((1.011806) <= (data["dist"]))*1.)))*1.)) +

            1.0*((((-1.0*(((((0.318310) > (data["z_1"]))*1.))))) * (((((((1.011806) > (data["dist"]))*1.)) > (data["atom_index_1"]))*1.)))) +

            0.922814*(np.minimum((((((1.008356) > (data["dist"]))*1.))), (((((((1.199319) * (data["z_1"]))) <= (((0.072832) * 2.0)))*1.))))))



def GP2(data):

    return (-0.268229 +

            0.937958*((-1.0*((((((((((((data["atom_index_1"]) > ((((data["y_0"]) <= (((1.194078) / 2.0)))*1.)))*1.)) * 2.0)) > (data["atom_index_1"]))*1.)) * 2.0))))) +

            1.0*(((((((((((0.994044) <= (data["dist_to_type_mean"]))*1.)) * 2.0)) - (2.0))) + ((((data["dist_to_type_mean"]) > (1.0))*1.)))/2.0)) +

            1.0*((((((1.032724) <= (data["dist_to_type_mean"]))*1.)) - (np.minimum((((((data["dist_to_type_mean"]) > (((((12.80295181274414062)) > (data["atom_index_0"]))*1.)))*1.))), ((0.166034)))))) +

            0.869565*(np.minimum(((data["dist_to_type_mean"])), ((((data["atom_index_1"]) * (((0.950018) - ((((0.950018) <= (data["dist_to_type_mean"]))*1.))))))))) +

            0.926234*(np.minimum(((((((1.570796) / 2.0)) / 2.0))), (((((((data["y_0"]) > (data["atom_index_1"]))*1.)) * (data["atom_index_1"])))))) +

            1.0*(((((((data["dist_to_type_mean"]) - (1.0))) * 2.0)) * 2.0)) +

            0.790914*(np.minimum(((data["atom_index_1"])), (((((data["atom_index_1"]) <= (np.minimum((((((data["y_0"]) > (((((data["z_0"]) / 2.0)) / 2.0)))*1.))), ((data["x_1"])))))*1.))))) +

            0.850024*(((data["x_0"]) * ((((-1.0*(((((0.166034) <= ((((data["atom_index_1"]) <= (((0.166034) / 2.0)))*1.)))*1.))))) / 2.0)))) +

            1.0*(((3.0) * (((0.982967) - ((((data["atom_index_0"]) > ((((7.0)) - (data["y_0"]))))*1.)))))) +

            1.0*((((-1.0*(((((((data["dist_to_type_mean"]) * (6.0))) <= (((((10.40789985656738281)) + ((((6.0) > (data["atom_index_0"]))*1.)))/2.0)))*1.))))) * 2.0)) +

            1.0*(((data["y_0"]) * ((((0.950018) + (((np.minimum(((((-2.0) - (data["dist_to_type_mean"])))), ((0.832110)))) + (2.080729))))/2.0)))) +

            0.999023*((((((data["x_0"]) > ((((((data["atom_index_0"]) + ((((data["y_0"]) > ((((data["x_0"]) + (data["z_0"]))/2.0)))*1.)))/2.0)) / 2.0)))*1.)) / 2.0)) +

            1.0*(np.minimum((((((data["dist"]) <= (2.080729))*1.))), (((((data["dist"]) > (((data["atom_index_1"]) - ((((2.080729) <= (data["atom_index_1"]))*1.)))))*1.))))) +

            1.0*((-1.0*(((((((data["dist_to_type_mean"]) <= (0.982494))*1.)) * (((data["dist"]) * (0.034596)))))))) +

            0.929653*((((-1.0*(((((((1.0) * 2.0)) <= (data["dist"]))*1.))))) * ((((((0.982494) > (data["dist_to_type_mean"]))*1.)) / 2.0)))) +

            0.999511*(((((((((data["dist"]) <= (2.080729))*1.)) + (((((((((data["z_0"]) <= (0.204195))*1.)) + (0.204195))/2.0)) / 2.0)))/2.0)) / 2.0)))



def GP3(data):

    return (-10.275834 +

            1.0*((-1.0*((((np.minimum(((0.636620)), (((((data["dist_to_type_mean"]) <= (((((4.59200382232666016)) <= (data["atom_index_0"]))*1.)))*1.))))) * 2.0))))) +

            1.0*(((((((np.minimum((((((data["dist_to_type_mean"]) > (1.004117))*1.))), ((data["dist_to_type_mean"])))) * (data["dist_to_type_mean"]))) * (2.0))) * (data["dist"]))) +

            0.974597*(((((((((data["dist_to_type_mean"]) * 2.0)) * ((((1.796855) <= (data["dist"]))*1.)))) * 2.0)) - (((((1.570796) / 2.0)) / 2.0)))) +

            0.787005*(((((((data["dist_to_type_mean"]) <= ((((1.765520) <= (data["dist"]))*1.)))*1.)) + ((((((1.765520) <= (data["dist"]))*1.)) - (data["dist"]))))/2.0)) +

            1.0*((((((((((((((-0.443103) <= (data["y_1"]))*1.)) > (data["dist_to_type_mean"]))*1.)) / 2.0)) + ((((1.200606) > (data["y_1"]))*1.)))/2.0)) / 2.0)) +

            0.994138*(((((((((np.minimum(((data["dist_to_type_mean"])), (((((data["dist"]) <= (1.719267))*1.))))) * (data["dist"]))) * 2.0)) * 2.0)) * (data["dist_to_type_mean"]))) +

            0.788471*(((((data["dist_to_type_mean"]) - (1.988152))) + (np.maximum((((((data["y_0"]) > (3.0))*1.))), (((((1.004117) > (data["dist_to_type_mean"]))*1.))))))) +

            1.0*((((data["dist"]) > (((1.570796) - ((((-1.0*((((((3.0) / 2.0)) * (((0.170694) * 2.0))))))) / 2.0)))))*1.)) +

            1.0*(((((np.minimum(((0.318310)), ((((((((data["dist_to_type_mean"]) - ((1.0)))) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +

            0.804592*(((((((data["y_0"]) - (((data["dist_to_type_mean"]) * (((1.004117) * (data["y_0"]))))))) * 2.0)) * 2.0)) +

            0.981925*(((0.147920) * (((((((((data["y_0"]) / 2.0)) > (0.147920))*1.)) <= ((((((data["y_0"]) / 2.0)) > (1.004117))*1.)))*1.)))) +

            0.890572*((((((-1.0*(((((((((data["y_1"]) * (data["y_1"]))) > (1.570796))*1.)) / 2.0))))) / 2.0)) / 2.0)) +

            1.0*(((((-1.0*((1.004117)))) > ((-1.0*((((data["dist_to_type_mean"]) - (((((((1.947232) <= (data["dist_to_type_mean"]))*1.)) + (0.054476))/2.0))))))))*1.)) +

            0.894480*((-1.0*((np.minimum(((0.636620)), (((((data["dist_to_type_mean"]) > (1.004117))*1.)))))))) +

            0.870542*(((((((((((data["dist"]) - (1.770059))) * (data["dist"]))) * (data["dist"]))) * (data["dist"]))) * (data["dist"]))) +

            1.0*(np.minimum((((((11.0) <= (data["atom_index_0"]))*1.))), ((((0.318310) * ((((((data["z_1"]) / 2.0)) <= (data["y_1"]))*1.))))))))



def GP4(data):

    return (3.128881 +

            1.0*((((((2.264586) > (data["dist"]))*1.)) * ((((-1.0*((data["dist"])))) + (((((-1.0*((1.207241)))) + (data["atom_index_1"]))/2.0)))))) +

            1.0*((((np.minimum(((((data["y_0"]) + (data["atom_index_1"])))), ((0.318310)))) <= ((((data["atom_index_0"]) <= (((data["atom_index_1"]) * 2.0)))*1.)))*1.)) +

            0.949194*((((((((3.141593) <= (data["x_0"]))*1.)) - ((((-0.441697) <= (data["y_0"]))*1.)))) / 2.0)) +

            1.0*(np.maximum((((((data["atom_index_1"]) <= (data["y_1"]))*1.))), ((np.minimum((((((-2.0) > (data["z_0"]))*1.))), (((((data["atom_index_1"]) <= ((5.21898651123046875)))*1.)))))))) +

            1.0*((-1.0*((((((((data["dist_to_type_mean"]) <= (1.022687))*1.)) <= ((-1.0*(((((data["atom_index_1"]) <= (data["dist_to_type_mean"]))*1.))))))*1.))))) +

            1.0*((((((((((data["dist_to_type_mean"]) / 2.0)) <= ((((0.24176722764968872)) * 2.0)))*1.)) * (((data["dist_to_type_mean"]) * 2.0)))) * (((data["dist_to_type_mean"]) * 2.0)))) +

            1.0*(((((np.minimum((((((data["dist"]) > (2.264586))*1.))), (((((data["dist_to_type_mean"]) <= (data["atom_index_1"]))*1.))))) * (data["dist_to_type_mean"]))) * (data["dist_to_type_mean"]))) +

            1.0*(np.minimum(((np.maximum(((data["y_1"])), (((((-1.0*(((((data["atom_index_0"]) > (((14.0) - (-1.0))))*1.))))) / 2.0)))))), ((0.023229)))) +

            1.0*((((((((data["z_0"]) - (1.866194))) > (((-2.0) * ((((((data["z_0"]) * 2.0)) <= (0.318310))*1.)))))*1.)) / 2.0)) +

            1.0*((((np.maximum(((data["x_1"])), (((((((1.034271) > (data["x_1"]))*1.)) - (data["z_0"])))))) > (np.maximum(((data["atom_index_1"])), ((data["dist"])))))*1.)) +

            1.0*((((((((-1.0*(((((1.929792) > (data["dist"]))*1.))))) * 2.0)) * 2.0)) - ((((data["dist"]) <= (1.929792))*1.)))) +

            1.0*((((data["x_0"]) <= ((((-1.0*((((((((data["atom_index_0"]) + ((-1.0*((2.0)))))/2.0)) + ((-1.0*((2.0)))))/2.0))))) * 2.0)))*1.)) +

            0.929653*(((((0.523343) - ((((data["dist_to_type_mean"]) > ((((1.0) + ((((1.0)) - (0.029367))))/2.0)))*1.)))) / 2.0)) +

            0.971177*((((((np.minimum(((data["x_0"])), ((((((3.594635) - (data["atom_index_1"]))) + (3.594635)))))) > (1.138006))*1.)) / 2.0)) +

            1.0*((((((-1.0*(((((0.318310) > (((((0.318310) * ((((data["dist_to_type_mean"]) + (0.0))/2.0)))) * 2.0)))*1.))))) / 2.0)) / 2.0)) +

            1.0*((((((data["x_0"]) <= (((np.minimum(((data["x_1"])), ((np.minimum(((((data["y_0"]) / 2.0))), ((0.318310))))))) - (data["dist_to_type_mean"]))))*1.)) / 2.0)))



def GP5(data):

    return (3.690675 +

            0.868100*(((data["dist"]) * (((((((((3.142503) > (data["dist"]))*1.)) <= ((((data["dist"]) > (3.141593))*1.)))*1.)) - (0.636620))))) +

            1.0*((((((1.0) <= (np.maximum((((-1.0*((data["y_0"]))))), (((((-0.206448) <= (data["y_0"]))*1.))))))*1.)) * (((0.148571) / 2.0)))) +

            0.885686*((((2.719738) > (data["dist"]))*1.)) +

            0.888129*(((((np.maximum(((data["dist_to_type_mean"])), ((0.826032)))) - ((((data["dist_to_type_mean"]) > (0.826032))*1.)))) * 2.0)) +

            0.887640*(((data["dist_to_type_mean"]) * ((-1.0*((((((((data["y_1"]) > (((-0.228787) / 2.0)))*1.)) > ((((data["dist"]) <= (3.141593))*1.)))*1.))))))) +

            1.0*(np.minimum((((((data["dist_to_type_mean"]) > (1.090505))*1.))), ((np.minimum((((((data["z_0"]) <= (data["dist_to_type_mean"]))*1.))), (((((data["x_0"]) <= (data["dist_to_type_mean"]))*1.)))))))) +

            1.0*(((((((-0.181841) + (((-0.228787) / 2.0)))/2.0)) + ((((0.927964) > (np.maximum(((data["atom_index_1"])), ((data["dist_to_type_mean"])))))*1.)))/2.0)) +

            1.0*((((((((((((((data["y_1"]) * 2.0)) > (data["y_1"]))*1.)) + (((1.570796) / 2.0)))/2.0)) > (data["dist_to_type_mean"]))*1.)) / 2.0)) +

            1.0*(((((0.018557) * (((np.maximum(((data["x_0"])), ((((data["y_0"]) * (data["dist"])))))) - (((1.570796) * 2.0)))))) * 2.0)) +

            1.0*(np.minimum((((((((1.570796) > (((data["dist_to_type_mean"]) * 2.0)))*1.)) * 2.0))), ((1.570796)))) +

            0.999511*(((((np.minimum(((((((data["atom_index_1"]) + (-0.267775))) + (-1.0)))), (((((1.570796) <= (data["z_0"]))*1.))))) / 2.0)) / 2.0)) +

            1.0*(np.minimum((((((((0.00275373528711498)) * 2.0)) + ((0.00275373528711498))))), ((data["atom_index_1"])))) +

            1.0*(((((((((3.647356) > (data["dist"]))*1.)) * (data["dist"]))) + ((-1.0*((data["dist"])))))/2.0)) +

            0.999023*(np.maximum((((((0.0) + ((((data["dist"]) + (-3.0))/2.0)))/2.0))), ((0.0)))) +

            1.0*(((((((((((0.0) / 2.0)) - ((((data["x_0"]) > (((data["x_0"]) * (data["x_0"]))))*1.)))) / 2.0)) / 2.0)) / 2.0)) +

            1.0*(np.maximum((((((((((((data["dist_to_type_mean"]) <= (0.913097))*1.)) / 2.0)) / 2.0)) / 2.0))), ((((0.913097) - (data["dist_to_type_mean"])))))))



def GP6(data):

    return (4.772715 +

            1.0*(((((0.636620) - ((((data["dist"]) <= (np.maximum(((3.0)), ((data["y_0"])))))*1.)))) - ((((data["dist"]) <= (3.0))*1.)))) +

            1.0*(((((((((((-3.0) + (data["dist"]))/2.0)) > ((((data["y_1"]) > (((0.218830) * 2.0)))*1.)))*1.)) * 2.0)) * 2.0)) +

            1.0*((((((data["dist_to_type_mean"]) <= ((((1.0) + (0.922659))/2.0)))*1.)) - ((((2.492783) <= (data["dist"]))*1.)))) +

            1.0*((((((((data["dist"]) > (3.0))*1.)) - ((((data["dist"]) > (3.102541))*1.)))) - ((((data["dist"]) > (3.102541))*1.)))) +

            0.962384*(((2.453189) - (((data["dist"]) + ((-1.0*(((((data["dist"]) <= (2.453189))*1.))))))))) +

            1.0*(((1.129494) * (((1.129494) * (((1.129494) * ((((np.maximum(((3.051597)), ((data["z_1"])))) <= (data["dist"]))*1.)))))))) +

            0.882755*(((((((((np.minimum(((0.636620)), ((((data["y_1"]) / 2.0))))) - ((((data["y_1"]) <= (0.636620))*1.)))) / 2.0)) / 2.0)) / 2.0)) +

            1.0*((((((1.146451) > (data["dist_to_type_mean"]))*1.)) - (data["dist_to_type_mean"]))) +

            1.0*((((-2.0) > ((-1.0*((((data["dist_to_type_mean"]) * (((data["dist_to_type_mean"]) * (((data["dist_to_type_mean"]) * (data["y_0"])))))))))))*1.)) +

            1.0*(((((((data["y_0"]) - ((((data["y_0"]) <= ((1.0)))*1.)))) * 2.0)) * (((data["dist_to_type_mean"]) - ((1.0)))))) +

            1.0*((((((((3.090264) <= (data["dist"]))*1.)) * 2.0)) * ((-1.0*((data["dist_to_type_mean"])))))) +

            1.0*(((((2.268008) / 2.0)) * ((((((data["dist_to_type_mean"]) > (1.132620))*1.)) / 2.0)))) +

            1.0*((((-0.278111) + (((((((-1.0) + ((((data["dist_to_type_mean"]) <= (0.941507))*1.)))/2.0)) + ((((data["dist_to_type_mean"]) <= (0.941507))*1.)))/2.0)))/2.0)) +

            1.0*(((np.minimum(((((((-0.047501) * (data["y_0"]))) - (-0.047501)))), ((-0.047501)))) * (((data["y_0"]) / 2.0)))) +

            1.0*((((-1.0) + ((((0.922659) <= (np.maximum(((data["x_1"])), (((((np.maximum(((data["z_1"])), ((data["dist_to_type_mean"])))) + (data["dist_to_type_mean"]))/2.0))))))*1.)))/2.0)) +

            1.0*((((((((3.046956) <= (np.maximum(((data["dist"])), ((data["z_0"])))))*1.)) / 2.0)) / 2.0)))



def GP7(data):

    return (0.990498 +

            0.938935*(np.minimum(((((0.318310) - (((data["dist_to_type_mean"]) * ((((data["dist_to_type_mean"]) <= (1.015225))*1.))))))), ((((data["dist"]) - (3.141593)))))) +

            0.933073*((-1.0*((np.maximum(((((((((data["y_1"]) * 2.0)) * (0.011385))) * (data["dist"])))), (((((data["dist"]) <= ((2.32503938674926758)))*1.)))))))) +

            0.932096*(np.minimum(((data["atom_index_1"])), ((((((((((-1.0*((data["dist_to_type_mean"])))) + (np.maximum(((0.983454)), ((data["dist_to_type_mean"])))))/2.0)) * 2.0)) * 2.0))))) +

            0.982413*(((np.minimum(((((-0.059342) / 2.0))), ((((((((-0.059342) / 2.0)) / 2.0)) * (data["atom_index_1"])))))) * (data["dist_to_type_mean"]))) +

            1.0*((-1.0*((((-0.108994) * (((np.maximum(((data["y_0"])), (((((((-1.0*((3.141593)))) / 2.0)) / 2.0))))) / 2.0))))))) +

            0.980948*(((((((1.0)) > (data["atom_index_1"]))*1.)) * (((data["dist"]) * (((1.055259) * (((-0.110218) / 2.0)))))))) +

            1.0*(((((((0.04956008121371269)) > (((1.055259) * (((data["x_0"]) * (data["x_1"]))))))*1.)) * ((0.04956008121371269)))) +

            1.0*((((((((((data["atom_index_0"]) <= ((5.20004987716674805)))*1.)) + ((((((((data["atom_index_1"]) <= (data["dist"]))*1.)) / 2.0)) / 2.0)))/2.0)) + (-0.059342))/2.0)) +

            1.0*((((data["dist_to_type_mean"]) <= (((-0.210597) + ((((data["y_1"]) > ((((((data["y_1"]) > (((data["dist_to_type_mean"]) / 2.0)))*1.)) * 2.0)))*1.)))))*1.)) +

            0.917929*((-1.0*(((((((data["dist_to_type_mean"]) <= (((1.570796) / 2.0)))*1.)) / 2.0))))) +

            1.0*((((data["dist"]) > (np.maximum(((3.0)), (((((data["atom_index_0"]) + (((-1.0) - ((-1.0*((data["x_1"])))))))/2.0))))))*1.)) +

            1.0*((((((((((1.177861) + ((-1.0*((data["y_0"])))))) <= (-1.0))*1.)) * (-0.108994))) * (data["dist_to_type_mean"]))) +

            1.0*((((-1.0*(((((((np.maximum(((data["y_0"])), ((((data["dist"]) * 2.0))))) > (7.0))*1.)) / 2.0))))) / 2.0)) +

            0.999023*((((((((data["atom_index_1"]) / 2.0)) <= (((data["dist"]) - (data["atom_index_0"]))))*1.)) / 2.0)) +

            0.706400*(((((((((((data["y_1"]) > ((((-0.108994) + (data["y_0"]))/2.0)))*1.)) > (((((5.40717029571533203)) + (data["y_0"]))/2.0)))*1.)) / 2.0)) / 2.0)) +

            1.0*((((((((((3.0) <= (((data["z_0"]) - ((((data["x_1"]) <= (data["y_0"]))*1.)))))*1.)) / 2.0)) / 2.0)) / 2.0)))



def GP(data):

    retValues = pd.DataFrame({'id':data.id})

    retValues['scalar_coupling_constant'] = 0

    retValues.loc[data.type==0,'scalar_coupling_constant'] =GP0(data[data.type==0])

    retValues.loc[data.type==1,'scalar_coupling_constant'] =GP1(data[data.type==1])

    retValues.loc[data.type==2,'scalar_coupling_constant'] =GP2(data[data.type==2])

    retValues.loc[data.type==3,'scalar_coupling_constant'] =GP3(data[data.type==3])

    retValues.loc[data.type==4,'scalar_coupling_constant'] =GP4(data[data.type==4])

    retValues.loc[data.type==5,'scalar_coupling_constant'] =GP5(data[data.type==5])

    retValues.loc[data.type==6,'scalar_coupling_constant'] =GP6(data[data.type==6])

    retValues.loc[data.type==7,'scalar_coupling_constant'] =GP7(data[data.type==7])

    return retValues
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()

predictions = GP(train)

group_mean_log_mae(train.scalar_coupling_constant,predictions.scalar_coupling_constant,train.type)
GP(test).to_csv('submission.csv',index=False)