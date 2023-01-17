%pylab inline --no-import-all
import pandas as pd
import seaborn as sns
import tqdm
import scipy 
import gc
import pickle
import re
import itertools
import os
import cufflinks as cf
cf.go_offline()
DATA_PATH = "../input/interactive-data-cleaning-and-preprocessing/"

with open(os.path.join(DATA_PATH, "train_clean.pkl"), 'rb') as f:
    train_data = pickle.load(f)
def category_columns(data: pd.DataFrame):
    cats = set(data.select_dtypes(['category', 'object']).columns.tolist())
    cats = cats.difference({'MachineIdentifier', 'HasDetection'})
    return list(cats)

ALL_COLS = category_columns(train_data)
def graph_category_distr(data: pd.DataFrame, col):
    col_data = data[col]
    
    infected_count = sum(data.HasDetections==1) / len(data)
    balancing_multiplier = 0.5 / infected_count
    infected_counts = col_data[data.HasDetections==1].value_counts() / col_data.value_counts() * balancing_multiplier

    ic = pd.DataFrame(list(infected_counts.items()), columns=['label', 'infected'])
    ic['label'] = '(' + ic['label'].astype(str) + ')'
    
    ic.iplot(kind='bar', x='label', y='infected', title=col)

def plot_distr(col):
    graph_category_distr(train_data, col)
plot_distr('SMode')
plot_distr( 'Census_ActivationChannel')
plot_distr('Census_ProcessorManufacturerIdentifier')
plot_distr('AVProductStatesIdentifier')
plot_distr('Census_OSBranch')
plot_distr('RtpStateBitfield')
plot_distr('SkuEdition')
plot_distr('Census_DeviceFamily')
plot_distr('OsBuildLab')
plot_distr('IeVerIdentifier')
plot_distr('Processor')
plot_distr('Platform')
plot_distr('Census_FirmwareVersionIdentifier')
plot_distr('AvSigVersion')
plot_distr('OsVer')
plot_distr('SmartScreen')
plot_distr('Census_ProcessorModelIdentifier')
plot_distr('Census_FlightRing')
plot_distr('Census_OEMModelIdentifier')
plot_distr('Census_GenuineStateName')
plot_distr('CityIdentifier')
plot_distr('Wdft_RegionIdentifier')
plot_distr('CountryIdentifier')
plot_distr('OrganizationIdentifier')
plot_distr('AppVersion')
plot_distr('OsPlatformSubRelease')
plot_distr('ProductName')
plot_distr('EngineVersion')
plot_distr('Census_OSInstallTypeName')
plot_distr('Census_PrimaryDiskTypeName')
plot_distr('Census_ChassisTypeName')
plot_distr('Census_OSVersion')
plot_distr('LocaleEnglishNameIdentifier')
plot_distr('PuaMode')
plot_distr('Census_OSUILocaleIdentifier')
plot_distr('GeoNameIdentifier')
plot_distr('UacLuaenable')
plot_distr('Census_FirmwareManufacturerIdentifier')
plot_distr('Census_OEMNameIdentifier')
plot_distr('Census_OSWUAutoUpdateOptionsName')
plot_distr('Census_OSInstallLanguageIdentifier')