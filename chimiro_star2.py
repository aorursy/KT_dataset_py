# Standard libraries

import numpy as np

import random

import pandas as pd

import time

import re

import gc 

from tqdm import tqdm  

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Pre-processing

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import cv2 



# Correlation

import scipy

from scipy.cluster import hierarchy as hc # dendrogram



# Model

from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf

from keras import backend as K

from keras.losses import mse, binary_crossentropy

from keras import optimizers, regularizers

from keras.layers import Input, Dense, Lambda

from keras.models import Sequential, Model, load_model 

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



from keras.models import Model



# Evaluate

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn import metrics

from sklearn.tree import export_graphviz

from sklearn.metrics import roc_auc_score

from keras.losses import mse, binary_crossentropy



pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option('display.max_columns', 3000)

pd.set_option('display.max_rows', 3000)

pd.set_option('display.max_colwidth', 3000)



# For notebook plotting

%matplotlib inline



import warnings                             

warnings.filterwarnings("ignore") 
# TPU 사용을 위한 초기화



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
# 전처리된 데이터셋 불러오기 

train = pd.read_feather("/kaggle/input/star2-processed-dataset/processed_train2.ftr")

train = reduce_mem_usage(train)
train.head()
train.describe().drop('count')
train.info()
# 한쪽 선수의 플레이 기록이 없는 경우에 제거



def check_missing_values(play_time):

    check = False

    if (play_time.min() <= 0) or (play_time.max() <= 0): 

        check = True

        

    return check
# Use unit information in StarCraft 2



# Protoss

def P_info():

    P_dict = {

    # VESPENE

    'BuildAssimilator':[75,0,0,'VESPENE'],

    

    # SUPPLY

    'BuildPylon':[100,0,0,'SUPPLY'],

    

    # WORKER

    'TrainProbe':[50,0,1,'WORKER'],

    

    # BASE 

    'BuildNexus':[400,0,0,'BASE'],

    

    # BUILDING

    'BuildGateway':[150,0,0,'BUILDING'],       'BuildTemplarArchive':[150,200,0,'BUILDING'],   'BuildDarkShrine':[150,150,0,'BUILDING'], 

    'BuildRoboticsBay':[200,200,0,'BUILDING'], 'BuildRoboticsFacility':[150,100,0,'BUILDING'], 'BuildStargate':[150,150,0,'BUILDING'], 

    'BuildFleetBeacon':[300,200,0,'BUILDING'], 'BuildForge':[150,0,0,'BUILDING'],              'BuildCyberneticsCore':[150,0,0,'BUILDING'], 

    'BuildTwilightCouncil':[150,100,0,'BUILDING'],'BuildTemplarArchive':[150,200,0,'BUILDING'],'BuildDarkShrine':[150,150,0,'BUILDING'], 

    'BuildFleetBeacon':[300,200,0,'BUILDING'], 'BuildRoboticsBay':[150,150,0,'BUILDING'],

    

    # DEFENSE

    'BuildPhotonCannon':[150,0,0,'DEFENSE'], 'BuildShieldBattery':[100,0,0,'DEFENSE'],

    

    # ARMY

    'TrainZealot':[100,0,2,'ARMY'],        'TrainSentry':[50,100,2,'ARMY'],      'TrainStalker':[125,50,2,'ARMY'],  'TrainHighTemplar':[50,150,2,'ARMY'], 

    'TrainDarkTemplar':[125,125,2,'ARMY'], 'TrainImmortal':[250,100,4,'ARMY'],   'TrainColossus':[300,200,6,'ARMY'],'TrainArchon':[0,0,4,'ARMY'],

    'TrainObserver':[25,75,1,'ARMY'],      'TrainWarpPrism':[200,0,2,'ARMY'],    'TrainPhoenix':[150,100,2,'ARMY'], 'TrainMothershipCore':[100,100,2,'ARMY'],

    'TrainVoidRay':[250,150,4,'ARMY'],     'TrainOracle':[150,150,3,'ARMY'],     'TrainTempest':[250,175,5,'ARMY'], 'TrainCarrier':[350,250,6,'ARMY'],

    'TrainInterceptor':[15,0,0,'ARMY'],    'TrainMothership':[400,400,8,'ARMY'], 'TrainAdept':[100,25,2,'ARMY'],    'TrainDisruptor':[150,150,3,'ARMY'],

    

    # UPGRADE    

    'UpgradeGroundWeapons1':[100,100,0,'UPGRADE'], 'UpgradeGroundWeapons2':[150,150,0,'UPGRADE'],'UpgradeGroundWeapons3':[200,200,0,'UPGRADE'], 

    'UpgradeGroundArmor1':[100,100,0,'UPGRADE'],   'UpgradeGroundArmor2':[150,150,0,'UPGRADE'],  'UpgradeGroundArmor3':[200,200,0,'UPGRADE'],

    'UpgradeShields1':[150,150,0,'UPGRADE'],       'UpgradeShields2':[225,225,0,'UPGRADE'],      'UpgradeShields3':[300,300,0,'UPGRADE'],

    'UpgradeAirWeapons1':[100,100,0,'UPGRADE'],    'UpgradeAirWeapons2':[175,175,0,'UPGRADE'],   'UpgradeAirWeapons3':[250,250,0,'UPGRADE'],

    'UpgradeAirArmor1':[150,150,0,'UPGRADE'],      'UpgradeAirArmor2':[225,225,0,'UPGRADE'],     'UpgradeAirArmor3':[300,300,0,'UPGRADE'],

    'ResearchCharge':[100,100,0,'UPGRADE'],        'ResearchBlink':[100,100,0,'UPGRADE'],        'ResearchResonatingGlaives':[100,100,0,'UPGRADE'], 

    'ResearchPsiStormTech':[200,200,0,'UPGRADE'],  'ResearchGraviticBoosters':[100,100,0,'UPGRADE'], 'ResearchGraviticDrive':[100,100,0,'UPGRADE'],      

    'ResearchExtendedThermalLance':[150,150,0,'UPGRADE'],  'ResearchAnionPulseCrystals':[150,150,0,'UPGRADE'],    'ResearchFluxVanes':[100,100,0,'UPGRADE']

    

    }

               

    return P_dict
# Terran

def T_info():

    T_dict = {

    # VESPENE

    'BuildRefinery':[75,0,0,'VESPENE'],

    

    # SUPPLY

    'BuildSupplyDepot':[100,0,0,'SUPPLY'],

   

    # WORKER

    'TrainSCV':[50,0,1,'WORKER'], 'TrainMule':[0,0,0,'WORKER'],

    

    # BASE

    'BuildCommandCenter':[400,0,0,'BASE'], 'UpgradeToPlanetaryFortress':[150,150,0,'BASE'], 'UpgradeToOrbitalCommand':[150,0,0,'BASE'],

    

    # BUILDING

    'BuildBarracks':[150,0,0,'BUILDING'],        'BuildFactory':[150,100,0,'BUILDING'],      'BuildGhostAcademy':[150,50,0,'BUILDING'], 'BuildArmory':[150,100,0,'BUILDING'],

    'BuildStarport':[150,100,0,'BUILDING'],      'BuildFusionCore':[150,150,0,'BUILDING'],   'BuildEngineeringBay':[125,0,0,'BUILDING'], 

    'BuildSensorTower':[125,100,0,'BUILDING'],   'BuildFactoryTechLab':[50,25,0,'BUILDING'], 'BuildFactoryReactor':[50,50,0,'BUILDING'],  

    'BuildBarracksTechLab':[50,25,0,'BUILDING'], 'BuildBarracksReactor':[50,50,0,'BUILDING'],

   

    # DEFENSE

    'BuildMissileTurret':[100,0,0,'DEFENSE'], 'BuildAutoTurret':[0,0,0,'DEFENSE'], 'BuildPointDefenseDrone':[0,0,0,'DEFENSE'], 'BuildBunker':[100,0,0,'DEFENSE'],

    

    # ARMY

    'TrainMarine':[50,0,1,'ARMY'],   'TrainMarauder':[100,25,2,'ARMY'],       'TrainReaper':[50,50,1,'ARMY'],    'TrainGhost':[150,125,2,'ARMY'], 

    'BuildHellion':[100,0,2,'ARMY'], 'BuildHellbat':[100,0,2,'ARMY'],         'BuildWidowMine':[75,25,2,'ARMY'], 'BuildSiegeTank':[150,125,3,'ARMY'],

    'BuildThor':[300,200,6,'ARMY'],  'TrainViking':[150,75,2,'ARMY'],         'TrainMedivac':[100,100,2,'ARMY'], 'TrainBanshee':[150,100,3,'ARMY'],

    'TrainRaven':[100,200,2,'ARMY'], 'TrainBattlecruiser':[400,300,6,'ARMY'], 'TrainCyclone':[150,100,3,'ARMY'], 'TrainLiberator':[150,150,3,'ARMY'],

    

    # UPGRADE

    'ResearchNeosteelArmor':[150,150,0,'UPGRADE'],                    'ResearchNeosteelFrame':[100,100,0,'UPGRADE'],           'ResearchHiSecAutoTracking':[100,100,0,'UPGRADE'],

    'UpgradeTerranInfantryWeapons1':[100,100,0,'UPGRADE'],            'UpgradeTerranInfantryWeapons2':[175,175,0,'UPGRADE'],   'UpgradeTerranInfantryWeapons3':[250,250,0,'UPGRADE'],

    'UpgradeTerranInfantryArmor1':[100,100,0,'UPGRADE'],              'UpgradeTerranInfantryArmor2':[175,175,0,'UPGRADE'],     'UpgradeTerranInfantryArmor3':[250,250,0,'UPGRADE'],

    'UpgradeStructureArmor':[150,150,0,'UPGRADE'],                    'ResearchPersonalCloaking':[150,150,0,'UPGRADE'],        'ResearchEnhancedShockwaves':[150,150,0,'UPGRADE'],

    'TrainNuke':[100,100,0,'UPGRADE'],                                'ResearchRapidReignitionSystem':[100,100,0,'UPGRADE'],

    'UpgradeVehicleWeapons1':[100,100,0,'UPGRADE'],                   'UpgradeVehicleWeapons2':[175,175,0,'UPGRADE'],           'UpgradeVehicleWeapons3':[250,250,0,'UPGRADE'],

    'UpgradeShipWeapons1':[100,100,0,'UPGRADE'],                      'UpgradeShipWeapons2':[175,175,0,'UPGRADE'],              'UpgradeShipWeapons3':[250,250,0,'UPGRADE'],

    'ResearchTerranVehicleAndShipArmorsLevel1':[100,100,0,'UPGRADE'], 'ResearchTerranVehicleAndShipArmorsLevel2':[175,175,0,'UPGRADE'],

    'ResearchTerranVehicleAndShipArmorsLevel1':[250,250,0,'UPGRADE'], 'ResearchWeaponRefit':[150,150,0,'UPGRADE'],                 'ResearchAdvancedBallistics':[150,150,0,'UPGRADE'], 

    'ResearchCombatShield':[100,100,0,'UPGRADE'],                     'ResearchStimpack':[100,100,0,'UPGRADE'],                    'ResearchConcussiveShells':[50,50,0,'UPGRADE'],

    'ResearchInfernalPreIgniter':[100,100,0,'UPGRADE'],               'ResearchMagFieldAccelerator':[100,100,0,'UPGRADE'],         'ResearchDrillingClaws':[75,75,0,'UPGRADE'],

    'ResearchSmartServos':[100,100,0,'UPGRADE'],                      'ResearchCorvidReactor':[150,150,0,'UPGRADE'],               'ResearchCloakingField':[100,100,0,'UPGRADE'],

    'ResearchHyperflightRotors':[150,150,0,'UPGRADE'],                'ResearchRavenRecalibratedExplosives':[150,150,0,'UPGRADE'], 'ResearchRapidFireLaunchers':[75,75,0,'UPGRADE']

    

    }

    

    return T_dict
# Zerg 

def Z_info():

    Z_dict = {

    # VESPENE

    'BuildExtractor':[25,0,0,'VESPENE'],



    # SUPPLY

    'MorphOverlord':[100,0,0,'SUPPLY'],

    

    # WORKER

    'MorphDrone':[50,0,1,'WORKER'],

    

    # BASE

    'BuildHatchery':[300,0,0,'BASE'], 'UpgradeToLair':[150,100,0,'BASE'], 'UpgradeToHive':[200,150,0,'BASE'],

    

    # BUILDING

    'BuildSpawningPool':[200,0,0,'BUILDING'],      'BuildRoachWarren':[150,0,0,'BUILDING'],      'BuildBanelingNest':[100,50,0,'BUILDING'], 

    'BuildUltraliskCavern':[150,200,0,'BUILDING'], 'BuildHydraliskDen':[100,100,0,'BUILDING'],   'BuildInfestationPit':[100,100,0,'BUILDING'],

    'BuildSpire':[200,200,0,'BUILDING'],           'MorphToGreaterSpire':[100,150,0,'BUILDING'], 'UpgradeToLurkerDenMP':[100,150,0,'BUILDING'], 'BuildCreepTumor':[0,0,0,'BUILDING'],

    'BuildEvolutionChamber':[75,0,0,'BUILDING'],   'BuildNydusNetwork':[150,150,0,'BUILDING'],   'BuildNydusWorm':[50,50,0,'BUILDING'],  

    

    # DEFENSE

    "BuildSpineCrawler":[100,0,0,'DEFENSE'], "BuildSporeCrawler":[75,0,0,'DEFENSE'],

    

    # ARMY

    'TrainQueen':[150,0,2,'ARMY'],       'MorphZergling':[25,0,0.5,'ARMY'],  'TrainBaneling':[25,25,0.5,'ARMY'],    'MorphRoach':[75,25,2,'ARMY'], 

    'MorphHydralisk':[100,50,2,'ARMY'],  'MorphInfestor':[100,150,2,'ARMY'], 'MorphSwarmHost':[100,75,3,'ARMY'], 

    'MorphUltralisk':[300,200,6,'ARMY'], 'MorphToOverseer':[50,50,0,'ARMY'], 'MorphMutalisk':[100,100,2,'ARMY'],    'MorphToLurker':[50,100,3,'ARMY'],

    'MorphCorruptor':[150,100,2,'ARMY'], 'MorphViper':[100,200,3,'ARMY'],    'MorphToBroodLord':[150,150,4,'ARMY'], 'MorphToRavage':[25,75,3,'ARMY'],

    

    # UPGRADE

    'EvolveFlyerAttacks1':[100,100,0,'UPGRADE'],             'EvolveFlyerAttacks2':[175,175,2,'UPGRADE'],             'EvolveFlyerAttacks3':[250,250,2,'UPGRADE'],

    'EvolveFlyerCarapace1':[150,150,0,'UPGRADE'],            'EvolveFlyerCarapace2':[225,225,2,'UPGRADE'],            'EvolveFlyerCarapace3':[300,300,2,'UPGRADE'],

    'EvolveBurrow':[100,100,0,'UPGRADE'],                    'EvolvePneumatizedCarapace':[100,100,0,'UPGRADE'],

    'EvolvePathogenGlands':[150,150,0,'UPGRADE'],            'EvolveAdrenalGlands':[200,200,0,'UPGRADE'],             'EvolveMetabolicBoost':[100,100,0,'UPGRADE'],

    'ResearchZergMeleeWeaponsLevel1':[100,100,0,'UPGRADE'],  'ResearchZergMeleeWeaponsLevel2':[150,150,0,'UPGRADE'],  'ResearchZergMeleeWeaponsLevel3':[200,200,0,'UPGRADE'],

    'ResearchZergMissileWeaponsLevel1':[100,100,0,'UPGRADE'],'ResearchZergMissileWeaponsLevel2':[150,150,0,'UPGRADE'],'ResearchZergMissileWeaponsLevel3':[200,200,0,'UPGRADE'],

    'ResearchZergGroundArmorsLevel1':[150,150,0,'UPGRADE'],  'ResearchZergGroundArmorsLevel2':[225,225,0,'UPGRADE'],  'ResearchZergGroundArmorsLevel3':[300,300,0,'UPGRADE'],

    'EvolveTunnelingClaws':[100,100,0,'UPGRADE'],            'EvolveGlialReconstitution':[100,100,0,'UPGRADE'],       'EvolveCentrifugalHooks':[150,150,0,'UPGRADE'],

    'ResearchEvolveMuscularAugments':[100,100,0,'UPGRADE'],  'ResearchAdaptiveTalons':[150,150,0,'UPGRADE'],          'ResearchSeismicSpines':[150,150,0,'UPGRADE'], 

    'EvolveNeuralParasite':[150,150,0,'UPGRADE'],            'EvolveChitinousPlating':[150,150,0,'UPGRADE'],          'EvolveAnabolicSynthesis':[150,150,0,'UPGRADE']

            

    }

    

    return Z_dict
def feature_engineering(player_info, player_num, species, play_time, event_names, event_count, player_sight):

    cols = ['VESPENE', 'SUPPLY', 'WORKER', 'BASE', 'BUILDING', 'DEFENSE', 'ARMY', 'UPGRADE', 'Minerals', 'Gas', 'Supply', 'Control_key']

    df = pd.DataFrame({col:[0] for col in cols})

    

    ability_check = player_info[player_info['event'] == 'Ability']

    onehot_species = pd.get_dummies(pd.Series(list('TPZ')))

    

    if species == 'T':

        onehot_species = onehot_species.loc[onehot_species['T'] == 1].reset_index(drop=True)

        species_dict = T_info()

        idx = 0

    elif species == 'P':

        onehot_species = onehot_species.loc[onehot_species['P'] == 1].reset_index(drop=True)

        species_dict = P_info()

        idx = 1

    elif species == 'Z':   

        onehot_species = onehot_species.loc[onehot_species['Z'] == 1].reset_index(drop=True)

        species_dict = Z_info()

        idx = 2

        

    ##################### Add Basic Feature #####################

    cols = list(event_count.columns)  

    for col in event_names:

        if col not in cols:

            df[col] = pd.Series(0)

        else:

            df[col] = event_count[col]

         

    ##################### Add New Feature #####################

    p = re.compile("- (\w+)", re.I)

    for i in range(len(ability_check)):

        match = p.findall(ability_check['event_contents'].iloc[i])

        if match: 

            key = match[0]

            if key in species_dict:

                categories = species_dict[key][3]

                df[categories] += 1

                df['Minerals'] += species_dict[key][0]

                df['Gas'] += species_dict[key][1] 

                df['Supply'] += species_dict[key][2]

        else:

            pass



    ### Check Control keys ###

    control_keys = ['Attack', 'Stop', 'Patrol', 'Move', 'HoldPosition', 'Rally', 'Gather', 'ReturnCargo', 'HaltBuilding', 'Cancel', 'Repair']

    for control_key in control_keys:

        cnt = len(ability_check[ability_check['event_contents'].str.contains(control_key, regex=True) == True])

        if cnt > 0:

            df['Control_key'] += cnt

         

    df['Micro'] = df['AddToControlGroup'] + df['GetControlGroup'] + df['SetControlGroup'] + df['ControlGroup'] + df['Control_key']

    df['Macro'] = df['VESPENE'] + df['SUPPLY'] + df['WORKER'] + df['BASE'] + df['BUILDING'] + df['DEFENSE'] + df['ARMY'] + df['UPGRADE']

    df['APM'] = df['Selection'] + df['Ability'] + df['Right Click'] + df['Micro']

    df['Player_sight'] = pd.Series(player_sight)

    df['Resource'] = df['Minerals'] + df['Gas']

    df['Play_time'] = pd.Series(play_time)



    cols = df.columns

    df = pd.concat([df, onehot_species], axis=1)

    df = df.rename(columns=lambda x:x+'_'+str(player_num))

    

    return df, cols
# 게임 유저의 맵의 이동 경로 활용



def transform_coordinate(data, play_time, t_interval=1, visualize=False):



    # Assume the maximum size of the map is 200.

    player_map = np.zeros([200,200,3], np.float)

    filter_size = (20,40)

    

    for n in range(0, play_time, t_interval):

        #display(data[(n <= data['time']) & (data['time'] <= n+t_interval)])

        sample = data[(n <= data['time']) & (data['time'] <= n+t_interval)]

        

        # Camera

        Cam = sample[sample['event'] == 'Camera']['event_contents'].str.extract(r'((\d+).(\d+)), ((\d+).(\d+))')

        Cam_x = list(map(float, Cam[0].dropna().values))

        Cam_y = list(map(float, Cam[3].dropna().values))

        

        # Right click

        Right_click = sample[sample['event'] == 'Right Click']['event_contents'].str.extract(r'((\d+).(\d+)), ((\d+).(\d+))')

        Right_click_x = list(map(float, Right_click[0].dropna().values))

        Right_click_y = list(map(float, Right_click[3].dropna().values))



        # Ability

        Ability = sample[sample['event'] == 'Ability']



        Build = Ability[Ability['event_contents'].str.contains('Build',regex=True) == True]['event_contents'].str.extract(r'((\d+).(\d+)), ((\d+).(\d+))')

        Build_x = list(map(float, Build[0].dropna().values))

        Build_y = list(map(float, Build[3].dropna().values))



        Attack = Ability[Ability['event_contents'].str.contains('Attack',regex=True) == True]['event_contents'].str.extract(r'((\d+).(\d+)), ((\d+).(\d+))')

        Attack_x = list(map(float, Attack[0].dropna().values))

        Attack_y = list(map(float, Attack[3].dropna().values))

        

        Patrol = Ability[Ability['event_contents'].str.contains('Patrol',regex=True) == True]['event_contents'].str.extract(r'((\d+).(\d+)), ((\d+).(\d+))')

        Patrol_x = list(map(float, Patrol[0].dropna().values))

        Patrol_y = list(map(float, Patrol[3].dropna().values))

        

        for x, y in zip(Cam_x, Cam_y):

            player_map[int(y)][int(x)] += 1

        

        for x, y in zip(Right_click_x, Right_click_y):

            player_map[int(y)][int(x)] += 1

      

        if visualize:

            labels = ['Cam', 'Right_click']

            sns.scatterplot(x=Cam_x,         y=Cam_y)

            sns.scatterplot(x=Right_click_x, y=Right_click_y)

            plt.xlim(0,200)

            plt.ylim(0,200)

            plt.legend(labels)

            plt.show()

    

    if visualize:

        player_map = cv2.flip(player_map, 0)

        player_map = cv2.dilate(player_map, np.ones(filter_size))

        # G: 2,0 / R: 2,1 / B: 1,0

        player_map[:,:,2] /= 255

        player_map[:,:,1] /= 255

        player_map[:,:,0] /= 255

        plt.imshow(player_map)

        plt.show()

    

    player_sight = sum(player_map.flatten())

    

    return player_map, player_sight
def data_preparation(data, split_time, testset=False):

    event_names = ['Ability', 'AddToControlGroup', 'Camera', 'ControlGroup', 'GetControlGroup', 

                   'Right Click', 'Selection', 'SetControlGroup']

        

    g = data.groupby(['game_id', 'player'])

    event_counts = g.event.value_counts()

    m_time = g.time.max()

    

    if testset == False:

        winners = g.winner.unique()

    

    x_data = pd.DataFrame()

    y_data = pd.DataFrame()

    gameIds = data['game_id'].unique()

    

    gc.collect()

    

    for gameId in tqdm(gameIds):

        play_time = m_time[gameId].max()

        

        if check_missing_values(m_time[gameId]):

            continue

            

        if testset == False:    

            winner = pd.Series(winners[gameId,0])

            

        df = pd.DataFrame()

        for player_num in range(2):

            player_info = g.get_group((gameId, player_num))

            species = player_info.species.unique()[0]

            play_time = player_info.time.max()

            event_count = event_counts[gameId, player_num].to_frame().T.reset_index(drop=True)

            

            _, player_sight = transform_coordinate(player_info, int(play_time)+1, split_time, visualize=False)

            processed_data, cols = feature_engineering(player_info, player_num, species, play_time, event_names, event_count, player_sight)

            

            df = pd.concat([df, processed_data], axis=1)

        

        for col in cols:

            df['delta_' + col] = df[col + '_1'][0] - df[col + '_0'][0]

        

        df['game_id'] = pd.Series(gameId)

        x_data = pd.concat([x_data, df], axis=0)

        

        if testset == False:    

            y_data = pd.concat([y_data, winner], axis=0)

            

        gc.collect()



    x_data = x_data.set_index('game_id')

    display(x_data.head(5))   

    display(y_data.head(5))    



    return x_data, y_data
# Data Cleansing + Feature Engineering 완료된 데이터를 저장해둠



'''

split_time = 100

x_data, y_data = data_preparation(test, split_time, testset=True)



train_data = x_data.reset_index()

train_data['winner'] = y_data.values

train_data.to_feather('./processed_train2.ftr')



test_data = x_data.reset_index()

test_data.to_feather('./processed_test2.ftr')

'''
x_data = train.drop(['game_id', 'winner'], axis=1)

y_data = train['winner']

train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

print(train_x.shape, train_y.shape, val_y.shape, val_x.shape)



gc.collect()
from lightgbm import LGBMRegressor

params = {

        'objective':'regression',

        'metric':'auc',

        'learning_rate':0.01,

        'n_estimators': 500,

}



model = LGBMRegressor(**params)

model.fit(

    train_x, train_y,

    eval_set=[(val_x, val_y)],

    eval_metric='auc',

    verbose=100,

)
feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, train_x.columns)), columns=['Value','Feature'])



plt.figure(figsize=(10, 10))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False)[:30])

plt.title('LightGBM Features')

plt.show()
# Dendrogram 



# Keep only significant features

to_keep = feature_importance.sort_values(by='Value', ascending=False)[:50].Feature



## Create a Dendrogram to view highly correlated features

corr = np.round(scipy.stats.spearmanr(train_x[to_keep]).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(14,20))

dendrogram = hc.dendrogram(z, labels=train_x[to_keep].columns, orientation='left', leaf_font_size=16)

plt.plot()
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=42).fit(val_x, val_y)

eli5.show_weights(perm, feature_names=list(val_x.columns))
import shap

shap.initjs()

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(val_x)

shap.summary_plot(shap_values, val_x, feature_names=list(val_x.columns))
use_num_features = 0 # use_num_features = 0 인 경우 모든 컬럼의 데이터 사용



scaler = StandardScaler()

if use_num_features:

    im_features = feature_importance.sort_values(by='Value', ascending=False)[:use_num_features].Feature 

    X_train = scaler.fit_transform(train_x[im_features].astype(np.float32))

    Y_train = train_y.values

    X_val = scaler.fit_transform(val_x[im_features].astype(np.float32))

    Y_val = val_y.values

else:

    X_train = scaler.fit_transform(train_x.astype(np.float32))

    Y_train = train_y.values

    X_val = scaler.fit_transform(val_x.astype(np.float32))

    Y_val = val_y.values
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):

    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose)
def get_model():

    penalties = 0.01

    stddev = 0.05



    hidden_layer = tf.keras.layers.GaussianNoise(stddev)(inputs)

    

    hidden_layer = tf.keras.layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(penalties))(hidden_layer)

    hidden_layer = tf.keras.activations.elu(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    hidden_layer = tf.keras.layers.GaussianNoise(stddev)(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(penalties))(hidden_layer)

    hidden_layer = tf.keras.activations.elu(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    hidden_layer = tf.keras.layers.GaussianNoise(stddev)(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(32, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(penalties))(hidden_layer)

    hidden_layer = tf.keras.activations.elu(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    hidden_layer = tf.keras.layers.GaussianNoise(stddev)(hidden_layer)



    outputs = tf.keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid')(hidden_layer) 

    

    return outputs
np.random.seed(0)

tf.random.set_seed(0)



epochs = 1000

lr = 0.01

batch_size = 8  * strategy.num_replicas_in_sync * 16 * 4



steps = len(X_train) // batch_size



optimizer = tf.keras.optimizers.Adam()



lr_sched = step_decay_schedule(initial_lr=lr, decay_factor=0.9, step_size=10, verbose=0)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=400, verbose=1)



callbacks_list = [lr_sched, early_stopping]



with strategy.scope():

    inputs = tf.keras.Input(shape=(X_train.shape[1],))

    output_lst = []

    

    outputs = get_model()

    model = tf.keras.Model(inputs=inputs, outputs=outputs)



    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])



    history = model.fit(

            X_train.astype(np.float32), Y_train.astype(np.float32),

            shuffle=True,

            epochs=epochs,

            batch_size=batch_size,

            validation_data = (X_val.astype(np.float32), Y_val.astype(np.float32)),

            #validation_split=0.15,

            steps_per_epoch=steps,

            callbacks=callbacks_list,

            verbose=0)
his = list(history.history)



plt.figure(figsize=(7,5))

plt.plot(history.history[his[0]])

plt.plot(history.history[his[2]])

plt.title('AUC')

plt.ylabel('AUC')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.figure(figsize=(7,5))

plt.plot(history.history[his[1]])

plt.plot(history.history[his[3]])

plt.title('MLP model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# 한쪽 선수의 플레이 기록이 없는 경우, 플레이 기록이 있는 선수가 승리한 것으로 후처리 

def delete_missing_values(data): 

    g = data.groupby(['game_id', 'player'])

    m_time = g.time.max()

   

    df = pd.DataFrame()

    gameIds = data['game_id'].unique()

    

    id_lst = []

    winner_lst = []

    

    for gameId in tqdm(gameIds):

        if (m_time[gameId].min() <= 0) or (m_time[gameId].max() <= 0): 

            player_time = m_time[gameId]

            id_lst.append(gameId)

            winner_lst.append(np.where(player_time[0] >= player_time[1], 0, 1))

            

    df['game_id'] = id_lst

    df['winner'] = winner_lst

    

    return df
origin_test = pd.read_feather("/kaggle/input/star2-dataset/test.ftr")

origin_test = reduce_mem_usage(origin_test)

test_df = delete_missing_values(origin_test)



test = pd.read_feather("/kaggle/input/star2-processed-dataset/processed_test2.ftr")

test = reduce_mem_usage(test)

display(test.head())



X_test = test.drop(['game_id'], axis=1)

X_test = scaler.fit_transform(X_test.astype(np.float32))



# Predict using DNN

pred = model.predict(X_test.astype(np.float32))

print('Prediction values range: {0} ~ {1}'.format(pred.min(), pred.max()))



test['winner'] = pred

test_result = test[['game_id', 'winner']]
submission = test_result.append(test_df, ignore_index=True).sort_values(by='game_id')

submission.to_csv('submission.csv', index=False)
# Last check of submission

print('Head of submission: ')

display(submission.head())