# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dtypes = {

        "MachineIdentifier": "category",

        "ProductName": "category",

        "EngineVersion": "category",

        "AppVersion": "category",

        "AvSigVersion": "category",

        "IsBeta": "int8",

        "RtpStateBitfield": "float16",

        "IsSxsPassiveMode": "int8",

        "DefaultBrowsersIdentifier": "float32",

        "AVProductStatesIdentifier": "category",

        "AVProductsInstalled": "category",

        "AVProductsEnabled": "float16",

        "HasTpm": "int8",

        "CountryIdentifier": "category",

        "CityIdentifier": "category",

        "OrganizationIdentifier": "category",

        "GeoNameIdentifier": "category",

        "LocaleEnglishNameIdentifier": "category",

        "Platform": "category",

        "Processor": "category",

        "OsVer": "category",

        "OsBuild": "category",

        "OsSuite": "category",

        "OsPlatformSubRelease": "category",

        "OsBuildLab": "category",

        "SkuEdition": "category",

        "IsProtected": "float16",

        "AutoSampleOptIn": "int8",

        "PuaMode": "category",

        "SMode": "float16",

        "IeVerIdentifier": "category",

        "SmartScreen": "category",

        "Firewall": "float16",

        "UacLuaenable": "float64",

        "Census_MDC2FormFactor": "category",

        "Census_DeviceFamily": "category",

        "Census_OEMNameIdentifier": "category",

        "Census_OEMModelIdentifier": "category",

        "Census_ProcessorCoreCount": "float16",

        "Census_ProcessorManufacturerIdentifier": "float16",

        "Census_ProcessorModelIdentifier": "category",

        "Census_ProcessorClass": "category",

        "Census_PrimaryDiskTotalCapacity": "float64",

        "Census_PrimaryDiskTypeName": "category",

        "Census_SystemVolumeTotalCapacity": "float64",

        "Census_HasOpticalDiskDrive": "int8",

        "Census_TotalPhysicalRAM": "float32",

        "Census_ChassisTypeName": "category",

        "Census_InternalPrimaryDiagonalDisplaySizeInInches": "float32",

        "Census_InternalPrimaryDisplayResolutionHorizontal": "float32",

        "Census_InternalPrimaryDisplayResolutionVertical": "float32",

        "Census_PowerPlatformRoleName": "category",

        "Census_InternalBatteryType": "category",

        "Census_InternalBatteryNumberOfCharges": "category",

        "Census_OSVersion": "category",

        "Census_OSArchitecture": "category",

        "Census_OSBranch": "category",

        "Census_OSBuildNumber": "category",

        "Census_OSBuildRevision": "category",

        "Census_OSEdition": "category",

        "Census_OSSkuName": "category",

        "Census_OSInstallTypeName": "category",

        "Census_OSInstallLanguageIdentifier": "category",

        "Census_OSUILocaleIdentifier": "category",

        "Census_OSWUAutoUpdateOptionsName": "category",

        "Census_IsPortableOperatingSystem": "int8",

        "Census_GenuineStateName": "category",

        "Census_ActivationChannel": "category",

        "Census_IsFlightingInternal": "float16",

        "Census_IsFlightsDisabled": "float16",

        "Census_FlightRing": "category",

        "Census_ThresholdOptIn": "float16",

        "Census_FirmwareManufacturerIdentifier": "category",

        "Census_FirmwareVersionIdentifier": "category",

        "Census_IsSecureBootEnabled": "int8",

        "Census_IsWIMBootEnabled": "float16",

        "Census_IsVirtualDevice": "float16",

        "Census_IsTouchEnabled": "int8",

        "Census_IsPenCapable": "int8",

        "Census_IsAlwaysOnAlwaysConnectedCapable": "float16",

        "Wdft_IsGamer": "float16",

        "Wdft_RegionIdentifier": "category",

        "HasDetections": "int8"

}
## change nrows

train_df = pd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv', dtype=dtypes, nrows=10000)
test_df = pd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv', dtype=dtypes, nrows=10000)
class ReducerAndCleaner:



    def __init__(self, rm_mostly_empty=False):

        self.remove_mostly_empty = rm_mostly_empty



    def reduce_and_clean(self, df):

        if self.remove_mostly_empty:

            df = self._remove_mostly_empty(df)



        df = self._battery(df)

        df = self._edition(df)

        df = self._other(df)

        df = self.fe(df)



        return df



    def _remove_cols(self, df, cols_to_remove):

        """

        Will remove cols from df



        :param df: dataframe that you want to remove columns from

        :param cols_to_remove: list of columns you would like removed

        :return: the df after the cols are removed

        """

        return df.drop(cols_to_remove, axis=1)



    def _remove_mostly_empty(self, df):

        """

        If any row of the df has one value for over 90% of the items it will get removed



        :param df:

        :return: cleaned df

        """

        good_cols = list(df.columns)

        for col in df.columns:

            rate = df[col].value_counts(normalize=True, dropna=False).values[0]

            if rate > 0.9:

                good_cols.remove(col)



        return df[good_cols]



    def _battery(self, df):

        def group_battery(x):

            x = x.lower()

            if 'li' in x:

                return 1

            else:

                return 0



        df['Census_InternalBatteryType'] = df['Census_InternalBatteryType'].apply(group_battery)

        return df



    def _edition(self, df):

        def rename_edition(x):

            x = x.lower()

            if 'core' in x:

                return 'Core'

            elif 'pro' in x:

                return 'pro'

            elif 'enterprise' in x:

                return 'Enterprise'

            elif 'server' in x:

                return 'Server'

            elif 'home' in x:

                return 'Home'

            elif 'education' in x:

                return 'Education'

            elif 'cloud' in x:

                return 'Cloud'

            else:

                return x



        df['Census_OSEdition'] = df['Census_OSEdition'].astype(str)

        df['Census_OSEdition'] = df['Census_OSEdition'].apply(rename_edition)

        df['Census_OSEdition'] = df['Census_OSEdition'].astype('category')



        df['Census_OSSkuName'] = df['Census_OSSkuName'].astype(str)

        df['Census_OSSkuName'] = df['Census_OSSkuName'].apply(rename_edition)

        df['Census_OSSkuName'] = df['Census_OSSkuName'].astype('category')

        return df



    def _other(self, df):

        """

        Cleaning from: https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated#Data-exploration



        :param df:

        :return:

        """



        df['OsBuildLab'] = df['OsBuildLab'].cat.add_categories(['0.0.0.0.0-0'])

        df['OsBuildLab'] = df['OsBuildLab'].fillna('0.0.0.0.0-0')



        df.loc[df['SkuEdition'] != 'Home', 'SkuEdition'] = 'Pro'

        df['SkuEdition'] = df['SkuEdition'].cat.remove_unused_categories()



        # df.loc[df['SmartScreen'].isnull(), 'SmartScreen'] = 'ExistsNotSet'

        # df.loc[df['SmartScreen'].isin(['RequireAdmin', 'ExistsNotSet', 'Off', 'Warn']) == False, 'SmartScreen'] = 'Prompt'

        #

        # df['SmartScreen'] = df['SmartScreen'].cat.remove_unused_categories()



        top_cats = list(df['Census_MDC2FormFactor'].value_counts().index[:5])

        df.loc[df['Census_MDC2FormFactor'].isin(top_cats) == False, 'Census_MDC2FormFactor'] = 'PCOther'



        df['Census_MDC2FormFactor'] = df['Census_MDC2FormFactor'].cat.remove_unused_categories()



        df.loc[df['Census_PrimaryDiskTypeName'].isin(['HDD', 'SSD']) == False, 'Census_PrimaryDiskTypeName'] = 'UNKNOWN'

        df['Census_PrimaryDiskTypeName'] = df['Census_PrimaryDiskTypeName'].cat.remove_unused_categories()



        df.loc[df['Census_ProcessorManufacturerIdentifier'].isin([5.0, 1.0]) == False, 'Census_ProcessorManufacturerIdentifier'] = 0.0

        df['Census_ProcessorManufacturerIdentifier'] = df['Census_ProcessorManufacturerIdentifier'].astype('category')



        df.loc[df['Census_PowerPlatformRoleName'].isin(['Mobile', 'Desktop', 'Slate']) == False, 'Census_PowerPlatformRoleName'] = 'UNKNOWN'



        df['Census_PowerPlatformRoleName'] = df['Census_PowerPlatformRoleName'].cat.remove_unused_categories()



        top_cats = list(df['Census_OSWUAutoUpdateOptionsName'].value_counts().index[:3])

        df.loc[df['Census_OSWUAutoUpdateOptionsName'].isin(top_cats) == False, 'Census_OSWUAutoUpdateOptionsName'] = 'Off'



        df['Census_OSWUAutoUpdateOptionsName'] = df['Census_OSWUAutoUpdateOptionsName'].cat.remove_unused_categories()



        df.loc[df['Census_GenuineStateName'] == 'UNKNOWN', 'Census_GenuineStateName'] = 'OFFLINE'



        df['Census_GenuineStateName'] = df['Census_GenuineStateName'].cat.remove_unused_categories()



        df.loc[df['Census_ActivationChannel'].isin(['Retail', 'OEM:DM']) == False, 'Census_ActivationChannel'] = 'Volume:GVLK'



        df['Census_ActivationChannel'] = df['Census_ActivationChannel'].cat.remove_unused_categories()

        return df



    def fe(self, df):

        df['EngineVersion_2'] = df['EngineVersion'].apply(lambda x: x.split('.')[2]).astype('category')

        df['EngineVersion_3'] = df['EngineVersion'].apply(lambda x: x.split('.')[3]).astype('category')



        df['AppVersion_1'] = df['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')

        df['AppVersion_2'] = df['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')

        df['AppVersion_3'] = df['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')



        df['AvSigVersion_0'] = df['AvSigVersion'].apply(lambda x: x.split('.')[0]).astype('category')

        df['AvSigVersion_1'] = df['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')

        df['AvSigVersion_2'] = df['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')



        df['OsBuildLab_0'] = df['OsBuildLab'].apply(lambda x: x.split('.')[0]).astype('category')

        df['OsBuildLab_1'] = df['OsBuildLab'].apply(lambda x: x.split('.')[1]).astype('category')

        df['OsBuildLab_2'] = df['OsBuildLab'].apply(lambda x: x.split('.')[2]).astype('category')

        df['OsBuildLab_3'] = df['OsBuildLab'].apply(lambda x: x.split('.')[3]).astype('category')

        # df['OsBuildLab_40'] = df['OsBuildLab'].apply(lambda x: x.split('.')[-1].split('-')[0]).astype('category')

        # df['OsBuildLab_41'] = df['OsBuildLab'].apply(lambda x: x.split('.')[-1].split('-')[1]).astype('category')



        df['Census_OSVersion_0'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[0]).astype('category')

        df['Census_OSVersion_1'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[1]).astype('category')

        df['Census_OSVersion_2'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[2]).astype('category')

        df['Census_OSVersion_3'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[3]).astype('category')



        # https://www.kaggle.com/adityaecdrid/simple-feature-engineering-xd

        df['primary_drive_c_ratio'] = df['Census_SystemVolumeTotalCapacity'] / df['Census_PrimaryDiskTotalCapacity']

        df['non_primary_drive_MB'] = df['Census_PrimaryDiskTotalCapacity'] - df['Census_SystemVolumeTotalCapacity']



        df['aspect_ratio'] = df['Census_InternalPrimaryDisplayResolutionHorizontal'] / df[

            'Census_InternalPrimaryDisplayResolutionVertical']



        df['monitor_dims'] = df['Census_InternalPrimaryDisplayResolutionHorizontal'].astype(str) + '*' + df[

            'Census_InternalPrimaryDisplayResolutionVertical'].astype('str')

        df['monitor_dims'] = df['monitor_dims'].astype('category')



        df['dpi'] = ((df['Census_InternalPrimaryDisplayResolutionHorizontal'] ** 2 + df[

            'Census_InternalPrimaryDisplayResolutionVertical'] ** 2) ** .5) / (

                    df['Census_InternalPrimaryDiagonalDisplaySizeInInches'])



        df['dpi_square'] = df['dpi'] ** 2



        df['MegaPixels'] = (df['Census_InternalPrimaryDisplayResolutionHorizontal'] * df[

            'Census_InternalPrimaryDisplayResolutionVertical']) / 1e6



        df['Screen_Area'] = (df['aspect_ratio'] * (df['Census_InternalPrimaryDiagonalDisplaySizeInInches'] ** 2)) / (

                    df['aspect_ratio'] ** 2 + 1)



        df['ram_per_processor'] = df['Census_TotalPhysicalRAM'] / df['Census_ProcessorCoreCount']



        df['new_num_0'] = df['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / df['Census_ProcessorCoreCount']



        df['new_num_1'] = df['Census_ProcessorCoreCount'] * df['Census_InternalPrimaryDiagonalDisplaySizeInInches']



        df['Census_IsFlightingInternal'] = df['Census_IsFlightingInternal'].fillna(1)

        df['Census_ThresholdOptIn'] = df['Census_ThresholdOptIn'].fillna(1)

        df['Census_IsWIMBootEnabled'] = df['Census_IsWIMBootEnabled'].fillna(1)

        df['Wdft_IsGamer'] = df['Wdft_IsGamer'].fillna(0)



        return df
r = ReducerAndCleaner(rm_mostly_empty=True)

cleaned_train_df = r.reduce_and_clean(df=train_df)

cleaned_test_df = r.reduce_and_clean(df=test_df)
cat_cols = [col for col in cleaned_train_df.columns if col not in ['MachineIdentifier', 'Census_SystemVolumeTotalCapacity', 'HasDetections'] and str(cleaned_train_df[col].dtype) == 'category']

to_encode = []

for col in cat_cols:

    if cleaned_train_df[col].nunique() > 100:

        print(col, cleaned_train_df[col].nunique())

        to_encode.append(col)
!pip install tqdm
from tqdm import tqdm
class Encoder:



    def __init__(self, encode_frequencies=True, encode_labels=True):

        """

        Initializes an encoder object with specificication on how you want to encode the data

        :param encode_frequencies: Boolean indicating whether frequencies should be encoded

        :param encode_labels: Boolean indicating whether labels should be encoded

        """



        self.encode_frequencies = encode_frequencies

        self.encode_labels = encode_labels



    @staticmethod

    def _build_frequency_encoding_dict(variable, train, test):

        """

        Creates a frequency encoding dictionary

        :param variable: The variable under consideration for frequency encoding

        :param train: A Pandas dataframe containing the training data that needs to be frequency encoded

        :param test: A Pandas dataframe containing the test data that needs to be frequency encoded

        :return: A frequency encoding dictionary for the given variable.

        """



        t = pd.concat([train[variable], test[variable]]).value_counts().reset_index()

        t = t.reset_index()

        t.loc[t[variable] == 1, 'level_0'] = np.nan

        t.set_index('index', inplace=True)

        max_label = t['level_0'].max() + 1

        t.fillna(max_label, inplace=True)

        return t.to_dict()['level_0']



    def _encode_freq(self, train, test, frequency_encoded_variables, categorical_columns):

        """

        Performs frequency encoding

        :param train: A Pandas dataframe containing the training data that needs to be frequency encoded

        :param test: A Pandas dataframe containing the test data that needs to be frequency encoded

        :param frequency_encoded_variables: List of features with frequency data

        :param categorical_columns: List of columns with categorical data

        :return: Frequency encoded train data, test data, and pruned categorical columns

        """



        for variable in tqdm(frequency_encoded_variables):

            freq_enc_dict = self._build_frequency_encoding_dict(variable=variable, train=train, test=test)

            train[variable] = train[variable].map(lambda x: freq_enc_dict.get(x, np.nan))

            test[variable] = test[variable].map(lambda x: freq_enc_dict.get(x, np.nan))

            categorical_columns.remove(variable)



        return train, test, categorical_columns



    @staticmethod

    def _encode_labels(train, test, categorical_columns):

        """

        Performs a label encoding

        :param train: A Pandas dataframe containing the training data that needs to be label encoded

        :param test: A Pandas dataframe containing the test data that needs to be label encoded

        :param categorical_columns: List of columns with categorical data

        :return: Label encoded data

        """



        indexer = {}

        for col in tqdm(categorical_columns):

            if col == 'MachineIdentifier':

                continue

            _, indexer[col] = pd.factorize(train[col])



        for col in tqdm(categorical_columns):

            if col == 'MachineIdentifier':

                continue

            train[col] = indexer[col].get_indexer(train[col])

            test[col] = indexer[col].get_indexer(test[col])



        return train, test



    def encode(self, train, test, categorical_columns=None, frequency_encoded_variables=None):

        """

        Encodes the given data according to the user specifications provided

        :param train: A Pandas dataframe containing the training data that needs to be encoded

        :param test: A Pandas dataframe containing the test data that needs to be encoded

        :param categorical_columns: OPTIONAL: List of columns with categorical data

        :param frequency_encoded_variables: OPTIONAL: List of features with frequency data

        :return: Encoded data

        """



        if not self.encode_labels and not self.encode_frequencies:

            return train, test



        if self.encode_frequencies:

            train, test, categorical_columns = self._encode_freq(train=train, test=test,

                                            frequency_encoded_variables=frequency_encoded_variables,

                                            categorical_columns=categorical_columns)



        if self.encode_labels:

            train, test = self._encode_labels(train=train, test=test, categorical_columns=categorical_columns)



        return train, test
e = Encoder(encode_frequencies=True, encode_labels=True)

encoded_train_df, encoded_test_df = e.encode(train=cleaned_train_df, test=cleaned_test_df, categorical_columns=cat_cols, frequency_encoded_variables=to_encode)