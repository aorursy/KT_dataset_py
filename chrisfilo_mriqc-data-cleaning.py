# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def clean_dollarsigns(df):

    replacements = {'_created.$date':'_created', '_id.$oid':'_id', '_updated.$date':'_updated'}

    df = df.rename(columns=replacements)

    return df
df_t1w = pd.read_csv('../input/mriqc/t1w.csv', index_col=0, low_memory=False)

df_t2w = pd.read_csv('../input/mriqc/t2w.csv', index_col=0, low_memory=False)

df_bold = pd.read_csv('../input/mriqc/bold.csv', index_col=0, low_memory=False)



df_t1w = clean_dollarsigns(df_t1w)

df_t2w = clean_dollarsigns(df_t2w)

df_bold = clean_dollarsigns(df_bold)
df_t1w['null_count'] = df_t1w.isnull().sum(1)

df_t2w['null_count'] = df_t2w.isnull().sum(1)

df_bold['null_count'] = df_bold.isnull().sum(1)



# Sort so that rows with fewest nulls and most recent creation are towards the top

df_t1w_unique = df_t1w.sort_values(['null_count','_created'], ascending=[True, False]).drop_duplicates(subset=['provenance.md5sum'])

df_t2w_unique = df_t2w.sort_values(['null_count','_created'], ascending=[True, False]).drop_duplicates(subset=['provenance.md5sum'])

df_bold_unique = df_bold.sort_values(['null_count','_created'], ascending=[True, False]).drop_duplicates(subset=['provenance.md5sum'])
print(df_t1w.shape, df_t1w_unique.shape)

print(df_t2w.shape, df_t2w_unique.shape)

print(df_bold.shape, df_bold_unique.shape)
dl_df = pd.read_csv('../input/datalad-superdataset-metadata/datalad_metadata.csv', low_memory=False)

dl_df['md5sum'] = dl_df.loc[dl_df.metadata__annex__key.notnull(),'metadata__annex__key'].str.split('--').str[-1].str.split('.').str[0]

dl_df['hashing_algo'] = dl_df.loc[dl_df.metadata__annex__key.notnull(),'metadata__annex__key'].str.split('--').str[0].str.split('-').str[0]

dl_df=dl_df.drop_duplicates(subset=['hashing_algo', 'md5sum'])



df_t1w_dl_merge = df_t1w_unique.loc[~df_t1w_unique['provenance.settings.testing'], :].merge(dl_df.loc[dl_df.hashing_algo == 'MD5E', :], indicator=True, how='left', left_on='provenance.md5sum', right_on='hash')

df_t2w_dl_merge = df_t2w_unique.loc[~df_t2w_unique['provenance.settings.testing'], :].merge(dl_df.loc[dl_df.hashing_algo == 'MD5E', :], indicator=True, how='left', left_on='provenance.md5sum', right_on='hash')

df_bold_dl_merge = df_bold_unique.merge(dl_df.loc[dl_df.hashing_algo == 'MD5E', :], indicator=True, how='left', left_on='provenance.md5sum', right_on='md5sum')
df_t1w_dl_merge.groupby('_merge')[['_etag']].count()
df_t2w_dl_merge.groupby('_merge')[['_etag']].count()
df_bold_dl_merge.groupby('_merge')[['_etag']].count()
dsst_datasets = pd.read_csv('../input/datalad-superdataset-metadata/additional_metadata.csv')

print(dsst_datasets.dataset.unique())

df_t1w_merge = df_t1w_dl_merge.merge(dsst_datasets, how='left', left_on='provenance.md5sum', right_on='md5sum', suffixes=('', '_HPC'), indicator='_dsst')

df_t2w_merge = df_t2w_dl_merge.merge(dsst_datasets, how='left', left_on='provenance.md5sum', right_on='md5sum', suffixes=('', '_HPC'), indicator='_dsst')

df_bold_merge = df_bold_dl_merge.merge(dsst_datasets, how='left', left_on='provenance.md5sum', right_on='md5sum', suffixes=('', '_HPC'), indicator='_dsst')
def clean_factor(df, new_names, column):

    mlist = []

    bad = []

    for old_name in df[column]:

        try:

            mlist.append(new_names[old_name])

        except KeyError:

            if pd.notnull(old_name):

                bad.append(old_name)

            mlist.append(np.nan)

    return bad, mlist



model_dict = {'Signa HDe': 'Signa HDe',

              'Signa_HDxt': 'Signa HDxt',

              'Signa HDxt': 'Signa HDxt',

              'SIGNA_HDx': 'Signa HDx',

              'SIGNA_PET_MR': 'Signa PET-MR',

              'SIGNA_EXCITE': 'Signa Excite',

              'Signa Twin Speed Excite HD scanne': 'Signa Excite',

              'SIGNA_Premier': 'Signa Premier',

              'Signa Premier': 'Signa Premier',

              'Signa': 'Signa',

              'GENESIS_SIGNA': 'Signa Genesis',

              'Symphony': 'Symphony',

              'SymphonyTim': 'Symphony',

              'Tim TRIO': 'Tim Trio',

              'TrioTim': 'Tim Trio',

              'Magnetom Trio' : 'Tim Trio',

              'TIM TRIO': 'Tim Trio',

              'MAGNETOM Trio': 'Tim Trio',

              'TRIOTIM': 'Tim Trio',

              'Trio': 'Tim Trio',

              'Trio TIM': 'Tim Trio',

              'Tim Trio': 'Tim Trio',

              'TRIO': 'Tim Trio',

              'MAGNETOM Trio A Tim': 'Tim Trio',

              'TimTrio': 'Tim Trio',

              'TriTim': 'Tim Trio',

              'MAGNETOM and Jerry': 'MAGNETOM and Jerry',

              'Trio Magnetom': 'Tim Trio',

              'Prisma_fit': 'Prisma',

              'Prisma': 'Prisma',

              'Magnetom Skyra Fit': 'Skyra',

              'MAGNETOM Skyra': 'Skyra',

              'Skyra': 'Skyra',

              'Intera': 'Intera',

              'Allegra': 'Allegra',

              'Verio': 'Verio',

              'Avanto': 'Avanto',

              'Sonata': 'Sonata',

              'Espree': 'Espree',

              'SonataVision': 'Sonata Vision',

              'Spectra':'Spectra',

              'Ingenia' : 'Ingenia',

              'DISCOVERY MR750': 'Discovery MR750',

              'DISCOVERY_MR750': 'Discovery MR750',

              'DISCOVERY_MR750w': 'Discovery MR750',

              'Discovery MR750': 'Discovery MR750',

              'MR750': 'Discovery MR750',

              'Achieva_dStream': 'Achieva dStream',

              'Achieva dStream': 'Achieva dStream',

              'Achieva Ds': 'Achieva dStream',

              'Achieva': 'Achieva',

              'Achieva TX': 'Achieva TX',

              'Intera_Achieva': 'Achieva',

              'Intera Achieva': 'Achieva',

              'Philips Achieva': 'Achieva',

              'GEMINI': 'Gemini',

              'Ingenuity': 'Ingenuity',

              'Gyroscan_Intera': 'Gyroscan Intera',

              'Biograph_mMR': 'Biograph mMR',

              'NUMARIS_4': 'Numaris 4',

              'Investigational_Device_7T': 'Investigational 7T',

              'N/A': np.nan,

              '': np.nan,

              'DicomCleaner': np.nan}



mfg_dict = {'Siemens': 'Siemens',

            'SIEMENS': 'Siemens',

            'Simiens': 'Siemens',

            'Siemans': 'Siemens',

            'Simens': 'Siemens',

            'GE': 'GE',

            'G.E.': 'GE',

            'GE MEDICAL SYSTEMS': 'GE',

            'GE_MEDICAL_SYSTEMS': 'GE',

            'General Electric': 'GE',

            'General Electrics': 'GE',

            'GE 3 Tesla MR750': 'GE',

            'Philips':'Philips',

            'Philips Ingenia 3.0T': 'Philips',

            'Philips Achieva Intera 3 T Scanner': 'Philips', 

            'Philips Medical Systems': 'Philips',

           }



def clean_table(res_df):

    # find all the rows with a mfg of 'GE 3 Tesla MR750' and make sure they've got a model value

    res_df.loc[res_df['bids_meta.Manufacturer'] == 'GE 3 Tesla MR750', 'bids_meta.ManufacturersModelName'] = 'Discovery MR750'

    res_df.loc[res_df['bids_meta.Manufacturer'] == 'GE 3 Tesla MR750', 'bids_meta.MagneticFieldStrength'] = 3.0

    res_df.loc[res_df['bids_meta.Manufacturer'] == 'Philips Ingenia 3.0T', 'bids_meta.ManufacturersModelName'] = 'Ingenia'

    res_df.loc[res_df['bids_meta.Manufacturer'] == 'Philips Ingenia 3.0T', 'bids_meta.MagneticFieldStrength'] = 3.0

    res_df.loc[res_df['bids_meta.Manufacturer'] == 'Philips Achieva Intera 3 T Scanner', 'bids_meta.ManufacturersModelName'] = 'Achieva'

    res_df.loc[res_df['bids_meta.Manufacturer'] == 'Philips Achieva Intera 3 T Scanner', 'bids_meta.MagneticFieldStrength'] = 3.0



    bad, mlist = clean_factor(res_df, model_dict, 'bids_meta.ManufacturersModelName')

    assert len(pd.unique(bad)) == 0

    res_df['bids_meta.ManufacturersModelName'] = mlist



    bad, mlist = clean_factor(res_df, mfg_dict , 'bids_meta.Manufacturer')

    assert len(pd.unique(bad)) == 0

    res_df['bids_meta.Manufacturer'] = mlist

    

    res_df['dataset_dl'] = res_df.path.str.split('/').str[6]

    res_df['subdataset_dl'] = res_df.path.str.split('/').str[7]

    res_df.loc[(res_df.dataset_dl == "indi"), 'subdataset_dl'] = res_df.path.str.split('/').str[7:9].str.join('__')

    res_df['dataset'] = res_df.dataset.str.lower()

    res_df.dataset = res_df.dataset.fillna(res_df.dataset_dl)

    try:

        res_df.dataset = res_df.dataset.fillna(res_df.dataset_lr)

    except AttributeError:

        pass

    res_df.loc[res_df.dataset == 'openneuro', 'subdataset'] = res_df.path_HPC.str.split('/').str[4]

    res_df.subdataset = res_df.subdataset.fillna(res_df.subdataset_dl)

    return res_df

df_t1w_merge = clean_table(df_t1w_merge)

df_t2w_merge = clean_table(df_t2w_merge)

df_bold_merge = clean_table(df_bold_merge)
df_bold_merge.loc[df_bold_merge['bids_meta.EchoTime'] > df_bold_merge['bids_meta.RepetitionTime'], 'bids_meta.EchoTime'] *= 1e-3
df_t1w_merge.to_csv('t1w.csv', index=False)

df_t2w_merge.to_csv('t2w.csv', index=False)

df_bold_merge.to_csv('bold.csv', index=False)