# Libs to deal with tabular data

import numpy as np

import pandas as pd



# Plotting packages

import seaborn as sns

sns.set_context("notebook")

sns.set_style("whitegrid")

import matplotlib.pyplot as plt



# To display stuff in notebook

from IPython.display import display, Markdown



# Auxiliar packages

import re
aircrafts = pd.read_csv('../input/abril.xls', sep='\t', encoding='latin-1', skiprows=1)

classes_map = pd.read_csv('../input/usage_classification.csv')

anac_classes = pd.read_csv('../input/anac_class.csv')

icao_classes = pd.read_csv('../input/ICAO_Aircraft_Type_Designators_2017-05-15.csv')
print('Shape:', aircrafts.shape)

aircrafts.sample(5, random_state=42)
aircrafts = aircrafts.rename(columns={

    'MARCA':'tail_number',

    'PROPRIETARIO':'owner',

    'OUTROS_PROPRIETARIOS':'other_owners',

    'SG_UF':'owner_state',

    'CPF_CNPJ':'owner_id',

    'NM_OPERADOR':'operator',

    'OUTROS_OPERADORES':'other_operators',

    'UF_OPERADOR':'operator_state',

    'CPF_CGC':'operator_id',

    'NR_CERT_MATRICULA':'registration_number',

    'NR_SERIE':'serial_number',

    'CD_CATEGORIA':'usage_classification',

    'CD_TIPO':'certification_type',

    'DS_MODELO':'model',

    'NM_FABRICANTE':'manufacturer',

    'CD_CLS':'anac_type_classification',

    'NR_PMD':'max_weight_takeoff',

    'CD_TIPO_ICAO':'icao_type_classification',

    'NR_TRIPULACAO_MIN':'min_number_crew',

    'NR_PASSAGEIROS_MAX':'max_number_passengers',

    'NR_ASSENTOS':'seats',

    'NR_ANO_FABRICACAO':'year_manufacture',

    'DT_VALIDADE_IAM':'exp_date_annual_inspection',

    'DT_VALIDADE_CA':'exp_date_special_inspection',

    'DT_CANC':'date_registry_cancellation',

    'DS_MOTIVO_CANC':'cancellation_cause',

    'CD_INTERDICAO':'airworthiness',

    'CD_MARCA_NAC1':'prev_tail_number_1',

    'CD_MARCA_NAC2':'prev_tail_number_2',

    'CD_MARCA_NAC3':'prev_tail_number_3',

    'CD_MARCA_ESTRANGEIRA':'prev_int_tail_number',

    'DS_GRAVAME':'use_rights'

})



aircrafts = aircrafts.drop_duplicates()
aircrafts.dtypes
missing = aircrafts.isnull().sum().div(aircrafts.shape[0]).sort_values(ascending=False)



plt.figure(figsize=(9,9))

plt.title('Missing values (%)', fontsize=16, fontweight='bold')

sns.barplot(x = missing.values, y = missing.index, palette='Blues_r')

plt.xlim(0,1)

plt.show()
# Checking if our primary key is duplicated

duplicated_mask = aircrafts.duplicated(['tail_number'], keep=False)

print('We have {} rows with duplicated primary key.'.format(duplicated_mask.sum()))



# Getting duplicated rows

duplicated_rows = aircrafts[duplicated_mask]
# Analyzing in which columns there is difference.

duplicated_rows.loc[:, duplicated_rows.iloc[0,:] != duplicated_rows.iloc[1,:]]
# Dropping row

aircrafts = aircrafts.drop(10304, axis=0)
# Checking tail number size

aircrafts['tail_number'].str.len().value_counts()
# Checking if the two first letters of the tail number represents Brazil

aircrafts['tail_number'].apply(lambda x: x[:2]).unique()
# Setting tail number as index

aircrafts = aircrafts.set_index('tail_number')
# Checking if there are numbers where it should be names.

print('Owner:', aircrafts['owner'].str.contains('\b[0-9]+\b').sum())

print('Other owners:', aircrafts['other_owners'].str.contains('\b[0-9]+\b').sum())

print('Operator:', aircrafts['operator'].str.contains('\b[0-9]+\b').sum())

print('Other operators:', aircrafts['other_operators'].str.contains('\b[0-9]+\b').sum())
# Checking if there is more than one owner in the owner column

aircrafts[aircrafts['owner'].str.contains(';').fillna(False)]
# Checking if there is more than one operator in the operator column

aircrafts[aircrafts['operator'].str.contains(';').fillna(False)]
# Removing &#8203; 

aircrafts['owner'] = aircrafts['owner'].str.replace('&#8203;', '')

aircrafts['other_owners'] = aircrafts['other_owners'].str.replace('&#8203;', '')

aircrafts['operator'] = aircrafts['operator'].str.replace('&#8203;', '')

aircrafts['other_operators'] = aircrafts['other_operators'].str.replace('&#8203;', '')
# Create a new table given a column containing multiple values.

def create_aux_table(df, col):

    aux_table = df[col].str.split(';', expand=True).stack().rename(col)

    aux_table.index = aux_table.index.droplevel(1)

    return aux_table



other_owners = create_aux_table(aircrafts, 'other_owners').reset_index()

other_operators = create_aux_table(aircrafts, 'other_operators').reset_index()



# Showing a sample of the new table

other_owners.head()
# Counting number of values. I'm filling NaNs with -1 to make the transformation works

aircrafts['num_other_owners'] = aircrafts['other_owners'].str.count(';').fillna(-1) + 1

aircrafts['num_other_operators'] = aircrafts['other_operators'].str.count(';').fillna(-1) + 1



aircrafts = aircrafts.drop(['other_owners', 'other_operators'], axis=1)
# Checking if state codes are correct

aircrafts['owner_state'].unique()
# Checking if state codes are correct

aircrafts['operator_state'].unique()
aircrafts[aircrafts['owner_state'] == 'GB']
aircrafts['owner_state'] = aircrafts['owner_state'].replace('GB', 'RJ')

aircrafts['operator_state'] = aircrafts['operator_state'].replace('GB', 'RJ')
# CPF validator. It gets an int and returns a boolean telling whether the CPF is valid or not.

def cpf_validator(number):

    number = list(map(int, str(int(number)).rjust(11, '0'))) # left padding with zeros to get 11 digits

    root = number[:9]

    d1, d2 = number[9], number[10]

    

    d1_, d2_ = 0, 0

    for idx, digit in enumerate(reversed(root)):

        d1_ += digit * (9 - (idx % 10))

        d2_ += digit * (9 - ((idx + 1) % 10))

        

    d1_ = (d1_ % 11) % 10

    if(d1_ != d1):

        return False

    

    d2_ = d2_ + (d1_ * 9)

    d2_ = (d2_ % 11) % 10

    return d2_ == d2



# CNPJ validator. It gets an int and returns a boolean telling whether the CNPJ is valid or not.

def cnpj_validator(number):

    number = list(map(int, str(int(number)).rjust(14, '0'))) # left padding with zeros to get 11 digits

    root = number[:12]

    d1, d2 = number[12], number[13]

    

    d1_, d2_ = 0, 0

    d1_ += 5*root[0] + 4*root[1]  + 3*root[2]  + 2*root[3]

    d1_ += 9*root[4] + 8*root[5]  + 7*root[6]  + 6*root[7]

    d1_ += 5*root[8] + 4*root[9] + 3*root[10] + 2*root[11]

    d1_ = 11 - (d1_ % 11)

    d1_ = 0 if d1_ >= 10 else d1_

    

    if(d1_ != d1):

        return False



    d2_ += 6*root[0] + 5*root[1]  + 4*root[2]  + 3*root[3]

    d2_ += 2*root[4] + 9*root[5]  + 8*root[6]  + 7*root[7]

    d2_ += 6*root[8] + 5*root[9] + 4*root[10] + 3*root[11]

    d2_ += 2*d1_

    d2_ = 11 - (d2_ % 11)

    d2_ = 0 if d2_ >= 10 else d2_

    

    return d2_ == d2



def check_id(number):

    if(pd.isnull(number)):

        return np.nan

    cpf_check = cpf_validator(number)

    cnpj_check = cnpj_validator(number)

    if(cpf_check and cnpj_check):

        return 'Not identified'

    elif(not cpf_check and not cnpj_check):

        return np.nan

    elif(cpf_check):

        return 'Natural'

    else:

        return 'Legal'
# Replacing IDs equals to 0 with NaN

aircrafts['owner_id'] = aircrafts['owner_id'].replace(0.0, np.nan)

aircrafts['operator_id'] = aircrafts['operator_id'].replace(0.0, np.nan)

# Apply validation check

aircrafts['owner_type'] = aircrafts['owner_id'].apply(check_id)

aircrafts['operator_type'] = aircrafts['operator_id'].apply(check_id)
print('Total of not identified types of person:', 

          (aircrafts['owner_type'].isnull().sum() + (aircrafts['owner_type'] == 'Not identified').sum()) / aircrafts.shape[0])

print('Nan:', aircrafts['owner_type'].isnull().sum())

aircrafts['owner_type'].value_counts()
print('Total of not identified types of person:', 

          (aircrafts['operator_type'].isnull().sum() + (aircrafts['operator_type'] == 'Not identified').sum()) / aircrafts.shape[0])

print('Nan:', aircrafts['operator_type'].isnull().sum())

aircrafts['operator_type'].value_counts()
gb = aircrafts.groupby('owner_id')['owner'].nunique() 

gb.sort_values(ascending=False)
# Percentage of owner_id's with more than one name.

gb.ge(2).sum() / len(gb)
aircrafts.loc[aircrafts['owner_id'] == gb.sort_values(ascending=False).index[0], ['owner_id', 'owner']]
# This column should be unique to each aircrafts, so I'm checking this.

reg_counts = aircrafts['registration_number'].value_counts()

reg_counts
reg_counts.describe()
# Percentage of registration numbers which are assigned to more than one aircraft.

reg_counts.ge(2).sum() / len(reg_counts)
aircrafts.loc[aircrafts['registration_number'] == 0.0, [

    'owner', 'owner_state', 'operator', 'operator_state', 'registration_number',

    'serial_number', 'model', 'year_manufacture', 'airworthiness',

    'anac_type_classification', 'manufacturer'

]]
aircrafts.loc[aircrafts['registration_number'] == 10000000.0, [

    'owner', 'owner_state', 'operator', 'operator_state', 'registration_number',

    'serial_number', 'model', 'year_manufacture', 'airworthiness',

    'anac_type_classification', 'manufacturer'

]]
# Examining an example

aircrafts.loc[aircrafts['registration_number'] == reg_counts.index[2], [

    'owner', 'owner_state', 'operator', 'operator_state', 'registration_number',

    'serial_number', 'model', 'year_manufacture', 'airworthiness',

    'anac_type_classification', 'manufacturer'

]]
# Examining an example

aircrafts.loc[aircrafts['registration_number'] == reg_counts.index[3], [

    'owner', 'owner_state', 'operator', 'operator_state', 'registration_number',

    'serial_number', 'model', 'year_manufacture', 'airworthiness',

    'anac_type_classification', 'manufacturer'

]]
aircrafts['airworthiness'].value_counts()
# Checking how many rows have more than one status

aircrafts['airworthiness'].str.count('[a-zA-Z]').gt(1.0).sum() / len(aircrafts)
aircrafts['airworthiness'].str.contains('M').sum() / len(aircrafts)
aircrafts = aircrafts.loc[~ aircrafts['airworthiness'].str.contains('M').astype(bool), :]
aircrafts['registration_number'].value_counts()
reg_problems = aircrafts.groupby('registration_number').apply(len) - aircrafts.groupby('registration_number')['serial_number'].nunique()
reg_problems[reg_problems.gt(0.0)]
aircrafts['serial_number'].nunique()
serial_reg = aircrafts['serial_number'].value_counts()

serial_reg
# Percentage of serial numbers assigned to more than one aircraft.

serial_reg.ge(2).sum() / len(serial_reg)
# Examining an example

aircrafts.loc[aircrafts['serial_number'] == '002', [

    'owner', 'owner_state', 'operator', 'operator_state', 'registration_number',

    'serial_number', 'model', 'year_manufacture', 'airworthiness',

    'anac_type_classification', 'manufacturer'

]].head(10)
aircrafts['usage_classification'].value_counts()
# Number of rows with number codes

aircrafts['usage_classification'].str.contains('[1-9]').astype(bool).sum()
not_mapped_mask = ~aircrafts['usage_classification'].isin(classes_map.iloc[:, 1])

aircrafts[not_mapped_mask & aircrafts['usage_classification'].notnull()]
# Preparing the auxiliar table to be joined

classes_map = classes_map.rename(columns={

    'Código Categoria':'code',

    'Tipo':'usage_classification',

    'Propriedade':'property_type'

})



classes_map = classes_map.loc[:, [

    'code', 'usage_classification', 'property_type'

]]



classes_map = classes_map.drop_duplicates().set_index('code')
# Adding more useful information about usage classification

aircrafts = aircrafts.join(classes_map, on='usage_classification', how='left', rsuffix='_new')

aircrafts = aircrafts.drop('usage_classification', axis=1).rename(columns={'usage_classification_new':'usage_classification'})
aircrafts['anac_type_classification'].value_counts()
# Checking if there are unmapped

not_mapped_mask = ~aircrafts['anac_type_classification'].isin(anac_classes.iloc[:, 0])

aircrafts.loc[not_mapped_mask & aircrafts['anac_type_classification'].notnull(), 'anac_type_classification'].unique()
anac_classes = anac_classes.rename(columns={

    'CLASSE':'anac_type_classification',

    'Tipo de Pouso':'landing_type',

    'Número de Motores':'engine_count',

    'Tipo de Motor':'engine_type'

})



anac_classes = anac_classes.set_index('anac_type_classification')
anac_classes.loc['L1E', :] = ['Conventional Landing', 1, 'Electric']

anac_classes.loc['A1T', :] = ['Anfiby', 1, 'Turbo Propeller']
aircrafts = aircrafts.join(anac_classes, on='anac_type_classification', how='left', rsuffix='_left')
letter_mask = aircrafts['max_weight_takeoff'].str.contains('[a-zA-Z]').astype(bool)

aircrafts.loc[letter_mask & aircrafts['max_weight_takeoff'].notnull(), 'max_weight_takeoff']
aircrafts['max_weight_takeoff'] = aircrafts['max_weight_takeoff'].str.replace('[ a-zA-Z]', '').str.replace(',', '.')

aircrafts['max_weight_takeoff'] = pd.to_numeric(aircrafts['max_weight_takeoff'])
aircrafts['icao_type_classification'].value_counts()
aircrafts['icao_type_classification'].isnull().sum()
# Fixing auxiliar table

icao_classes = icao_classes.rename(columns={

    'Type Designator':'type',

    'Manufacturer':'manufacturer',

    'Model':'model',

    'Description':'landing_type',

    'Engine Type':'engine_type',

    'Engine Count':'engine_count'

})



icao_classes['landing_type'] = icao_classes['landing_type'].replace({

    'LandPlane':'Land Plane',

    'SeaPlane':'Sea Plane'

})



# Dropping manufacturer and model because they can have multiple values to the same code.

# Also dropping final row (error).

icao_classes = icao_classes.drop(['manufacturer', 'model', 'WTC'], axis=1).drop(9925, axis=0)



icao_classes = icao_classes.drop_duplicates().set_index('type')
# Joining tables

aircrafts = aircrafts.join(icao_classes, on='icao_type_classification', how='left', rsuffix='_new')



# Combining columns

aircrafts['landing_type'] = aircrafts['landing_type_new'].combine_first(aircrafts['landing_type'])

aircrafts['engine_type'] = aircrafts['engine_type_new'].combine_first(aircrafts['engine_type'])

aircrafts['engine_count'] = pd.to_numeric(aircrafts['engine_count_new'].combine_first(aircrafts['engine_count']))



aircrafts = aircrafts.drop(['landing_type_new', 'engine_type_new', 'engine_count_new'], axis=1)
# Checking if there are textual information

aircrafts.loc[aircrafts['exp_date_annual_inspection'].str.contains('[^0-9]').astype(bool), 'exp_date_annual_inspection'].value_counts()
# Showing the number os digits

dates = aircrafts.loc[~aircrafts['exp_date_annual_inspection'].str.contains('[^0-9]').astype(bool), 'exp_date_annual_inspection']

date_size = dates.apply(lambda x: len(x) if pd.notnull(x) else np.nan)

date_size.value_counts()
dates[date_size == 7]
# Fixing mistyped date

aircrafts.loc['PTKQE', 'exp_date_annual_inspection'] = '05022019'
dates[date_size == 6]
# Fixing status words and creating the status column.

def get_status(x):

    if(pd.isnull(x)):

        return np.nan

    else:

        x = x.upper()

        

    if('BORDO' in x):

        return 'ABORDO'

    elif('SENT' in x):

        return 'ISENTO'

    elif(re.search('[^0-9]', x)):

        return np.nan

    else:

        return 'NORMAL'

    

aircrafts['annual_inspec_status'] = aircrafts['exp_date_annual_inspection'].apply(get_status)
# Fixing dates

def fix_dates(x):

    if(pd.isnull(x)):

        return np.nan

    elif(len(x) == 6):

        year = int(x[-2:])

        if(30 <= year <= 99):

            return x[:4] + '19' + x[-2:]

        else:

            return x[:4] + '20' + x[-2:]

    else:

        return x



aircrafts.loc[aircrafts['exp_date_annual_inspection'].str.contains('[^0-9]').astype(bool), 'exp_date_annual_inspection'] = np.nan

aircrafts['exp_date_annual_inspection'] = pd.to_datetime(aircrafts['exp_date_annual_inspection'].apply(fix_dates), format='%d%m%Y')
# Checking if there are textual information

aircrafts.loc[aircrafts['exp_date_special_inspection'].str.contains('[^0-9]').astype(bool), 'exp_date_special_inspection'].value_counts()
# Showing the number os digits

dates = aircrafts.loc[~aircrafts['exp_date_special_inspection'].str.contains('[^0-9]').astype(bool), 'exp_date_special_inspection']

date_size = dates.apply(lambda x: len(x) if pd.notnull(x) else np.nan)

date_size.value_counts()
dates[date_size == 6]
# Fixing status words and creating the status column.

def get_status(x):

    if(pd.isnull(x)):

        return np.nan

    else:

        x = x.upper()

        

    if('RESRA' in x):

        return 'RESRAB'

    elif('SENTO' in x):

        return 'ISENTO'

    elif(re.search('[^0-9]', x)):

        return np.nan

    else:

        return 'NORMAL'

    

aircrafts['special_inspec_status'] = aircrafts['exp_date_special_inspection'].apply(get_status)
# Fixing dates

def fix_dates(x):

    if(pd.isnull(x)):

        return np.nan

    elif(len(x) == 6):

        year = int(x[-2:])

        if(30 <= year <= 99):

            return x[:4] + '19' + x[-2:]

        else:

            return x[:4] + '20' + x[-2:]

    else:

        return x



aircrafts.loc[aircrafts['exp_date_special_inspection'].str.contains('[^0-9]').astype(bool), 'exp_date_special_inspection'] = np.nan

aircrafts['exp_date_special_inspection'] = pd.to_datetime(aircrafts['exp_date_special_inspection'].apply(fix_dates), format='%d%m%Y')
aircrafts['date_registry_cancellation'].notnull().sum()
# Checking if there are textual information

aircrafts.loc[aircrafts['date_registry_cancellation'].str.contains('[^0-9]').astype(bool), 'date_registry_cancellation'].value_counts()
aircrafts['cancellation_cause'].notnull().sum()
aircrafts['cancellation_cause'].value_counts()
aircrafts.loc[aircrafts['date_registry_cancellation'].notnull(), :]
aircrafts = aircrafts.drop('PTJRT', axis=0)

aircrafts = aircrafts.drop(['date_registry_cancellation', 'cancellation_cause'], axis=1)
aircrafts['prev_tail_number_1'].notnull().sum()
aircrafts['prev_int_tail_number'].notnull().sum()
aircrafts[col].notnull()
tail_cols = [

    'prev_tail_number_1',

    'prev_tail_number_2',

    'prev_tail_number_3',

    'prev_int_tail_number'

]



stack = []

reseted = aircrafts.reset_index()

for col in tail_cols:

    prev = reseted.loc[reseted[col].notnull(), ['tail_number', col]]

    prev = prev.rename(columns={col:'prev_tail'})

    prev['nationality'] = 'International' if 'int' in col else 'National' 

    stack.append(prev)

    

prev_tails = pd.concat(stack, axis=0)
prev_tails
aircrafts = aircrafts.drop(tail_cols, axis=1) 
aircrafts['use_rights'].notnull().sum()
aircrafts['use_rights'].value_counts()
aircrafts.head(5)
aircrafts.shape
aircrafts.dtypes
aircrafts.isnull().sum().sort_values()
# Organizing columns in a way that makes sense

aircrafts = aircrafts.loc[:,[

    # owner

    'owner',

    'owner_id',

    'owner_type',

    'owner_state',

    'num_other_owners',

    'property_type',

    # operator

    'operator',

    'operator_id',

    'operator_type',

    'operator_state',

    'num_other_operators',

    'use_rights',

    # registration/legal info

    'airworthiness',

    'registration_number',

    'certification_type',

    'anac_type_classification',

    'icao_type_classification',

    'usage_classification',

    'annual_inspec_status',

    'exp_date_annual_inspection',

    'special_inspec_status',

    'exp_date_special_inspection',

    # aircraft info

    'model',

    'manufacturer',

    'serial_number',

    'year_manufacture',

    'max_weight_takeoff',

    'min_number_crew',

    'max_number_passengers',

    'seats',

    'landing_type',

    'engine_count',

    'engine_type'

]]
aircrafts['landing_type'].value_counts()
aircrafts['landing_type'] = aircrafts['landing_type'].replace({

    'Land Plane':'Conventional Landing',

    'Anfiby':'Amphibian'

})
aircrafts['engine_type'].value_counts()
aircrafts['engine_type'] = aircrafts['engine_type'].replace({

    'Conventional':'Piston',

    'Jat / Turbofan':'Jet',

    'Turbo Propeller':'Turboprop/Turboshaft',

    'Turbo axy':'Turboprop/Turboshaft'

})
aircrafts['manufacturer'] = aircrafts['manufacturer'].str.replace(';', '')
other_owners.to_csv('/kaggle/working/other_owners.csv', index=False)

other_operators.to_csv('/kaggle/working/other_operators.csv', index=False)

prev_tails.to_csv('/kaggle/working/prev_tails.csv', index=False)

aircrafts.to_csv('/kaggle/working/aircrafts.csv', index=False)