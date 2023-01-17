import pandas as pd
import os
!ls ../input/cpe-data
ACS_variable_descriptions = pd.read_csv("../input/cpe-data/{}.csv".format("ACS_variable_descriptions"), index_col=0, header=None)
ACS_variable_descriptions.shape
ACS_variable_descriptions.head()
ACS_variable_descriptions.loc['HC01_VC03']
!ls -R ../input/cpe-data/Dept_11-00091
def read_metadata(file):
    return pd.read_csv(file, index_col=0, header=None)
    
def read_data(file):
    return pd.read_csv(file, index_col=[0, 1, 2], header=[0,1], na_values=['(X)', '-', '**'])

def read_prepped(file):
    return pd.read_csv(file, header=[0,1])

def ingnore_DS_Store(directory):
    return filter(lambda f: f != '_DS_Store', os.listdir(directory))

def collect_info_for_dep (dept_dir):
    """
    This function collects the '.csv' files into pandas dataframes.
    The return value is a hash where the keys refer to the original file names.
    """
    base_dir = "../input/cpe-data/{}".format(dept_dir)
    data_directories = list(filter(lambda f: f.endswith("_data"), os.listdir(base_dir)))
    info = {'dept' : dept_dir}
    assert len(data_directories) == 1, "found {} data directories".format(len(data_directories))
    for dd in data_directories:
        directory = "{}/{}".format(base_dir, dd)
        dd_directories = ingnore_DS_Store(directory)
        #print(dd_directories)
        for ddd in dd_directories:
            ddd_directory = "{}/{}".format(directory, ddd)
            files = list(ingnore_DS_Store(ddd_directory))
            #print(files)
            assert len(files) == 2, "found {} files in {}".format(len(files), directory)
            full_file_names = ["{}/{}".format(ddd_directory, file) for file in files]
            dataframes = [read_metadata(file) if file.endswith('_metadata.csv') else read_data(file) for file in full_file_names]
            info[ddd] = dict(zip(files, dataframes))
    prepped_files = list(filter(lambda f: f.endswith("_prepped.csv"), os.listdir(base_dir)))
    for pf in prepped_files:
        info[pf] = read_prepped("{}/{}".format(base_dir, pf))
    return info
Dept_11_00091_info = collect_info_for_dep('Dept_11-00091')

Dept_11_00091_info.keys()
Dept_11_00091_info['11-00091_ACS_education-attainment'].keys()
Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_metadata.csv'].info()
Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_metadata.csv'].head()
Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].info()
Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].head()
Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].loc['1400000US25027700100', 'HC01_MOE_VC02']
desc = Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].describe()
desc.loc[:, desc.loc['count', :] > 0]
Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].shape
# 768 - 606 columns are empty (all values are -NA-)
def investigate_dept(dept):
    print(dept['dept'])
    print('=' * 20)
    print(dept.keys())
investigate_dept(Dept_11_00091_info)
department_names = [
    'Dept_11-00091',
    'Dept_23-00089',
    'Dept_35-00103',
    'Dept_37-00027',
    'Dept_37-00049',
    'Dept_49-00009',
]

departments = {dep: collect_info_for_dep(dep) for dep in department_names}
for dep in departments.keys():
    investigate_dept(departments[dep])
    print()
!ls -R ../input/cpe-data/Dept_35-00103
departments['Dept_37-00049']['37-00049_UOF-P_2016_prepped.csv']
departments['Dept_37-00049']['37-00049_UOF-P_2016_prepped.csv'].info()
